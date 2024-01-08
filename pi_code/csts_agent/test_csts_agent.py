
import random
import subprocess
import time

import numpy as np
from smarts.core.agent import Agent
from smarts.core.controllers.action_space_type import ActionSpaceType
from smarts.core.road_map import RoadMap
from smarts.core.sumo_road_network import SumoRoadNetwork
from smarts.core.observations import EgoVehicleObservation, Observation, VehicleObservation

import rospy
import threading
from csts_msgs.msg import perception_prediction, prediction_traj, object_prediction, ego_state, map_lanes, lane
from geometry_msgs.msg import Vector3
import sys
from pathlib import Path
import time
from typing import Dict, Final

import gymnasium as gym


AGENT_ID: Final[str] = "Agent"


from smarts.core.road_map import RoadMap
from smarts.core.sumo_road_network import SumoRoadNetwork
from smarts.env.gymnasium.hiway_env_v1 import HiWayEnvV1
from smarts.env.gymnasium.wrappers.single_agent import SingleAgent
from smarts.env.utils.observation_conversion import ObservationOptions, ObservationSpacesFormatter

SMARTS_REPO_PATH = Path(__file__).parents[2].absolute()
sys.path.insert(0, str(SMARTS_REPO_PATH))
from examples.tools.argument_parser import minimal_argument_parser
from smarts.core.agent import Agent
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.controllers.action_space_type import ActionSpaceType
from smarts.core.utils.episodes import episodes
from smarts.sstudio.scenario_construction import build_scenarios
from smarts.core.observations import Observation

class CSTSAgent(Agent):
    def __init__(self, init_ros=True, planning=True):
        self.action_type = ActionSpaceType.RelativeTargetPose
        if init_ros:
            rospy.init_node('csts_agent', anonymous=True)
        self.pub_perception_prediction = rospy.Publisher("map_server/perception_prediction", perception_prediction, queue_size=1)
        self.pub_ego_state = rospy.Publisher("map_server/ego_state", ego_state, queue_size=1)
        self.pub_lanes = rospy.Publisher("/map_server/map_lanes", map_lanes, queue_size=1)
        self.sub_ego_state = rospy.Subscriber("/search_server/ego_state", ego_state, callback=self.ego_state_callback, queue_size=1)
        
        self.ego_state_msg = ego_state()
        self.ego_state_msg_sub = ego_state()
        self.map_lanes_msg = map_lanes()
        self.perception_prediction_msg = perception_prediction()
        
        
        self.spin_thread = threading.Thread(target = self.thread_job)
        self.roslaunch_cmd = "roslaunch contingency_st_search test_search.launch"
        
        self.planning = planning
        if self.planning:
            self.roslaunch_proc = subprocess.Popen(self.roslaunch_cmd.split(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)        
        else:
            self.roslaunch_proc = subprocess.Popen("ls", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)       
        rospy.sleep(1)
            
        self.spin_thread.start()
        
        self.ego_state_msg_sub: ego_state = None
        self.receive_ego_state = False
        self.dt = 0.1
        self.timestamp = 0
        
    def __del__(self):
        self.roslaunch_proc.terminate()
    
    def thread_job(self):
        rospy.spin()
        
    def ego_state_callback(self, state: ego_state):
        if self.receive_ego_state == True:
            return
        self.ego_state_msg_sub = state
        self.receive_ego_state = True
    
    def act(self, obs, map:RoadMap):
        print(f"timestamp: {self.timestamp}")
        # print(obs.keys(), )
        # dict_keys(['active', 'steps_completed', 'distance_travelled', 'ego_vehicle_state', 'events', 'drivable_area_grid_map', 'lidar_point_cloud', 'neighborhood_vehicle_states', 'occupancy_grid_map', 'top_down_rgb', 'waypoint_paths', 'signals'])
        
        ego = obs.get("ego_vehicle_state")
        if not ego:
            pass
        else:
            # print(ego.keys())
            # dict_keys(['angular_velocity', 'box', 'heading', 'lane_id', 'lane_index', 'linear_velocity', 'position', 'speed', 'steering', 'yaw_rate', 'mission', 'angular_acceleration', 'angular_jerk', 'linear_acceleration', 'linear_jerk'])  
            ego["bounding_box"] = ego.pop("box")
            ego["id"] = 0
            ego["road_id"] = None
               
            # print(f"ego before:{type(ego)}", end="")  
            ego = EgoVehicleObservation(**ego)  
            # print(f"after={type(ego)}")  
            
        objs = obs.get("neighborhood_vehicle_states")
        if not objs:
            pass
        else:
            # print(objs.keys())
            # dict_keys(['box', 'heading', 'id', 'interest', 'lane_id', 'lane_index', 'position', 'speed'])
            objs["bounding_box"] = objs.pop("box")
            len_objs = len(objs["bounding_box"])
            # print(len(objs["bounding_box"]), type(objs["bounding_box"]))
            objs["road_id"] = np.ndarray([len_objs,1], str)
            
            # print(f"objs before:{type(objs)}", end="")  
            
            keys = objs.keys()
            new_objs: Dict[str, VehicleObservation] = {}
            for i in range(len(objs["id"])):
                id = objs["id"][i]
                if(id == ""):
                    # print("empty")
                    continue
                obj = VehicleObservation(objs["id"][i], objs["position"][i], objs["speed"][i], objs["heading"][i], objs["bounding_box"][i], objs["lane_id"][i], objs["lane_index"][i], objs["road_id"][i])
                box = objs["bounding_box"][i]
                # print(f"i = {i}, obj_id: {id}, box: {box}, type: {type(obj)}")
                new_objs[id] = obj
            
            objs = VehicleObservation(**objs)
            # print(f"after={type(objs)}")
            # raise
        
        self.ego_state_msg = self.get_ego_state(ego, map)
        self.map_lanes_msg = self.get_map_lanes(ego, map)
        self.perception_prediction_msg = self.get_pp(objs, map)
        
        
        self.publish_all()
        if self.planning:
            while(self.receive_ego_state == False and rospy.is_shutdown() == False):
                continue
        else:
            pass
        
        action = self.get_action(self.ego_state_msg_sub)
        
        
        dx, dy, dyaw = 0,0,0
        action = (dx, dy, dyaw)
        self.timestamp += 1
        return action
    
    def const_v_action(self, obs):
        pass
    
    def get_ego_state(self, ego: EgoVehicleObservation, map: RoadMap):
        ego_state = self.ego_state_msg
        
        
        return ego_state
    
    def get_map_lanes(self, ego: EgoVehicleObservation, map: RoadMap):
        map_lanes = self.map_lanes_msg
        
        return map_lanes 
    
    def get_pp(self, objs: VehicleObservation, map: RoadMap):
        pp = self.perception_prediction_msg
        
        return pp 
    
    
    def publish_all(self,):
        self.pub_ego_state.publish(self.ego_state_msg)
        self.pub_lanes.publish(self.map_lanes_msg)
        self.pub_perception_prediction.publish(self.perception_prediction_msg)
          
    def get_action(self, ego_state_msg_sub: ego_state):
        dx, dy, dyaw = 0,0,0
        action = (dx, dy, dyaw)
        
        
        
        return action
    
    

def main(scenarios, headless, num_episodes, max_episode_steps=None):
    # This interface must match the action returned by the agent
    agent_interface = AgentInterface.from_type(
        AgentType.Full, 
        max_episode_steps=max_episode_steps, 
    )
    # agent_interface.action = ActionSpaceType.Lane
    agent_interface.action = ActionSpaceType.RelativeTargetPose   
    
    agent_interfaces = {AGENT_ID: agent_interface}
    env:HiWayEnvV1 = gym.make(
        "smarts.env:hiway-v1",
        scenarios=scenarios,
        agent_interfaces=agent_interfaces,
        headless=headless,
    )
    
    env.observation_space = ObservationOptions.multi_agent
    
    env = SingleAgent(env)
    
    for episode in episodes(n=num_episodes):
        agent = CSTSAgent(True, False)
        # agent = KeepLaneAgent()
        observation, _ = env.reset()
        episode.record_scenario(env.unwrapped.scenario_log)

        map = env.get_map()
        terminated = False
        while not terminated:
            # print(type(observation))
            # obs = observation.get(AGENT_ID)
            # if(obs is None):
            #     break
            
            action = agent.act(observation, map)
            
            # action = {AGENT_ID: action}
            observation, reward, terminated, truncated, info = env.step(action)
            episode.record_step(observation, reward, terminated, truncated, info)

    env.close()


if __name__ == "__main__":
    parser = minimal_argument_parser(Path(__file__).stem)
    args = parser.parse_args()

    if not args.scenarios:
        args.scenarios = [
            str(SMARTS_REPO_PATH / "scenarios" / "sumo" / "loop"),
            str(SMARTS_REPO_PATH / "scenarios" / "sumo" / "figure_eight"),
        ]

    build_scenarios(scenarios=args.scenarios)

    main(
        scenarios=args.scenarios,
        headless=args.headless,
        num_episodes=args.episodes,
        max_episode_steps=args.max_episode_steps,
    )
