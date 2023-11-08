
import argparse
import random
import sys
from pathlib import Path
import time
from typing import Final, Sequence
import gymnasium as gym
import rospy

from smarts.core.controllers.action_space_type import ActionSpaceType

SMARTS_REPO_PATH = Path(__file__).parents[1].absolute()
sys.path.insert(0, str(SMARTS_REPO_PATH))
from examples.tools.argument_parser import minimal_argument_parser
from smarts.core.agent import Agent
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.observations import Observation
from smarts.core.utils.episodes import episodes
from smarts.env.utils.observation_conversion import ObservationOptions
from smarts.env.gymnasium.wrappers.single_agent import SingleAgent
from smarts.sstudio.scenario_construction import build_scenarios

from csts_msgs.msg import ego_state, perception_prediction, object_prediction, prediction_traj, map_lanes, lane
from geometry_msgs.msg import Vector3

AGENT_ID: Final[str] = "Agent"

class CSTSAgent(Agent):
    # Action=(delta_x, delta_y, delta_heading)
    action: Sequence[float,float, float] = (0,0,0)
    
    def __init__(self) -> None:
        super().__init__()
        # init ROS node
        rospy.init_node('csts_agent', anonymous=True)
        # init ROS publisher
        
        # init ROS subscriber
        
        # init ros launch file
        
        
    
    def act(self, obs: Observation) -> Sequence[float,float, float]:
        time.sleep(0.1)  # import time
        action: Sequence[float,float, float] = (0,0,0)
        
        pub_map_lanes()
        pub_perception()
        pub_ego_state()
        
        sub_action()
                
        return action
    
    def publish_ego_state(self, ):       
        if self.timestampe == self.start_frame:
            msg = ego_state()
            msg.ego_id = 0
            msg.header.frame_id = 'world'
            msg.header.stamp = rospy.Time().now()
            msg.frame_now = self.timestampe
            
            # lane_id = data_frame.ego_state.lane_id
            msg.lane_id = 0
            ego_box = Vector3(5, 2, 2)
            msg.ego_box = ego_box

            # ego_polygon = Polygon()
            # msg.ego_polygon = ego_polygon
            # self.map_state = data_frame.ego_state

            world_coord = Vector3(self.map_state.x, self.map_state.y, self.map_state.psi_rad)
            msg.world_coord = world_coord
            
            lane_coord = self.ref_path.coord_world_to_lane((self.map_state.x, self.map_state.y))
            msg.lane_coord = Vector3(lane_coord[0], lane_coord[1], 0)
            ego_speed = Vector3(self.map_state.v, 0, 0)
            msg.ego_speed = ego_speed

            ego_acc = Vector3(self.map_state.a, 0, 0)
            msg.ego_acc = ego_acc

            # ego_planning_trajectory_xyt = []
            # ego_planning_trajectory_xyt.append(Vector3(1,0,1))
            # msg.ego_planning_trajectory_xyt = ego_planning_trajectory_xyt

            # ego_planning_trajectory_yawva = []
            # ego_planning_trajectory_yawva.append(Vector3(0,1,0))
            # msg.ego_planning_trajectory_yawva = ego_planning_trajectory_yawva
        else:
            msg = self.state_receive
        self.ego_state_pub.publish(msg)
    
    
    
class KeepLaneAgent(Agent):
    def act(self, obs, **kwargs):
        return random.randint(0, 3)
    
class ChaseViaPointsAgent(Agent):
    def act(self, obs: Observation):
        if (
            len(obs.via_data.near_via_points) < 1
            or obs.ego_vehicle_state.road_id != obs.via_data.near_via_points[0].road_id
        ):
            return (obs.waypoint_paths[0][0].speed_limit, 0)

        nearest = obs.via_data.near_via_points[0]
        if nearest.lane_index == obs.ego_vehicle_state.lane_index:
            return (nearest.required_speed, 0)

        return (
            nearest.required_speed,
            1 if nearest.lane_index > obs.ego_vehicle_state.lane_index else -1,
        )


def main(scenarios, headless, num_episodes, max_episode_steps=None):
    # This interface must match the action returned by the agent
    # agent_interface = AgentInterface.from_type(
    #     AgentType.Laner, max_episode_steps=max_episode_steps
    # )
    # agent_interface = AgentInterface.from_type(
    #     AgentType.LanerWithSpeed,
    #     max_episode_steps=max_episode_steps,
    # )
    agent_interface = AgentInterface.from_type(
        # https://smarts.readthedocs.io/en/latest/sim/agent.html#pre-configured-interface
        AgentType.Full,
        max_episode_steps=max_episode_steps,
    )
    agent_interface.action = ActionSpaceType.RelativeTargetPose
    # https://smarts.readthedocs.io/en/latest/api/smarts.core.controllers.action_space_type.html#smarts.core.controllers.action_space_type.ActionSpaceType

    env = gym.make(
        "smarts.env:hiway-v1",
        scenarios=scenarios,
        agent_interfaces={AGENT_ID: agent_interface},
        headless=headless,
        observation_options=ObservationOptions.multi_agent,
    )
    # env = SingleAgent(env)

    for episode in episodes(n=num_episodes):
        agent = KeepLaneAgent()
        observation, _ = env.reset()
        episode.record_scenario(env.unwrapped.scenario_log)

        terminated = False
        while not terminated:
            action = agent.act(observation)
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
