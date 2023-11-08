from calendar import c
import logging
import math
import re
import subprocess
import time
from typing import Dict, List, Tuple, Type, Union

from matplotlib import pyplot as plt

from nuplan.common.actor_state.dynamic_car_state import DynamicCarState

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.tracked_objects import TrackedObject
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.pi_code.AgentFormer_main.predictor import Predictor
from nuplan.planning.simulation.controller.motion_model.kinematic_bicycle import KinematicBicycleModel
from nuplan.planning.simulation.observation.idm.utils import create_path_from_se2, path_to_linestring
from nuplan.planning.simulation.observation.observation_type import Observation
from nuplan.planning.simulation.planner.abstract_idm_planner import AbstractIDMPlanner
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput, AbstractPlanner
from nuplan.planning.simulation.planner.utils.breadth_first_search import BreadthFirstSearch
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.simulator.ref_path import Ref_path

import rospy
import numpy as np
from nuplan.simulator.new_sim_class import ScenarioSimulation
from nuplan.simulator.decode_utils import MotionState
from nuplan.simulator.state2ptcloud_class import Pred_state
from shapely.geometry import LineString, Point, Polygon

from nuplan.common.actor_state.state_representation import Point2D, StateSE2, StateVector2D, TimePoint

logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings("ignore")

class HAStarIDMPlanner(AbstractIDMPlanner):
    """
    The IDM planner is composed of two parts:
        1. Path planner that constructs a route to the same road block as the goal pose.
        2. IDM policy controller to control the longitudinal movement of the ego along the planned route.
    """

    # Inherited property, see superclass.
    requires_scenario: bool = False

    def __init__(
        self,
        target_velocity: float = 10,
        min_gap_to_lead_agent: float = 1,
        headway_time: float = 1.5,
        accel_max: float = 1,
        decel_max: float = 3,
        planned_trajectory_samples: int = 16,
        planned_trajectory_sample_interval: float = 0.5,
        occupancy_map_radius: float = 40,
    ):
        """
        Constructor for IDMPlanner
        :param target_velocity: [m/s] Desired velocity in free traffic.
        :param min_gap_to_lead_agent: [m] Minimum relative distance to lead vehicle.
        :param headway_time: [s] Desired time headway. The minimum possible time to the vehicle in front.
        :param accel_max: [m/s^2] maximum acceleration.
        :param decel_max: [m/s^2] maximum deceleration (positive value).
        :param planned_trajectory_samples: number of elements to sample for the planned trajectory.
        :param planned_trajectory_sample_interval: [s] time interval of sequence to sample from.
        :param occupancy_map_radius: [m] The range around the ego to add objects to be considered.
        """
        super(HAStarIDMPlanner, self).__init__(
            target_velocity = 10,
            min_gap_to_lead_agent = 1.0, #3.0, #1.0,
            headway_time = 1, #2, # 1.6
            accel_max = 1.0,
            decel_max = 1.0, #2.0,
            planned_trajectory_samples = 16,
            planned_trajectory_sample_interval = 0.5,
            occupancy_map_radius = 40,
        ) 
        #best close-loop 10, 3.0, 2, 1, 2, 16, 0.5, 40
        #best open-loop  10, 1.0, 1.6, 1.0, 1.0, 16, 0.5, 40
                # pred length = length_ + time_*0.5,  length_ = obj.box.length + 2 

        self._initialized = False
        self.vehicle = get_pacifica_parameters()
        self.motion_model = KinematicBicycleModel(self.vehicle)
        self.last_traj=None
        
        # self.pred_model = Predictor()
        self.pred_model = None

    def initialize(self, initialization: PlannerInitialization) -> None:
        """Inherited, see superclass."""
        start_time = time.time()
        roslaunch_cmd = "roslaunch ha_planner testing_launch.launch"

        self.roslaunch_proc = subprocess.Popen("ls", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.roslaunch_proc = subprocess.Popen(roslaunch_cmd.split(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


        self._map_api = initialization.map_api
        self._initialize_route_plan(initialization.route_roadblock_ids)
        self._initialized = False
        
        print("init ScenarioSimulation")
        self.sim = ScenarioSimulation(None, True, False)
        d_t = max(0, 4.8 - (time.time()-start_time))
        time.sleep(d_t)
        print("init over, sleep:", d_t)
        
        

    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        """Inherited, see superclass."""
        # Ego current state
        Print = False
        Debug_Print = False
        iteration = current_input.iteration
        print(iteration.index, end=' ')
        # print(f"iteration.index={iteration.index}")
        start_time = time.time()
        
        ego_history = current_input.history._ego_state_buffer
        obs_history = current_input.history._observations_buffer
        
        ego_state, observations = current_input.history.current_state
        
        
        self.iteration = iteration
        if iteration.index >= 149:
            print(f"index out of range!")
            logging.warning(f"index out of range!")
        
        if not self._initialized:
            get_path_start = time.time()
            self._initialize_ego_path(ego_state)
            self._initialized = True
            # print(f"001: get path time cost={(time.time()-start_time)*1000}ms")
            ref_path_pts = self.get_ref_path(ego_state)
            # print(ref_path_pts)
            # raise Exception("stop")
            x = ego_state.car_footprint.center.x
            y = ego_state.car_footprint.center.y
            heading = ego_state.car_footprint.center.heading
            self.sim.init_map(ref_path_pts, x, y, heading)
            # print(f"first frame, pub ref_path, len:{len(ref_path_pts)}")
            self.sim.start_sim()
            self.sim.got_best_path = False
            # print(f"2: init first frame, time cost={(time.time()-start_time)*1000}ms")
        
        points = []
        if iteration.index != 0:
            # pred map
            time0 = time.time()
            # pred_traj = self.pred_model.get_predict_traj(ego_history, obs_history)
            # print(type(pred_traj), f", time cost={(time.time()-time0)*1000}ms")
            # print(pred_traj)
            # raise Exception("stop")
        
            pred_states_dict = self.get_pred_states(ego_state, observations)
            
            
            
            points, pred_dict = self.sim.get_pred_pts(pred_states_dict)
            # print(f"3:get map :{(time.time()-start_time)*1000}ms")
            # idm map
            # Create occupancy map
            occupancy_map, unique_observations = self._construct_occupancy_map(ego_state, observations)

            # Traffic light handling
            traffic_light_data = current_input.traffic_light_data
            self._annotate_occupancy_map(traffic_light_data, occupancy_map)
            
            idm_traj = self._get_planned_trajectory(ego_state, occupancy_map, unique_observations)
            
            dt = (idm_traj._trajectory[4].time_point.time_s - iteration.time_s)
            speed_idm_2s = idm_traj._trajectory[4].to_split_state().linear_states[3]
            speed_idm_3s = idm_traj._trajectory[6].to_split_state().linear_states[3]
            
            ego_progress = self._ego_path_linestring.project(Point(*ego_state.center.point.array))
            speed_now = ego_state.to_split_state().linear_states[3]

            state_3s = self._ego_path.get_state_at_progress(ego_progress + 3*speed_now)
            ego_lane_id, ego_lane_type = self.get_lane_from_pt(state_3s)
            ego_lane_3s = self._map_api.get_map_object(ego_lane_id, ego_lane_type)
            
            desired_speed = min(ego_lane_3s.speed_limit_mps*0.8, speed_idm_3s*1.2)
            # !
            # desired_speed = ego_lane_3s.speed_limit_mps*0.8
            # print(f"idm speed_idm={ego_state.dynamic_car_state.speed}, {speed_idm_2s}, {speed_idm_3s}, {desired_speed}")
        else:
            idm_traj = None
            points = []
            speed_now = ego_state.to_split_state().linear_states[3]
            desired_speed = speed_now*0.8
            
            
        if len(points) == 0:
            points.append((0,0,5,1))

        # print(f"4:pub map, time cost={(time.time()-start_time)*1000}ms")

        msg = self.sim.State_to_Ptcloud.prob_pts_to_4dmsg(points)
        # print(f"4.5:pub map, time cost={(time.time()-start_time)*1000}ms")
        self.sim.map_pub.publish(msg)

        map_state = self.get_map_state(ego_state)
        time.sleep(0.001)
                
        
        
        # print(f"5:pub state, time cost={(time.time()-start_time)*1000}ms")
        msg = self.sim.publish_car_state(map_state, desired_speed)
        
        self.sim.publish_goals(msg)
        self.sim.got_best_path = False
        if Debug_Print:
            print(map_state)
            print(f"pub map, len:{len(points)}")
            print(f"pub car_state, bool:{self.sim.got_best_path}")
        
        
        
        while_start_time = time.time()
        # print(f"6:in while, time cost={(time.time()-start_time)*1000}ms")

        # if iteration.index in range(1, 6):
        #     print()
        #     print(self.same_pt(ego_state))
        #     print()
            
        
        if iteration.index != 0:
            if idm_traj:
                back_traj = idm_traj
            else:
                self._policy.target_velocity = ego_lane_3s.speed_limit_mps*0.8
                back_traj = self._get_planned_trajectory(ego_state, occupancy_map, unique_observations)
        else:
            back_traj = self.simple_plan(ego_state)
            
        
        while(self.sim.got_best_path == False and not rospy.is_shutdown()):
            time_cost = (time.time() - start_time)*1000
            if time_cost > 930:
                # print(f"timeout, use back_traj, iteration.index={iteration.index}")
                self.last_traj = back_traj
                print("+", end="")
                if iteration.index == 148:
                    print(f"over, iteration.index={iteration.index}")
                    self.close_node()
                return back_traj
            pass
        # print(f"7:get_path, time cost={(time.time()-start_time)*1000}ms")
        # print(self.sim.traj)
        trajectory: List[EgoState] = self.get_traj(self.sim.traj, ego_state.car_footprint.vehicle_parameters)
        # print(len(trajectory), )
        self.sim.got_best_path = False
        
        if iteration.index == 148:
            print(f"over, iteration.index={iteration.index}")
            self.close_node()
        
            
        time_cost = (time.time() - start_time)*1000
        if Print:
            print(f"iteration.index:{iteration.index}, time_cost={time_cost:.2f}ms")
        if time_cost < 300:
            time.sleep((300-time_cost)/1000)
            
        traj = InterpolatedTrajectory(trajectory)
        self.last_traj = traj
        # print(f"8:len:{len(trajectory)}, index:{iteration.index}, run time:{(time.time() - start_time)*1000:.2f}ms")
        # input()
        return traj
    
    def close_node(self):
        self.sim = None
        self.roslaunch_proc.terminate()
        self.roslaunch_proc = None
        # time.sleep(3)
        self.pred_model = None
        
    def same_pt(self, ego_state: EgoState):
        state = self.last_traj.get_state_at_time(self.iteration.time_point)
        print(ego_state.to_split_state())
        print(state.to_split_state())
        if ego_state.to_split_state() == state.to_split_state():
            return True
        return False
        

    def get_ref_path(self, ego_state:EgoState):
        
        return self.xy_list

    def _initialize_ego_path(self, ego_state: EgoState) -> None:
        """
        Initializes the ego path from the ground truth driven trajectory
        :param ego_state: The ego state at the start of the scenario.
        """
        route_plan, path_found = self._breadth_first_search(ego_state)
        print(f"path_found:{path_found}")
        ego_speed = ego_state.dynamic_car_state.rear_axle_velocity_2d.magnitude()
        speed_limit = route_plan[0].speed_limit_mps or self._policy.target_velocity
        self._policy.target_velocity = speed_limit if speed_limit > ego_speed else ego_speed
        discrete_path = []
        for edge in route_plan:
            discrete_path.extend(edge.baseline_path.discrete_path)
        
        self.xy_list = [(pt.x, pt.y) for pt in discrete_path]
            
            
        self._ego_path = create_path_from_se2(discrete_path)
        self._ego_path_linestring = path_to_linestring(discrete_path)
        ego_pt = Point(ego_state.center.x, ego_state.center.y)
        # return
        if ego_pt.distance(self._ego_path_linestring) > 4:
            new_time = time.time()
            old_discrete_path = discrete_path
            # print("ego_pt.distance(self._ego_path_linestring) > 2")
            
            point = Point(ego_state.center.x, ego_state.center.y)
            foot = self._ego_path_linestring.interpolate(self._ego_path_linestring.project(point))
            dx = point.x - foot.x
            dy = point.y - foot.y
            discrete_path = []
            for pt in old_discrete_path:
                discrete_path.append(StateSE2(pt.x+dx, pt.y+dy, pt.heading))
            
            # print(f"new_time={time.time()-new_time}")
            # print(f"ego_pt=({ego_state.center.x},{ego_state.center.y})")
            # print(f"old_discrete_path={[(pt.x, pt.y) for pt in old_discrete_path]}")
            # print(f"new_discrete_path={[(pt.x, pt.y) for pt in discrete_path]}")
            # raise ValueError("Ego state is not on the route plan!")    
            self._ego_path = create_path_from_se2(discrete_path)
            self._ego_path_linestring = path_to_linestring(discrete_path)
            self.xy_list = [(pt.x, pt.y) for pt in discrete_path]
            # print(f"time cost={time.time()-new_time}")
        

    def _get_starting_edge(self, ego_state: EgoState) -> LaneGraphEdgeMapObject:
        """
        Get the starting edge based on ego state. If a lane graph object does not contain the ego state then
        the closest one is taken instead.
        :param ego_state: Current ego state.
        :return: The starting LaneGraphEdgeMapObject.
        """
        assert (
            self._route_roadblocks is not None
        ), "_route_roadblocks has not yet been initialized. Please call the initialize() function first!"
        assert len(self._route_roadblocks) >= 2, "_route_roadblocks should have at least 2 elements!"

        starting_edge = None
        closest_distance = math.inf

        # Check for edges in about first and second roadblocks
        for edge in self._route_roadblocks[0].interior_edges + self._route_roadblocks[1].interior_edges:
            if edge.contains_point(ego_state.center):
                starting_edge = edge
                break

            # In case the ego does not start on a road block
            distance = edge.polygon.distance(ego_state.car_footprint.geometry)
            if distance < closest_distance:
                starting_edge = edge
                closest_distance = distance

        assert starting_edge, "Starting edge for IDM path planning could not be found!"
        return starting_edge

    def _breadth_first_search(self, ego_state: EgoState) -> Tuple[List[LaneGraphEdgeMapObject], bool]:
        """
        Performs iterative breath first search to find a route to the target roadblock.
        :param ego_state: Current ego state.
        :return:
            - A route starting from the given start edge
            - A bool indicating if the route is successfully found. Successful means that there exists a path
              from the start edge to an edge contained in the end roadblock. If unsuccessful a longest route is given.
        """
        assert (
            self._route_roadblocks is not None
        ), "_route_roadblocks has not yet been initialized. Please call the initialize() function first!"
        assert (
            self._candidate_lane_edge_ids is not None
        ), "_candidate_lane_edge_ids has not yet been initialized. Please call the initialize() function first!"

        starting_edge = self._get_starting_edge(ego_state)
        graph_search = BreadthFirstSearch(starting_edge, self._candidate_lane_edge_ids)
        # Target depth needs to be offset by one if the starting edge belongs to the second roadblock in the list
        offset = 1 if starting_edge.get_roadblock_id() == self._route_roadblocks[1].id else 0
        route_plan, path_found = graph_search.search(self._route_roadblocks[-1], len(self._route_roadblocks[offset:]))

        if not path_found:
            logger.warning(
                "IDMPlanner could not find valid path to the target roadblock. Using longest route found instead"
            )

        return route_plan, path_found


    def get_lane_from_pt(self, pt:Union[Point2D, StateSE2]):
        # return lane_id, lane_type, 'ego_lane = self.map_api.get_map_object(ego_lane_id, ego_lane_type)' to get lane object
        assert isinstance(pt, Point2D) or isinstance(pt, StateSE2)
        if isinstance(pt, StateSE2):
            pt = pt.point
        
        nearest_lane = self._map_api.get_distance_to_nearest_map_object(pt, SemanticMapLayer.LANE)
        nearest_lane_c = self._map_api.get_distance_to_nearest_map_object(pt, SemanticMapLayer.LANE_CONNECTOR)
        # print(nearest_lane, nearest_lane_c)
        
        if nearest_lane[1]!=None and nearest_lane_c[1]!=None:
            if nearest_lane[1] <= nearest_lane_c[1]:
                lane_id = nearest_lane[0]
                lane_type = SemanticMapLayer.LANE
            else:
                lane_id = nearest_lane_c[0]
                lane_type = SemanticMapLayer.LANE_CONNECTOR
        elif nearest_lane[1] == None:
            lane_id = nearest_lane_c[0]
            lane_type = SemanticMapLayer.LANE_CONNECTOR
        elif nearest_lane_c[1] == None:
            lane_id = nearest_lane[0]
            lane_type = SemanticMapLayer.LANE
        else:
            raise ValueError("no lane")
        # print(lane_id)
        # raise ValueError("test")
        return lane_id, lane_type
    
    def get_pred_states(self, ego_state, observation:Observation) -> Dict[int, List[Dict[float, Pred_state]]]:
        pred_states_dict = {}
        pred_states_dict = self.get_cv_pred_states(ego_state, observation)
        # observation.tracked_objects : TrackedObjects
        # TrackedObjects.tracked_objects : List[TrackedObject]
        # # obj class
        # Agent
        # StaticObject
        # SceneObject
        # AgentTemporalState
        
        # for obj in observation.tracked_objects:
        #     pass
        #     # print(obj)
        #     # <nuplan.common.actor_state.agent.Agent object at 0x7fbdb5684820>
        #     # <nuplan.common.actor_state.static_object.StaticObject object at 0x7fbdb5680f40>
        
        
        return pred_states_dict
    
    def get_cv_pred_states(self, ego_state:EgoState, observation:Observation) -> Dict[int, Dict[float, Pred_state]]:
        pred_states_dict = {}
        observation.tracked_objects : List[TrackedObject]
        for obj in observation.tracked_objects:
            
            if obj.tracked_object_type.value > 2:
                # print(f"obj: {obj.tracked_object_type}, {obj.tracked_object_type.value}")
                continue
            
            car_state = np.array((obj.center.x, obj.center.y))
            ego_s = np.array((ego_state.center.x, ego_state.center.y))
            vector = ego_s - car_state
            angle = np.arctan2(vector[1], vector[0])
            d_angle = abs(angle - ego_state.center.heading)
            if d_angle < np.pi/6:
                continue
            
            # TrackedObjectType
            pred_obs = {}
            obj_id = obj.metadata.track_id
            pos_x = obj.center.x
            pos_y = obj.center.y
            speed_ = obj.velocity.magnitude()
            v_x = obj.velocity.x
            v_y = obj.velocity.y
            yaw_ = obj.center.heading
            length_ = obj.box.length + 2 # 4
            width_ = obj.box.width
            type_ = obj.metadata.category_name
            # print(f"sim.transfer: {type(self.sim.transfer)}")
            (pos_x, pos_y), yaw_, (v_x, v_y) = self.sim.transfer.from_dataset_to_map((pos_x, pos_y), yaw_, (v_x, v_y))
            
            
            for time_ in np.arange(0.0, 5.5, 0.5):
                x = pos_x + v_x * time_
                y = pos_y + v_y * time_
                                
                Pred_state_ = Pred_state(
                id = obj_id, 
                x = x,
                y = y,
                yaw = yaw_,
                length = length_ + time_*0, # 1
                width = width_,
                speed = speed_,
                _time = time_,
                _type= type_,
                )
                pred_obs[time_] = Pred_state_
            pred_states_dict[obj_id] = [pred_obs]
        # print(f"pred_states_dict: {pred_states_dict}")
        return pred_states_dict
    
    
    def get_traj(self, map_traj_list, vehicle_parameters):
        trajectory: List[EgoState] = []
        time_us_now = self.iteration.time_us
        
        for traj_pt in map_traj_list:
            x, y, t, angle, v, a = traj_pt
            
            (x, y), angle, _ = self.sim.transfer.from_map_to_dataset((x,y),angle)
            vx = v * math.cos(angle) * 0
            vy = v * math.sin(angle) * 0
            ax = a * math.cos(angle) * 0
            ay = a * math.sin(angle) * 0
            
            t_us = t*1e6
            
            state = EgoState.build_from_center(
                center = StateSE2(x, y, angle),
                center_velocity_2d = StateVector2D(vx,vy),
                center_acceleration_2d = StateVector2D(ax,ay),
                tire_steering_angle = 0, 
                time_point = TimePoint(time_us_now + t_us),
                vehicle_parameters = vehicle_parameters,
                is_in_auto_mode = True,
                angular_vel = 0,
                angular_accel = 0
            )
            trajectory.append(state)
        # print(t, t_us)
        # raise Exception("stop")
        # state.dynamic_car_state.center_velocity_2d.y = 0
        # state.dynamic_car_state.center_acceleration_2d = StateVector2D(0,0)
        # for t in np.arange(5.1, 8.1, 0.1):
        #     t_us = t*1e6
            
        #     state = self.motion_model.propagate_state(state, state.dynamic_car_state, TimePoint(0.1*1e6))
        #     trajectory.append(state)
        
        return trajectory
    
    def get_map_state(self, ego_state:EgoState):
        map_state = MotionState()
        pt_data = (ego_state.car_footprint.center.x, ego_state.car_footprint.center.y)
        angle_data = ego_state.car_footprint.center.heading
        # ego_state.car_footprint.center.x, ego_state.car_footprint.center.y, ego_state.car_footprint.center.heading
        
        speed_v = (ego_state.dynamic_car_state.center_velocity_2d.x, ego_state.dynamic_car_state.center_velocity_2d.y)
        accel_v = (ego_state.dynamic_car_state.center_acceleration_2d.x, ego_state.dynamic_car_state.center_acceleration_2d.y)
        
        pt, angle, speed = self.sim.transfer.from_dataset_to_map(pt_data, angle_data, speed_v)
        # pt, angle, accel = self.sim.transfer.from_dataset_to_map(pt_data, angle_data, accel_v)
        
        map_state.x = pt[0]
        map_state.y = pt[1]
        map_state.psi_rad = angle
        map_state.vx = ego_state.dynamic_car_state.center_velocity_2d.x
        map_state.vy = ego_state.dynamic_car_state.center_velocity_2d.y
        map_state.ax = ego_state.dynamic_car_state.center_acceleration_2d.x
        map_state.ay = ego_state.dynamic_car_state.center_acceleration_2d.y
        map_state.ay = 0
        map_state.v = ego_state.dynamic_car_state.speed
        map_state.a = ego_state.dynamic_car_state.acceleration
        
        
        return map_state
    
    
    
    def simple_plan(self, ego_state:EgoState):
        state = EgoState(
            car_footprint=ego_state.car_footprint,
            dynamic_car_state=DynamicCarState.build_from_rear_axle(
                ego_state.car_footprint.rear_axle_to_center_dist,
                ego_state.dynamic_car_state.rear_axle_velocity_2d,
                StateVector2D(0,0),
            ),
            tire_steering_angle=0,
            is_in_auto_mode=True,
            time_point=ego_state.time_point,
        )
        trajectory: List[EgoState] = [state]
        horizon_seconds = TimePoint(int(10 * 1e6))
        sampling_time = TimePoint(int(0.25 * 1e6))
        max_velocity = ego_state.to_split_state().linear_states[3]
        for _ in range(int(horizon_seconds.time_us / sampling_time.time_us)):
            if state.dynamic_car_state.speed > max_velocity:
                accel = max_velocity - state.dynamic_car_state.speed
                state = EgoState.build_from_rear_axle(
                    rear_axle_pose=state.rear_axle,
                    rear_axle_velocity_2d=state.dynamic_car_state.rear_axle_velocity_2d,
                    rear_axle_acceleration_2d=StateVector2D(accel, 0),
                    tire_steering_angle=state.tire_steering_angle,
                    time_point=state.time_point,
                    vehicle_parameters=state.car_footprint.vehicle_parameters,
                    is_in_auto_mode=True,
                    angular_vel=state.dynamic_car_state.angular_velocity,
                    angular_accel=state.dynamic_car_state.angular_acceleration,
                )

            state = self.motion_model.propagate_state(state, state.dynamic_car_state, sampling_time)
            trajectory.append(state)
        back_traj = InterpolatedTrajectory(trajectory)
        return back_traj
    
    
    
    
class PredPlanner(AbstractIDMPlanner):
    def __init__(
        self,
    ):
        super(PredPlanner, self).__init__(
            target_velocity = 10,
            min_gap_to_lead_agent = 1.0, #1.0
            headway_time = 1.6, # 1.6
            accel_max = 1.0,
            decel_max = 2.0,
            planned_trajectory_samples = 16,
            planned_trajectory_sample_interval = 0.5,
            occupancy_map_radius = 50,
        ) #best 10, 3.0, 2, 1, 2, 16, 0.5, 40

        self._initialized = False
        self.vehicle = get_pacifica_parameters()
        self.motion_model = KinematicBicycleModel(self.vehicle)
        
        self.start_pred = None
        self.ego_pts = []
        

    def initialize(self, initialization: List[PlannerInitialization]) -> None:
        
        self._map_api = initialization.map_api
        self._initialize_route_plan(initialization.route_roadblock_ids)
        self._initialized = False
        self.pred_model = Predictor()
        self.ego_pts = []
        self.last_traj = None
        self.last_traj_time_us = 0
        self.ax = plt.gca()
        plt.ion()
        plt.show()
    
    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def observation_type(self) -> Type[Observation]:
        """Inherited, see superclass."""
        return DetectionsTracks  # type: ignore

    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        start_time = time.time()
        iteration = current_input.iteration
        print(iteration.index)
        ego_history = current_input.history._ego_state_buffer
        obs_history = current_input.history._observations_buffer
        
        ego_state, observations = current_input.history.current_state
        
        pred_traj = self.pred_model.get_predict_traj(ego_history, obs_history)
        ego_trajs = []
        margin_x, margin_y = ego_history[-1].center.x, ego_history[-1].center.y
        if not self._initialized:
            get_path_start = time.time()
            self._initialize_ego_path(ego_state)
            self._initialized = True
            # print(f"001: get path time cost={(time.time()-start_time)*1000}ms")
            ref_path_pts = self.get_ref_path(ego_state)
            self.ref_path = Ref_path(ref_path_pts)
        
        min_dis = 100
        for i in range(len(pred_traj)):
            traj = []
            for j in range(len(pred_traj[i])):
                if pred_traj[i][j][1] == -1:
                    pt = [pred_traj[i][j][0], pred_traj[i][j][1], pred_traj[i][j][2]-( - margin_x + 200), pred_traj[i][j][3]-( - margin_y + 200)]
                    traj.append(pt)
                # traj.append(list(pred_traj[i][j]))
            ll_vector = np.array((traj[-2][2], traj[-2][3])) - np.array((traj[-3][2], traj[-3][3]))
            last_vector = np.array((traj[-1][2], traj[-1][3])) - np.array((traj[-2][2], traj[-2][3]))
            d_v = np.linalg.norm(ll_vector - last_vector)/np.linalg.norm(last_vector)
            
            # for i in range(1, 7):
            #     new_xy = np.array((traj[-1][2], traj[-1][3])) + last_vector*2*max(0,(1-i*d_v))
            #     x, y = new_xy
            #     traj.append([15+i, traj[-1][1], x, y])
            ego_trajs.append(traj)


        if iteration.index == 0:
            self.start_pred = ego_trajs
            # print(f"ego_trajs={ego_trajs}")
        # raise Exception("stop")
        
        # traj = ego_trajs[0]
        # if iteration.index <= 80:
        #     self.ego_pts.append((ego_state.center.x, ego_state.center.y))
        # else:
        #     print(f"ego_trajs={self.start_pred}")
        #     print(f"ego_pts={self.ego_pts}")
        #     raise Exception("stop")
        
        useful_trajs = []
        useful_trajs = ego_trajs
        for traj in ego_trajs:
            continue
            goal_pt = Point(traj[-1][2], traj[-1][3])
            pt_3s_2d = Point2D(traj[5][2], traj[5][3])
            dis = goal_pt.distance(self._ego_path_linestring)
            if dis >= 8:
                print(f"out of this lane")
                continue
            if not self._map_api.is_in_layer(pt_3s_2d, SemanticMapLayer.DRIVABLE_AREA):
                print(f"out of map")
                continue
            last_vector = np.array((traj[-1][2], traj[-1][3])) - np.array((traj[-2][2], traj[-2][3]))
            last_heading = np.arctan2(last_vector[1], last_vector[0])
            if abs(last_heading - ego_state.center.heading) > np.pi/2:
                print(f"heading wrong")
                continue
            useful_trajs.append(traj)
        
        
        
        if useful_trajs:
            
            traj = np.array(useful_trajs[0])
            for i in range(len(useful_trajs)-1):
                traj += np.array(useful_trajs[i+1])
            best_traj = traj/len(useful_trajs)
                

        else:
            min_dis = 30
            best_traj = ego_trajs[0]
            for traj in ego_trajs:
                goal_pt = Point(traj[-1][2], traj[-1][3])
                dis = goal_pt.distance(self._ego_path_linestring)
                if dis < min_dis:
                    min_dis = dis
                    best_traj = traj

        for i in range(len(best_traj)):
            pt = best_traj[i]
            point = Point(pt[2], pt[3])
            projected_point = self._ego_path_linestring.interpolate(self._ego_path_linestring.project(point))
            
            best_traj[i] = [pt[0], pt[1], projected_point.x, projected_point.y]


        self.last_traj = best_traj
        self.last_traj_time_us = iteration.time_us
        time_us_now = iteration.time_us
            
        # Display
        # self.ax.cla()
        # self.ax.scatter(ego_state.center.x, ego_state.center.y, c='r', s=10)
        # for i in range(len(ego_trajs)):
        #     traj = ego_trajs[i]
        #     x = [p[2] for p in traj]
        #     y = [p[3] for p in traj]
        #     if traj in useful_trajs:
        #         color = 'r'
        #     else:
        #         color = 'b'
        #     self.ax.plot(x, y, color=color)
        #     # break
        # x = [p[2] for p in self.last_traj]
        # y = [p[3] for p in self.last_traj]
        # color = 'g'
        # self.ax.plot(x, y, color=color)
        
        # for route in self.route_plan[:5]:
        #     x, y = route.polygon.exterior.xy

        #     # 使用 Polygon 对象的坐标创建一个多边形对象，并将其添加到图形对象中
        #     poly = plt.Polygon(list(zip(x, y)), edgecolor='r', alpha=0.5)
        #     self.ax.add_patch(poly)
        #     # route.polygon
        # self.ax.axis('equal')
        # plt.pause(0.5)
        
        
        trajectory: List[EgoState] = [ego_state]
        
        
        
        for traj_pt in best_traj[:]:
            t, id, x, y = traj_pt
            t = (t-3)*0.5
            angle = ego_state.center.heading
            t_us = t*1e6
            # print(angle)
            state = EgoState.build_from_center(
                center = StateSE2(x, y, angle),
                center_velocity_2d = StateVector2D(0,0),
                center_acceleration_2d = StateVector2D(0,0),
                tire_steering_angle = 0, 
                time_point = TimePoint(time_us_now + t_us),
                vehicle_parameters = ego_state.car_footprint.vehicle_parameters,
                is_in_auto_mode = True,
                angular_vel = 0,
                angular_accel = 0
            )
            trajectory.append(state)
        
            
        if iteration.index == 148:
            self.pred_model = None    
            
            
            
        return InterpolatedTrajectory(trajectory)
    
        
    def get_ref_path(self, ego_state:EgoState):
        
        return self.xy_list
    
    def _initialize_ego_path(self, ego_state: EgoState) -> None:
        """
        Initializes the ego path from the ground truth driven trajectory
        :param ego_state: The ego state at the start of the scenario.
        """
        route_plan, path_found = self._breadth_first_search(ego_state)
        self.route_plan = route_plan
        print(f"path_found:{path_found}")
        ego_speed = ego_state.dynamic_car_state.rear_axle_velocity_2d.magnitude()
        speed_limit = route_plan[0].speed_limit_mps or self._policy.target_velocity
        self._policy.target_velocity = speed_limit if speed_limit > ego_speed else ego_speed
        discrete_path = []
        for edge in route_plan:
            discrete_path.extend(edge.baseline_path.discrete_path)
        
        self.xy_list = [(pt.x, pt.y) for pt in discrete_path]
            
            
        self._ego_path = create_path_from_se2(discrete_path)
        self._ego_path_linestring = path_to_linestring(discrete_path)
        ego_pt = Point(ego_state.center.x, ego_state.center.y)
        # return
        if ego_pt.distance(self._ego_path_linestring) > 2:
            new_time = time.time()
            old_discrete_path = discrete_path
            # print("ego_pt.distance(self._ego_path_linestring) > 2")
            
            point = Point(ego_state.center.x, ego_state.center.y)
            foot = self._ego_path_linestring.interpolate(self._ego_path_linestring.project(point))
            dx = point.x - foot.x
            dy = point.y - foot.y
            discrete_path = []
            for pt in old_discrete_path:
                discrete_path.append(StateSE2(pt.x+dx, pt.y+dy, pt.heading))
            
            self._ego_path = create_path_from_se2(discrete_path)
            self._ego_path_linestring = path_to_linestring(discrete_path)
            self.xy_list = [(pt.x, pt.y) for pt in discrete_path]

        
        
    def _breadth_first_search(self, ego_state: EgoState) -> Tuple[List[LaneGraphEdgeMapObject], bool]:
        """
        Performs iterative breath first search to find a route to the target roadblock.
        :param ego_state: Current ego state.
        :return:
            - A route starting from the given start edge
            - A bool indicating if the route is successfully found. Successful means that there exists a path
              from the start edge to an edge contained in the end roadblock. If unsuccessful a longest route is given.
        """
        assert (
            self._route_roadblocks is not None
        ), "_route_roadblocks has not yet been initialized. Please call the initialize() function first!"
        assert (
            self._candidate_lane_edge_ids is not None
        ), "_candidate_lane_edge_ids has not yet been initialized. Please call the initialize() function first!"

        starting_edge = self._get_starting_edge(ego_state)
        graph_search = BreadthFirstSearch(starting_edge, self._candidate_lane_edge_ids)
        # Target depth needs to be offset by one if the starting edge belongs to the second roadblock in the list
        offset = 1 if starting_edge.get_roadblock_id() == self._route_roadblocks[1].id else 0
        route_plan, path_found = graph_search.search(self._route_roadblocks[-1], len(self._route_roadblocks[offset:]))

        if not path_found:
            logger.warning(
                "IDMPlanner could not find valid path to the target roadblock. Using longest route found instead"
            )

        return route_plan, path_found
    
    def _get_starting_edge(self, ego_state: EgoState) -> LaneGraphEdgeMapObject:
        """
        Get the starting edge based on ego state. If a lane graph object does not contain the ego state then
        the closest one is taken instead.
        :param ego_state: Current ego state.
        :return: The starting LaneGraphEdgeMapObject.
        """
        assert (
            self._route_roadblocks is not None
        ), "_route_roadblocks has not yet been initialized. Please call the initialize() function first!"
        assert len(self._route_roadblocks) >= 2, "_route_roadblocks should have at least 2 elements!"

        starting_edge = None
        closest_distance = math.inf

        # Check for edges in about first and second roadblocks
        for edge in self._route_roadblocks[0].interior_edges + self._route_roadblocks[1].interior_edges:
            if edge.contains_point(ego_state.center):
                starting_edge = edge
                break

            # In case the ego does not start on a road block
            distance = edge.polygon.distance(ego_state.car_footprint.geometry)
            if distance < closest_distance:
                starting_edge = edge
                closest_distance = distance

        assert starting_edge, "Starting edge for IDM path planning could not be found!"
        return starting_edge