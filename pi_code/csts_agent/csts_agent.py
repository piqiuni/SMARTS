
import math
import random
import subprocess
import time
from typing import Dict, List

import numpy as np
from smarts.core.agent import Agent
from smarts.core.controllers.action_space_type import ActionSpaceType
from smarts.core.road_map import RoadMap
from smarts.core.sumo_road_network import SumoRoadNetwork
from smarts.core.observations import EgoVehicleObservation, Observation, VehicleObservation
from smarts.core.coordinates import Point, RefLinePoint

import rospy
import threading
from csts_msgs.msg import perception_prediction, prediction_traj, object_prediction, ego_state, map_lanes, lane
from geometry_msgs.msg import Vector3

from collections import namedtuple

Pt=namedtuple("Pt", ["x", "y", "t", "yaw", "v", "a"])
class CSTSAgent(Agent):
    def __init__(self, init_ros=True, planning=True):
        self.action_type = ActionSpaceType.RelativeTargetPose
        if init_ros:
            rospy.init_node('csts_agent', anonymous=True)
        self.pub_perception_prediction = rospy.Publisher(
            "/map_server/perception_prediction", perception_prediction, queue_size=1)
        self.pub_ego_state = rospy.Publisher(
            "/map_server/ego_state", ego_state, queue_size=1)
        self.pub_lanes = rospy.Publisher(
            "/map_server/map_lanes", map_lanes, queue_size=1)
        self.sub_ego_state = rospy.Subscriber(
            "/search_server/ego_state", ego_state, callback=self.ego_state_callback, queue_size=1)

        self.ego_state_msg = ego_state()
        self.ego_state_msg_sub = ego_state()
        self.map_lanes_msg = map_lanes()
        self.perception_prediction_msg = perception_prediction()

        self.spin_thread = threading.Thread(target=self.thread_job)
        self.roslaunch_cmd = "roslaunch contingency_st_search test_search.launch"

        self.planning = planning
        if self.planning:
            self.roslaunch_proc = subprocess.Popen(
                "ls", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # self.roslaunch_proc = subprocess.Popen(self.roslaunch_cmd.split(
            #     ), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            self.roslaunch_proc = subprocess.Popen(
                "ls", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        rospy.sleep(1)

        self.spin_thread.start()

        self.ego_state_msg_sub: ego_state = None
        self.receive_ego_state = False
        self.dt = 0.1
        self.timestamp = 0
        self.ego_obs = EgoVehicleObservation
        self.objs_obs = Dict[str, VehicleObservation]
        self.ego_lane: RoadMap.Lane = None

        self.lane_id_to_idx_dict: Dict[str, int] = {}
        self.obj_id_to_idx_dict: Dict[str, int] = {}
        

    def __del__(self):
        self.roslaunch_proc.terminate()

    def thread_job(self):
        rospy.spin()

    def ego_state_callback(self, state: ego_state):
        if self.receive_ego_state == True:
            return
        self.ego_state_msg_sub = state
        self.receive_ego_state = True

    def act(self, obs, map: RoadMap):
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
            ego["heading"] += np.pi/2
            ego_speed = math.sqrt(ego["linear_velocity"][0]**2 + ego["linear_velocity"][1]**2)
            ego["linear_velocity"][2] = ego_speed
            # print(f"ego before:{type(ego)}", end="")
            ego = EgoVehicleObservation(**ego)
            # print(f"after={type(ego)}")
        self.ego_obs = ego
        objs = obs.get("neighborhood_vehicle_states")
        if not objs:
            pass
        else:
            # print(objs.keys())
            # dict_keys(['box', 'heading', 'id', 'interest', 'lane_id', 'lane_index', 'position', 'speed'])
            objs["bounding_box"] = objs.pop("box")
            len_objs = len(objs["bounding_box"])
            # print(len(objs["bounding_box"]), type(objs["bounding_box"]))
            objs["road_id"] = np.ndarray([len_objs, 1], str)

            # print(f"objs before:{type(objs)}", end="")

            keys = objs.keys()
            new_objs: Dict[str, VehicleObservation] = {}
            for i in range(len(objs["id"])):
                id = objs["id"][i]
                if (id == ""):
                    # print("empty")
                    continue
                objs["heading"][i] += np.pi/2
                obj = VehicleObservation(objs["id"][i], objs["position"][i], objs["bounding_box"][i], objs["heading"][i], objs["speed"][i], objs["road_id"][i], objs["lane_id"][i], objs["lane_index"][i])
                # box = objs["bounding_box"][i]
                # print(f"i = {i}, obj_id: {id}, box: {box}, type: {type(obj)}")
                new_objs[id] = obj

            # objs = VehicleObservation(**new_objs)
            # print(f"after={type(objs)}")
            # raise
        self.objs_obs = new_objs
        self.ego_state_msg = self.get_ego_state(ego, map)
        self.map_lanes_msg = self.get_map_lanes(ego, map)
        self.perception_prediction_msg = self.get_pp(new_objs, map)

        self.publish_all()
        if self.planning:
            while (self.receive_ego_state == False and rospy.is_shutdown() == False):
                continue
        else:
            rospy.sleep(0.1)
        if self.planning:
            action = self.get_action(self.ego_state_msg_sub)
        else:
            action = self.const_v_action()
        print(action)
        self.timestamp += 1
        self.receive_ego_state = False
        return action


    def get_ego_state(self, ego: EgoVehicleObservation, map: RoadMap):
        if self.timestamp == 0 or not self.planning:
            msgs = self.ego_state_msg
            msgs.header.frame_id = "world"
            msgs.header.stamp = rospy.Time.now()
            msgs.header.seq = self.ego_state_msg.header.seq+1
            msgs.ego_id = 0
            msgs.ego_box.x = ego.bounding_box[0]
            msgs.ego_box.y = ego.bounding_box[1]
            msgs.ego_box.z = ego.bounding_box[2]
            msgs.ego_acc.x = ego.linear_acceleration[0]
            msgs.ego_acc.y = ego.linear_acceleration[1]
            msgs.ego_acc.z = 0
            msgs.world_coord.x = ego.position[0]
            msgs.world_coord.y = ego.position[1]
            msgs.world_coord.z = ego.heading
            msgs.ego_speed.x=ego.linear_velocity[0]
            msgs.ego_speed.y=ego.linear_velocity[1]
            ego_speed = math.sqrt(ego.linear_velocity[0]**2 + ego.linear_velocity[1]**2)
            msgs.ego_speed.z=ego_speed
            # msgs.lane_coord = Vector3()
            # msgs.ego_polygon
            msgs.frame_now = self.ego_state_msg.header.seq+1
            msgs.ego_planning_trajectory_xyyaw = list()
            msgs.ego_planning_trajectory_tva = list()
        else:
            msgs = self.ego_state_msg_sub
        
        return msgs

    def lane_to_idx(self, lane_id: str):
        if lane_id not in self.lane_id_to_idx_dict:
            self.lane_id_to_idx_dict[lane_id] = len(self.lane_id_to_idx_dict)+1
        return self.lane_id_to_idx_dict[lane_id]

    def dfs_lane(self, this_lane: RoadMap.Lane, lane_accessed: set, msgs):
        if this_lane.lane_id in lane_accessed:
            return
        else:
            lane_accessed.add(this_lane.lane_id)

            msg = lane()
            msg.lane_id = self.lane_to_idx(this_lane.lane_id)
            msg.incoming_id = [self.lane_to_idx(
                _lane.lane_id) for _lane in this_lane.incoming_lanes]
            msg.outgoing_id = [self.lane_to_idx(
                _lane.lane_id) for _lane in this_lane.outgoing_lanes]
            if this_lane.lane_to_left[1]:
                msg.left_lane_id = self.lane_to_idx(this_lane.lane_to_left[0])
            if this_lane.lane_to_right[1]:
                msg.right_lane_id = self.lane_to_idx(
                    this_lane.lane_to_right[0])
            msg.length = this_lane.length
            msg.width = this_lane.width_at_offset(0)[0]
            msg.speed_limit = this_lane.speed_limit
            for s in np.arange(0, this_lane.length, 0.5):
                waypt = this_lane.from_lane_coord(RefLinePoint(s))
                msg.waypoints.append(Vector3(waypt.x, waypt.y, waypt.z))

            msgs.lanes.append(msg)
            for _lane in this_lane.outgoing_lanes:
                self.dfs_lane(_lane, lane_accessed, msgs)

    def get_map_lanes(self, ego: EgoVehicleObservation, map: RoadMap):
        msgs = self.map_lanes_msg
        msgs.header.frame_id = "world"
        msgs.header.stamp = rospy.Time.now()
        msgs.header.seq = self.map_lanes_msg.header.seq+1
        lane_accessed=set()

        lanes: List[RoadMap.Lane] = []
        nearest_lane = map.nearest_lane(Point(ego.position[0], ego.position[1]))
        self.ego_lane = nearest_lane
        # print(nearest_lane.lane_id)
        lanes.append(nearest_lane)
        left_lane = nearest_lane.lane_to_left
        # print(left_lane)
        if left_lane[0] and left_lane[1]:
            lanes.append(left_lane[0])
            # print(left_lane[0].lane_id)
        right_lane = nearest_lane.lane_to_right
        # print(right_lane)
        if right_lane[0] and right_lane[1]:
            lanes.append(right_lane[0])
            # print(right_lane[0].lane_id)
        
        msg_id = 0
        msgs.lanes.clear()
        forward_distance = 200
        # print(f"len of lanes: {len(lanes)}")
        for map_lane in lanes:
            msg = lane()
            msg.lane_id = msg_id
            msg.speed_limit = map_lane.speed_limit
            msg.width = 3
            ego_lane_coord = map_lane.to_lane_coord(Point(ego.position[0], ego.position[1]))
            lane_list = [map_lane]
            length_list= [map_lane.length]
            distance = int(forward_distance + ego_lane_coord.s)
            while(sum(length_list) < distance and lane_list[-1].outgoing_lanes):
                lane_list.append(lane_list[-1].outgoing_lanes[0])
                length_list.append(lane_list[-1].length)
            # print(f"lane_list: {[lane_list[i].lane_id for i in range(len(lane_list))]}" )
            # print(f"sum_length: {sum(length_list)}, length: {length_list}")
            
            index = 0
            way_pt_list = list()
            vector3_list = list()
            for s in range (0, distance, 1):
                if s > sum(length_list[:index+1]):
                    index += 1
                waypt = lane_list[index].from_lane_coord(RefLinePoint(s-sum(length_list[:index])))
                way_pt_list.append(waypt)
                vector3_list.append(Vector3(waypt.x, waypt.y, waypt.z))
                # print(waypt, s-sum(length_list[:index]))
            # print(way_pt_list)
            msg.waypoints = vector3_list
            msgs.lanes.append(msg)
            msg_id += 1
            # break
        self.map_lanes_msg = msgs
        # print(len(msgs.lanes))
        # raise
        return msgs
        
        # # print(ego_lane_coord)
        # ego_lane_pts = nearest_lane.project_along(ego_lane_coord.s, distance)
        # lane_set = set()
        # lane_ss = list()
        # for pt in ego_lane_pts:
        #     lane_set.add(pt[0])
        #     lane_ss.append(pt[1])
        #     print(pt[0].lane_id, pt[1])
        # print(lane_set)
        # print(lane_ss)
        raise 
        
        # self.dfs_lane(nearest_lane, lane_accessed, msgs)
        return msgs
    
    def step_vehicle_model(self, now: Pt, delta, dt):
        L_wheel_base = 2.9
        new_x = now.x + now.v * np.cos(now.yaw) * dt
        new_y = now.y + now.v * np.sin(now.yaw) * dt
        new_v = max(0.0, now.v + now.a * dt)
        new_yaw = now.yaw + now.v / L_wheel_base * np.tan(delta) * dt
        return Pt(new_x, new_y, now.t, new_yaw, new_v, now.a)

    def cal_pure_pursuit(self, ego_yaw, d_angle, L_ahead):
        L_wheel_base = 2.9
        alpha = d_angle - ego_yaw
        delta = np.arctan2(2.0 * L_wheel_base * np.sin(alpha) /
                         L_ahead, 1.0) if alpha != 0 else 0
        return delta
    
    def path_predict(self, obs_state: VehicleObservation, obs_lane: RoadMap.Lane, map: RoadMap):
        follow_path = []
        # collision_check_xytyaw_list = []

        s_now = obs_lane.to_lane_coord(Point(obs_state.position[0],obs_state.position[1]))
        
        x_now = obs_state.position[0]
        y_now = obs_state.position[1]
        yaw_now = obs_state.heading
        v_now = obs_state.speed
        a_now = 0
        max_jerk = 1  # 示例值
        break_acc = -1  # 示例值

        follow_path.append(Pt(x_now, y_now, 0, yaw_now, v_now, 0))
        # collision_check_xytyaw_list.append([x_now, y_now, 0.0, yaw_now])

        delta_i = 1
        for i in np.arange(delta_i, 51, delta_i):# 前瞻50点
            time_now = i * 0.1
            track_dis = obs_state.speed * 1.5
            track_pt = obs_lane.from_lane_coord(RefLinePoint(s_now.s + track_dis))
            d_angle = np.arctan2(track_pt.y - y_now, track_pt.x - x_now)
            delta = self.cal_pure_pursuit(yaw_now, d_angle, track_dis)
            a_now = max(break_acc, a_now - max_jerk * delta_i / 10)

            next_xyyawva = self.step_vehicle_model(Pt(
                x_now, y_now, time_now,yaw_now, v_now, a_now), delta, delta_i / 10)

            x_now, y_now, time_now, yaw_now, v_now, a_now = next_xyyawva
            s_now = obs_lane.to_lane_coord(Point(x_now, y_now, 0))
            follow_path.append(Pt(x_now, y_now, time_now, yaw_now, v_now, 0))

            # if i % 5 == 0:
                # collision_check_xytyaw_list.append(
                    # [x_now, y_now, time_now, yaw_now])

        return follow_path#, collision_check_xytyaw_list
    
    def obj_to_idx(self, obj_id: str):
        if obj_id not in self.obj_id_to_idx_dict:
            self.obj_id_to_idx_dict[obj_id] = len(self.obj_id_to_idx_dict)+1
        return self.obj_id_to_idx_dict[obj_id]

    def get_pp(self, objs: Dict[str, VehicleObservation], map: RoadMap):
        def get_traj(obj,lane,map):
            traj = prediction_traj()
            follow_path: list[Pt] = self.path_predict(
                obj, lane, map)
            for pt in follow_path:
                traj.predicted_traj_xyt.append(Vector3(pt.x, pt.y, pt.t))
                traj.predicted_traj_yawva.append(Vector3(pt.yaw, pt.v, pt.a))
                traj.prediction_probability = 0.33
            return traj
        
        msgs = self.perception_prediction_msg
        msgs.header.frame_id = "world"
        msgs.header.stamp = rospy.Time.now()
        msgs.header.seq = self.perception_prediction_msg.header.seq+1
        msgs.object_predictions.clear()
        for obj_id in objs.keys():
            msg=object_prediction()
            msg.obj_id=self.obj_to_idx(objs[obj_id].id)
            msg.obj_box.x = objs[obj_id].bounding_box[0]
            msg.obj_box.y = objs[obj_id].bounding_box[1]
            msg.obj_box.z = objs[obj_id].bounding_box[2]
            msg.obj_speed.x = objs[obj_id].speed
            msg.obj_speed.z = objs[obj_id].speed
            msg.world_coord.x = objs[obj_id].position[0]
            msg.world_coord.y = objs[obj_id].position[1]
            # msg.world_coord.z = objs[obj_id].position[2]
            msg.world_coord.z = objs[obj_id].heading

            # this_lane = map.lane_by_id(objs[obj_id].lane_id)
            this_lane = map.nearest_lane(
                Point(objs[obj_id].position[0], objs[obj_id].position[1], objs[obj_id].position[2]))
            left_lane = this_lane.lane_to_left
            right_lane = this_lane.lane_to_right

            msg.prediction_trajs.append(get_traj(objs[obj_id],this_lane,map))
            if left_lane[1] and (type(left_lane[0]) is not type(None)):
                msg.prediction_trajs.append(
                    get_traj(objs[obj_id], left_lane[0], map))
            if right_lane[1] and (type(right_lane[0]) is not type(None)):
                msg.prediction_trajs.append(
                    get_traj(objs[obj_id], right_lane[0], map))
            
            # msg.lane_coord
            
            msgs.object_predictions.append(msg)
        return msgs

    def publish_all(self,):
        self.pub_ego_state.publish(self.ego_state_msg)
        self.pub_lanes.publish(self.map_lanes_msg)
        # self.pub_lanes.publish(map_lanes())
        self.pub_perception_prediction.publish(self.perception_prediction_msg)
        # self.pub_perception_prediction.publish(perception_prediction())

    def get_action(self, ego_state_msg_sub: ego_state):
        dx, dy, dyaw = 0, 0, 0
        dx = ego_state_msg_sub.world_coord.x - self.ego_state_msg.world_coord.x
        dy = ego_state_msg_sub.world_coord.y - self.ego_state_msg.world_coord.y
        dyaw = ego_state_msg_sub.world_coord.z - self.ego_state_msg.world_coord.z
        while(dyaw > np.pi):
            dyaw -= 2*np.pi
        while(dyaw < -np.pi):
            dyaw += 2*np.pi
        
        action = (dx, dy, dyaw)
        return action

    def const_v_action(self):
        dx, dy, dyaw = 0, 0, 0
        lane_pt = self.ego_lane.to_lane_coord(Point(self.ego_obs.position[0], self.ego_obs.position[1]))
        yaw = self.ego_obs.heading
        ds = self.ego_obs.linear_velocity[2]*self.dt
        if lane_pt.s + ds > self.ego_lane.length:
            ds = lane_pt.s + ds - self.ego_lane.length
            next_lane = self.ego_lane.outgoing_lanes[0]
            new_lane_coord = RefLinePoint(0 + ds, lane_pt.t) 
            yaw_vector = next_lane.vector_at_offset(new_lane_coord.s)
            new_world_coord = next_lane.from_lane_coord(new_lane_coord)
            
        else:
            new_lane_coord = RefLinePoint(lane_pt.s + ds, lane_pt.t)
            yaw_vector = self.ego_lane.vector_at_offset(new_lane_coord.s)
            new_world_coord = self.ego_lane.from_lane_coord(new_lane_coord)
        new_yaw = math.atan2(yaw_vector[1], yaw_vector[0])
        dx = new_world_coord.x - self.ego_obs.position[0]
        dy = new_world_coord.y - self.ego_obs.position[1]
        dyaw = new_yaw - yaw
        while(dyaw > np.pi):
            dyaw -= 2*np.pi
        while(dyaw < -np.pi):
            dyaw += 2*np.pi
        action = (dx, dy, dyaw)
        return action