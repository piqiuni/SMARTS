#!/usr/bin/env python3.8

from collections import Counter
import contextlib
import math
import os
import pickle
from re import S
import pandas as pd
import time
from typing import Dict, List, Tuple
from matplotlib import pyplot as plt
import matplotlib
from matplotlib.animation import FFMpegWriter
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse
import numpy as np
from shapely.geometry import Polygon
import rosbag
from simulator.decode_utils import FrameControlButton
from simulator.ref_path import Ref_path
import matplotlib.colors as colors

# class ModelState:
#     def __init__(self, x, y, v, vs, vd, a, s, d, yaw, lane_id=-1, spline_id=-1):
#         self.x = x
#         self.y = y
#         self.v = v
#         self.vs = vs
#         self.vd = vd
#         self.a = a
#         self.s = s
#         self.d = d
#         self.yaw = yaw
#         self.lane_id = lane_id
#         self.spline_id = spline_id

class Prediction(object):
    def __init__(self, ) -> None:
        self.pred_path = [] # [[x,y,yaw,v],...]
        self.pred_prob = [] # [0.8,0.2]
        
    def add_pred(self, paths, probs):
        if not paths:
            return
        for path,prob in zip(paths, probs):
            self.pred_path.append(path)
            self.pred_prob.append(prob)

class Car_Struct(object):
    def __init__(self, id, car_type, x, y, yaw, v, a, j, length, width, lane_id=-1, spline_id=-1):
        self.id = id
        self.type = car_type
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.a = a
        self.j = j
        self.length = length
        self.width = width
        self.lane_id = lane_id
        self.spline_id = spline_id
        self.prediction = Prediction()
    
    def add_pred(self, paths, probs):
        self.prediction.add_pred(paths,probs)
        

class PlannedPt(object):
    def __init__(self, x, y, t, yaw, v, a):
        self.x = x
        self.y = y
        self.t = t
        self.yaw = yaw
        self.v = v
        self.a = a
    def __getitem__(self, index): # for visit with PlannedPt[0]
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        elif index == 2:
            return self.t
        elif index == 3:
            return self.yaw
        elif index == 4:
            return self.v
        elif index == 5:
            return self.a
        else:
            raise IndexError("Invalid index")
    def __iter__(self): # for zip(*List[PlannedPt])
        return iter((self.x, self.y, self.t, self.yaw, self.v, self.a))

class Data_Struct(object):
    def __init__(self, ):
        self.scene_name = None
        self.scene_id = None
        self.saved_time = None
        self.result = None
        self.collision_frame = []
        self.first_frame = 0
        self.max_frame = 0
        self.start_time = 0 
        self.max_dataset_time = 0
        self.stop_time = 0
        self.social_dict : Dict[int, Dict[int, Car_Struct]] = {}
        self.ego_dict : Dict[int, Car_Struct] = {}
        self.planning_path_dict : Dict[int, List[PlannedPt]] = {}
        self.lane_dict : Dict[int, Tuple[float, float]] = {} # center line of all lanes, left_edge, right_edge
        self.ego_speed : Dict[int, float] = {}
        self.ego_acc : Dict[int, float] = {}
        self.ego_jerk : Dict[int, float] = {}
        self.decision_dict = {}
        self.decision_time_ms = []
        self.state_str = ''
        
        


class Drive_Test(object):
    def __init__(self, root_dir, case_name:str):
        # self.ego_length = 5.0
        # self.ego_width = 2.0
        
        time_now = time.strftime('%Y-%m-%d-%H-%M-%S')
        self.start_time = time_now
        # 'test_logs_hastar'
        self.root_dir = os.path.join(root_dir, case_name)
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir)
        self.log_dir = case_name + '_' + time_now
        self.folder_path = os.path.join(self.root_dir, self.log_dir)
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
        self.data : Data_Struct = None
        self.save_dis = 100
        self.last_collision_id = None
        self.scene_info_dict = {}
        self._init_info_dict()
        pass
    def _init_info_dict(self):
        self.scene_info_dict['scene_id'] = []
        self.scene_info_dict['scene_name'] = []
        self.scene_info_dict['dataset timestamp'] = []
        self.scene_info_dict['sim timestamp'] = []
        self.scene_info_dict['stop time(ms)'] = []
        self.scene_info_dict['result'] = []
        self.scene_info_dict['collision'] = []
        self.scene_info_dict['collision state'] = []
        self.scene_info_dict['start speed(m/s)'] = []
        self.scene_info_dict['average speed(m/s)'] = []
        self.scene_info_dict['max acc +(m/s2)'] = []
        self.scene_info_dict['max acc -(m/s2)'] = []
        self.scene_info_dict['max jerk(m/s3)'] = []
        self.scene_info_dict['mid state'] = []
        self.scene_info_dict['v_std_dev'] = []
        self.scene_info_dict['a_std_dev'] = []
        self.scene_info_dict['UD'] = []
        # self.scene_info_dict
        
    
    def new_scene(self, scene_id, scene_name, start_frame, max_time, start_speed=0):
        self.data = Data_Struct()
        # time_now = time.strftime('%H-%M-%S')
        saved_name = f"{scene_id}_t{start_frame}_v{start_speed:.1f}_{scene_name}.pkl"
        self.saved_file = os.path.join(self.root_dir, self.log_dir, saved_name)
        
        self.data.scene_name = saved_name
        self.data.scene_id = scene_id
        self.data.start_time = start_frame
        self.data.max_dataset_time =  max_time
        
    def save_scene(self, result, max_frame):
        time_now = time.strftime('%Y-%m-%d-%H-%M-%S')
        self.data.saved_time = time_now
        self.data.result = result
        self.data.max_frame = max_frame
        self.add_info_dict()
        with open(self.saved_file, 'wb') as f:
            pickle.dump(self.data, f)
        
        
                 
    def push_this_frame(self, frame_id, ego_state: Car_Struct, social_states: List[Car_Struct], path: List[PlannedPt], decision=None, decision_time_ms=None):
        ego_polygon = self.get_polygon(ego_state)
        self.data.ego_dict[frame_id] = ego_state
        self.data.ego_speed[frame_id] = ego_state.v
        if ego_state.v < 0.5:
            self.data.stop_time += 100
        self.data.ego_acc[frame_id] = ego_state.a
        self.data.ego_jerk[frame_id] = ego_state.j
        self.data.planning_path_dict[frame_id] = path
        self.data.decision_dict[frame_id] = decision
        self.data.decision_time_ms.append(decision_time_ms)
        state_dict = {}
        for state in social_states:
            if self.cal_dis((ego_state.x, ego_state.y), (state.x, state.y)) > self.save_dis:
                continue
            polygon = self.get_polygon(state)
            state_dict[state.id] = state
            if state.id == 0 or state.id == -1:
                continue
            if state.type == 32:
                continue
            collision_flag, collision_state = self.is_collision(ego_polygon, ego_state.yaw, polygon)
            if collision_flag == True :
                collision_id = state.id
                if collision_id != self.last_collision_id:
                    self.data.collision_frame.append([frame_id, collision_state])
                self.last_collision_id = collision_id
                # print(self.last_collision_id)
        self.data.social_dict[frame_id] = state_dict
        
    def add_lane(self, id, lane):
        self.data.lane_dict[id] = lane
        
    def get_polygon(self, ego_state):
        x, y, yaw, l, w = ego_state.x, ego_state.y, ego_state.yaw, ego_state.length, ego_state.width
        polygon = Polygon([[x - l/2 * math.cos(yaw) + w/2 * math.sin(yaw),
                                                y - l/2 * math.sin(yaw) - w/2 * math.cos(yaw)],
                                                [x - l/2 * math.cos(yaw) - w/2 * math.sin(yaw),
                                                y - l/2 * math.sin(yaw) + w/2 * math.cos(yaw)],
                                                [x +l/2 * math.cos(yaw) - w/2 * math.sin(yaw),
                                                y + l/2 * math.sin(yaw) + w/2 * math.cos(yaw)],
                                                [x + l/2 * math.cos(yaw) + w/2 * math.sin(yaw),
                                                y + l/2 * math.sin(yaw) - w/2 * math.cos(yaw)]])
        return polygon
    
    def is_collision(self, ego_polygon : Polygon, ego_heading, polygon):
        res = ego_polygon.intersects(polygon)
        if res == False:
            return False, 'None'
        else:
            ego_center = ego_polygon.centroid
            social_center = polygon.centroid
            vector = np.array((ego_center.x, ego_center.y)) - np.array((social_center.x, social_center.y))
            angle = np.arctan2(vector[1], vector[0])
            angle_diff = angle - ego_heading
            angle_diff = np.mod(angle_diff + np.pi, 2*np.pi) - np.pi
            #front
            if abs(angle_diff) > np.pi - 0.5:
                result = 'front'
            # backward
            elif abs(angle_diff) < 0.5:
                result = 'backward'
            else:
                result = 'side'
            return True, result

    
    def cal_dis(self, car1, car2):
        return math.sqrt((car1[0] - car2[0])**2 + (car1[1] - car2[1])**2)
    
    def add_info_dict(self, ):
        self.scene_info_dict['scene_id'].append(self.data.scene_id)
        self.scene_info_dict['scene_name'].append(self.data.scene_name)
        self.scene_info_dict['dataset timestamp'].append(f"0-{self.data.max_dataset_time}")
        self.scene_info_dict['sim timestamp'].append(f"{self.data.start_time}-{self.data.max_frame}")
        self.scene_info_dict['stop time(ms)'].append(self.data.stop_time)
        self.scene_info_dict['result'].append(self.data.result)
        self.scene_info_dict['collision'].append(self.data.collision_frame)
        self.scene_info_dict['collision state'].append(None)
        self.scene_info_dict['start speed(m/s)'].append(self.data.ego_dict[self.data.start_time].v)
        self.scene_info_dict['average speed(m/s)'].append(np.mean(list(self.data.ego_speed.values())))
        self.scene_info_dict['max acc +(m/s2)'].append(max(list(self.data.ego_acc.values())))
        self.scene_info_dict['max acc -(m/s2)'].append(min(list(self.data.ego_acc.values())))
        self.scene_info_dict['max jerk(m/s3)'].append(max(np.abs(list(self.data.ego_jerk.values()))))
        self.scene_info_dict['mid state'].append(self.data.state_str)
        self.scene_info_dict['v_std_dev'].append(self.cal_std_deviation(self.data.ego_speed.values()))
        self.scene_info_dict['a_std_dev'].append(self.cal_std_deviation(self.data.ego_acc.values()))
        self.scene_info_dict['UD'].append(self.cal_uncomfortable_deceleration(list(self.data.ego_acc.values())))
        
    def add_info_dict_failed(self, ):
        self.scene_info_dict['scene_id'].append(self.data.scene_id)
        self.scene_info_dict['scene_name'].append(self.data.scene_name)
        self.scene_info_dict['dataset timestamp'].append(f"0-{self.data.max_dataset_time}")
        self.scene_info_dict['sim timestamp'].append(f"{self.data.start_time}-{self.data.max_frame}")
        self.scene_info_dict['stop time(ms)'].append(None)
        self.scene_info_dict['result'].append('Break')
        self.scene_info_dict['collision'].append(None)
        self.scene_info_dict['collision state'].append(None)
        self.scene_info_dict['start speed(m/s)'].append(None)
        self.scene_info_dict['average speed(m/s)'].append(None)
        self.scene_info_dict['max acc +(m/s2)'].append(None)    
        self.scene_info_dict['max acc -(m/s2)'].append(None)
        self.scene_info_dict['max jerk(m/s3)'].append(None)
        self.scene_info_dict['mid state'].append(self.data.state_str)
        self.scene_info_dict['v_std_dev'].append(None)
        self.scene_info_dict['a_std_dev'].append(None)
        self.scene_info_dict['UD'].append(None)
        
    def add_info_dict_read(self, ):
        self.scene_info_dict['scene_id'].append(self.data.scene_id)
        self.scene_info_dict['scene_name'].append(self.data.scene_name)
        self.scene_info_dict['dataset timestamp'].append(f"0-{self.data.max_dataset_time}")
        self.scene_info_dict['sim timestamp'].append(f"{self.data.start_time}-{self.data.max_frame}")
        self.scene_info_dict['stop time(ms)'].append(self.data.stop_time)
        self.scene_info_dict['result'].append(self.data.result)
        self.scene_info_dict['collision'].append(self.data.collision_frame)
        self.scene_info_dict['collision state'].append(None)
        self.scene_info_dict['start speed(m/s)'].append(self.data.ego_dict[self.data.start_time].v)
        self.scene_info_dict['average speed(m/s)'].append(np.mean(list(self.data.ego_speed.values())))
        self.scene_info_dict['max acc +(m/s2)'].append(max(list(self.data.ego_acc.values())))
        self.scene_info_dict['max acc -(m/s2)'].append(min(list(self.data.ego_acc.values())))
        self.scene_info_dict['max jerk(m/s3)'].append(max(np.abs(list(self.data.ego_jerk.values()))))
        self.scene_info_dict['mid state'].append(self.data.state_str)
        self.scene_info_dict['v_std_dev'].append(self.cal_std_deviation(self.data.ego_speed.values()))
        self.scene_info_dict['a_std_dev'].append(self.cal_std_deviation(self.data.ego_acc.values()))
        self.scene_info_dict['UD'].append(self.cal_uncomfortable_deceleration(list(self.data.ego_acc.values())))
        
    def cal_std_deviation(self, values):
        value_list = list(values)
        values = np.array(value_list)
        return np.std(values)
    
    def cal_uncomfortable_deceleration(self, acc_list):
       # count continuous deceleration < -1.6 as one time
        accs = np.array(acc_list)
        uncomfortable_deceleration_count = 0
        i = 0
        while i < len(accs):
            if accs[i] < -1.6:
                uncomfortable_deceleration_count += 1
                while 1:
                    if i == len(accs)-1:
                        break
                    if accs[i+1] > -1.6:
                        break
                    i += 1
            i += 1
        return uncomfortable_deceleration_count
    
    def save_to_excel(self):
        saved_name =  'result' + '_' + self.start_time + '.xlsx'
        saved_file = os.path.join(self.root_dir, self.log_dir, saved_name)
        
        # 将数据转换为DataFrame对象
        df = pd.DataFrame(self.scene_info_dict)
        
        # 将DataFrame写入Excel文件
        df.to_excel(saved_file, index=False)
        pass
        
class ReadLog(object):
    def __init__(self, data:Data_Struct):
        self.data = data
        self.lane_spline : Dict[int, Ref_path] = {}
        # print(len(self.data.ego_speed), len(self.data.ego_dict))
        self.mean_speed = np.mean(list(self.data.ego_speed.values()))
        self.max_speed = max(np.abs(list(self.data.ego_speed.values())))
        self.max_acc = max(np.abs(list(self.data.ego_acc.values())))
        self.get_collision()
        self.get_stop_time()
        self.info_str = self.get_info()
        self.first_frame = list(self.data.ego_dict.keys())[0]
        self.max_frame = list(self.data.ego_dict.keys())[-1]
    
        
        
        pass
    
    def get_collision(self):
        if len(self.data.collision_frame) != 0:
            self.is_collision = True
        else:
            self.is_collision = False
        print("collision : [frame, direction]")
        print(self.data.collision_frame)
    
    def get_info(self):
        
        print("scene name:{}, result:{}, collision times:{}".format(self.data.scene_name, \
            self.data.result, len(self.data.collision_frame)))
        print("average speed:{:.1f}, max speed:{:.1f}, max acceleration:{:.1f}, max jerk:{:.1f}".format(self.mean_speed, \
                self.max_speed, self.max_acc, self.max_acc))
        # print("average decision time:{:.1f}ms".format(np.mean(self.data.decision_time_ms)))
        
        info_str = "Collision:" + str(self.is_collision)
        if self.is_collision == True:
            info_str += ", " + str(self.data.collision_frame[0])
        info_str += "  Stop time(ms):" + str(self.stop_time)
        return info_str
    
    def get_stop_time(self):
        stop_count = 0
        for key, speed in self.data.ego_speed.items():
            if speed < 0.5:
                stop_count += 1
        self.stop_time = int(stop_count * 100)
        print("Stop time: {}ms".format(self.stop_time))
        
    def get_dataset_ego_va(self):
        v_list = []
        
        for timestamp in range(len(self.data.social_dict)):
            if 0 not in self.data.social_dict[timestamp].keys():
                continue
            state = self.data.social_dict[timestamp][0]
            v_list.append(state.v)
            
        a_list = []
        for i in range(len(v_list) - 1):
            a_list.append((v_list[i+1] - v_list[i])/0.1)
        j_list = []
        for i in range(len(a_list) - 1):
            j_list.append((a_list[i+1] - a_list[i])/0.1)
        print([round(v,2) for v in v_list])
        print([round(v,2) for v in a_list], max(a_list), min(a_list))
        print([round(v,2) for v in j_list], max(j_list), min(j_list))
    
    def _init_display(self, only_ax1=False):
        
        
        self.ego_patch = None
        self.planning_dict = {}
        self.patches_dict = {}
        self.text_dict = {}
        self.arrow_dict = {}
        self._get_disp_type()
        self.pred_dict = {}
        
        if only_ax1:
            self.fig = plt.figure(figsize=(19.2, 10.8))
            self.ax1 = self.fig.add_subplot()
            # self.ax1.set_facecolor((0.9, 0.9, 0.9))
            return 
        
        self.fig = plt.figure(figsize=(19.2, 10.8))
        # self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(1, 2, figsize=(10, 5))
        # self.fig.set_size_inches(16*1.3,9*1.3)
        gs = GridSpec(90, 160)
        # plt.rcParams['axes.facecolor'] = (0.9, 0.9, 0.9)#'gray'
        self.ax1 = self.fig.add_subplot(gs[:, :90])
        self.ax1.set_facecolor((0.9, 0.9, 0.9))
        self.ax2 = self.fig.add_subplot(gs[0:40, 95:125])
        self.ax3 = self.fig.add_subplot(gs[50:, 95:125])
        self.ax4 = self.fig.add_subplot(gs[0:40, 130:])
        self.ax5 = self.fig.add_subplot(gs[50:, 130:])
        
        
        self.fig.text(0.65, 0.91, 'History', ha='center', fontsize=18, color='black')
        self.fig.text(0.83, 0.91, 'Planned', ha='center', fontsize=18, color='black')
        
        
        
    def display_scene(self):
        self._init_display()
        self.draw_lane()
        
        # plt.text(10, 10, str("speed"), fontsize=10, ha='center')
        
        plt.suptitle(self.info_str,fontsize=15)
        
        plt.ion()
        plt.show()
        
        time.sleep(1)
        # print("input to start or after 1s")
        # input()
        
        first_frame = list(self.data.ego_dict.keys())[0]
        max_frame = list(self.data.ego_dict.keys())[-1]
        
        # timestamp_min, timestamp_max = first_frame, max_frame
        # button_pp = FrameControlButton([0.2, 0.01, 0.05, 0.05], '<<')
        # button_p = FrameControlButton([0.27, 0.01, 0.05, 0.05], '<')
        # button_f = FrameControlButton([0.4, 0.01, 0.05, 0.05], '>')
        # button_ff = FrameControlButton([0.47, 0.01, 0.05, 0.05], '>>')
        # button_play = FrameControlButton([0.6, 0.01, 0.1, 0.05], 'play')
        # button_pause = FrameControlButton([0.71, 0.01, 0.1, 0.05], 'pause')
        
        
        for i in range(first_frame, max_frame):
            t0 = time.time()
            percentage = int(100*(i-first_frame)/(max_frame - first_frame))
            self.ax1.set_title("{}\ntime(ms) = {} / {} ({}%)".format(self.data.scene_name, (i)*100, self.data.max_frame*100, percentage))
            self.draw_ego(i)
            self.draw_path(i)
            self.draw_socials(i)
            self.draw_v_a(i-first_frame)
            self.draw_planned_v_a(i)
            self.fig.canvas.draw()
            t_cost = time.time() - t0
            # print(t_cost)
            if t_cost >= 0.1:
                plt.pause(0.01)
            else:
                plt.pause(0.1 - t_cost)
            # print()
            # input()
        plt.close()
         
    def display_scene_new(self):
        self._init_display()
        self.draw_lane()
        
        # plt.text(10, 10, str("speed"), fontsize=10, ha='center')
        
        plt.suptitle(self.info_str,fontsize=15)
        
        plt.ion()
        plt.show()
        
        time.sleep(1)
        # print("input to start or after 1s")
        # input()
        
        first_frame = list(self.data.ego_dict.keys())[0]
        max_frame = list(self.data.ego_dict.keys())[-1]
        for i in range(first_frame, max_frame):
            t0 = time.time()
            percentage = int(100*(i-first_frame)/(max_frame - first_frame))
            self.ax1.set_title("{}\ntime(ms) = {} / {} ({}%)".format(self.data.scene_name, (i)*100, self.data.max_frame*100, percentage))
            self.draw_ego(i)
            self.draw_path(i)
            self.draw_socials(i)
            self.draw_v_a(i-first_frame)
            self.draw_planned_v_a(i)
            self.fig.canvas.draw()
            t_cost = time.time() - t0
            # print(t_cost)
            if t_cost >= 0.1:
                plt.pause(0.01)
            else:
                plt.pause(0.1 - t_cost)
            # print()
            # input()
        plt.close()
         
    def record_video(self, title, root_path, video_name):
        self._init_display()
        self.draw_lane()
        metadata = dict(title=title, artist='Matplotlib',comment='')
        writer = FFMpegWriter(fps=10, metadata=metadata)
        
        
        video_dir_path = os.path.join(root_path, 'video')
        if not os.path.exists(video_dir_path):
            os.makedirs(video_dir_path)
        
        video_name = video_name + '.mp4'
        video_path = os.path.join(video_dir_path, video_name)
        
        plt.suptitle(self.info_str,fontsize=15)
        plt.ion()
        # plt.show()
        
        first_frame = list(self.data.ego_dict.keys())[0]
        max_frame = list(self.data.ego_dict.keys())[-1]
        with writer.saving(self.fig, video_path, 300):
            for i in range(first_frame, max_frame):
                percentage = int(100*(i-first_frame)/(max_frame - first_frame))
                self.ax1.set_title("{}\ntime(ms) = {} / {} ({}%)".format(self.data.scene_name, (i)*100, self.data.max_frame*100, percentage))
                self.draw_ego(i)
                self.draw_path(i)
                self.draw_socials(i)
                self.draw_v_a(i-first_frame)
                self.draw_planned_v_a(i)
                self.fig.canvas.draw()
                plt.pause(0.01)
                writer.grab_frame()
                # print()
            plt.close()
        print("record ok")
        
    def record_video_show(self, title, root_path, video_name):
        self._init_display(True)
        self.draw_lane()
        metadata = dict(title=title, artist='Matplotlib',comment='')
        writer = FFMpegWriter(fps=10, metadata=metadata)
        
        
        video_dir_path = os.path.join(root_path, 'video')
        if not os.path.exists(video_dir_path):
            os.makedirs(video_dir_path)
        
        video_name = video_name + '.mp4'
        video_path = os.path.join(video_dir_path, video_name)
        
        plt.suptitle(self.info_str,fontsize=15)
        plt.ion()
        # plt.show()
        
        first_frame = list(self.data.ego_dict.keys())[0]
        max_frame = list(self.data.ego_dict.keys())[-1]
        with writer.saving(self.fig, video_path, 300):
            for i in range(first_frame, max_frame):
                percentage = int(100*(i-first_frame)/(max_frame - first_frame))
                self.ax1.set_title("{}\ntime(ms) = {} / {} ({}%)".format(self.data.scene_name, (i)*100, self.data.max_frame*100, percentage))
                self.draw_ego(i)
                self.draw_path(i)
                self.draw_socials(i)
                # self.draw_v_a(i-first_frame)
                # self.draw_planned_v_a(i)
                self.fig.canvas.draw()
                plt.pause(0.01)
                writer.grab_frame()
                # print()
            plt.close()
        print("record ok")    
       
        
    def draw_lane(self, draw_edge = True):
            
        for id, value in self.data.lane_dict.items():
            # print(value)
            # xy_list = [(pt['route_point_x_relative'], pt['route_point_y_relative']) for pt in value]
            x ,y = list(zip(*value))
            self.lane_spline[id] = Ref_path(value)
            left_edge = []
            right_edge = []
            lane_width = 3.5
            
            type_dict = dict(color='y', linewidth=1, linestyle = '--', zorder=10)
            self.ax1.plot(x, y, **type_dict)
            
            if draw_edge:
                for s in self.lane_spline[id].ref_path_s:
                    left_edge.append(self.lane_spline[id].coord_lane_to_world((s, -lane_width/2)))
                    right_edge.append(self.lane_spline[id].coord_lane_to_world((s, +lane_width/2)))
                type_dict = dict(color='g', linewidth=2, linestyle = '-', zorder=10)
                x,y = list(zip(*left_edge))
                self.ax1.plot(x, y, **type_dict)
                x,y = list(zip(*right_edge))
                self.ax1.plot(x, y, **type_dict)
            mid_s = self.lane_spline[id].max_s / 2
            mid_pt = self.lane_spline[id].coord_lane_to_world((mid_s, 0))
            self.ax1.text(mid_pt[0], mid_pt[1], str(id), horizontalalignment='center', zorder=30)
            
            
    def draw_ego(self, i):
        if self.ego_patch != None:
            self.ego_patch.remove()
        
        ego_state = self.data.ego_dict[i]
        # print(ego_state.yaw)
        # rect = plt.Rectangle((ego_state.x-2.5, ego_state.y-1.25), 5, 2, angle=np.rad2deg(ego_state.yaw), color = 'g', fill=True)
        rect = matplotlib.patches.Polygon(self.get_vehicle_boundary(ego_state), closed=True, facecolor='g', 
                                                            zorder=40)
        self.ax1.add_patch(rect)
        self.ego_patch = rect
        DisplayDIS = 50
        self.ax1.set_xlim([ego_state.x-DisplayDIS-0, ego_state.x+DisplayDIS+0])
        # self.ax1.set_xlim([ego_state.x-DisplayDIS-20, ego_state.x+DisplayDIS+20])
        self.ax1.set_ylim([ego_state.y-DisplayDIS-10, ego_state.y+DisplayDIS+10])
        # self.ax1.set_ylim([ego_state.y-DisplayDIS, ego_state.y+DisplayDIS])
        self.ax1.set_aspect('equal')
        
    def draw_path(self, i):
        # print(self.data.planning_path_dict.keys())
        # print(self.data.planning_path_dict[i])
        path = self.data.planning_path_dict[i]
        if not path:
            return
        # print(path)
        # print(path, type(path))
        x,y,t,yaw,v,a = list(zip(*path))
        
        norm = colors.Normalize(vmin=0, vmax=15)
        
        if self.planning_dict == {}:
            self.planning_dict = self.ax1.scatter(x, y, c = v, cmap = 'viridis', norm = norm, s = [15 for i in range(len(x))], zorder = 50)
        else:
            self.planning_dict.remove()
            self.planning_dict = self.ax1.scatter(x, y, c = v, cmap = 'viridis', norm = norm, s = [15 for i in range(len(x))], zorder = 50)
      
        
    def draw_socials(self, i, draw_ellipse=False, draw_id=True, draw_arrow = False):
        keys = list(self.patches_dict.keys())
        for key in keys:
            if key in self.patches_dict:
                self.patches_dict[key].remove()
                self.patches_dict.pop(key)
            if key in self.text_dict:
                self.text_dict[key].remove()
                self.text_dict.pop(key)
            if key in self.arrow_dict:
                self.arrow_dict[key].remove()
                self.arrow_dict.pop(key)
        
        keys = list(self.pred_dict.keys())
        for key in keys:
            for scatter in self.pred_dict[key]:
                scatter.remove()
            self.pred_dict.pop(key)
        print(f"len: {len(self.data.social_dict[i])}")
        # print(f"len: {(self.data.social_dict[i])}")
        
        for key, state in self.data.social_dict[i].items():
            if state.type:
                car_type = int(state.type)
            else:
                car_type = None
            if car_type in self.vehicle_types:
                color = 'blue'
            elif car_type in self.pedestrian_types:
                color = 'orange'
            elif car_type in self.cyclist_types:
                color = 'orchid'
            else:
                color = 'gray'
            
            # if int(state.id) == 0:
            #     color = 'red'
            if int(state.id) == -1:
                color = 'red'    
                # continue
            
            # rect = plt.Rectangle((state.x-l/2, state.y-w/2), l, w, angle=np.rad2deg(state.yaw), color = color, fill=True)
            rect = matplotlib.patches.Polygon(self.get_vehicle_boundary(state), closed=True, facecolor=color, 
                                                            zorder=20)
                   
            self.ax1.add_patch(rect)
            self.patches_dict[key] = rect
            draw_id = False
            if draw_id:
                self.text_dict[key] = self.ax1.text(state.x, state.y, \
                    str(key), fontsize=8, horizontalalignment='center', zorder=30)
            draw_arrow = True
            if draw_arrow:
                if state.v < 1:
                    continue
                pos = (state.x + state.length*0.5*np.cos(state.yaw), state.y + state.length*0.5*np.sin(state.yaw))
                arrow_length = state.v * 0.7 + 3#2 + 1
                arrow_point = (state.x + arrow_length * np.cos(state.yaw), state.y + arrow_length * np.sin(state.yaw))
                arrow = matplotlib.patches.FancyArrowPatch(pos, arrow_point, arrowstyle='->', linewidth = 3, mutation_scale=20, color=color)
                # 添加箭头对象到轴
                self.ax1.add_patch(arrow)
                self.arrow_dict[key] = arrow


            pred = []
            if not state.prediction.pred_path:
                continue
            if not state.prediction.pred_path[0]:
                continue
            print(state.prediction.pred_path, len(state.prediction.pred_path))
            # print(state.prediction.pred_path)
            for path in state.prediction.pred_path:
                path_xyyaw = list(zip(*path))
                # print(path[0:2])
                pred.append(self.ax1.scatter(path_xyyaw[0], path_xyyaw[1], c = color, s = [5 for i in range(len(path_xyyaw[0]))], zorder=20))
                if len(path_xyyaw) == 5 and draw_ellipse:
                    for i in range(len(path_xyyaw[0])):
                        ellipse = Ellipse([path_xyyaw[0][i], path_xyyaw[1][i]], path_xyyaw[3][i], path_xyyaw[4][i], angle=path_xyyaw[2][i], facecolor='none', edgecolor='y', zorder=0)
                        pred.append(self.ax1.add_patch(ellipse))
                
                
            self.pred_dict[key] = pred
                    
            
    def draw_v_a(self, i):
        t_list = np.arange(i) / 10
        speed_list = list(self.data.ego_speed.values())[0:i]
        acc_list = list(self.data.ego_acc.values())[0:i]
        self.ax2.clear()
        self.ax2.plot(t_list, speed_list, label='speed')
        self.ax2.set_xlabel('time')
        self.ax2.set_ylabel('speed')
        self.ax2.legend()
        if speed_list != []:
            self.ax2.set_ylim(0, self.max_speed+1)
        self.ax2.set_title('Ego Speed  max={:.1f}'.format(self.max_speed))
        
        self.ax3.clear()
        self.ax3.plot(t_list, acc_list, label='acceleration')
        self.ax3.set_xlabel('time')
        self.ax3.set_ylabel('acc')
        self.ax3.set_ylim(-3,3)
        self.ax3.legend()
        self.ax3.set_title('Ego Acceleration  max={:.1f}'.format(self.max_acc))
        pass
    
    def draw_planned_v_a(self, i):
        if not self.data.planning_path_dict[i]:
            return
        if len(self.data.planning_path_dict[i][0]) == 5:
            return
        # t_list = np.arange(0, 5.1, 0.1)
        x,y,t,yaw,v,a = list(zip(*self.data.planning_path_dict[i]))
        self.ax4.clear()
        self.ax4.plot(t, v, label='speed')
        self.ax4.set_xlabel('time')
        self.ax4.set_ylabel('speed')
        self.ax4.legend()
        self.ax4.set_ylim(0, self.max_speed+1)
        self.ax4.set_title('Ego Speed')
        
        self.ax5.clear()
        self.ax5.plot(t[1:], a[1:], label='acceleration')
        self.ax5.set_xlabel('time')
        self.ax5.set_ylabel('acc')
        self.ax5.set_ylim(-3,3)
        self.ax5.legend()
        self.ax5.set_title('Ego Acceleration')
    
    def _get_disp_type(self):
        self.vehicle_types = [1, 2, 3, 4, 5, 6, 22]
        self.cyclist_types = [8, 9, 10,11, 23, 21]
        self.pedestrian_types = [7]
        
    def get_vehicle_boundary(self, state:Car_Struct):
        """
        根据车辆坐标、偏航角、车长和车宽计算车辆四个边界点位置
        :return: 四个边界点的位置列表[(x1,y1),(x2,y2),(x3,y3),(x4,y4)]
        """
        length, width, x, y, yaw = state.length, state.width, state.x, state.y, state.yaw
        front_right = np.array([length/2, -width/2])
        front_left = np.array([length/2, width/2])
        rear_left = np.array([-length/2, width/2])
        rear_right = np.array([-length/2, -width/2])

        rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw)],
                                    [np.sin(yaw), np.cos(yaw)]])
        front_right = np.matmul(rotation_matrix, front_right)
        front_left = np.matmul(rotation_matrix, front_left)
        rear_left = np.matmul(rotation_matrix, rear_left)
        rear_right = np.matmul(rotation_matrix, rear_right)

        front_right += np.array([x, y])
        front_left += np.array([x, y])
        rear_left += np.array([x, y])
        rear_right += np.array([x, y])

        return [front_right, front_left, rear_left, rear_right]

    def update_plot(self, frame):
        first_frame = list(self.data.ego_dict.keys())[0]
        max_frame = list(self.data.ego_dict.keys())[-1]
        percentage = int(100*(frame-first_frame)/(max_frame - first_frame))
        self.ax1.set_title("{}\ntime(ms) = {} / {} ({}%)".format(self.data.scene_name, (frame)*100, self.data.max_frame*100, percentage))
        with contextlib.redirect_stdout(None): 
            self.draw_ego(frame)
            self.draw_path(frame)
            self.draw_socials(frame)
            self.draw_v_a(frame-first_frame)
            self.draw_planned_v_a(frame)
            self.fig.canvas.draw()
            plt.pause(0.01)
        

class test_gt_info(object):
    def __init__(self, data:Data_Struct) -> None:
        self.data = data
        self.scene_info_dict = {}
        self._init_info_dict()
    
    def _init_info_dict(self):
        self.scene_info_dict['scene_id'] = []
        self.scene_info_dict['scene_name'] = []
        self.scene_info_dict['dataset timestamp'] = []
        self.scene_info_dict['sim timestamp'] = []
        self.scene_info_dict['stop time(ms)'] = []
        self.scene_info_dict['result'] = []
        self.scene_info_dict['collision'] = []
        self.scene_info_dict['collision state'] = []
        self.scene_info_dict['start speed(m/s)'] = []
        self.scene_info_dict['average speed(m/s)'] = []
        self.scene_info_dict['max acc +(m/s2)'] = []
        self.scene_info_dict['max acc -(m/s2)'] = []
        self.scene_info_dict['max jerk(m/s3)'] = []
        self.scene_info_dict['mid state'] = []
        self.scene_info_dict['v_std_dev'] = []
        self.scene_info_dict['a_std_dev'] = []
        self.scene_info_dict['UD'] = []
        # self.scene_info_dict
        
    def add_info_dict_read(self, ):
        self.scene_info_dict['scene_id'].append(self.data.scene_id)
        self.scene_info_dict['scene_name'].append(self.data.scene_name)
        self.scene_info_dict['dataset timestamp'].append(f"0-{self.data.max_dataset_time}")
        self.scene_info_dict['sim timestamp'].append(f"{self.data.start_time}-{self.data.max_frame}")
        self.scene_info_dict['stop time(ms)'].append(self.data.stop_time)
        self.scene_info_dict['result'].append(self.data.result)
        self.scene_info_dict['collision'].append(self.data.collision_frame)
        self.scene_info_dict['collision state'].append(None)
        self.scene_info_dict['start speed(m/s)'].append(self.data.ego_dict[self.data.start_time].v)
        self.scene_info_dict['average speed(m/s)'].append(np.mean(list(self.data.ego_speed.values())))
        self.scene_info_dict['max acc +(m/s2)'].append(max(list(self.data.ego_acc.values())))
        self.scene_info_dict['max acc -(m/s2)'].append(min(list(self.data.ego_acc.values())))
        self.scene_info_dict['max jerk(m/s3)'].append(max(np.abs(list(self.data.ego_jerk.values()))))
        self.scene_info_dict['mid state'].append(self.data.state_str)
        self.scene_info_dict['v_std_dev'].append(self.cal_std_deviation(self.data.ego_speed.values()))
        self.scene_info_dict['a_std_dev'].append(self.cal_std_deviation(self.data.ego_acc.values()))
        self.scene_info_dict['UD'].append(self.cal_uncomfortable_deceleration(list(self.data.ego_acc.values())))
        
    def cal_std_deviation(self, values):
        value_list = list(values)
        values = np.array(value_list)
        return np.std(values)
    
    def cal_uncomfortable_deceleration(self, acc_list):
        # count continuous deceleration < -1.6 as one time
        accs = np.array(acc_list)
        uncomfortable_deceleration_count = 0
        i = 0
        while i < len(accs):
            if accs[i] < -1.6:
                uncomfortable_deceleration_count += 1
                while 1:
                    if i == len(accs)-1:
                        break
                    if accs[i+1] > -1.6:
                        break
                    i += 1
            i += 1
        return uncomfortable_deceleration_count
    
    def save_to_excel(self, path):
        
        saved_name =  'result' + '_get' + '.xlsx'
        saved_file = os.path.join(path, saved_name)
        
        # 将数据转换为DataFrame对象
        df = pd.DataFrame(self.scene_info_dict)
        
        # 将DataFrame写入Excel文件
        df.to_excel(saved_file, index=False)
        pass
        

class EvaluateAll(object):
    def __init__(self, dir_name):
        self.dir_name = dir_name
        file_list = os.listdir(dir_name)
        self.file_list = []
        for name in file_list:
            log_path = os.path.join(self.dir_name, name)
            if log_path[-4:] != '.pkl':
                continue
            if os.path.isfile(log_path):
                self.file_list.append(name)
        
        self.file_list.sort(key=lambda x:int(x.split('_')[0]))
        
        self.scenarios : Dict[int, Data_Struct] = {}
        for file_name in self.file_list:
            log_path = os.path.join(self.dir_name, file_name)
            if not os.path.isfile(log_path):
                continue
            with open(log_path, 'rb') as f:
                data_dict = pickle.load(f)
            index = int(file_name.split('_')[0])
            self.scenarios[index] = data_dict
        pass

    
    def evaluate_all(self,):
        self.get_results()
        
        
        for index, result in self.results.items():
            print(index, result, self.scenarios[index].scene_name)
        # print([(a[0],a[1]) for a in enumerate(self.results)])
        str_res = [str(a) for a in self.results.values()]
        
        arr=Counter(str_res)
        print("counter:")
        print(arr)
        
        print("average speed in all tests:{:.1f}".format(np.mean(self.average_speed)))
        # print()
        # self.get_collision()
    
    def get_results(self):
        self.results = {}
        self.average_speed = []
        for index, scenario in self.scenarios.items():
            mean_v = np.mean(list(scenario.ego_speed.values()))
            self.average_speed.append(mean_v)
            if len(scenario.collision_frame) != 0:
                self.results[index] = [scenario.result, scenario.collision_frame[0][1]]
            else:
                self.results[index] = [scenario.result]
        # self.results.sort(key=lambda x:x[0])
        # print('results:\r\n', self.results)
    
    def get_decision_time_ms(self):
        self.decision_time_ms = []
        for index, scenario in self.scenarios.items():
            self.decision_time_ms.append(np.mean(scenario.decision_time_ms))
           
    # def get_max_value(self):
    #     self.max_speeds = []
    #     self.ave_speeds = []
    #     self.max_acc = []
    #     self.ave_acc = []
    #     for scenario in self.scenarios:
    #         self.max_speeds.append(np.max(scenario.ego_speed.values()))
    #         self.ave_speeds.append(np.mean(scenario.ego_speed.values()))
    #         self.max_acc.append(max(scenario.ego_acc.values()))
    #         self.ave_acc.append(np.mean(scenario.ego_acc.values()))
    #     # print('speed:\r\n', )
        
        
    # def print_info_to_excel(self):
    #     # number file_name result collision stop_time ave_speed max_speed max_acc max_jerk 
    #     pass
    
    # def record_all_videos(self):
    #     pass
    

        
    

class RecordRosbag(object):
    def __init__(self, root_path, bag_name):
        bag_path = os.path.join(root_path, bag_name+'.bag')
        self.bag = rosbag.Bag(bag_path, 'w')
            
    def add_message(self,topic:str, data):
        self.bag.write(topic,data)
    
    def save_rosbag(self):
        self.bag.close()
        
