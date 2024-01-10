#!/usr/bin/env python3.8

import contextlib
import pickle
import time
from typing import Tuple

import yaml
from pi_code.csts_agent.csts_agent import CSTSAgent

import rospy
import os
import sys
from pathlib import Path
from typing import List

sys.path.append(os.path.dirname(sys.path[0]))
from tqdm import tqdm

import subprocess

from testing.testing_function import Drive_Test, Data_Struct, Car_Struct, PlannedPt, ReadLog

SMARTS_REPO_PATH = Path(__file__).parents[1].absolute()
sys.path.insert(0, str(SMARTS_REPO_PATH))
from examples.tools.argument_parser import minimal_argument_parser
from smarts.core.agent import Agent
from smarts.core.agent_interface import AgentInterface, AgentType
from smarts.core.controllers.action_space_type import ActionSpaceType
from smarts.core.utils.episodes import episodes
from smarts.env.gymnasium.driving_smarts_2023_env import driving_smarts_2023_env
from smarts.env.gymnasium.wrappers.single_agent import SingleAgent
from smarts.sstudio.scenario_construction import build_scenarios
from smarts.core.road_map import RoadMap

dt = 0.1

# TODO: 从agent和map中获取数据，填充ego_state, social_states, ego_planning
def get_data(agent:CSTSAgent, map: RoadMap)-> Tuple[Car_Struct,  List[Car_Struct], List[PlannedPt]]:
    ego_state: Car_Struct = None
    social_states: List[Car_Struct] = []
    ego_planning: List[PlannedPt] = []
    
    
    
    
    # x = data_frame.ego_state.obj_x_relative
    # y = data_frame.ego_state.obj_y_relative
    # yaw = data_frame.ego_state.obj_theta_map
    # v = sim.map_state.v
    # a = sim.map_state.a
    # j = sim.map_state.j
    # length = data_frame.ego_state.length
    # width = data_frame.ego_state.width
    # lane_id = data_frame.ego_state.lane_id
    # ego_state = Car_Struct(0, 0, x,y,yaw,v,a,j,length,width,lane_id)
    # social_states = []
    # for key, value in data_frame.objects.items():
    #     if math.sqrt((value.obj_x_relative - x)**2 + (value.obj_y_relative - y)**2) > 100:
    #         continue
        
    #     state = Car_Struct(key, value.obj_class, value.obj_x_relative, value.obj_y_relative, value.obj_heading_relative, value.obj_v_map, 0,0,value.length,value.width)
        
    #     pred = pred_dict.get(key, None)
    #     if pred:
    #         probs, paths = list(zip(*pred))
    #         transfered_paths = []
    #         for path in paths:
    #             path_pts = []
    #             for pt in path:
    #                 new_pt, yaw, _ = sim.transfer.from_map_to_dataset((pt[0], pt[1]), pt[2])
    #                 if len(pt) == 5:
    #                     path_pts.append((new_pt[0], new_pt[1], yaw, pt[3], pt[4]))
    #                 else:
    #                     path_pts.append((new_pt[0], new_pt[1], yaw))
    #             transfered_paths.append(path_pts)
    #         state.add_pred(transfered_paths, probs)
    #     social_states.append(state)
    
    
    return ego_state, social_states, ego_planning



if __name__ == "__main__":
    
    log_folder_absolute_path = '/home/rancho/2-ldl/Huawei-dataset/LOG'
    envision_record_data_replay_path = None
    case_name = "SMARTS"
    get_videos = True
    
    
    record_max_frame = None
    
    Display = False
    RCV_State = True
    start_flag = False
    
    rospy.init_node("sim_testing", anonymous=True)
    
    drive_test = Drive_Test(log_folder_absolute_path, case_name)
    test_start_time = time.time()
    
    fail_list = []
    count = 0 

    testing_cases = ["scenarios/sumo/straight/3lane_cut_in_agents_1/", 
                     "scenarios/sumo/straight/cutin_2lane_agents_1/"]
    testing_nums = [5, 5]
    start_seed = 10
    save_senario_num = 100
    
    parser = minimal_argument_parser(Path(__file__).stem)
    max_episode_steps = 200
    agent_interface = AgentInterface.from_type(
        AgentType.Full, 
        action = ActionSpaceType.RelativeTargetPose,
        max_episode_steps=max_episode_steps, 
    )
    
    
    
    
    max_no_social_vehicle_count = 20
    seed = start_seed
    saving_count = 0
    testing_cases_index = 0
    with tqdm(total=save_senario_num) as pbar:
        
        while(testing_cases_index < len(testing_cases)):
            case = testing_cases[testing_cases_index]
            num_episodes = testing_nums[testing_cases_index]
        
            env = driving_smarts_2023_env(
                scenario=case,
                seed = seed,
                agent_interface=agent_interface,
                envision_record_data_replay_path=envision_record_data_replay_path
            )
            env = SingleAgent(env)
            scene_name = case.split('/')[-2]
            scene_number = 0
            for episode in episodes(n=num_episodes):
                timestamp = 0
                with contextlib.redirect_stdout(None):
                    agent= CSTSAgent(False, True, False)
                    observation, _ = env.reset()
                    episode.record_scenario(env.unwrapped.scenario_log)
                    
                    drive_test.new_scene(saving_count, scene_name, 0, max_episode_steps, )  
                    root_path = os.path.join(drive_test.root_dir, drive_test.log_dir)
                    result = 'Break'

                    map = env.get_map()
                    
                    # TODO: drive_test.add_lane 添加车道，每次添加一条
                    # 参考
                    with 1:
                        key = 0
                        points= [(0, 0), (0, 1000)]
                        drive_test.add_lane(key, points)
                    
                    terminated = False
                    no_social_vehicle_count = 0
                    while (not terminated) and not (rospy.is_shutdown()) :
                        # obs = observation[AGENT_ID]
                        action = agent.act(observation, map)
                        # action = {AGENT_ID: action}
                        
                        # TODO: 获取flag，自车行驶到了车道末端，或者没有社会车辆
                        no_social_vehicle_flag, end_of_lane_flag = agent.
                        
                        ego_state, social_states, ego_planning = get_data(agent)
                        
                        drive_test.push_this_frame(timestamp, ego_state, social_states, ego_planning)
                        
                        timestamp += 1
                        if(action == False):
                            break
                        observation, reward, terminated, truncated, info = env.step(action)
                        episode.record_step(observation, reward, terminated, truncated, info)
                        
                        if no_social_vehicle_flag:
                            no_social_vehicle_count += 1
                            if no_social_vehicle_count > max_no_social_vehicle_count:
                                result = 'No Social Vehicle'
                                break
                        else:
                            no_social_vehicle_count = 0
                            
                        if end_of_lane_flag:
                            result = 'End of Lane'
                            break
                        
                    agent.__del__()
                    # out of while
                    
                if len(drive_test.data.collision_frame) != 0:
                    if(drive_test.data.collision_frame[0][1] == 'backward'):
                        result = 'Backward Collision'
                    else:
                        result = 'Collision'
                elif abs(timestamp - max_episode_steps) <= 1:
                    result = 'Over'
                else:                   
                    result = 'Break at {}'.format(timestamp)
                    drive_test.data.collision_frame.append([timestamp, 'Break'])
                   
                
                # TODO: 通过数据获得仿真中是否发生了社会车变道
                # 判断 drive_test.data.social_dict 中的车辆是否发生了变道——根据车辆y坐标变化（），或者根据地图，判断车辆车道是否改变过
                
                with 1: # 参考
                    frame = 0
                    car_id = 1
                    drive_test.data.social_dict[frame][car_id].y
                    drive_test.data.social_dict[frame][car_id].lane_id

                vehicle_lane_change_flag = 
                
                
                if not vehicle_lane_change_flag:
                    print(f"not asving, ", end='')    
                elif result == 'No Social Vehicle':
                    print(f"not asving, ", end='')
                else:
                    drive_test.data.state_str = ''
                    drive_test.save_scene(result, timestamp)
                    print(f"saving {saving_count}th, ", end='')    
                    pbar.update(1)
                    saving_count += 1
                
                print("scene {}th over, result: ".format(scene_number) + result)
                    
                
            env.close()
    
        
    drive_test.save_to_excel()
    rospy.signal_shutdown("closed!")
    
    
    if get_videos:
        log_path = os.path.join(drive_test.root_dir, drive_test.log_dir)
        pkl_file_list = os.listdir(log_path)
        files = []
        for i in range(len(pkl_file_list)):
            file_path = os.path.join(log_path, pkl_file_list[i])
            if os.path.isfile(file_path) and str(file_path)[-3:] == 'pkl':
                files.append(pkl_file_list[i])
        print("begin recording")
        # print(files)
        for i in tqdm(range(0, len(files))):
        # for i in tqdm(range(2)):
            file_path = os.path.join(log_path, files[i])
            
            video_dir_path = os.path.join(log_path, 'video')
            video_name = files[i] + '.mp4'
            video_path = os.path.join(video_dir_path, video_name)
            # print(video_path)
            # print(video_path, os.path.exists(video_path))
            if os.path.exists(video_path):
            # if os.path.isfile(video_path):
                continue
            
            with open(file_path, 'rb') as f:
                data_dict = pickle.load(f)
            # with contextlib.redirect_stdout(None):
            log_reader = ReadLog(data_dict)
            log_reader.record_video("SMARTS", log_path, files[i])
            
        print("test over, all time cost: {:.1f}s".format((time.time() - test_start_time)))
        
        
    print(f"fail list: {fail_list}")
    print(f"file name: {os.path.join(drive_test.root_dir, drive_test.log_dir)}")
    