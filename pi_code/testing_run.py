#!/usr/bin/env python3.8

import contextlib
import pickle
import time
from tracemalloc import start
from turtle import st
from typing import Tuple

import yaml

import rospy
import os
import sys


sys.path.append(os.path.dirname(sys.path[0]))
from tqdm import tqdm

import subprocess

dt = 0.1



def get_data(sim:ScenarioSimulation, data_frame:OneFrame, pred_dict:Dict)-> Tuple[Car_Struct,  List[Car_Struct]]:
    x = data_frame.ego_state.obj_x_relative
    y = data_frame.ego_state.obj_y_relative
    yaw = data_frame.ego_state.obj_theta_map
    v = sim.map_state.v
    a = sim.map_state.a
    j = sim.map_state.j
    length = data_frame.ego_state.length
    width = data_frame.ego_state.width
    lane_id = data_frame.ego_state.lane_id
    ego_state = Car_Struct(0, 0, x,y,yaw,v,a,j,length,width,lane_id)
    social_states = []
    for key, value in data_frame.objects.items():
        if math.sqrt((value.obj_x_relative - x)**2 + (value.obj_y_relative - y)**2) > 100:
            continue
        
        state = Car_Struct(key, value.obj_class, value.obj_x_relative, value.obj_y_relative, value.obj_heading_relative, value.obj_v_map, 0,0,value.length,value.width)
        
        pred = pred_dict.get(key, None)
        if pred:
            probs, paths = list(zip(*pred))
            transfered_paths = []
            for path in paths:
                path_pts = []
                for pt in path:
                    new_pt, yaw, _ = sim.transfer.from_map_to_dataset((pt[0], pt[1]), pt[2])
                    if len(pt) == 5:
                        path_pts.append((new_pt[0], new_pt[1], yaw, pt[3], pt[4]))
                    else:
                        path_pts.append((new_pt[0], new_pt[1], yaw))
                transfered_paths.append(path_pts)
            state.add_pred(transfered_paths, probs)
        social_states.append(state)
    
    return ego_state, social_states



if __name__ == "__main__":
    # ./testing_run.py --case_name case_0706 --get_videos
    data_folder_absolute_path = '/home/rancho/2-ldl/Huawei-dataset/DATA'
    log_folder_absolute_path = '/home/rancho/2-ldl/Huawei-dataset/LOG'
    # case_name = 'case_0901' 
    parser = argparse.ArgumentParser()
    choices=['case_old', 'case_0601', 'case_0706', 'case_0901', 'case_gen']
    parser.add_argument("--case_name", type=str, choices=choices, required=True, help="case_name", )
    parser.add_argument('--get_videos', action='store_true', default=False, help='get_videos')
    args = parser.parse_args()
    case_name = args.case_name
    get_videos = args.get_videos
    
    # case_name = ''
    
    if case_name == 'case_gen':
        # except_case = 'case1'
        except_case = 'case2'
        record_max_frame = 130
        if except_case == 'case1': # case2 -> Change Lane
            start_speeds = [20/3.6]
            start_frames = [30]
        elif except_case == 'case2': # case1 -> Left Turn
            start_speeds = np.array([15, 12, 18, 20, 10]) / 3.6
            start_frames = [40, 40, 40, 30, 50]
    else:
        start_speeds = [None]
        start_frames = [None]
        except_case = 'nonono'
        record_max_frame = None
    
    
    
    if len(start_speeds) != len(start_frames):
        raise ValueError("start_speeds and start_frames should have the same length")
    
    
    case_folder_absolute_path = data_folder_absolute_path + '/' + case_name
    full_file_names = [f for f in os.listdir(case_folder_absolute_path) if 
                  (os.path.isfile(os.path.join(case_folder_absolute_path, f)) and not f.endswith(('.yaml', '.md')))]
    if case_name == 'case_old':
        full_file_names = old_case_file_names
        # print(file_names)
    file_names = [name for name in full_file_names if except_case not in name]
    print(f"Testing start, with {len(start_speeds)} runs, {len(file_names)} scenarios")
    
    # test_scenes = [[a,b,c] for b,c in zip(start_speeds, start_frames) for a in file_names]
    test_scenes = [[a,b,c] for a in file_names for b,c in zip(start_speeds, start_frames) ]
    # [(name, speed, frame)]
    print(f"test_scenes: ")
    for item in test_scenes:
        print(f"\t{item}")
    record_bag = None
    Display = False
    RCV_State = True
    start_flag = False
    
    PredModel, use_map_config = get_case_config(case_folder_absolute_path)
    rospy.init_node("sim_testing", anonymous=True)
    # init record class
    drive_test = Drive_Test(log_folder_absolute_path, case_name)
    test_start_time = time.time()
    # s2p = State2PtcloudNew()
    s2p = "000"
    fail_list = []
    count = 0 
    
    for i in tqdm(range(0, len(test_scenes))):
    # for i in tqdm(range(12, 13)):
        scene_name, start_speed, start_frame = test_scenes[i]
        scene_number = full_file_names.index(scene_name)
        
        # if scene_number not in [9, 10, 11, 13, 18, 20, 27, 34, 36]:
        #     continue

        # print(f"Testing scene {scene_name}, {scene_number}")
        start_speed_str = f"{(start_speed*3.6):.1f}km/h" if start_speed is not None else 'Ego speed'
        start_frame_str = f"{start_frame}" if start_frame is not None else '0'
        # print(f"Start speed: {start_speed_str}, start frame: {start_frame_str}")
            
        if use_map_config:
            map_config = get_map_param(case_folder_absolute_path, scene_number)
        else:
            map_config = MapConfig()
        # map_config.use_ego_path = False
        if start_speed:
            map_config.start_speed = start_speed
        if start_frame:
            map_config.start_frame = start_frame
        
        result = 'Break'
        scene_count = 0
        while result == 'Break' and scene_count < 5:
            
            scene_count += 1
            # start launch file
            if CSTS:
                roslaunch_cmd = "roslaunch contingency_st_search test_search.launch"
            else:
                roslaunch_cmd = "roslaunch hybrid_A_search testing_launch.launch"
            roslaunch_proc = subprocess.Popen(roslaunch_cmd.split(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)        
            # roslaunch_proc = subprocess.Popen(roslaunch_cmd.split())
            rospy.sleep(1)
            state = True
            with contextlib.redirect_stdout(None):
                sim = ScenarioSimulation(s2p, False, False)
                sim.get_files(scene_name, scene_number, case_folder_absolute_path)
                sim.init_flags(Display, PredModel, RCV_State)
                Succsee, state_str = sim.init_config(map_config, None, 60/3.6)
                
            if not Succsee:
                print(f"scene {scene_number} failed, skip, :{state_str}")
                result = 'Continue'
                continue
            
            with contextlib.redirect_stdout(None):
                rospy.sleep(1)
                sim.start_sim()

            # start Record RosBag
            drive_test.new_scene(scene_number, scene_name, sim.start_frame, sim.max_frame+1, sim.map_state.v)  
            root_path = os.path.join(drive_test.root_dir, drive_test.log_dir)
            # record_bag = RecordRosbag(root_path, scene_name)
            
            # for rosbag
            # rospy.Subscriber("/best_path", Path, RecordCallback, ("/best_path", record_bag))
            # rospy.Subscriber("/map", PointCloud2, RecordCallback, ("/map", record_bag))
            # rospy.Subscriber("/goals", PointCloud2, RecordCallback, ("/goals", record_bag))
            # rospy.Subscriber("/grid_path_vis", MarkerArray, RecordCallback, ("/grid_path_vis", record_bag))
            # rospy.Subscriber("/ref_path", Path, RecordCallback, ("/ref_path", record_bag))
            # if start_flag == False:
            #     spin_thread.start()
            #     start_flag = True
            
            # print(sim.all_route_ids)
            # print(sim.all_routes)
            for key, route in zip(sim.all_route_ids, sim.all_routes):
                drive_test.add_lane(key, route.points)
                
            # if except_case == 'case1':
            #     max_time = start_frame + 100
            end_flag = 0
            timestamp = 0
            for timestamp in tqdm(range(sim.start_frame, sim.max_frame+1)):
                print(timestamp)
                start_frame_time = time.time()
                while(sim.got_best_path == False and not rospy.is_shutdown()):
                    time_cost = time.time() - start_frame_time
                    if time_cost > 10:
                        end_flag = 1
                        break
                if end_flag == 1:
                    break

                with contextlib.redirect_stdout(None):
                    data_frame, pred_dict = sim.run_once()
                
                # data_frame = data_dict[timestamp]
                ego_state, social_states = get_data(sim, data_frame, pred_dict)
                drive_test.push_this_frame(timestamp, ego_state, social_states, sim.draw_best_path_pts)
                
                sim.got_best_path = False
            # end 
            
            # close launch run
            roslaunch_proc.terminate()
            rospy.sleep(3)
            
            # # end bag record and data record
            # record_bag.save_rosbag()
            # record_bag = None
            
            if len(drive_test.data.collision_frame) != 0:
                if(drive_test.data.collision_frame[0][1] == 'backward'):
                    result = 'Backward Collision'
                # print(drive_test.last_collision_id)
                else:
                    result = 'Collision'
            elif timestamp == sim.max_frame:
                result = 'Over'
            else:
                if timestamp - sim.start_frame < 5:
                    # fail_list.append(scene_number)
                    result = 'Break'
                    print("scene {}th break at start, restart".format(scene_number))
                else:
                    result = 'Break at {}'.format(timestamp)
                drive_test.data.collision_frame.append([timestamp, 'Break'])
            if result != 'Break':
                
                
                drive_test.data.state_str = ''
                
                drive_test.save_scene(result, timestamp)
                print("scene {}th over, result: ".format(scene_number) + result)
                
            if scene_count == 5:
                fail_list.append(scene_number)
                drive_test.add_info_dict_failed()
    
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
            log_reader.record_video("HAstar cv", log_path, files[i])
            
        print("test over, all time cost: {:.1f}s".format((time.time() - test_start_time)))
        
        
    print(f"fail list: {fail_list}")
    print(f"file name: {os.path.join(drive_test.root_dir, drive_test.log_dir)}")
    