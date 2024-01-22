#!/usr/bin/env python3.8

import contextlib
import pickle
import time
from typing import Tuple

import yaml
from csts_agent.csts_agent import CSTSAgent

import rospy
import os
import sys
from pathlib import Path
from typing import List,Dict
import numpy as np

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
from smarts.core.coordinates import Point,RefLinePoint
from smarts.core.utils.episodes import episodes
from smarts.env.gymnasium.driving_smarts_2023_env import driving_smarts_2023_env
from smarts.env.gymnasium.wrappers.single_agent import SingleAgent
from smarts.sstudio.scenario_construction import build_scenarios
from smarts.core.road_map import RoadMap

if __name__ == "__main__":
    get_videos = True
    
    log_folder_absolute_path = '/home/rancho/2-ldl/Huawei-dataset/LOG/SMARTS'
    log_name = 'SMARTS_2024-01-15-20-46-45'
    test_start_time = time.time()
    if get_videos:
        log_path = os.path.join(log_folder_absolute_path, log_name)
        pkl_file_list = os.listdir(log_path)
        files = []
        # for i in range(len(pkl_file_list)):
        #     file_path = os.path.join(log_path, pkl_file_list[i])
        #     if os.path.isfile(file_path) and str(file_path)[-3:] == 'pkl':
        #         files.append(pkl_file_list[i])
        files = [file for file in pkl_file_list if file.endswith(".pkl")]
        print(f"begin recording, total {len(files)} files")
        
        # raise
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
        
        
    rospy.signal_shutdown("closed!")
    