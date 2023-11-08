source .venv/bin/activate


scl run --envision examples/e2_single_agent.py scenarios/sumo/loop
scl run --envision pi_code/e2_single_agent.py scenarios/sumo/straight/cruise_2lane_agents_1/
























for ROS  (smarts/ros/README.md)  
    follow the instructions in the README.md file in the smarts/ros folder  
        change version in package.xml, I'm using 0.0.0

    source smarts/ros/install/setup.bash




rostopic pub /SMARTS/reset smarts_ros/SmartsReset "scenario: '/home/rancho/2lidelun/SMARTS/scenarios/NGSIM/i80'
initial_agents:
- agent_id: ''
  veh_type: 0
  veh_length: 0.0
  veh_width: 0.0
  veh_height: 0.0
  start_pose:
    position: {x: 0.0, y: 0.0, z: 0.0}
    orientation: {x: 0.0, y: 0.0, z: 0.0, w: 0.0}
  start_speed: 0.0
  tasks:
  - {task_ref: '', task_ver: '', params_json: ''}
  end_pose:
    position: {x: 0.0, y: 0.0, z: 0.0}
    orientation: {x: 0.0, y: 0.0, z: 0.0, w: 0.0}"