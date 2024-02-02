Setup:
    https://smarts.readthedocs.io/en/latest/setup.html

## Running

1. `cd ~/2-ldl/SMARTS`
2. `source .venv/bin/activate`

## Build

observation `scl scenario build-all scenarios/sumo/straight/`

## TODO

1. 测试相关

   1. 测试使用 `pi_code/csts_agent/high_way_env.py`
      1. `python3 ~/2-ldl/SMARTS/pi_code/csts_agent/high_way_env.py`
   2. 测试类 `pi_code/csts_agent/csts_agent.py`
2. 完成 `csts_agent.py`

   1. self.ego_state_msg=self.get_ego_state(ego, map)
   2. self.map_lanes_msg=self.get_map_lanes(ego, map)
   3. self.perception_prediction_msg=self.get_pp(objs, map)
3. 变道场景

relay:

sudo apt install ffmpeg
