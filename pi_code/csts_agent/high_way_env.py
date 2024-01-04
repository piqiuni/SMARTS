"""This example shows how you might run a SMARTS environment for single-agent work. SMARTS is
natively multi-agent so a single-agent wrapper is used."""
import argparse
import random
import sys
from pathlib import Path
import time
from typing import Final

import gymnasium as gym

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

AGENT_ID: Final[str] = "Agent"

from csts_agent import CSTSAgent


class KeepLaneAgent(Agent):
    def act(self, obs, map):
        return random.randint(0, 3)


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
        agent = CSTSAgent()
        # agent = KeepLaneAgent()
        observation, _ = env.reset()
        episode.record_scenario(env.unwrapped.scenario_log)

        map = env.get_map()
        terminated = False
        while not terminated:
            print(type(observation))
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
