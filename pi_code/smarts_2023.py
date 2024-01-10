"""This example shows how you might run a SMARTS environment for single-agent work. SMARTS is
natively multi-agent so a single-agent wrapper is used."""
import sys
from pathlib import Path
import time
from typing import Final

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

AGENT_ID: Final[str] = "Agent_0"

from csts_agent.csts_agent import CSTSAgent

class KeepLaneAgent(Agent):
    def act(self, obs, **kwargs):
        return (0,0,0)


def main(scenarios, headless, num_episodes, max_episode_steps=None):
    # This interface must match the action returned by the agent
    agent_interface = AgentInterface.from_type(
        # AgentType.Laner, 
        AgentType.Full, 
        action = ActionSpaceType.RelativeTargetPose,
        max_episode_steps=max_episode_steps, 
    )

    env = driving_smarts_2023_env(
        scenario=scenarios[0],
        seed = 13,
        agent_interface=agent_interface,
        headless=headless,
    )

    env = SingleAgent(env)
    
    for episode in episodes(n=num_episodes):
        # agent = KeepLaneAgent()
        agent= CSTSAgent(True, True, False)
        observation, _ = env.reset()
        episode.record_scenario(env.unwrapped.scenario_log)

        map = env.get_map()
        terminated = False
        while not terminated:
            # obs = observation[AGENT_ID]
            action = agent.act(observation, map)
            # action = {AGENT_ID: action}
            if(action == False):
                break
            observation, reward, terminated, truncated, info = env.step(action)
            episode.record_step(observation, reward, terminated, truncated, info)

    env.close()


if __name__ == "__main__":
    parser = minimal_argument_parser(Path(__file__).stem)
    args = parser.parse_args()

    if not args.scenarios:
        args.scenarios = [
            # "scenarios/sumo/straight/3lane_cut_in_agents_1/", 
            "scenarios/sumo/straight/cutin_2lane_agents_1/",
            # str(SMARTS_REPO_PATH / "scenarios" / "sumo" / "loop"),
            # str(SMARTS_REPO_PATH / "scenarios" / "sumo" / "figure_eight"),
        ]
    args.max_episode_steps = 200

    build_scenarios(scenarios=args.scenarios)
   

    main(
        scenarios=args.scenarios,
        headless=args.headless,
        num_episodes=args.episodes,
        max_episode_steps=args.max_episode_steps,
    )
