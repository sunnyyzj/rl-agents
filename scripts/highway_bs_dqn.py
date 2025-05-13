import highway_env
import gymnasium as gym
from pathlib import Path
from rl_agents.agents.common.factory import load_environment, load_agent
from rl_agents.trainer.evaluation import Evaluation
from rl_agents.agents.common.factory import agent_factory
import sys
from tqdm.notebook import trange
from datetime import datetime
sys.path.insert(0, './highway-env/scripts/')
# from utils import record_videos, show_videos

# env = gym.make("highway-fast-v0")
env = gym.make("highway-bs-v0", render_mode="rgb_array")
(obs, info), done = env.reset(), False

agent_config = {
    "__class__": "<class 'rl_agents.agents.deep_q_network.pytorch.DQNAgent'>",
    "model": {
        "type": "MultiLayerPerceptron",
        "layers": [256, 256]
    },
    "double": False,
    "loss_function": "l2",
    "optimizer": {
        "lr": 5e-4
    },
    "gamma": 0.8,
    "n_steps": 1,
    "batch_size": 32,
    "memory_capacity": 15000,
    "target_update": 50,
    "exploration": {
        "method": "EpsilonGreedy",
        "tau": 6000,
        "temperature": 1.0,
        "final_temperature": 0.05
    }
}
agent = agent_factory(env, agent_config)
evaluation = Evaluation(env,
                        agent,
                        run_directory=Path("./logsBS/"),
                        num_episodes=6000,
                        sim_seed=0,
                        recover=False,
                        # display_env=True,
                        # display_agent=True,
                        # display_rewards=True,
                        display_env=False,
                        display_agent=False,
                        display_rewards=True,
                        step_callback_fn=None)
evaluation.train()
# now we will open a file for writin