# ---------------------------------------------------------------
# 0. Imports
# ---------------------------------------------------------------
import gymnasium as gym
import highway_env
import numpy as np
from gymnasium import spaces
from gymnasium.wrappers import FlattenObservation          # <- works
from pathlib import Path
from rl_agents.agents.common.factory import agent_factory
from rl_agents.trainer.evaluation import Evaluation

# ---------------------------------------------------------------
# 0‑bis.  Mini replacement for the missing FlattenAction
# ---------------------------------------------------------------
class FlattenAction(gym.ActionWrapper):
    """
    Convert Tuple(Discrete(n1), Discrete(n2), …) -> MultiDiscrete([n1, n2, …])
    Nothing else changes: we just pass the action straight through.
    """
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.action_space, spaces.Tuple), \
            "FlattenAction expects a Tuple action space"
        self.nvec = np.array([space.n for space in env.action_space], dtype=np.int32)
        self.action_space = spaces.MultiDiscrete(self.nvec)

    def action(self, act):
        # SB3 / rl‑agents will give us a list‑like array; convert to tuple for env
        return tuple(int(a) for a in act)

    def reverse_action(self, act):
        # not used, but keeps the API symmetrical
        return np.asarray(act, dtype=np.int32)

# ---------------------------------------------------------------
# 1. Build a true multi‑agent highway env (2 egos)
# ---------------------------------------------------------------
multi_agent_cfg = {
    "controlled_vehicles": 2,
    "vehicles_count": 15,
    "duration": 60,
    "observation": {
        "type": "MultiAgentObservation",
        "observation_config": {"type": "Kinematics"}
    },
    "action": {
        "type": "MultiAgentAction",
        "action_config": {"type": "DiscreteMetaAction"}
    }
}

# env = gym.make("highway-v0", render_mode=None, config=multi_agent_cfg)
env = gym.make("highway-v0", render_mode="rgb_array", config=multi_agent_cfg)

# ---------------------------------------------------------------
# 2. Wrap:  Tuple  →  MultiDiscrete  →  Discrete(25)
# ---------------------------------------------------------------
class MultiDiscreteToDiscrete(gym.ActionWrapper):
    """Collapse MultiDiscrete([n1, n2]) -> Discrete(n1*n2)."""
    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.action_space, spaces.MultiDiscrete)
        self.nvec = env.action_space.nvec
        self.action_space = spaces.Discrete(int(np.prod(self.nvec)))

    def action(self, act):
        idx0 = act // self.nvec[1]
        idx1 = act %  self.nvec[1]
        return (int(idx0), int(idx1))

    def reverse_action(self, act):
        return int(act[0] * self.nvec[1] + act[1])

env = FlattenObservation(env)     #  (2 × features) -> flat vector
env = FlattenAction(env)          #  Tuple          -> MultiDiscrete([5,5])
env = MultiDiscreteToDiscrete(env)#  MultiDiscrete  -> Discrete(25)

# quick sanity check
obs, info = env.reset(seed=0)
print("obs shape:", obs.shape, "| action space:", env.action_space)

# ---------------------------------------------------------------
# 3.  Build the DQN agent exactly as before
# ---------------------------------------------------------------
agent_config = {
    "__class__": "<class 'rl_agents.agents.deep_q_network.pytorch.DQNAgent'>",
    "model": {"type": "MultiLayerPerceptron", "layers": [256, 256]},
    "double": True,
    "loss_function": "l2",
    "optimizer": {"lr": 5e-4},
    "gamma": 0.8,
    "n_steps": 1,
    "batch_size": 32,
    "memory_capacity": 15_000,
    "target_update": 50,
    "exploration": {
        "method": "EpsilonGreedy",
        "tau": 6_000,
        "temperature": 1.0,
        "final_temperature": 0.05
    }
}

agent      = agent_factory(env, agent_config)
evaluation = Evaluation(
    env,
    agent,
    run_directory=Path("./logs_multi/"),
    num_episodes=6_000,
    sim_seed=0,
    display_env=False,
    display_agent=False,
    display_rewards=True
)
evaluation.train()