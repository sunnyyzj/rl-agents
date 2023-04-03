import highway_env
from pathlib import Path
from rl_agents.agents.common.factory import load_environment, load_agent
from rl_agents.trainer.evaluation import Evaluation

env_config = {
    "id": "highway-v0",
    "lanes_count": 3,
    "vehicles_count": 15,
    "observation": {
        "type": "GrayscaleObservation",
        "observation_shape": (128, 64),
        "stack_size": 4,
        "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
        "scaling": 1.75,
    },
    "policy_frequency": 2,
    "duration": 40,
}

agent_config = {
    "__class__": "<class 'rl_agents.agents.deep_q_network.pytorch.DQNAgent'>",
    "double": True,
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
    },
    "model": {
        "type": "ConvolutionalNetwork",
        "activation": "RELU",
        "head_mlp": {
            "type": "MultiLayerPerceptron",
            "layers": [20],
            "activation": "RELU",
            "reshape": "True"
        }
    }
}

env = load_environment(env_config)
agent = load_agent(agent_config, env)
run_directory = None
evaluation = Evaluation(env,
                        agent,
                        run_directory=Path("./logs/"),
                        num_episodes=1000,
                        sim_seed=0,
                        recover=False,
                        display_env=True,
                        display_agent=True,
                        display_rewards=True)
evaluation.train()