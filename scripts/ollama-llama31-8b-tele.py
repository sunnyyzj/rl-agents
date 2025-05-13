import os
import pandas as pd
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
import gymnasium as gym
import highway_env
import ollama
from datetime import datetime
import cProfile

# Initialize the Ollama client (replace with actual client initialization if needed)
# client = ollama.Client(api_token=os.environ['OLLAMA_API_TOKEN'])

# Function to discretize the observations
def discretize_observation(observation, bins):
    discretized = np.digitize(observation, bins=bins)
    return discretized

# Define custom bins for discretization
bins = [
    [-0.1, 0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.1],   # Bins for column 1
    [-0.2, -0.1, 0, 0.1, 0.2, 0.4, 0.6, 0.8],  # Bins for column 2
    [-0.1, -0.05, 0, 0.1, 0.2, 0.3, 0.4],      # Bins for column 3
    [-0.00004, -0.00002, 0, 0.00002, 0.00004, 0.002, 0.004, 0.006, 0.008],  # Bins for column 4
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # Bins for column 5 (already discretized) rf_cnt
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # Bins for column 6 (already discretized) thz_cnt
]

ACTIONS_ALL = {
        0: ['LANE_LEFT','t1'],
        1: ['IDLE','t1'],
        2: ['LANE_RIGHT','t1'],
        3: ['FASTER','t1'],
        4: ['SLOWER','t1'],
        5: ['LANE_LEFT','t2'],
        6: ['IDLE','t2'],
        7: ['LANE_RIGHT','t2'],
        8: ['FASTER','t2'],
        9: ['SLOWER','t2'],
        10: ['LANE_LEFT','t3'],
        11: ['IDLE','t3'],
        12: ['LANE_RIGHT','t3'],
        13: ['FASTER','t3'],
        14: ['SLOWER','t3'],
    }

# Reverse the ACTIONS_ALL dictionary to map actions back to their corresponding index
ACTIONS_ALL_REVERSE = {tuple(value): key for key, value in ACTIONS_ALL.items()}

def preprocess_observation(observation):
    if isinstance(observation, tuple):
        observation_array = np.array(observation[0])
    else:
        observation_array = np.array(observation)
    observation_array = np.array([subarray[1:] for subarray in observation_array])
    # print(observation_array)
    # Ensure the number of columns matches the number of bins
    # num_columns = observation_array.shape[1]
    # assert num_columns <= len(bins), str(num_columns) + " Number of columns in observation  exceeds number of bin sets."    

    discretized_obs = np.zeros_like(observation_array)
    for i in range(observation_array.shape[1]):
        discretized_obs[:, i] = discretize_observation(observation_array[:, i], bins[i])
    return discretized_obs.flatten()

def discretize_process_observation(observation):
    if isinstance(observation, tuple):
        observation_array = np.array(observation[0])
    else:
        observation_array = np.array(observation)
    observation_array = np.array([subarray[1:] for subarray in observation_array])
    discretized_obs = np.zeros_like(observation_array)
    for i in range(observation_array.shape[1]):
        discretized_obs[:, i] = discretize_observation(observation_array[:, i], bins[i])
    return discretized_obs

def get_top_5_similar_good_examples(input_state, good_examples, threshold=100.0):
    if good_examples.size == 0:
        print("good_examples is empty. No similar examples to find.")
        return np.array([])

    def euclidean_distance(observation, input_state):
        try:
            return distance.euclidean(observation, input_state)
        except Exception as e:
            print(f"Error calculating distance: {e}")
            return float('inf')

    distances = np.array([euclidean_distance(row[0], input_state) for row in good_examples])
    similar_indices = np.where(distances <= threshold)[0]
    if similar_indices.size > 0:
        top_5_indices = similar_indices[np.argsort(distances[similar_indices])[:5]]
        return good_examples[top_5_indices]
    else:
        return np.array([])

def get_action_from_llama(observation, good_training_step, good_examples):
    feature_description = """
    Features of the environment include:
    - 'x': Horizontal offset of the vehicle relative to the ego vehicle along the x-axis.
    - 'y': Vertical offset of the vehicle relative to the ego vehicle along the y-axis.
    - 'vx': Velocity of the vehicle along the x-axis.
    - 'vy': Velocity of the vehicle along the y-axis. A non-zero value indicates lane changes.
    The first row of the observation table represents the ego vehicle.
    Observations, if normalized, are within a fixed range [100, 100, 20, 20] for 'x', 'y', 'vx', 'vy' respectively.
    """

    task_description = """
    Task Description: Assist in driving the ego vehicle on a highway.
    """
    
    task_goal = """
    Task Goal:
    - Achieve maximum velocity for the ego vehicle while minimizing collisions.
    - Reduce unnecessary lane changes (LANE_RIGHT, LANE_LEFT) unless required for safety.
    - Prefer keeping the vehicle in the right-most lane when safe to do so.
    """
    
    decision = """
    Decision: Choose one action from FASTER, SLOWER, LANE_RIGHT, LANE_LEFT, or IDLE.
    """
    
    experience_replay = f"""
    Experience Replay: The last step was {'a good' if good_training_step else 'not a good'} training step.
    """
    
    action_map = {
        0: "LANE_LEFT",
        1: "IDLE",
        2: "LANE_RIGHT",
        3: "FASTER",
        4: "SLOWER"
    }

    if good_examples.size > 0:
        example_states = "\n".join([f"State: {ex[0]}, Action: {action_map.get(ex[1], 'IDLE')}, Reward: {ex[2]}" for ex in good_examples])
        good_examples_section = f"""
        Here are some examples of good previous experiences, I suggest you try a higher reward action based on these examples:
        {example_states}
        """
    else:
        good_examples_section = ""

    prompt = f"""
    {task_description}

    {task_goal}

    {feature_description}

    Given the current state of the environment with the following observations: {discretize_process_observation(observation)}

    {experience_replay}

    {good_examples_section}

    {decision}

    Please provide only the chosen action in the response.

    """
    
    try:
        response = ollama.chat(
            model="llama3.1",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )
        response_string = response["message"]["content"]
        action = get_action_from_response(response_string)
    except Exception as e:
        print(f"Error encountered: {e}")
        action = "IDLE"
    return action

def get_telecom_action_from_llama(transport_action, rf_cnt, thz_cnt):
    telecom_task_description = """
    Task Description: You are assisting in optimizing telecommunication decisions for a vehicular network environment, 
    where the goal is to jointly optimize autonomous driving and network selection policies.   
    """
    
    telecom_task_goal = """
    Task Goal:
    - Maximize the communication data rate while ensuring safe driving and minimal handovers (HOs).
    - Balance the load across base stations (BSs) to reduce the impact of network congestion.
    """
    
    telecom_decision = """
    Decision: Select one action from t1, t2, or t3, based on the given objectives.
    t1 - Select the next base station to maximize the weighted data rate with consideration of traffic load balancing and HO penalties.
    t2 - Select the next base station to maximize the weighted data rate, assuming the BS's user quota is not exceeded, ignoring HO penalties.
    t3 - Select the next base station purely based on the highest achievable data rate, without considering traffic load or HO penalties.
    """
    
    prompt = f"""
    {telecom_task_description}

    {telecom_task_goal}

    The current transportation action is: {transport_action}
    The number of nearby Radio Frequency (RF) base stations with data rate exceeding the threshold is: {rf_cnt}
    The number of nearby Terahertz (THz) base stations with data rate exceeding the threshold is: {thz_cnt}
    {telecom_decision}

    Please provide only the chosen action in the response.
    """
    
    try:
        response = ollama.chat(
            model="llama3.1",
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )
        response_string = response["message"]["content"]
        action = get_action_from_response(response_string)
    except Exception as e:
        print(f"Error encountered: {e}")
        action = "t1"
    return action

def get_action_from_response(response):
    actions = ["LANE_LEFT", "IDLE", "LANE_RIGHT", "FASTER", "SLOWER", "t1", "t2", "t3"]

    for action in actions:
        if action in response:
            return action
    return "IDLE"

def select_action(observation, epsilon, good_training_step, top_5_similar_good_examples):
    if np.random.rand() < epsilon:
        return np.random.randint(0, 15)  # Exploration within action space 0-14
    else:
        action_text = get_action_from_llama(observation, good_training_step, top_5_similar_good_examples)
        transport_action = action_text
        # bs_cnt = observation[0][-1]  # Assuming bs_cnt is the last feature in the observation array
        rf_cnt = observation[0][-2]  # Assuming rf_cnt is the second last feature in the observation array
        thz_cnt = observation[0][-1]  # Assuming bs_cnt is the last feature in the observation array
        telecom_action_text = get_telecom_action_from_llama(transport_action, rf_cnt, thz_cnt)
        # Combine both actions into one action space index
        dual_action = [transport_action, telecom_action_text]

        # dual_action_index = list(DiscreteDualObjectMetaAction.ACTIONS_ALL.keys())[list(DiscreteDualObjectMetaAction.ACTIONS_ALL.values()).index(dual_action)]
        # return dual_action

        # Convert the dual action into an index using the reversed dictionary
        dual_action_index = ACTIONS_ALL_REVERSE.get(tuple(dual_action), 1)  # Default to IDLE action index (1) if not found
        return dual_action_index  # Exploitation
    
# Initialize good examples as an empty numpy array
good_examples = np.empty((0, 4), dtype=object)

# Ensure the Feather file is empty at the start of training
csv_path = "csv_test/csv_log_"+datetime.now().strftime('%Y%m%d_%H%M%S')+".csv"
feather_path= "csv_test/csv_log_"+datetime.now().strftime('%Y%m%d_%H%M%S')+".feather"
fig_path= "csv_test/csv_log_"+datetime.now().strftime('%Y%m%d_%H%M%S')+".pdf"

# Training loop
# logs = []

env = gym.make("highway-bs-v0")#, render_mode="rgb_array" ,render_mode="human" highway-fast-v0
num_episodes = 1000  # Adjust as needed
epsilon = 1.0  # Initial exploration rate
epsilon_min = 0.01  # Minimum exploration rate
epsilon_decay = 0.995  # Decay rate of exploration

all_logs = []  # Initialize a list to store logs for all episodes

for episode in range(num_episodes):
    observation = env.reset()
    total_reward = 0
    done = False
    truncated = False
    step = 0
    good_training_step = True
    episode_logs = []  # Initialize logs for the current episode

    while not (done or truncated) and step <= 30:
        env.render()

        # Get top 5 similar good examples
        input_state = preprocess_observation(observation)
        top_5_similar_good_examples = get_top_5_similar_good_examples(input_state, good_examples)

        dual_action = select_action(observation, epsilon, good_training_step, top_5_similar_good_examples)
        observation, reward, done, truncated, info = env.step(dual_action)
        total_reward += reward
        step += 1

        # Extract additional data from the info dictionary
        agents_ho_prob = info.get('agents_ho_prob', [None])[0]
        agents_tran_all_rewards = info.get('agents_tran_all_rewards', [None])[0]
        agents_tele_all_rewards = info.get('agents_tele_all_rewards', [None])[0]
        agents_rewards = info.get('agents_rewards', [None])[0]
        agents_collided = info.get('agents_collided', [None])[0]
        distance_travelled = info.get('distance_travelled', [None])[0]

        # Log all relevant data
        episode_logs.append([
            episode,
            step,
            dual_action,
            reward,
            total_reward,
            step,
            # observation.tolist(),
            agents_ho_prob,
            agents_tran_all_rewards,
            agents_tele_all_rewards,
            agents_rewards,
            agents_collided,
            distance_travelled
        ])

        # Update good examples if the step was successful
        if not truncated and reward > 0:
            discretized_observation = preprocess_observation(observation)
            new_example = np.array([[discretized_observation.tolist(), dual_action, reward, step]], dtype=object)
            good_examples = np.vstack([good_examples, new_example])

        good_training_step = not truncated

    # Append the current episode's logs to the main all_logs list
    all_logs.extend(episode_logs)
    print(f"Episode {episode} finished on step {step} with total reward: {total_reward}")

    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    # Convert all logs to a DataFrame and save them to a CSV file
    df_logs = pd.DataFrame(all_logs, columns=[
        'episode',
        'Step',
        'Action',
        'Reward',
        'Total Reward',
        'TimeStep',
        # 'Observation',
        'agents_ho_prob',
        'agents_tran_all_rewards',
        'agents_tele_all_rewards',
        'agents_rewards',
        'agents_collided',
        'distance_travelled'
    ])
    df_logs.to_csv(csv_path, index=False)

env.close()

def main():
    # Save logs to Feather file
    # df_logs = pd.DataFrame(logs, columns=['Episode', 'Step', 'Action', 'Reward', 'Total Reward', 'TimeStep', 'Observation'])
    # df_logs = pd.DataFrame(logs, columns=[
    #     'Episode',
    #     'Step',
    #     'Action',
    #     'Reward',
    #     'Total Reward',
    #     'TimeStep',
    #     'Observation',
    #     'HO',
    #     'tran_rewards',
    #     'tele_rewards',
    #     'total_rewards',
    #     'collides',
    #     'distance_travelled'
    # ])
    # df_logs.to_csv(csv_path, index=False)

    # Grouping the data by episodes and calculating the total reward for the last step of each episode
    last_step_rewards = df_logs.groupby('Episode').apply(lambda x: x.loc[x['Step'].idxmax(), 'Total Reward'])

    # Defining the bins for episode ranges
    bins = range(0, last_step_rewards.index.max() + 20, 20)

    # Grouping the last step rewards by the defined bins
    binned_rewards = last_step_rewards.groupby(pd.cut(last_step_rewards.index, bins)).sum()

    # Plotting the binned rewards
    plt.figure(figsize=(10, 6))
    plt.plot(binned_rewards.index.astype(str), binned_rewards, marker='o')
    plt.xlabel('Episode Range')
    plt.ylabel('Total Reward (Last Step)')
    plt.title('Total Reward for Last Step per Episode Range')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fig_path)
    # Show the plot
    plt.show()

if __name__ == "__main__":
    # main()
    cProfile.run('main()')
