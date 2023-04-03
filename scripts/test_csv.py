import csv

data = {
    'speed': 20.00019801064021,
    'crashed': False,
    'action': 10,
    'rewards': {
        'collision_reward': 0.0,
        'right_lane_reward': 0.0,
        'high_speed_reward': 1.9801064021152114e-05,
        'on_road_reward': 1.0,
        'tele_reward': 0.0005917219637241545
    },
    'other_vehicle_collision': 0,
    'agents_ho_prob': (0.17777777777777778,),
    'agents_tran_all_rewards': (0.6666719469504057,),
    'agents_tele_all_rewards': (0.0005917219637241545,),
    'agents_rewards': (0.6672636689141298,),
    'agents_collided': (False,),
    'distance_travelled': (227.03100720971636,),
    'agents_survived': False
}

# Flatten the rewards dictionary
rewards = data.pop('rewards')
for key, value in rewards.items():
    data[key] = value

# Open the CSV file in append mode
with open('data.csv', 'a', newline='') as file:
    writer = csv.writer(file)

    # Write the header row if the file is empty
    if file.tell() == 0:
        writer.writerow(data.keys())

    # Write the data row
    writer.writerow(data.values())