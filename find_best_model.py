import torch
import os
import numpy as np
from Policy import Policy
from DQN import Network

# import the environment you want to use
# from SimplifiedLunarLander import LunarLander
from LunarLander import LunarLander
"""
Evaluates each model in a folder and returns the model with the most wins and fuel
"""

# 'dqn' or 'policy'
# model_type = 'dqn'
model_type = 'policy'
model_directory = "policies/22-1-2021_13-44"

# number of episodes to run each model
episodes = 20

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


def test_model(episodes):
    wins = 0
    frames = []
    fuel_left = []

    env = LunarLander()
    for i in range(episodes):
        frame_count = 0
        env.reset()
        state = env.get_state()
        while True:
            frame_count += 1
            action = model(torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)).argmax()
            state, reward, done = env.step(action)

            if done:
                if env.won:
                    wins += 1
                    frames.append(frame_count)
                    fuel_left.append(env.rocket.fuel)
                break

    if len(fuel_left) > 0:
        return np.mean(fuel_left), wins
    else: 
        return 0, 0


if __name__ == '__main__':
    files = os.listdir(model_directory)

    best_mean_fuel = 0
    best_file = ""
    best_wins = 0
    env = LunarLander()
    for file in files:
        model_path = f"{model_directory}/{file}"

        if model_type == 'policy':
            model = Policy(env.observation_dim, env.action_dim)
        elif model_type == 'dqn':
            model = Network(env.observation_dim, env.action_dim)
        model.to(device)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        print(f"Testing model {file}")
        mean_fuel, wins = test_model(episodes)

        if wins == best_wins:
            if mean_fuel > best_mean_fuel:
                best_wins = wins
                best_mean_fuel = mean_fuel
                best_file = file
                print(f"{best_file} is the current best model with {best_wins} wins and a fuel mean of {best_mean_fuel}")
        elif wins > best_wins:
            best_wins = wins
            best_mean_fuel = mean_fuel
            best_file = file
            print(f"{best_file} is the current best model with {best_wins} wins and a fuel mean of {best_mean_fuel}")
    env.close()
    if wins > 0:
        print(f"{best_file} is the best model with {best_wins} wins and a fuel mean of {best_mean_fuel}")
    else:
        print("No model managed to win")
