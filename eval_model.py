import torch
from Policy import Policy
import numpy as np
from DQN import Network

# import the environment you want to use
# from SimplifiedLunarLander import LunarLander
from LunarLander import LunarLander

# 'policy' or 'dqn' to choose which type of model to evaluate
model_type = 'policy'
# model_type = 'dqn'
model_path = "policies/22-1-2021_13-44/policy0.tar"


def eval(model_type=model_type, model_path=model_path):
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    env = LunarLander()

    if model_type == 'policy':
        model = Policy(env.observation_dim, env.action_dim)
    elif model_type == 'dqn':
        model = Network(env.observation_dim, env.action_dim)
    model.to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    episodes = 50
    wins = 0
    frames = []
    fuel_left = []
    for i in range(episodes):
        if i % 10 == 0:
            print(f"On episode {i}")
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
        env.close()

    if wins > 0:
        print(f"wins: {wins}")
        print(f"mean frames on wins {np.mean(frames)}")
        print(f"std frames on wins {np.std(frames, ddof=1)}")
        print(f"min frames on wins {np.min(frames)}")
        print(f"max frames on wins {np.max(frames)}")

        print(f"mean fuel on wins {np.mean(fuel_left)}")
        print(f"std fuel on wins {np.std(fuel_left, ddof=1)}")
        print(f"min fuel on wins {np.min(fuel_left)}")
        print(f"max fuel on wins {np.max(fuel_left)}")
    else:
        print("The model had 0 wins. Statistics can't be calculated")


if __name__ == '__main__':
    eval()
