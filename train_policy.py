import numpy as np
from itertools import count
from Policy import Policy
# import wandb
import random
import torch
import datetime
import os
import torch.optim as optim
from torch.distributions import Categorical

# import the environment you want to use
# from SimplifiedLunarLander import LunarLander
from LunarLander import LunarLander

seed = 0
gamma = 0.99
log_interval = 10
lr = 1e-2
save_interval = 200


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


seed_everything(seed)

env = LunarLander()
policy = Policy(env.observation_dim, env.action_dim)
optimizer = optim.Adam(policy.parameters(), lr=lr)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()


def finish_episode():
    R = 0
    policy_loss = []
    returns = []
    for r in policy.rewards[::-1]:
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


def main():
    # Make working directory
    time = datetime.datetime.now().timetuple()
    path = f"policies/{time[2]}-{time[1]}-{time[0]}_{time[3]}-{time[4]}"
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)

    # config = {'seed': seed, 'gamma': gamma, 'log_interval': log_interval, 'learning_rate': lr, 'directory': path,
    #           'type': 'policies', 'environment': 'normal'}
    # wandb.init(project='is_os', entity='pydqn', config=config, notes=env.reward_function, tags=['report'])
    # wandb.watch(policies)

    model_counter = 0
    running_reward = 10

    for i_episode in count(1):
        env.reset()
        state = env.get_state()
        ep_reward = 0
        for t in range(1, 10000):  # Don't infinite loop while learning
            action = select_action(state)
            state, reward, done = env.step(action)

            policy.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        finish_episode()

        if i_episode % save_interval == 0:
            print(f"saving model policy{model_counter}.tar")
            print(f"@ {path}/policy{model_counter}.tar")
            torch.save(policy.state_dict(), f'{path}/policy{model_counter}.tar')
            model_counter += 1

        if i_episode % log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))
            # wandb.log({'episode': i_episode, 'average_reward': running_reward})


if __name__ == '__main__':
    main()
