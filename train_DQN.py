import torch
import random
import numpy as np
import os
from DQN_agent import DQNAgent

# import the environment you want to use
# from SimplifiedLunarLander import LunarLander
from LunarLander import LunarLander

# number of frames to train on
num_frames = 20_000
memory_size = 1000
batch_size = 32
target_update = 100
seed = 0
env = LunarLander()

# whether or not to use wandb. All wandb code is commented out so that code can be run without it
log = False


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

agent = DQNAgent(env, memory_size, batch_size, target_update, log=log, seed=seed)
agent.train(num_frames)
