import torch
from Policy import Policy
from DQN import Network
import pygame

# import the environment you want to use
# from SimplifiedLunarLander import LunarLander
from LunarLander import LunarLander

# 'policy' or 'dqn' to choose which type of model to evaluate
model_type = 'policy'
# model_type = 'dqn'
model_path = "policies/22-1-2021_13-44/policy0.tar"

env = LunarLander()
env.reset()
exit_program = False

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

if model_type == 'policy':
    model = Policy(env.observation_dim, env.action_dim)
elif model_type == 'dqn':
    model = Network(env.observation_dim, env.action_dim)
model.to(device)
model.load_state_dict(torch.load(model_path))
model.eval()
state = env.get_state()

while not exit_program:
    env.render()
    action = model(torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)).argmax()

    state, reward, done = env.step(action)

    # Process game events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit_program = True
        if event.type == pygame.KEYDOWN:
            if event.key in [pygame.K_ESCAPE, pygame.K_q]:
                exit_program = True
            if event.key == pygame.K_UP:
                boost = True
            if event.key == pygame.K_DOWN:
                boost = False
            if event.key == pygame.K_RIGHT:
                left = False if right else True
                right = False
            if event.key == pygame.K_LEFT:
                right = False if left else True
                left = False
            if event.key == pygame.K_r:
                boost = False
                left = False
                right = False
                env.reset()

    if done:
        env.reset()
        state = env.get_state()


env.close()
