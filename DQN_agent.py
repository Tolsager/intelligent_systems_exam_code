from replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import os
import datetime
from DQN import Network
# import wandb


class DQNAgent:
    def __init__(
            self,
            env,
            memory_size,
            batch_size,
            target_update=100,
            gamma=0.99,
            # replay parameters
            alpha=0.2,
            beta=0.6,
            prior_eps=1e-6,
            # Categorical DQN parameters
            v_min=0,
            v_max=200,
            atom_size=51,
            # N-step Learning
            n_step=3,
            start_train=32,
            save_weights=True,
            log=True,
            lr=0.001,
            seed=0,
            episodes=200

    ):

        self.env = env

        obs_dim = self.env.observation_dim
        action_dim = self.env.action_dim

        self.batch_size = batch_size
        self.target_update = target_update
        self.gamma = gamma
        self.lr = lr
        self.memory_size = memory_size
        self.seed = seed

        # device: cpu / gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        # memory for 1-step Learning
        self.beta = beta
        self.prior_eps = prior_eps
        self.memory = PrioritizedReplayBuffer(
            obs_dim, memory_size, batch_size, alpha=alpha
        )

        # memory for N-step Learning
        self.use_n_step = True if n_step > 1 else False
        if self.use_n_step:
            self.n_step = n_step
            self.memory_n = ReplayBuffer(obs_dim, memory_size, batch_size, n_step=n_step, gamma=gamma)

        # Categorical DQN parameters
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.support = torch.linspace(
            self.v_min, self.v_max, self.atom_size
        ).to(self.device)

        # networks: dqn, dqn_target
        self.dqn = Network(
            obs_dim, action_dim, self.atom_size, self.support
        ).to(self.device)
        self.dqn_target = Network(
            obs_dim, action_dim, self.atom_size, self.support
        ).to(self.device)

        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=self.lr)

        # transition to store in memory
        self.transition = list()

        self.fig, (self.ax1, self.ax2) = plt.subplots(2, figsize=(10, 10))

        self.start_train = start_train

        self.save_weights = save_weights

        self.time = datetime.datetime.now().timetuple()
        self.path = f"weights/{self.time[2]}-{self.time[1]}-{self.time[0]}_{self.time[3]}-{self.time[4]}"

        self.log = log
        self.episode_cnt = 0
        self.episodes = episodes

        if self.save_weights is True:
            self.create_save_directory()

        plt.ion()

    def create_save_directory(self):
        try:
            os.mkdir(self.path)
        except OSError:
            print("Creation of the directory %s failed" % self.path)
        else:
            print("Successfully created the directory %s " % self.path)

    def select_action(self, state):
        """Select an action from the input state."""
        # NoisyNet: no epsilon greedy action selection
        selected_action = self.dqn(torch.FloatTensor(state).to(self.device)).argmax()
        selected_action = selected_action.detach().cpu().numpy()

        self.transition = [state, selected_action]

        return selected_action

    def step(self, action):
        """Take an action and return the response of the env."""
        next_state, reward, done = self.env.step(action)

        self.transition += [reward, next_state, done]

        # N-step transition
        if self.use_n_step:
            one_step_transition = self.memory_n.store(*self.transition)
        # 1-step transition
        else:
            one_step_transition = self.transition

        # add a single step transition
        if one_step_transition:
            self.memory.store(*one_step_transition)

        return next_state, reward, done

    def update_model(self):
        """Update the model by gradient descent."""
        # PER needs beta to calculate weights
        samples = self.memory.sample_batch(self.beta)
        weights = torch.FloatTensor(
            samples["weights"].reshape(-1, 1)
        ).to(self.device)
        indices = samples["indices"]

        # 1-step Learning loss
        elementwise_loss = self._compute_dqn_loss(samples, self.gamma)

        # PER: importance sampling before average
        loss = torch.mean(elementwise_loss * weights)

        # N-step Learning loss
        # we are gonna combine 1-step loss and n-step loss so as to
        # prevent high-variance. The original rainbow employs n-step loss only.
        if self.use_n_step:
            gamma = self.gamma ** self.n_step
            samples = self.memory_n.sample_batch_from_idxs(indices)
            elementwise_loss_n_loss = self._compute_dqn_loss(samples, gamma)
            elementwise_loss += elementwise_loss_n_loss

            # PER: importance sampling before average
            loss = torch.mean(elementwise_loss * weights)

        self.optimizer.zero_grad()
        loss.backward()
        # print(loss)
        clip_grad_norm_(self.dqn.parameters(), 10.0)
        self.optimizer.step()

        # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)

        # NoisyNet: reset noise
        self.dqn.reset_noise()
        self.dqn_target.reset_noise()

        return loss.item()

    def train(self, num_frames, plotting_interval=100):
        """Train the agent."""

        if self.log:
            pass
            # config = {'gamma': self.gamma, 'log_interval': plotting_interval, 'learning_rate': self.lr,
            #           'directory': self.path, 'type': 'dqn', 'replay_memory': self.memory_size, 'environment': 'normal', 'seed': self.seed}
            # wandb.init(project='is_os', entity='pydqn', config=config, notes=self.env.reward_function, reinit=True, tags=['report'])
            # wandb.watch(self.dqn)

        self.env.reset()
        state = self.env.get_state()
        won = False
        update_cnt = 0
        losses = []
        scores = []
        score = 0
        frame_cnt = 0
        self.episode_cnt = 0
        
        for frame_idx in range(1, num_frames + 1):
            frame_cnt += 1
            action = self.select_action(state)
            next_state, reward, done = self.step(action)

            state = next_state
            score += reward

            fraction = min(frame_cnt / num_frames, 1.0)
            self.beta = self.beta + fraction * (1.0 - self.beta)

            # if agent has trained 500 frames, terminate
            if frame_cnt == 500:
                done = True

            # if episode ends
            if done:
                if reward > 0:
                    won = True
                self.env.reset()
                state = self.env.get_state()
                self.episode_cnt += 1
                scores.append(score)
                score = 0
                frame_cnt = 0

            # if training is ready
            if len(self.memory) >= self.batch_size:
                loss = self.update_model()
                losses.append(loss)
                update_cnt += 1

                # if hard update is needed
                if update_cnt % self.target_update == 0:
                    self._target_hard_update()

            # plotting
            if frame_idx % plotting_interval == 0:
                self._plot(frame_idx, scores, losses)

            if frame_idx % 1000 == 0:
                torch.save(self.dqn.state_dict(), f'{self.path}/{frame_idx}.tar')
                print(f"model saved at:\n {self.path}/{frame_idx}.tar")


        # wandb.run.summary['won'] = won
        self.env.close()

    def _compute_dqn_loss(self, samples, gamma):
        """Return categorical dqn loss."""
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"]).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        # Categorical DQN algorithm
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            # Double DQN
            next_action = self.dqn(next_state).argmax(1)
            next_dist = self.dqn_target.dist(next_state)
            next_dist = next_dist[range(self.batch_size), next_action]

            t_z = reward + (1 - done) * gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(
                    0, (self.batch_size - 1) * self.atom_size, self.batch_size
                ).long()
                    .unsqueeze(1)
                    .expand(self.batch_size, self.atom_size)
                    .to(self.device)
            )

            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        dist = self.dqn.dist(state)
        log_p = torch.log(dist[range(self.batch_size), action])
        elementwise_loss = -(proj_dist * log_p).sum(1)

        return elementwise_loss

    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.dqn_target.load_state_dict(self.dqn.state_dict())

    def _plot(self, frame_cnt, scores, losses):
        self.ax1.cla()
        self.ax1.set_title(f'frames: {frame_cnt} score: {np.mean(scores[-10:])}')
        self.ax1.plot(scores[-999:], color='red')
        self.ax2.cla()
        self.ax2.set_title(f'loss: {np.mean(losses[-10:])}')
        self.ax2.plot(losses[-999:], color='blue')
        plt.show()
        plt.pause(0.1)

        # needed for wandb to not log nans
        # if frame_cnt < self.start_train + 11:
        #     loss = 0
        # else:
        #     loss = np.mean(losses[-10:])

        if self.log:
            pass
            # wandb.log({'score': np.mean(scores[-10:]), 'loss': loss, 'Frame Count': frame_cnt, 'episode': self.episode_cnt})
