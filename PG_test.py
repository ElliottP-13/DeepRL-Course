import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
from torch.autograd import Variable
from itertools import count
import matplotlib.pyplot as plt
import numpy as np
import gym
import imageio
import pdb
import os


class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()

        self.fc1 = nn.Linear(4, 24)
        self.fc2 = nn.Linear(24, 36)
        self.fc3 = nn.Linear(36, 1)  # Prob of Left

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


def main():
    
    # Plot duration curve: 
    # From http://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
    episode_durations = []
    def plot_durations():
        plt.figure(2)
        plt.clf()
        durations_t = torch.FloatTensor(episode_durations)
        plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated

    # Parameters
    num_episode = 500
    batch_size = 5
    learning_rate = 0.01
    gamma = 0.999

    env = gym.make('CartPole-v1')
    #env = gym.wrappers.RecordVideo(env, 'video_dir')
    if not os.path.exists('video_dir'):
        os.makedirs('video_dir')
    video_dir = 'video_dir'  # Replace with the directory where you want to save the recorded videos
    def should_record_episode(episode_idx):
        return episode_idx % 50 == 0
    
    policy_net = PolicyNet()
    
#ToDo 1
    optimizer = 

    # Batch History
    state_pool = []
    action_pool = []
    reward_pool = []
    steps = 0

    
    recording = False
    frames = []


    for e in range(num_episode):

        print(e)
        state = env.reset()
        state = torch.from_numpy(state).float()
        state = Variable(state)
        env.render(mode='rgb_array')
        #env.render = True

        for t in count():
#ToDo 2
            probs = 
            m = 
            action = 

            action = action.data.numpy().astype(int)[0]
            next_state, reward, done, _ = env.step(action)
            env.render(mode='rgb_array')
            #env.render = True

            # To mark boundarys between episodes
            if done:
                reward = 0

            state_pool.append(state)
            action_pool.append(float(action))
            reward_pool.append(reward)

            state = next_state
            state = torch.from_numpy(state).float()
            state = Variable(state)

            steps += 1

            if done:
                episode_durations.append(t + 1)
                plot_durations()
                break
                
            if should_record_episode(e):
                if not recording:
                    recording = True

            frame = env.render(mode='rgb_array')
            frames.append(frame)

        if recording:
            recording = False

            # Save frames as a video using imageio
            video_file = f'{video_dir}/episode_{e}.mp4'
            imageio.mimsave(video_file, frames, fps=30 , macro_block_size=1)  # frames per second
            frames = []
                
        # Update policy
        if e > 0 and e % batch_size == 0:

            # Discount reward
            running_add = 0
            for i in reversed(range(steps)):
                if reward_pool[i] == 0:
                    running_add = 0
                else:
#ToDo 3
                    running_add = 
                    reward_pool[i] = running_add

            # Normalize reward
            reward_mean = np.mean(reward_pool)
            reward_std = np.std(reward_pool)
            for i in range(steps):
                reward_pool[i] = (reward_pool[i] - reward_mean) / reward_std

            # Gradient Desent
            optimizer.zero_grad()

            for i in range(steps):
                state = state_pool[i]
                action = Variable(torch.FloatTensor([action_pool[i]]))
                reward = reward_pool[i]
#ToDo 4
                probs = 
                m = 
                loss = 
                loss

            optimizer

            state_pool = []
            action_pool = []
            reward_pool = []
            steps = 0
            # Show the plot in a window
            plt.show()

            # Save the plot as a PNG image
            plt.savefig('my_plot1.png')


if __name__ == '__main__':
    main()