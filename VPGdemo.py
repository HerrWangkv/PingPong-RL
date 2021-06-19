import torch
from torch import nn
import numpy as np
import gym
import wandb
import os
import datetime

def prepro(I):
    """ prepro 210x160x3 uint8 frame into 80x80 2D float vector """
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    I = I.reshape(1, 1, 80, 80)
    return torch.FloatTensor(I)
    
class Policy_Network(nn.Module):
    def __init__(self, act_size):
        super(Policy_Network, self).__init__()
        self.conv = nn.Sequential(
            # 1*80*80->3*26*26
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=5, stride=3),
            nn.ReLU(),
            # 3*26*26->3*13*13
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(3*13*13, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, act_size),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv(x)
        prob = self.fc(x.view(1, -1))
        return prob

class VPG():
    def __init__(self, env, device, save_dir,alpha=1e-4, gamma=0.99):
        self.act_size=env.action_space.n
        self.env = env
        self.device = device
        self.save_dir = save_dir
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount rate
        self.policy = Policy_Network(self.act_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=alpha)
        self.responsible_probs = []
        self.rewards = []

    def choose_action(self, state):
        act_probs = self.policy(state.to(self.device))
        # can't convert CUDA tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
        action = np.random.choice(self.act_size, 1, p=act_probs.cpu().data.numpy().reshape(-1))[0]
        self.responsible_probs.append(act_probs[0, action])
        return action
    
    def store_reward(self, reward):
        self.rewards.append(reward)

    def discount_rewards(self):
        discounted_rewards = torch.zeros(len(self.rewards)).to(self.device)
        running_add = 0
        for t in reversed(range(len(self.rewards))):
            if (self.rewards[t] != 0):
                running_add = 0
            running_add = running_add * self.gamma + self.rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def learn(self):
        rewards = self.discount_rewards()
        rewards = (rewards - torch.mean(rewards)) / (torch.std(rewards) + 1e-7)
        loss = 0
        for p, r in zip(self.responsible_probs, self.rewards):
            loss += torch.log(p) * r
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        wandb.log({'loss': loss})
        self.responsible_probs, rewards = [], []
    
    def train(self, num_episodes):
        for i in range(num_episodes):
            state = self.env.reset()
            reward_sum = 0
            done = False
            while not done:
                self.env.render()
                state = prepro(state)
                action = self.choose_action(state)
                state, reward, done, _ = self.env.step(action)
                self.store_reward(reward)
                reward_sum += reward
                if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
                    print (f'ep {i}: game finished, reward: {"-1" if reward == -1 else "1 !!!!!!!!"}')
            self.learn()
            wandb.log({'reward_sum': reward_sum})
            print(f'Episode: {i} | total reward: {reward_sum}')
            if (i % 100 == 0):
                torch.save(self.policy.state_dict(), os.path.join(self.save_dir, f'checkpoint_{i}.pt'))
if __name__ == "__main__":
    file_dir = os.path.join(os.path.dirname(__file__), f'./{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")}')
    os.mkdir(file_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb.init(project='Ping Pong')
    num_episodes = 1000
    env = gym.make("Pong-v0")
    pg_agent = VPG(env, device, file_dir)
    pg_agent.train(num_episodes)
