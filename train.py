import numpy as np
#OpenAI's library that provides environments to test RL algorithms in, Universe adds
#even more environments
import gym
import torch
from torch import nn
import wandb
import datetime

# hyperparameters
learning_rate = 1e-4 #for convergence (too low- slow to converge, too high,never converge)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
        # 1*80*80 -> 3*76*76
        nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(5, 5)),
        nn.ReLU(),
        # 3*76*76 -> 3*38*38
        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        # 3*38*38 -> 6*34*34
        nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(5, 5)),
        nn.ReLU(),
        # 6*34*34 -> 6*17*17
        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )
        self.fc = nn.Sequential(
        nn.Linear(6*17*17, 200),
        nn.ReLU(),
        nn.Linear(200, 10),
        nn.ReLU(),
        nn.Linear(10, 1),
        nn.Sigmoid()
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(-1))
        return output

def prepro(I):
    """ prepro 210x160x3 uint8 frame into 80x80 2D float tensor """
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    I = I.reshape(1, 1, 80, 80)
    return torch.FloatTensor(I)


#wandb.init(project='Ping Pong')
# network  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Net().to(device)
#environment
env = gym.make("Pong-v0")
#Each timestep, the agent chooses an action, and the environment returns an observation and a reward.
#The process gets started by calling reset, which returns an initial observation
observation = env.reset()
prev_x = None # used in computing the difference frame
#current reward
running_reward = None
#sum rewards
reward_sum = 0
#where are we?
episode_number = 0
# optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
# loss
l = 0

#begin training!
while True:

    # preprocess the observation, set input to network to be difference image
    #Since we want our policy network to detect motion
    cur_x = prepro(observation)
    #difference image = subtraction of current and last frame
    x = cur_x - prev_x if prev_x is not None else torch.zeros_like(cur_x)
    prev_x = cur_x

    # forward the policy network and sample an action from the returned probability
    UpProb = net(x.to(device))
    #this is the stochastic part 
    action = 2 if np.random.uniform() < UpProb else 3 # 2 UP, 3 Down

    # step the environment and get new measurements
    env.render()
    observation, reward, done, info = env.step(action)
    reward_sum += reward # (+1 if moves past AI, -1 if missed ball, 0 otherwise)
    l += torch.log(UpProb) * reward
    if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
        print ('ep %d: game finished, reward: %f' % (episode_number, reward) + ('' if reward == -1 else ' !!!!!!!!'))
    if done: # an episode finished, but already many game boundaries
        episode_number += 1
        optimizer.zero_grad()
        l.backward()
        optimizer.step() # update the parameters

        # boring book-keeping
        running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
        print('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
        reward_sum = 0
        observation = env.reset() # reset env
        prev_x = None
        l = 0
        if episode_number % 100 == 0:
            torch.save(net.state_dict(), f'./{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")}/checkpoint_{episode_number}.pt')