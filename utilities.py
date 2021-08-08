import torch
from torch import nn
from collections import namedtuple
from functools import reduce
import numpy as np
import os
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from scipy import optimize
import wandb

class Policy_Network(nn.Module):
    def __init__(self, obs_size, act_size):
        super(Policy_Network, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_size, 200),
            nn.ReLU(),
            nn.Linear(200, act_size)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc(x)
        probs = self.softmax(x)

        return probs

class Value_Network(nn.Module):
    def __init__(self, obs_size):
        super(Value_Network, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_size, 200),
            nn.ReLU(),
            nn.Linear(200, 1)
        )

    def forward(self, x):
        value = self.fc(x)
        return value

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'prob'])

class Memory:
    def __init__(self):
        self.memory = []

    def push(self, *args):
        self.memory.append(Transition(*args))

    def collect(self):
        return Transition(*zip(*self.memory))

        
    def __getitem__(self, position):
        return self.memory[position]

    def __len__(self):
        return len(self.memory)

def get_flat_params_from(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))

    flat_params = torch.cat(params)
    return flat_params

def set_flat_params_to(model, flat_params):
    prev_ind = 0
    flat_params = torch.from_numpy(flat_params)
    for param in model.parameters():
        flat_size = int(reduce(lambda a, b: a*b, param.shape))
        param = flat_params[prev_ind:prev_ind + flat_size].view(param.shape)
        prev_ind += flat_size

def get_flat_grad_from(model):
    grads = []
    for param in model.parameters():
        grads.append(param.grad.view(-1))

    flat_grad = torch.cat(grads)
    return flat_grad

################# Mountain Car #################
def prepro_for_MountainCar(state):
    state = state.reshape(1, -1) * np.array([1, 400])
    ret = torch.FloatTensor(state)
    ret.requires_grad_(True)
    return ret

def reward_for_MountainCar(state, reward):
    if state[0] >= 0.5:
        return 100
    if state[0] >= 0.4:
        return 10
    if state[0] >= 0.2:
        return 5
    if state[0] >= -0.4:
            return 2
    elif -0.6 < state[0] < -0.4:
        return -2
    else:
        return -1
################# Pong #################
def prepro_for_Pong(I):
    """ prepro 210x160x3 uint8 frame into 80x80 2D float vector """
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    I = I.reshape(1, -1)
    ret = torch.FloatTensor(I)
    ret.requires_grad_(True)
    return ret

def reward_for_Pong(state, reward):
    """ 
    +1 if ball is in a small neighbor of our racket, 
    +5 if hit the ball
    +10 if win, -10 if lose, 0 for every step
    """
    if reward < 0:
        return -100
    elif reward > 0:
        return 100
    else:
        s = state[35:195:2, ::2, 0].copy()
        s[s==144] = 0
        s[s==109] = 0
        up = np.min(np.where(s[:, 70] != 0))
        down = np.max(np.where(s[:, 70] != 0))
        if np.any(s[up: down+1, 69] != 0):
            return 5
        elif np.any(s[up: down+1, 65:70] != 0):
            return 1
        else:
            return 0