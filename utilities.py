import torch
from torch import nn
from collections import namedtuple

def prepro(I):
    """ prepro 210x160x3 uint8 frame into 80x80 2D float vector """
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    I = I.reshape(1, 1, 80, 80)
    ret = torch.FloatTensor(I)
    ret.requires_grad_(True)
    return ret

class Policy_Network(nn.Module):
    def __init__(self, act_size):
        super(Policy_Network, self).__init__()
        self.conv = nn.Sequential(
            # 1*80*80->1*26*26
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, stride=3),
            nn.ReLU(),
            # 1*26*26->1*13*13
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(13*13, 64),
            nn.ReLU(),
            nn.Linear(64, act_size),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv(x)
        probs = self.fc(x.view(x.shape[0], -1))
        return probs

class Value_Network(nn.Module):
    def __init__(self):
        super(Value_Network, self).__init__()
        self.conv = nn.Sequential(
            # 1*80*80->1*26*26
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, stride=3),
            nn.ReLU(),
            # 1*26*26->1*13*13
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(13*13, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.conv(x)
        value = self.fc(x.view(x.shape[0], -1))
        return value

Transition = namedtuple('Transition', ['state', 'value', 'action', 'reward', 'prob'])

class Memory:
    def __init__(self):
        self.memory = []
        self.trajectories = 0
        self.last_boundary = 0

    def push(self, *args):
        self.memory.append(Transition(*args))
        # reward
        if (args[3] != 0):
            self.trajectories += 1
        if (args[3] == 0 and self.memory[len(self.memory)-2].reward != 0):
            self.last_boundary = len(self.memory) - 1
    
    def push_transition(self, t):
        self.memory.append(t)
        if (t.reward != 0):
            self.trajectories += 1
            self.last_boundary = len(self.memory)

    def collect(self):
        return Transition(*zip(*self.memory))

    def size(self):
        return self.trajectories
        
    def __getitem__(self, position):
        return self.memory[position]

    def __len__(self):
        return len(self.memory)