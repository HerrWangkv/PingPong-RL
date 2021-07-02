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
            nn.Linear(obs_size, 64),
            nn.ReLU(),
            nn.Linear(64, act_size),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        probs = self.fc(x)
        return probs

class Value_Network(nn.Module):
    def __init__(self, obs_size):
        super(Value_Network, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
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
################# TRPO #################
class TRPO:
    def __init__(self, env, save_dir, prepro, device=torch.device('cpu'), \
                    reward_func=lambda state, reward: reward, \
                    threshold=0, batch_size=1, gamma=0.6, max_kl=0.1, mu = 0.5, max_iter=20):
        self.act_size = env.action_space.n
        self.obs_size = reduce(lambda a,b:a*b, env.reset().shape)
        self.env = env
        self.device = device
        self.save_dir = save_dir
        self.prepro = prepro    #Preprocessing
        self.reward_func = reward_func
        self.threshold = threshold
        self.batch_size = batch_size    #batch size before learning
        self.gamma = gamma      # discount rate
        self.max_kl = max_kl    # kl divergence limit
        self.mu = mu            # backtracking coefficient
        self.max_iter=max_iter  # maximum number of backtracking steps
        self.policy = Policy_Network(self.obs_size, self.act_size).to(self.device)
        self.value = Value_Network(self.obs_size).to(self.device)
        self.buffer = Memory()
        self.unchanged = 0

    # Need to output all act_probs for KL divergence
    def choose_action(self, state):
        """int, [1, self.act_size]"""
        act_probs = self.policy(state)
        # can't convert CUDA tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
        index = int(Categorical(act_probs).sample())
        #index = torch.argmax(act_probs)
        return index, act_probs

    def discount_rewards(self, rewards):
        """[N]"""
        with torch.no_grad():
            discounted_rewards = torch.zeros(rewards.shape[0])
            running_add = 0
            for t in reversed(range(rewards.shape[0])):
                running_add = running_add * self.gamma + rewards[t]
                discounted_rewards[t] = running_add
            return discounted_rewards

    def learn(self):
        # [N,obs_size]
        states = torch.cat(self.buffer.collect().state).to(self.device)
        # [N,act_size]
        old_probs = torch.cat(self.buffer.collect().prob).to(self.device) # probabilities of all actions
        actions = torch.LongTensor(self.buffer.collect().action).view(-1) # one-hot only accepts LongTensor
        # [N,act_size]
        actions = F.one_hot(actions, self.act_size).float().to(self.device)
        # [N]
        old_pis = torch.sum(old_probs * actions, dim=1).to(self.device) # probabilities of the sampled actions
        # [N]
        rewards = torch.FloatTensor(self.buffer.collect().reward).view(-1)
        # [N]
        values = self.value(states)[:,0]
        def get_diff_and_kl(new_policy):
            # sum of pi_new(a|s) / pi(a|s) A^\pi(s,a)
            with torch.no_grad():
                new_probs = new_policy(states)
                diff = torch.sum(((torch.sum(new_probs * actions, dim=1) / old_pis.data)-1) * advantages)/ self.batch_size
                kl = torch.sum(new_probs * torch.log((new_probs+1e-7) / (old_probs+1e-7)))
                return diff, kl / len(self.buffer)

        def backtracking():
            alpha = 1
            for _ in range(self.max_iter):
                new_policy = Policy_Network(self.obs_size, self.act_size).to(self.device)
                for old_params, new_params in zip(self.policy.parameters(),\
                    new_policy.parameters()):
                    new_params.data = old_params + alpha * old_params.grad
                diff, new_kl = get_diff_and_kl(new_policy)
                print(f"diff = {diff}, new_kl = {new_kl}")
                if diff >= 0 and torch.abs(new_kl) < self.max_kl:
                    self.policy = new_policy
                    print(f"success, alpha = {alpha}, diff={diff}, kl_mean={new_kl}")
                    self.unchanged = 0
                    return
                else:
                    alpha = self.mu * alpha
            self.unchanged += 1
            if (self.unchanged >= 100):
                self.max_iter += 10
                print(f"line search failed, policy remains unchanged, increase max_iter to {self.max_iter}")
                self.unchanged = 0
            else:
                print(f"line search failed, policy remains unchanged.")
            
            
        def get_value_loss(flat_params):
            va = Value_Network(self.obs_size).to(self.device)
            set_flat_params_to(va, flat_params)
            values_ = va(states)[:,0]
            value_loss = (torch.sum(torch.pow(values_ - rewards_to_go.data, 2))) / len(self.buffer)
            # weight decay
            for params in va.parameters():
                value_loss += torch.sum(torch.pow(params, 2)) * 0.1
            va.zero_grad()
            value_loss.backward()
            return (value_loss.data.cpu().double().numpy(), get_flat_grad_from(va).data.cpu().double().numpy())
        rewards_to_go = self.discount_rewards(rewards).to(self.device)
        advantages = rewards_to_go.data - values.data 
        # normalize the advantage function
        #advantages = (advantages - torch.mean(advantages)) / torch.std(advantages)
        #advantages = advantages / torch.std(advantages)
        expected_return = torch.sum((old_pis / old_pis.data) * advantages.data)/ self.batch_size
        # weight decay
        for params in self.policy.parameters():
            expected_return += torch.sum(torch.pow(params, 2)) * 0.1
        
        # Optimize Policy Network
        print(f"expected_return = {expected_return}")
        self.policy.zero_grad()
        expected_return.backward()
        backtracking()
        # Optimize Value Network
        flat_params, value_loss, opt_info = optimize.fmin_l_bfgs_b(func=get_value_loss, x0=get_flat_params_from(self.value).cpu().double().numpy(), maxiter=25)
        set_flat_params_to(self.value, flat_params)
        print(f"value_loss = {value_loss}")

        self.buffer = Memory()
        return expected_return, value_loss

    def train(self, episodes):
        running_average = None
        reward_sum = 0
        for i in range(1, episodes+1):
            done = False
            state = self.env.reset()
            pro_state = self.prepro(state).to(self.device)
            while not done:
                #self.env.render()
                action, act_probs = self.choose_action(pro_state)
                next_state, reward, done, _ = self.env.step(action)
                reward_sum += reward
                r = self.reward_func(state, reward)
                self.buffer.push(pro_state, action, r, act_probs)
                state = next_state
                pro_state = self.prepro(state).to(self.device)
            print("action_probs: ", act_probs[0])
            print(f'Episode: {i} | total reward: {reward_sum}')
            if (i % self.batch_size == 0):
                reward_avg = reward_sum / self.batch_size
                print(f"buffer size:{len(self.buffer)}")
                expected_return, value_loss = self.learn()
                running_average = reward_avg if running_average is None else 0.99 * running_average + 0.01 * reward_avg
                wandb.log({'expected_return': expected_return, 'value_loss': value_loss, 'reward_average': reward_avg, 'running average': running_average})
                if running_average > self.threshold:
                    self.save('BEST')
                    print('*********FINISHED*********')
                    return
                reward_sum = 0
            if (i % 100 == 0):
                self.save(i)
    def save(self, i):
        os.makedirs(os.path.join(self.save_dir, f'checkpoint_{i}'), exist_ok=True)
        torch.save(self.policy.state_dict(), os.path.join(self.save_dir, f'checkpoint_{i}/policy.pt'))
        torch.save(self.value.state_dict(), os.path.join(self.save_dir, f'checkpoint_{i}/value.pt'))

    def load_policy(self, model_dir):
        self.policy.load_state_dict(torch.load(model_dir, map_location=self.device))

    def load_value(self, model_dir):
        self.value.load_state_dict(torch.load(model_dir, map_location=self.device)) 

    def show(self, episodes):
        for i in range(episodes):
            reward_sum = 0
            done = False
            state = self.env.reset()
            state = self.prepro(state).to(self.device)
            #old_value = self.value(state)
            while not done:
                self.env.render()
                action, _ = self.choose_action(state)
                state, reward, done, _ = self.env.step(action)
                state = self.prepro(state).to(self.device)
                #new_value = self.value(state)
                #advantage = reward + self.gamma*new_value - old_value
                reward_sum += reward
                    
            print(f'Episode: {i} | total reward: {reward_sum}')

################# Mountain Car #################
def prepro_for_MountainCar(state):
    state = state.reshape(1, -1) * np.array([1, 400])
    ret = torch.FloatTensor(state)
    ret.requires_grad_(True)
    return ret

def reward_for_MountainCar(state, reward):
    if reward > 0:
        return 200
    if state[0] > -0.4 and state[1] > 0:
        return (1 + state[0]) * 10
    elif state[0] < -0.6 and state[1] < 0:
        return -5 * state[0]
    else:
        return -1