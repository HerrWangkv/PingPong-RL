import torch
import os
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from scipy import optimize
import wandb
from functools import reduce
from utilities import *

class TRPO:
    """
    simplified TRPO only uses first-order derivative for gradient descent
    """
    def __init__(self, env, save_dir, prepro, device=torch.device('cpu'), \
                    reward_func=lambda state, reward: reward, \
                    threshold=0, batch_size=1, gamma=0.99, max_kl=0.1, mu = 0.5, max_iter=20):
        self.act_size = env.action_space.n
        self.obs_size = reduce(lambda a,b:a*b, prepro(env.reset()).shape)
        self.env = env
        self.device = device
        self.save_dir = save_dir
        self.prepro = prepro                # Preprocessing function, environment specific. Modify the default state
        self.reward_func = reward_func      # Reward function, environment specific. Modify the default reward to accelerate training
        self.threshold = threshold          # Threshold of running_average to end the training
        self.batch_size = batch_size        # how many times we play the game before learning
        self.gamma = gamma                  # discount rate
        self.max_kl = max_kl                # kl divergence limit
        self.mu = mu                        # backtracking coefficient
        self.max_iter=max_iter              # maximum number of backtracking steps
        self.policy = Policy_Network(self.obs_size, self.act_size).to(self.device)
        self.value = Value_Network(self.obs_size).to(self.device)
        self.buffer = Memory()
        self.unchanged = 0
        self.states = None
        self.values = None
        self.rewards = None

    def choose_action(self, state):
        """
        index: chosen action according to the policy network. int
        act_probs: output of policy network. Tensor [1, self.act_size]
        """
        act_probs = self.policy(state)
        index = int(Categorical(act_probs.data).sample())
        return index, act_probs

    def discount_rewards(self, rewards):
        """
        discounted_rewards: accumulated discounted rewards. Tensor [N]
        """
        with torch.no_grad():
            discounted_rewards = torch.zeros(rewards.shape[0])
            running_add = 0
            for t in reversed(range(rewards.shape[0])):
                if self.env.spec.id.startswith('Pong') and (rewards[t] == 50 or rewards[t] == -50):
                    running_add = 0
                running_add = running_add * self.gamma + rewards[t]
                discounted_rewards[t] = running_add
            return discounted_rewards

    def evaluate(self):
        if self.states is None:
        # [N,obs_size]
            self.states = torch.cat(self.buffer.collect().state).to(self.device)

        if self.rewards is None:
            # [N]
            self.rewards = torch.FloatTensor(self.buffer.collect().reward).view(-1)
        if self.values is None:
            # [N]
            self.values = self.value(self.states)[:,0]
        def get_value_loss(flat_params):
            va = Value_Network(self.obs_size).to(self.device)
            set_flat_params_to(va, flat_params)
            values_ = va(self.states)[:,0]
            value_loss = (torch.sum(torch.pow(values_ - rewards_to_go.data, 2))) / len(self.buffer)
            # weight decay
            for params in va.parameters():
                value_loss += torch.sum(torch.pow(params, 2)) * 0.1
            va.zero_grad()
            value_loss.backward()
            return (value_loss.data.cpu().double().numpy(), get_flat_grad_from(va).data.cpu().double().numpy())

        rewards_to_go = self.discount_rewards(self.rewards).to(self.device)  
        # Optimize Value Network
        flat_params, value_loss, _ = optimize.fmin_l_bfgs_b(func=get_value_loss, x0=get_flat_params_from(self.value).cpu().double().numpy(), maxiter=25)
        set_flat_params_to(self.value, flat_params)
        print(f"value_loss = {value_loss}")
        self.states = None
        self.values = None
        self.rewards = None
        self.buffer = Memory()
        return value_loss

    def learn(self):
        # [N,obs_size]
        self.states = torch.cat(self.buffer.collect().state).to(self.device)
        # [N,act_size]
        old_probs = torch.cat(self.buffer.collect().prob).to(self.device) # probabilities of all actions
        actions = torch.LongTensor(self.buffer.collect().action).view(-1) # one-hot only accepts LongTensor
        # [N,act_size]
        actions = F.one_hot(actions, self.act_size).float().to(self.device)
        # [N]
        old_pis = torch.sum(old_probs * actions, dim=1).to(self.device) # probabilities of the sampled actions
        # [N]
        self.rewards = torch.FloatTensor(self.buffer.collect().reward).view(-1)
        # [N]
        self.values = self.value(self.states)[:,0]
        def get_diff_and_kl(new_policy):
            with torch.no_grad():
                new_probs = new_policy(self.states)
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
                if (diff == 0):
                    print("Fall into a local optimum, choose random parameters")
                    for params in self.policy.parameters():
                        params.data = torch.randn_like(params)
                    return
                if diff > 0 and torch.abs(new_kl) < self.max_kl:
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
    
        rewards_to_go = self.discount_rewards(self.rewards).to(self.device)
        advantages = rewards_to_go.data - self.values.data 
        expected_return = torch.sum((old_pis / old_pis.data) * advantages.data)/ self.batch_size
        # weight decay
        for params in self.policy.parameters():
            expected_return -= torch.mean(torch.pow(params, 2)) * 0.1
        
        # Optimize Policy Network
        print(f"expected_return = {expected_return}")
        self.policy.zero_grad()
        expected_return.backward()
        backtracking()
        
        value_loss =  self.evaluate()
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
                if (i <= 100):
                    action = self.env.action_space.sample()
                    act_probs = None
                else:
                    action, act_probs = self.choose_action(pro_state)
                next_state, reward, done, _ = self.env.step(action)
                reward_sum += reward
                r = self.reward_func(state, reward)
                self.buffer.push(pro_state, action, r, act_probs)
                state = next_state
                pro_state = self.prepro(state).to(self.device)
            if (i > 100): 
                print("action_probs: ", act_probs[0])
            print(f'Episode: {i} | total reward: {reward_sum}')
            if (i % self.batch_size == 0):
                reward_avg = reward_sum / self.batch_size
                running_average = reward_avg if running_average is None else 0.99 * running_average + 0.01 * reward_avg
                print(f"buffer size:{len(self.buffer)}")
                if (i <= 100):
                    value_loss = self.evaluate()
                    wandb.log({'value_loss': value_loss, 'reward_average': reward_avg, 'running average': running_average})
                else:
                    expected_return, value_loss = self.learn()
                    wandb.log({'expected_return': expected_return, 'value_loss': value_loss, 'reward_average': reward_avg, 'running average': running_average})
                if running_average > self.threshold:
                    self.save('BEST')
                    print('*********FINISHED*********')
                    print(f'running average: {running_average}')
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
            while not done:
                self.env.render()
                action, _ = self.choose_action(state)
                state, reward, done, _ = self.env.step(action)
                state = self.prepro(state).to(self.device)
                reward_sum += reward
                    
            print(f'Episode: {i} | total reward: {reward_sum}')