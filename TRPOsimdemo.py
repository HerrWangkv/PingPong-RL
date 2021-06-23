from utilities import Policy_Network, Value_Network, Transition, Memory, prepro
import torch
import numpy as np
import wandb
import os
import datetime
import gym
import argparse

class TRPO:
    def __init__(self, env, device, save_dir,alpha=1e-2, gamma=0.9, max_kl=1e-2, mu = 0.5, max_iter=5):
        self.act_size = env.action_space.n
        self.env = env
        self.device = device
        self.save_dir = save_dir
        self.alpha = alpha      # learning rate
        self.gamma = gamma      # discount rate
        self.max_kl = max_kl    # kl divergence limit
        self.mu = mu            # backtracking coefficient
        self.max_iter=max_iter  # maximum number of backtracking steps
        self.policy = Policy_Network(self.act_size).to(self.device)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=alpha)
        self.value = Value_Network().to(self.device)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=alpha)
        self.buffer = Memory()

    # Need to output all act_probs for KL divergence
    def choose_action(self, state):
        act_probs = self.policy(state.to(self.device))
        # can't convert CUDA tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
        action = np.random.choice(self.act_size, 1, p=act_probs.cpu().data.numpy().reshape(-1))[0]
        return action, act_probs[0]

    def discount_rewards(self):
        discounted_rewards = torch.zeros((len(self.buffer), 1))
        running_add = 0
        for t in reversed(range(len(self.buffer))):
            # smaller gamma means no need for resetting running_add
            #if (self.buffer[t].reward != 0):
            #    running_add = 0
            running_add = running_add * self.gamma + self.buffer[t].reward
            discounted_rewards[t] = torch.Tensor([running_add])
        return discounted_rewards

    def learn(self):
        def get_diff_and_kl(new_policy):
            # sum of pi_new(a|s) / pi(a|s) A^\pi(s,a)
            diff = torch.Tensor([0]).to(self.device)
            # sum of KL divergence
            kl = torch.Tensor([0]).to(self.device)
            for t in range(len(self.buffer)):
                new_probs = new_policy(self.buffer[t].state.to(self.device))[0]
                diff += new_probs[self.buffer[t].action] / self.buffer[t].prob[self.buffer[t].action] * advantages[t]
                for a in range(self.act_size):
                    kl += new_probs[a] * torch.log(new_probs[a] / self.buffer[t].prob[a])
            return diff, kl / len(self.buffer)

        def backtracking():
            from copy import deepcopy
            alpha = 1
            for i in range(self.max_iter):
                new_policy = deepcopy(self.policy)
                for old_params, new_params in zip(self.policy.parameters(),\
                    new_policy.parameters()):
                    new_params.data += alpha * old_params.grad
                diff, new_kl = get_diff_and_kl(new_policy)
                if diff > 0 and new_kl < self.max_kl:
                    self.policy = new_policy
                    print(f"success, alpha = {alpha}, diff={diff}, kl_mean={new_kl}")
                    return
                else:
                    alpha = self.mu * alpha
            print(f"line search failed, alpha = {alpha}")
            self.policy = new_policy     

        rewards_to_go = self.discount_rewards().to(self.device)
        advantages = torch.zeros(len(self.buffer))
        # Optimize Policy Network
        for t in range(len(self.buffer) - 1):
            # Compute advantage estimates
            # Use discounted reward instead of reward
            advantages[t] = self.buffer[t].reward + \
                            self.gamma * self.buffer[t+1].value - \
                            self.buffer[t].value
        # normalize the advantage function
        #advantages = (advantages - torch.mean(advantages)) / torch.std(advantages)
        expected_return = torch.Tensor([0]).to(self.device)
        value_loss = torch.Tensor([0]).to(self.device)
        for t in range(len(self.buffer)):
            expected_return += torch.log(self.buffer[t].prob[self.buffer[t].action]).to(self.device) * \
                advantages[t].to(self.device)
            value_loss += torch.pow((self.buffer[t].value[0] - \
                            rewards_to_go[t]), 2)
        # Optimize Policy Network
        self.policy.zero_grad()
        expected_return.backward(retain_graph=True)
        backtracking()

        # Optimize Value Network
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        self.buffer = Memory()
        return expected_return, value_loss

    def train(self, episodes):
        for i in range(episodes):
            done = False
            state = self.env.reset()
            reward_sum = 0
            while not done:
                #self.env.render()
                state = prepro(state)
                action, act_probs = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                value = self.value(state.to(self.device))
                self.buffer.push(state, value, action, reward, next_state, act_probs)
                state = next_state
                reward_sum += reward
                if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
                    print(f'ep {i}: game finished, reward: {"-1" if reward == -1 else "1 !!!!!!!!"}')
            
            print(action, act_probs[action])
            expected_return, value_loss = self.learn()
            wandb.log({'reward_sum': reward_sum, 'expected_return': expected_return, 'value_loss': value_loss})
            print(f'Episode: {i} | total reward: {reward_sum}')
            if (i % 100 == 0):
                os.makedirs(os.path.join(self.save_dir, f'checkpoint_{i}'), exist_ok=True)
                torch.save(self.policy.state_dict(), os.path.join(self.save_dir, f'checkpoint_{i}/policy.pt'))
                torch.save(self.value.state_dict(), os.path.join(self.save_dir, f'checkpoint_{i}/value.pt'))

    def load_policy(self, model_dir):
        self.policy.load_state_dict(torch.load(model_dir, map_location=self.device))

    def load_value(self, model_dir):
        self.value.load_state_dict(torch.load(model_dir, map_location=self.device)) 
    def show(self, episodes):
        for i in range(episodes):
            state = self.env.reset()
            reward_sum = 0
            done = False
            while not done:
                self.env.render()
                state = prepro(state)
                action, act_probs = self.choose_action(state)
                state, reward, done, _ = self.env.step(action)
                reward_sum += reward
                if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
                    print (f'ep {i}: game finished, reward: {"-1" if reward == -1 else "1 !!!!!!!!"}')
                    print(act_probs)
            print(f'Episode: {i} | total reward: {reward_sum}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sp = parser.add_subparsers(dest="mode")
    sp.required = True
    t = sp.add_parser("train")
    t.add_argument("--policy", '-p', default=None, help='Path to trained policy network')
    t.add_argument("--value", '-v', default=None, help='Path to trained value network')
    sp.add_parser("show").add_argument("--policy", '-p',required=True, default=None, help='Path to trained policy network')
    config = vars(parser.parse_args())
    print(config)
    if config['mode'] == 'train':
        policy = config.get('policy', None)
        value = config.get('value', None)
        file_dir = os.path.join(os.path.dirname(__file__), f'./{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")}')
        os.makedirs(file_dir, exist_ok=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        wandb.init(project='Pong')
        num_episodes = 50000
        env = gym.make("Pong-v0")
        trpo_agent = TRPO(env, device, file_dir)
        if policy is not None:
            trpo_agent.load_policy(policy)
        if value is not None:
            trpo_agent.load_value(value)
        trpo_agent.train(num_episodes)  
    elif config['mode'] == 'show':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy = config.get('policy', None)
        env = gym.make("Pong-v0")
        trpo_agent = TRPO(env, device, 'blabla')
        trpo_agent.load_policy(policy)
        trpo_agent.show(5)    
    else:
        raise ValueError("Your mode is wrong!")

