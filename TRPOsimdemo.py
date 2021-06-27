from utilities import Policy_Network, Value_Network, Transition, Memory, prepro
import torch
import numpy as np
import wandb
import os
import datetime
import gym
import argparse
import torch.nn.functional as F
import matplotlib.pyplot as plt
class TRPO:
    def __init__(self, env, device, save_dir, batch_size=1, alpha=1e-3, gamma=0.9, max_kl=1e-3, mu = 0.5, max_iter=15):
        self.act_size = 2# only accept action 2 and 3 for simplification
        self.env = env
        self.device = device
        self.save_dir = save_dir
        self.batch_size = batch_size    #batch size before learning
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
        self.idol = Memory()

    # Need to output all act_probs for KL divergence
    def choose_action(self, state):
        """int, [1, self.act_size]"""
        act_probs = self.policy(state)
        possible_actions = (2, 3)
        # can't convert CUDA tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
        index = np.random.choice(len(possible_actions), 1, p=act_probs.cpu().data.numpy().reshape(-1))[0]
        return possible_actions[index], act_probs

    def discount_rewards(self, memory):
        """[N]"""
        discounted_rewards = torch.zeros(len(memory))
        discounted_rewards.requires_grad_(True)
        running_add = 0
        for t in reversed(range(len(memory))):
            if (memory[t].reward != 0):
                running_add = 0
            running_add = running_add * self.gamma + memory[t].reward
            discounted_rewards[t].data = torch.FloatTensor([running_add])
        return discounted_rewards

    def learn(self):
        # [N,1,80,80]
        states = torch.cat(self.buffer.collect().state).to(self.device)
        states.requires_grad_(True)
        # [N,2]
        old_probs = torch.cat(self.buffer.collect().prob).to(self.device) # probabilities of all actions
        old_probs.requires_grad_(True)
        actions = torch.LongTensor(self.buffer.collect().action).view(-1) # one-hot only accepts LongTensor
        # [N,2]
        actions = F.one_hot(actions-2, self.act_size).float().to(self.device)
        # [N]
        old_pis = torch.sum(old_probs * actions, dim=1).to(self.device) # probabilities of the sampled actions
        assert old_pis.requires_grad == True
        # [N]
        values = torch.cat(self.buffer.collect().value)[:,0].to(self.device)
        values.requires_grad_(True)
        idol_values = torch.cat(self.idol.collect().value)[:,0].to(self.device)
        idol_values.requires_grad_(True)

        def get_diff_and_kl(new_policy):
            # sum of pi_new(a|s) / pi(a|s) A^\pi(s,a)
            with torch.no_grad():
                new_probs = new_policy(states)
                diff = torch.sum(torch.sum(new_probs * actions, dim=1) / old_pis * advantages)
                kl = torch.sum(new_probs * torch.log((new_probs+1e-7) / (old_probs+1e-7)))
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
                print(f"diff = {diff}, new_kl = {new_kl}")
                if diff > 0 and torch.abs(new_kl) < self.max_kl:
                    self.policy = new_policy
                    print(f"success, alpha = {alpha}, diff={diff}, kl_mean={new_kl}")
                    return
                else:
                    alpha = self.mu * alpha
            if new_kl <= 1e-10:
                print("line search failed, reset random layer")
                reset_index = np.random.randint(6, size=1)[0]
                for i, new_params in enumerate(new_policy.parameters()):
                    if i == reset_index:
                        new_params.data = torch.randn_like(new_params.data)
            else:
                print(f"line search failed, alpha = {alpha}")
            self.policy = new_policy

        rewards_to_go = self.discount_rewards(self.buffer).to(self.device)
        idol_rewards_to_go = self.discount_rewards(self.idol).to(self.device)
        advantages = torch.zeros(len(self.buffer)).to(self.device)
        for t in range(len(self.buffer) - 1):
            # Compute advantage estimates
            if (t > 0 and self.buffer[t-1].reward != 0 and self.buffer[t].reward == 0):
                pass
            else:
                advantages[t]= self.buffer[t].reward + \
                            self.gamma * self.buffer[t+1].value - \
                            self.buffer[t].value
        # normalize the advantage function
        advantages = (advantages - torch.mean(advantages)) / torch.std(advantages)
        expected_return = torch.sum(torch.log(1 + old_pis) * advantages.data)/ self.buffer.size()
        value_loss = (torch.sum(torch.pow(values - rewards_to_go, 2)) + \
                        torch.sum(torch.pow(idol_values - idol_rewards_to_go, 2))) / (len(self.buffer) + len(self.idol))
        # Optimize Value Network
        print(f"value_loss = {value_loss}")
        self.value.zero_grad()
        value_loss.backward(retain_graph=True)
        self.value_optimizer.step()
        # Optimize Policy Network
        print(f"expected_return = {expected_return}")
        self.policy.zero_grad()
        expected_return.backward()
        backtracking()

        self.buffer = Memory()
        return expected_return, value_loss

    def train(self, episodes):
        for i in range(1, episodes+1):
            done = False
            state = self.env.reset()
            state = prepro(state).to(self.device)
            reward_sum = 0
            while not done:
                #self.env.render()
                action, act_probs = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                value = self.value(state)
                next_state = prepro(next_state).to(self.device)
                #           [1,1,80,80],[1,1],  int,   float, [1, self.act_size] 
                self.buffer.push(state, value, action, reward, act_probs)
                state = next_state
                reward_sum += reward
                if (reward == 1 and self.idol.size() < 10):
                    for j in range(self.buffer.last_boundary, len(self.buffer)):
                        self.idol.push_transition(self.buffer[j])
                if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
                    print(f'ep {i}: game finished, reward: {"-1" if reward == -1 else "1 !!!!!!!!"}')
            print("action_probs: ", act_probs[0])
            print(f'Episode: {i} | total reward: {reward_sum}')
            if (i % self.batch_size == 0):
                expected_return, value_loss = self.learn()
                wandb.log({'expected_return': expected_return, 'value_loss': value_loss})
            wandb.log({'reward_sum': reward_sum})
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
            state = prepro(state).to(self.device)
            reward_sum = 0
            done = False
            old_value = self.value(state)
            while not done:
                self.env.render()
                action, act_probs = self.choose_action(state)
                state, reward, done, _ = self.env.step(action)
                state = prepro(state).to(self.device)
                new_value = self.value(state)
                advantage = reward + self.gamma*new_value - old_value
                reward_sum += reward
                if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
                    print (f'ep {i}: game finished, reward: {"-1" if reward == -1 else "1 !!!!!!!!"}')
                    print(act_probs)
                    print(advantage)
                    
            print(f'Episode: {i} | total reward: {reward_sum}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sp = parser.add_subparsers(dest="mode")
    sp.required = True
    t = sp.add_parser("train")
    t.add_argument("--policy", '-p', default=None, help='Path to trained policy network')
    t.add_argument("--value", '-v', default=None, help='Path to trained value network')
    s = sp.add_parser("show")
    s.add_argument("--policy", '-p',required=True, default=None, help='Path to trained policy network')
    s.add_argument("--value", '-v', default=None, help='Path to trained value network')
    config = vars(parser.parse_args())
    print(config)
    env = gym.make("Pong-v0")
    if config['mode'] == 'train':
        policy = config.get('policy', None)
        value = config.get('value', None)
        file_dir = os.path.join(os.path.dirname(__file__), f'./{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")}')
        os.makedirs(file_dir, exist_ok=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        wandb.init(project='Pong-v0')
        num_episodes = 50000
        trpo_agent = TRPO(env, device, file_dir)
        if policy is not None:
            trpo_agent.load_policy(policy)
        if value is not None:
            trpo_agent.load_value(value)
        trpo_agent.train(num_episodes)  
    elif config['mode'] == 'show':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy = config.get('policy', None)
        value = config.get('value', None)
        trpo_agent = TRPO(env, device, 'blabla')
        trpo_agent.load_policy(policy)
        if value is not None:
            trpo_agent.load_value(value) 
        trpo_agent.show(5) 
        ''' DEBUG
        state = trpo_agent.env.reset()
        done = False
        for i in range(100):
            action= trpo_agent.env.action_space.sample()
            state, reward, done, _ = trpo_agent.env.step(action)
            trpo_agent.env.render()  
        state = prepro(state).to(trpo_agent.device)      
        temp = trpo_agent.value.conv(state).cpu().data.numpy()[0]
        plt.imshow(temp[0]) 
        plt.show()
        plt.imshow(temp[1]) 
        plt.show()
        plt.imshow(temp[2])
        plt.show()
        '''
    else:
        raise ValueError("Your mode is wrong!")

