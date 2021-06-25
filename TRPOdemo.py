from utilities import Policy_Network, Value_Network, Transition, Memory, prepro
import torch
import numpy as np
import wandb
import os
import datetime
import gym
import argparse
import torch.nn.functional as F
class TRPO:
    def __init__(self, env, device, save_dir, alpha=0.1, gamma=0.99, max_kl=1e-2, mu = 0.5, max_iter=10):
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
        """int, [1, self.act_size]"""
        act_probs = self.policy(state)
        # can't convert CUDA tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
        action = np.random.choice(self.act_size, 1, p=act_probs.cpu().data.numpy().reshape(-1))[0]
        return action, act_probs

    def discount_rewards(self):
        """[N]"""
        discounted_rewards = torch.zeros(len(self.buffer))
        running_add = 0
        for t in reversed(range(len(self.buffer))):
            # smaller gamma means no need for resetting running_add
            if (self.buffer[t].reward != 0):
                running_add = 0
            running_add = running_add * self.gamma + self.buffer[t].reward
            discounted_rewards[t] = running_add
        return discounted_rewards

    def learn(self):
        # [N,1,80,80]
        states = torch.cat(self.buffer.collect().state).to(self.device)
        states.requires_grad_(True)
        # [N,6]
        old_probs = torch.cat(self.buffer.collect().prob).to(self.device) # probabilities of all actions
        actions = torch.LongTensor(self.buffer.collect().action).view(-1) # one-hot only accepts LongTensor
        # [N,6]
        actions = F.one_hot(actions, self.act_size).float().to(self.device)
        # [N]
        old_pis = torch.sum(old_probs * actions, dim=1).to(self.device) # probabilities of the sampled actions
        # [N]
        values = torch.cat(self.buffer.collect().value)[:,0].to(self.device)

        def get_kl_at_theta_k():
            kl = torch.sum(old_probs * torch.log((old_probs+1e-6) / (old_probs.data+1e-6)))
            return kl

        def get_diff_and_kl(new_policy):
            with torch.no_grad():
                new_probs = new_policy(states)
                diff = torch.sum(torch.sum(new_probs * actions, dim=1) / old_pis * advantages)
                kl = torch.sum(new_probs * torch.log((new_probs+1e-6) / (old_probs+1e-6)))
                return diff, kl / len(self.buffer)
        

        def compute_x(name, params, g, inner, msteps=100, damping=0.1, threshold=0.1):
            x = torch.rand_like(params.data, requires_grad=True).to(self.device)
            shape = params.data.shape
            # conjugate gradient
            old_rdotr = None
            outer = torch.sum(inner * x)
            self.policy.zero_grad()
            outer.backward(retain_graph=True)
            Hx = params.grad.view(-1, 1)
            # make H positive definite
            Hx += damping * x.view(-1, 1)
            for step in range(msteps):
                r = g - Hx if old_rdotr is None else r - alpha * z
                if (torch.mean(torch.abs(r)) < threshold):
                    print(f'Lucky conjungate gradient for {name}')
                    return x
                beta = None if old_rdotr is None else (r.T@r) / old_rdotr 
                d = r.clone() if old_rdotr is None else r + beta * d
                old_rdotr = r.T@r
                z = torch.autograd.grad(torch.sum(inner.clone().view(1, -1) @ d), params, retain_graph=True)[0].clone().view(-1, 1)
                alpha = old_rdotr / (d.T @ z)
                x = x.view(-1, 1)
                x = x + alpha*d
                x = x.view(shape)
                old_rdotr = r.T@r
            print(f'Poor {name}')
            return x

        def backtracking():
            from copy import deepcopy
            ps = []
            grad = [params.grad.clone() for params in self.policy.parameters()]
            kl_mean = get_kl_at_theta_k()
            self.policy.zero_grad()
            kl_mean.backward(retain_graph=True, create_graph=True)
            inner_grad = [params.grad.clone() for params in self.policy.parameters()]
            for i, (name, params) in enumerate(self.policy.named_parameters()):
                x = compute_x(name, params, grad[i].view(-1, 1), inner_grad[i])
                ps.append(x)
            # line search
            alpha = 1
            for i in range(self.max_iter):
                new_policy = deepcopy(self.policy)
                for index, params in enumerate(new_policy.parameters()):
                    params.data -= alpha * ps[index]
                diff, new_kl = get_diff_and_kl(new_policy)
                print(diff, new_kl)
                if diff > 0 and new_kl < self.max_kl:
                    print("success", alpha)
                    self.policy = new_policy
                    new_policy.zero_grad()
                    return
                else:
                    alpha = self.mu * alpha
            print("line search failed", alpha)
            self.policy = new_policy
            new_policy.zero_grad()        

        rewards_to_go = self.discount_rewards().to(self.device)
        advantages = torch.zeros(len(self.buffer)).to(self.device)
        # Optimize Policy Network
        for t in range(len(self.buffer) - 1):
            # Compute advantage estimates
            if (t > 0 and self.buffer[t-1].reward != 0 and self.buffer[t].reward == 0):
                advantages[t] = 0
            else:
                advantages[t] = self.buffer[t].reward + \
                            self.gamma * self.buffer[t+1].value - \
                            self.buffer[t].value
        # normalize the advantage function
        #advantages = (advantages - torch.mean(advantages)) / torch.std(advantages)

        expected_return = -1 * torch.sum(torch.log(old_pis) * advantages)
        value_loss = torch.sum(torch.pow(values - rewards_to_go, 2)) / len(self.buffer)
        print(f"expected_return = {expected_return}, value_loss = {value_loss}")
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
                if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
                    print(f'ep {i}: game finished, reward: {"-1" if reward == -1 else "1 !!!!!!!!"}')
            
            print(action, act_probs[0, action])
            print(f'Episode: {i} | total reward: {reward_sum}')
            expected_return, value_loss = self.learn()
            wandb.log({'expected_return': expected_return, 'value_loss': value_loss, 'reward_sum': reward_sum})
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
