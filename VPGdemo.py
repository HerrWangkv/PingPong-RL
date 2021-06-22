from utilities import Policy_Network, Value_Network, Transition, Memory, prepro
import torch
import numpy as np
import wandb
import os
import datetime
import gym

class VPG:
    def __init__(self, env, device, save_dir,alpha=1e-2, gamma=0.99):
        self.act_size=env.action_space.n
        self.env = env
        self.device = device
        self.save_dir = save_dir
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount rate
        self.policy = Policy_Network(self.act_size).to(self.device)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=alpha)
        self.value = Value_Network()
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=alpha)
        self.replay_buffer = Memory()

    def choose_action(self, state):
        act_probs = self.policy(state.to(self.device))
        # can't convert CUDA tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
        action = np.random.choice(self.act_size, 1, p=act_probs.cpu().data.numpy().reshape(-1))[0]
        return action, act_probs[0, action]

    def discount_rewards(self):
        discounted_rewards = torch.zeros((len(self.replay_buffer), 1))
        running_add = 0
        for t in reversed(range(len(self.replay_buffer))):
            if (self.replay_buffer[t].reward != 0):
                running_add = 0
            running_add = running_add * self.gamma + self.replay_buffer[t].reward
            discounted_rewards[t] = torch.Tensor([running_add])
        return discounted_rewards

    def learn(self):
        rewards_to_go = self.discount_rewards().to(self.device)
        policy_loss = torch.Tensor([0]).to(self.device)
        value_loss = torch.Tensor([0]).to(self.device)
        # Optimize Policy Network
        for t in range(len(self.replay_buffer)):
            # Compute advantage estimates
            if self.replay_buffer[t].reward != 0:
                break
            advantage = self.replay_buffer[t].reward + \
                        self.gamma * self.replay_buffer[t+1].value - \
                        self.replay_buffer[t].value
            value_loss += torch.pow((self.replay_buffer[t].value[0].to(self.device) - \
                            rewards_to_go[t]), 2)
            policy_loss -= torch.log(self.replay_buffer[t].prob).to(self.device) * \
                advantage[0].to(self.device)
        # Optimize Policy Network
        self.policy_optimizer.zero_grad()
        policy_loss.backward(retain_graph=True)
        self.policy_optimizer.step()

        # Optimize Value Network
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        self.replay_buffer = Memory()
        return policy_loss, value_loss

    def train(self, episodes):
        for i in range(episodes):
            done = False
            state = self.env.reset()
            reward_sum = 0
            while not done:
                #self.env.render()
                state = prepro(state)
                action, act_prob = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                value = self.value(state)
                self.replay_buffer.push(state, value, action, reward, next_state, act_prob)
                state = next_state
                reward_sum += reward
                if reward != 0: # Pong has either +1 or -1 reward exactly when game ends.
                    print(f'ep {i}: game finished, reward: {"-1" if reward == -1 else "1 !!!!!!!!"}')
            policy_loss, value_loss = self.learn()
            wandb.log({'reward_sum': reward_sum, 'policy_loss': policy_loss, 'value_loss': value_loss})
            print(f'Episode: {i} | total reward: {reward_sum}')
            if (i % 100 == 0):
                os.makedirs(os.path.join(self.save_dir, f'checkpoint_{i}'), exist_ok=True)
                torch.save(self.policy.state_dict(), os.path.join(self.save_dir, f'checkpoint_{i}/policy.pt'))
                torch.save(self.value.state_dict(), os.path.join(self.save_dir, f'checkpoint_{i}/value.pt'))

    def load_model(self, model_dir):
        self.policy.load_state_dict(torch.load(model_dir, map_location='cpu'))
    
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
    file_dir = os.path.join(os.path.dirname(__file__), f'./{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")}')
    os.makedirs(file_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb.init(project='Pong')
    num_episodes = 50000
    env = gym.make("Pong-v0")
    pg_agent = VPG(env, device, file_dir)
    pg_agent.train(num_episodes)           

