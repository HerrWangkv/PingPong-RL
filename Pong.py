from utilities import *
#from TRPOsim import TRPO
from TRPOwithValue import TRPO
import datetime
import gym
import wandb
import torch
import argparse
import os

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
    env.seed(42)
    if config['mode'] == 'train':
        policy = config.get('policy', None)
        value = config.get('value', None)
        file_dir = os.path.join(os.path.dirname(__file__), f'./{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")}')
        os.makedirs(file_dir, exist_ok=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        wandb.init(project='Pong-v0')
        num_episodes = 50000
        trpo_agent = TRPO(env, file_dir, prepro_for_Pong, device, reward_for_Pong, threshold=18, max_kl=0.01)
        if policy is not None:
            trpo_agent.load_policy(policy)
        if value is not None:
            trpo_agent.load_value(value)
        trpo_agent.train(num_episodes)  
    elif config['mode'] == 'show':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy = config.get('policy', None)
        value = config.get('value', None)
        trpo_agent = TRPO(env, "blabla", prepro_for_Pong, device)
        trpo_agent.load_policy(policy)
        if value is not None:
            trpo_agent.load_value(value) 
        trpo_agent.show(5) 
    else:
        raise ValueError("Your mode is wrong!")