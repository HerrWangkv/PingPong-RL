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
    config = vars(parser.parse_args())
    print(config)
    env = gym.make("MountainCar-v0")
    env.seed(42)
    file_dir = os.path.join(os.getcwd(), f'./MountainCar')
    os.makedirs(file_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config['mode'] == 'train':
        policy = config.get('policy', None)
        value = config.get('value', None)
        trpo_agent = TRPO(env, file_dir, prepro_for_MountainCar, device, reward_for_MountainCar,\
             gamma=0.9, threshold=-150, max_kl=0.01)
        if policy is not None:
            trpo_agent.load_policy(policy)
        if value is not None:
            trpo_agent.load_value(value)
        num_episodes = 50000
        wandb.init(project='OtherRLEnv')
        trpo_agent.train(num_episodes)  
    elif config['mode'] == 'show':
        policy = config.get('policy', None)
        trpo_agent = TRPO(env, "blabla", prepro_for_MountainCar, device)
        trpo_agent.load_policy(policy)
        trpo_agent.show(5) 
    else:
        raise ValueError("Your mode is wrong!")

