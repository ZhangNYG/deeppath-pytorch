import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import torchvision.transforms as T

import numpy as np 
from itertools import count
import os, sys
import time

from networks import PolicyNet
from utils import *
from env import Env 
from BFS.KB import KB 
from BFS.BFS import BFS 

relation = sys.argv[1]
graph_path = os.path.join(dataPath, 'tasks', relation, 'graph.txt')
relation_path = os.path.join(dataPath, 'tasks', relation, 'train_pos')


def train():
    policy_nn = PolicyNet(state_dim, action_space)
    optimizer = optim.Adam(policy_nn.parameters(), lr=0.001)

    with open(relation_path) as f:
        train_data = f.readlines()
        num_samples = len(train_data)

    if num_samples > 500:
        num_samples = 500

    for episode in range(num_samples):
        print("Episode %d "%episode)
        print("Training Sample: ", train_data[episode%num_samples][:-1])

        env = Env(dataPath, train_data[episode%num_samples])
        sample = train_data[episode%num_samples].split()

        try: 
            good_episodes = teacher(sample[0], sample[1], 5, env, graph_path)
        except: 
            print("Cannot find a path")
            continue

        for transitions in good_episodes:
            state_batch = []
            action_batch = []
            for transition in transitions:
                state_batch.append(transition.state)
                action_batch.append(transition.action)
            state_batch = np.squeeze(state_batch)
            state_batch = np.reshape(state_batch, [-1, state_dim])

            action_prob = policy_nn(state_batch)

            action_mask = torch.FloatTensor(len(action_batch), action_space)
            action_mask.zero_()
            action_mask.scatter_(1, action_batch, 1)

            picked_action_prob = torch.mm(action_prob, action_mask)
            loss = torch.sum(-torch.log(picked_action_prob))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    torch.save(policy_nn.state_dict(), 'models/policy_supervised_'+relation)


def test(test_episode):
    policy_nn = PolicyNet(state_dim, action_space)
    policy_nn.load_state_dict(torch.load('models/policy_supervised_'+relation))
    policy_nn.eval()

    with open(relation_path) as f:
        test_data = f.readlines()

    test_data = test_data[-test_episode:]
    print(len(test_data))

    success = 0
    for episode in range(len(test_data)):
        print("Test sample %d: %s" % episode, test_data[episode][:-1])
        env = Env(dataPath, test_data[episode])
        sample = test_data[episode].split()
        state_idx = [env.entity2id[sample[0]], env.entity2id[sample[1]], 0]
        
        for t in count():
            state_vec = env.idx_state(state_idx)
            action_probs = policy_nn(state_vec)
            action_chosen = np.random.choice(np.arange(action_space), p=np.squeeze(action_probs))
            reward, new_state, done = env.interact(state_idx, action_chosen)

            if done or t==max_steps_test:
                if done:
                    print("success")
                    success += 1
                print("episode ends\n")
                break
            
            state_idx = new_state

    print("success percentage: ", success/test_episode)

if __name__ == "__main__":
    train()
    test(50)