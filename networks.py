import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import torchvision.transforms as T 

class Policy_NN(nn.Module):
    def __init__(self, state_dim, action_dim):
        self.policy_nn = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, action_dim),
            nn.Softmax()
        )
    def forward(self, state):
        return self.policy_nn(state)

'''
def policy_nn(state, state_dim, action_dim):
    policy_nn = nn.Sequential(
        nn.Linear(state_dim, 512),
        nn.ReLU(),
        nn.Linear(512, 1024),
        nn.ReLU(),
        nn.Linear(1024, action_dim),
        nn.Softmax()
    )
    return policy_nn(state)
'''

class Value_NN(nn.Module):
    def __init__(self, state_dim):
        self.value_nn = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, state):
        return self.value_nn(state)

'''
def value_nn(state, state_dim):
    value_nn = nn.Sequential(
        nn.Linear(state_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )
    return value_nn
'''