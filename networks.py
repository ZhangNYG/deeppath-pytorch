import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import torchvision.transforms as T 

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

def value_nn(state, state_dim):
    value_nn = nn.Sequential(
        nn.Linear(state_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    )
    return value_nn
