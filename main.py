import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import torchvision.transforms as T

import numpy as np 
from itertools import count
import os, sys
import time

from networks import policy_nn
from utils import *
from env import Env 
from BFS.KB import KB 
from BFS.BFS import BFS 

relation = sys.argv[1]
graph_path = os.path.join(dataPath, 'tasks', relation, 'graph.txt')
relation_path = os.path.join(dataPath, 'tasks', relation, 'train_pos')

class SupervisePolicy(object):
    '''
    docstring for SupervisePolicy
    '''
    def __init__(self, learning_rate = 0.001):
        pass

    def 