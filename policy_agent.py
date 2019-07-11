import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import torchvision.transforms as T

import numpy as np 
import collections
from itertools import count
import os, sys
import time
from sklearn.metrics.pairwise import cosine_similarity

from networks import PolicyNet, ValueNet
from utils import *
from env import Env

relation = sys.argv[1]
task = sys.argv[2]
graph_path = os.path.join(dataPath, 'tasks', relation, 'graph.txt')
relation_path = os.path.join(dataPath, 'tasks', relation, 'train_pos')

def retrain():
    print("Start retraining")
    

def test():
    pass

if __name__ == "__main__":
	if task == 'test':
		test()
	elif task == 'retrain':
		retrain()
	else:
		retrain()
		test()

