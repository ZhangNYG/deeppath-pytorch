import numpy as np 
import random 
import os
from utils import * 

class Env(object):
    '''
    knowledge graph evironment definition
    '''
    def __init__(self, dataPath, task=None):

        self.entity2id = {}
        self.relation2id= {}
        self.relations = []

        with open(os.path.join(dataPath, 'entity2id.txt')) as f:
            lines = f.readlines()
            for line in lines:
                self.entity2id[line.split()[0]] = int(line.split()[1])

        with open(os.path.join(dataPath, 'relation2id.txt')) as f:
            lines = f.readlines()
            for line in lines:
                self.relation2id[line.split()[0]] = int(line.split()[1])
                self.relations.append(line.split()[0])

        self.entity2vec = np.loadtxt(os.path.join(dataPath, 'entity2vec.bern'))
        self.relation2vec = np.loadtxt(os.path.join(dataPath, 'relation2vec.bern'))

        self.path = []
        self.path_relations = []

        # knowledge graph for path finding
        with open(os.path.join(dataPath, 'kb_env_rl.txt')) as  f:
            kb_all = f.readlines()
        self.kb = []
        if task != None:
            relation = task.split()[2]
            for line in kb_all:
                rel = line.split()[2]
                if rel != relation and rel != relation+'_inv':
                    self.kb.append(line)

        self.die = 0 # record how many times does the agent choose an invalid path

    def interact(self, state, action):
        '''
        This function process the interact from the agent
        state: is [current_position, target_position]
        action: an integer
        return: (reward, [new_position, target_position], done)

        '''
        done = 0 # whether the episode has finished
        curr_pos = state[0]
        target_pos = state[1]
        chosed_relation = self.relations[action]
        choices = []
        for line in self.kb:
            triple = line.split()
            e1_idx = self.entity2id[triple[0]]

            if curr_pos = e1_idx and triple[2] == chosed_relation and triple[1] in self.entity2id:
                choices.append(triple)
        
        if len(choices) == 0:
            reward = -1
            self.die += 1
            next_state = state # state in the initial state
            next_state[-1] = self.die
            return (reward, next_state, done)
        else: # find a valid path
            path = random.choice(choices)
            self.path.append(path[2]+'->'+path[1])
            self.path_relations.append(path[2])
            # print('Find a valid path', path)
            # print('Action index', action)
            self.die = 0
            new_pos = self.entity2id[path[1]]
            reward = 0
            new_state = [new_pos, target_pos, self.die]

            if new_pos == target_pos:
                print('Find a path:', self.path)
                done = 1
                reward = 0
                new_state = None
            return (reward, new_state, done)

    def idx_state(self, idx_list):
        if idx_list != None:
            curr = self.entity2vec[idx_list[0], :]
            targ = self.entity2vec[idx_list[1], :]
            return np.expand_dims(np.concatenate([curr, targ-curr]), axis=0)
        else:
            return None

    
    def get_valid_actions(self, entityID):
        actions = set()
        for line in self.kb:
            triple = line.split()
            e1_idx = self.entity2id[triple[0]]
            if e1_idx == entityID:
                actions.add(self.relation2id[triple[2]])

        return np.array(list(actions))

    def path_embedding(self, path):
        embeddings = [self, relation2vec[self.relation2id[relation]. :] for relation in path]
        embeddings = np.reshape(embeddings, [-1, embedding_dim])
        path_encoding = np.sum(embeddings, axis=0)
        return np.reshape(path_encoding, [-1, embedding_dim])