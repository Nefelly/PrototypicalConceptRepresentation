import torch
import torch.nn.functional as F
from tqdm import tqdm
import pdb
import random
import math
import time
import os
import pickle
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc 


class Sampler:
    def __init__(self, concept_entity_info, typicalness, ent_per_con, mode='train'):
    	#self.dataset = dataset
    	self.concept_entity_info = concept_entity_info
    	self.typicalness = typicalness
    	self.ent_per_con = ent_per_con
    	self.mode = mode

    def sample(self, con):
    	#dataset = self.dataset
    	concept_entity_info = self.concept_entity_info
    	ent_per_con = self.ent_per_con

    	ents = concept_entity_info[con]
    	#if self.mode == 'train':
    	#	ents = ents[:8]
    	#else:
    	#	ents = ents[8:]
    	ents = random.sample(ents, ent_per_con)
    	return ents