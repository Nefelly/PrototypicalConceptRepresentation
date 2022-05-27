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
	def __init__(self, concept_entity_info, instance_info, typicalness, ent_per_con, mode='train', fixed_num_insts = False):
		self.concept_entity_info = concept_entity_info
		self.instance_info = instance_info
		self.typicalness = typicalness
		self.ent_per_con = ent_per_con
		self.mode = mode
		self.fixed_num_insts = fixed_num_insts

	def sample_single(self, con, exclude_ins=None, num=-1):
		ent_per_con = self.ent_per_con

		if num == -1:
			num = ent_per_con
		concept_entity_info = self.concept_entity_info
		instance_info = self.instance_info
		
		ents = [instance_info[ins] for ins in concept_entity_info[con] if ins != exclude_ins]
		ents = random.sample(ents, min(num, len(ents)))
		return ents

	def sample(self, con1, con2):
		ent_per_con = self.ent_per_con

		concept_entity_info = self.concept_entity_info
		instance_info = self.instance_info
		try:
			ents1 = [ instance_info[ins] for ins in concept_entity_info[con1] ]
			ents2 = [ instance_info[ins] for ins in concept_entity_info[con2] ]
		except:
			pdb.set_trace()

		if len(ents1) < ent_per_con:
			num1 = len(ents1)
			num2 = min(len(ents2), 2*ent_per_con - num1)
		else:
			num2 = min(len(ents2), ent_per_con)
			num1 = min(len(ents1), 2*ent_per_con - num2)

		ents1 = random.sample(ents1, num1)
		ents2 = random.sample(ents2, num2)
		return ents1, ents2
