import os
import argparse
from tqdm import tqdm
import torch
import pdb
import random
import pickle
import numpy as np

from transformers import BertTokenizer, BertModel
from models import ProtProtModel, Bert_Classifier
from trainer import Trainer
from utils import *

def ratio_split(concept_taxonomy):
	dataset_size = len(concept_taxonomy)
	split_idx = int(0.7 * dataset_size)

	train_triples = concept_taxonomy[:split_idx]
	test_triples = concept_taxonomy[split_idx:]

	return train_triples, test_triples

def cover_split(concept_taxonomy, concepts):
	dataset_size = len(concept_taxonomy)
	split_idx = int(0.7 * dataset_size)

	train_triples = []
	test_triples = []
	remaining_concepts = set(concepts)

	for triple in concept_taxonomy:
		con1 = triple[0]
		con2 = triple[1]
		if con1 in remaining_concepts or con2 in remaining_concepts:
			train_triples.append(triple)
			remaining_concepts.discard(con1)
			remaining_concepts.discard(con2)
		else:
			test_triples.append(triple)

	num_diff = split_idx - len(train_triples)
	random.shuffle(test_triples)
	train_triples = train_triples + test_triples[:num_diff]
	test_triples = test_triples[num_diff:]
	
	return train_triples, test_triples