import os
import argparse
from tqdm import tqdm
import torch
import pdb
import random
import pickle
import numpy as np
import math

def detect_nan_params(model):
		if math.isnan(sum([ i.sum() for i in model.parameters()])):
			return True
		else:
			return False

def clip_grad_norm_(parameters, max_norm, norm_type=2):
	r"""Clips gradient norm of an iterable of parameters.

	The norm is computed over all gradients together, as if they were
	concatenated into a single vector. Gradients are modified in-place.

	Arguments:
		parameters (Iterable[Tensor]): an iterable of Tensors that will have
			gradients normalized
		max_norm (float or int): max norm of the gradients
		norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
			infinity norm.

	Returns:
		Total norm of the parameters (viewed as a single vector).
	"""
	parameters = list(filter(lambda p: p.grad is not None, parameters))
	max_norm = float(max_norm)
	norm_type = float(norm_type)
	if norm_type == float('inf'):
		total_norm = max(p.grad.data.abs().max() for p in parameters)
	else:
		total_norm = 0
		for p in parameters:
			param_norm = p.grad.data.norm(norm_type)
			total_norm += param_norm ** norm_type
		total_norm = total_norm ** (1. / norm_type)

	clip_coef = max_norm / (total_norm + 1e-6)

	if clip_coef < 1:
		for p in parameters:
			p.grad.data.mul_(clip_coef)

		total_norm = 0
		for p in parameters:
			param_norm = p.grad.data.norm(norm_type)
			total_norm += param_norm ** norm_type
		total_norm = total_norm ** (1. / norm_type)
		print('After ',sum([ p.grad.norm() for p in parameters]), total_norm)
	else:
		print('No clip ', total_norm)
	return total_norm


def ratio_split(isSubclassOf_triples):
	dataset_size = len(isSubclassOf_triples)
	split_idx = int(0.7 * dataset_size)

	train_triples = isSubclassOf_triples[:split_idx]
	test_triples = isSubclassOf_triples[split_idx:]

	return train_triples, test_triples

def cover_split(isSubclassOf_triples, concepts, num_train_triples=-1):
	dataset_size = len(isSubclassOf_triples)
	
	if num_train_triples == -1:
		num_train_triples = int(0.7 * dataset_size)

	train_triples = []
	test_triples = []
	remaining_concepts = set(concepts)
	ent_ratio = 0.4

	total_ents_of_concepts = { con: 0 for con in concepts }

	for triple in isSubclassOf_triples:
		con1 = triple[0]
		con2 = triple[1]
		total_ents_of_concepts[con1] += 1
		total_ents_of_concepts[con2] += 1

	count_ents_of_concepts = { con: 0 for con in concepts }

	
	for triple in isSubclassOf_triples:
		con1 = triple[0]
		con2 = triple[1]
		if con1 in remaining_concepts or con2 in remaining_concepts:
			train_triples.append(triple)
			count_ents_of_concepts[con1] += 1
			count_ents_of_concepts[con2] += 1

			if count_ents_of_concepts[con1] >= math.ceil(ent_ratio * total_ents_of_concepts[con1]):
				remaining_concepts.discard(con1)
			if count_ents_of_concepts[con2] >= math.ceil(ent_ratio * total_ents_of_concepts[con2]):
				remaining_concepts.discard(con2)
		else:
			test_triples.append(triple)

	num_diff = max(num_train_triples - len(train_triples), 0)

	random.shuffle(test_triples)
	train_triples = train_triples + test_triples[:num_diff]
	test_triples = test_triples[num_diff:]
	
	num_test_triples = int(len(test_triples)/2)
	valid_triples = test_triples[:num_test_triples]
	test_triples = test_triples[num_test_triples:]

	print('Num Train Triples {0} Valid Triples {1} Test Triples {2}'.format(len(train_triples), len(valid_triples), len(test_triples)))
	return train_triples, valid_triples, test_triples


def load_data(folder_path):
	isSubclassOf_triples = []
	isInstanceOf_triples = []
	concepts = set()
	instances = set()
	subclass_train = []
	subclass_valid = []
	subclass_test = []
	concept_info = {}
	concept_instance_info = {}
	instance_info = {}
	instance_train = []
	instance_valid = []
	instance_test = []
	negative_isSubclassOf_triples = []
	negative_isInstanceOf_triples = []

	with open(folder_path + '/concepts.txt', 'r') as f:
		for line in f.readlines()[1:]:
			line = line.strip('\n')
			concepts.add(line)

	with open(folder_path + '/instances.txt', 'r') as f:
		for line in f.readlines()[1:]:
			line = line.strip('\n')
			instances.add(line)

	if os.path.exists(folder_path + '/description_concepts.txt'):
		with open(folder_path + '/description_concepts.txt', 'r') as f:
			for line in f.readlines():
				item, desc = line.split('\t', 1)
				concept_info[item] = desc
	else:
		concept_info = None

	with open(folder_path + '/description_instances.txt', 'r') as f:
		for line in f.readlines():
			item, desc = line.split('\t', 1)
			instance_info[item] = { 'ins_name': item, 'text': desc}

	with open(folder_path + '/instances_of_concepts.txt', 'r') as f:
		for line in f.readlines():
			con, insts = line.split('\t', 1)
			if 'CN-PB' in folder_path:
				insts = insts.split('\t')[:-1]
			else:
				insts = insts.split(',')[:-1]
			concept_instance_info[con] = insts 

	with open(folder_path + '/subclass_all.txt', 'r') as f:
		for line in f.readlines()[1:]:
			line = line.strip('\n')
			hypo, rel, hyper = line.split('\t')
			isSubclassOf_triples.append((hypo, hyper, 1))

	with open(folder_path + '/subclass_train.txt', 'r') as f:
		for line in f.readlines()[1:]:
			line = line.strip('\n')
			hypo, rel, hyper = line.split('\t')
			subclass_train.append((hypo, hyper, 1))

	with open(folder_path + '/subclass_valid.txt', 'r') as f:
		for line in f.readlines()[1:]:
			line = line.strip('\n')
			hypo, rel, hyper = line.split('\t')
			subclass_valid.append((hypo, hyper, 1))

	with open(folder_path + '/subclass_test.txt', 'r') as f:
		for line in f.readlines()[1:]:
			line = line.strip('\n')
			hypo, rel, hyper = line.split('\t')
			subclass_test.append((hypo, hyper, 1))

	with open(folder_path + '/subclass_neg.txt', 'r') as f:
		for line in f.readlines()[1:]:
			line = line.strip('\n')
			hypo, rel, hyper = line.split('\t')
			negative_isSubclassOf_triples.append((hypo, hyper, 0))
	
	with open(folder_path + '/instance_all.txt', 'r') as f:
		for line in f.readlines()[1:]:
			line = line.strip('\n')
			hypo, rel, hyper = line.split('\t')
			isInstanceOf_triples.append((hypo, hyper, 1))

	with open(folder_path + '/instance_train.txt', 'r') as f:
		for line in f.readlines()[1:]:
			line = line.strip('\n')
			hypo, rel, hyper = line.split('\t')
			instance_train.append((hypo, hyper, 1))

	with open(folder_path + '/instance_valid.txt', 'r') as f:
		for line in f.readlines()[1:]:
			line = line.strip('\n')
			hypo, rel, hyper = line.split('\t')
			instance_valid.append((hypo, hyper, 1))

	with open(folder_path + '/instance_test.txt', 'r') as f:
		for line in f.readlines()[1:]:
			line = line.strip('\n')
			hypo, rel, hyper = line.split('\t')
			instance_test.append((hypo, hyper, 1))

	with open(folder_path + '/instance_neg.txt', 'r') as f:
		for line in f.readlines()[1:]:
			line = line.strip('\n')
			hypo, rel, hyper = line.split('\t')
			negative_isInstanceOf_triples.append((hypo, hyper, 0))


	data_bundle = {
        'isSubclassOf_triples': isSubclassOf_triples,  # all positive samples of isSubclassOf
        'isInstanceOf_triples': isInstanceOf_triples, # all positive samples of isInstanceOf
        'concepts': concepts, #all concepts
        'instances': instances, # all instances
        'subclass_train': subclass_train, # isSubclassOf samples for train
        'subclass_valid': subclass_valid,
        'subclass_test': subclass_test,
        'concept_info': concept_info, # description of concepts 
        'concept_instance_info': concept_instance_info, # key: concept，value: a list of its instances
        'instance_info': instance_info, # description of instances
        'instance_train': instance_train, # isInstanceOf samples for train
        'instance_valid': instance_valid,
        'instance_test':instance_test,
        'negative_isSubclassOf_triples': negative_isSubclassOf_triples, # randomly constructed negative samples of isSubclassOf
        'negative_isInstanceOf_triples':negative_isInstanceOf_triples,
    }

	return data_bundle
