# coding=UTF-8
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
# 111
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--seed', type=int, default=42)
	parser.add_argument('--bert_lr', type=float, default=1e-5)
	parser.add_argument('--model_lr', type=float, default=1e-5)
	parser.add_argument('--batch_size', type=int, default=1)
	parser.add_argument('--epoch', type=int, default=1000000)
	parser.add_argument('--no_use_gpu', action='store_true', default=False)
	parser.add_argument('--weight_decay', type=float, default=1e-7)
	parser.add_argument('--load_path', type=str, default=None)
	parser.add_argument('--load_data', action='store_true', default=False)
	parser.add_argument('--model', type=str, default='prot_prot', choices=
		['supt_conf', 'bert_cls', 'prot_prot', 'kgbert'])
	parser.add_argument('--ent_per_con', type=int, default=4)
	parser.add_argument('--typicalness', type=str, default='none', choices = ['none', 'w_ent', 'max_label', 'sum_label'])
	#parser.add_argument('--cvsample', default=False, action = 'store_true') # cv = cross validation
	parser.add_argument('--language', type=str, default='zh', choices = ['zh', 'en'])
	parser.add_argument('--evaluate_threshold', type=float, default=0.5)
	parser.add_argument('--con_desc', default=False, action = 'store_true') # use description of concept
	parser.add_argument('--use_probase_text', default=False, action = 'store_true')

	arg = parser.parse_args()

	random.seed(arg.seed)
	np.random.seed(arg.seed)
	torch.manual_seed(arg.seed)

	if arg.no_use_gpu == False:
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')

	if arg.language == 'zh':
		with open('10000000_dataset_v2.pkl', 'rb') as fil:
			data_bundle = pickle.load(fil)
		bert_pretrained = "bert-base-chinese"
	elif arg.language == 'en':
		data_bundle = 1
		bert_pretrained = 1

		
	tokenizer = BertTokenizer.from_pretrained(bert_pretrained)
	bertmodel = BertModel.from_pretrained(bert_pretrained)

	trainable_models = ['prot_prot', 'bert_cls']
	if arg.model in trainable_models:
		if arg.model == 'prot_prot':
			model = ProtProtModel(bertmodel, arg.ent_per_con, data_bundle['concept_entity_info'].keys())
		elif arg.model == 'bert_cls':
			model = Bert_Classifier(bertmodel, arg.ent_per_con)
		#pdb.set_trace()
		no_decay = ["bias", "LayerNorm.weight"]

		param_group = [
			{'lr': arg.bert_lr, 'params': [p for n, p in model.named_parameters()
										   if ('bert' in n) and
										   (not any(nd in n for nd in no_decay)) ], # name中不包含bias和LayerNorm.weight
			 'weight_decay': arg.weight_decay},
			{'lr': arg.bert_lr, 'params': [p for n, p in model.named_parameters()
										   if ('bert' in n) and
										   (any(nd in n for nd in no_decay))],
			 'weight_decay': 0.0},
			{'lr': arg.model_lr, 'params': [p for n, p in model.named_parameters()
										   if ('bert' not in n) and
										   (not any(nd in n for nd in no_decay))],
			 'weight_decay': arg.weight_decay},
			{'lr': arg.model_lr, 'params': [p for n, p in model.named_parameters()
										   if ('bert' not in n) and
										   (any(nd in n for nd in no_decay))],
			 'weight_decay': 0.0},
		]	

		optimizer = torch.optim.AdamW(param_group)

		#data_bundle['train_triples'], data_bundle['test_triples'] = cover_split(data_bundle['concept_taxonomy'], data_bundle['concept_entity_info'].keys())#ratio_split(data_bundle['concept_taxonomy'])
		data_bundle['train_triples'], data_bundle['test_triples'] = ratio_split(data_bundle['concept_taxonomy'])
		#pdb.set_trace()

	hyperparams = {
		'batch_size': arg.batch_size,
		'epoch': arg.epoch,
		'model_name': arg.model,
		'ent_per_con': arg.ent_per_con,
		'typicalness': arg.typicalness,
		'cvsample': arg.cvsample,
		'con_desc': arg.con_desc,
		'evaluate_threshold': arg.evaluate_threshold,
		'load_path': arg.load_path,
		'use_probase_text': arg.use_probase_text,
		'evaluate_every': 1, 
		'update_every': 1
	}

	trainer = Trainer(data_bundle, model, tokenizer, optimizer, device, hyperparams)
	trainer.run()

	pdb.set_trace()


