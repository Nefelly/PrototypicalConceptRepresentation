import torch
import torch.nn.functional as F
from tqdm import tqdm
import pdb
import random
import time
import math
import os
import pickle
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc 
from sampler import Sampler
from models import subclass_loss

save_folder = './params/'
max_length = 256
trainable_models = ['prot_prot', 'bert_cls']
add_concept_hint = False #True
gradual_epochs = 0.001

class Trainer:
	def __init__(self, data_bundle, model, tokenizer, optimizer, device, hyperparams):

		self.data_bundle = data_bundle
		self.model = model
		self.model_name = hyperparams['model_name']
		if self.model_name in trainable_models:
			model.to(device)
		self.tokenizer = tokenizer
		self.optimizer = optimizer
		self.device = device
		
		self.hyperparams = hyperparams
		self.save_folder = save_folder
		self.best_f1 = 0
		self.best_epoch = -1

		load_path = hyperparams['load_path']
 
		if ((load_path != None) and (not load_path.startswith(save_folder))):
			load_path = save_folder + load_path
		if (self.model_name in trainable_models) and (load_path!=None and os.path.exists(load_path)):
			model.load_state_dict(torch.load(load_path))
			print('Parameters loaded from {0}.'.format(load_path))

	def run(self):
		if self.model_name in trainable_models:
			self.train()
		else:
			pass
			#self.use_statistic()

	def train(self):
		pos_triples = self.data_bundle['train_triples']#[:5]
		concept_entity_info = self.data_bundle['concept_entity_info']

		model = self.model
		tokenizer = self.tokenizer
		optimizer = self.optimizer
		device = self.device
		hyperparams = self.hyperparams
		   
		batch_size = hyperparams['batch_size'] 
		epoch = hyperparams['epoch'] 
		model_name = hyperparams['model_name']
		ent_per_con = hyperparams['ent_per_con']
		concepts = self.data_bundle['concept_entity_info'].keys()

		evaluate_threshold = hyperparams['evaluate_threshold']

		criterion = torch.nn.CrossEntropyLoss()

		model.train()

		sampler = Sampler(concept_entity_info, hyperparams['typicalness'], ent_per_con, 'train')
		
		# 计算需要多少个step来渐变
		dataset = complete_train_dataset(pos_triples, concepts)
		dataset_size = len(dataset)
		gradual_steps = gradual_epochs * dataset_size
		count_step = 0

		for epc in range(epoch):
			dataset = complete_train_dataset(pos_triples, concepts)
			dataset_size = len(dataset)

			random_map = random.sample(range(dataset_size), dataset_size)
			batch_list = [ random_map[i:i+batch_size] for i in range(0, dataset_size ,batch_size)] 

			total_loss = 0
			total_accuracy = 0
			TP = {0: 0, 1: 0, 2: 0}
			FP = {0: 0, 1: 0, 2: 0}
			FN = {0: 0, 1: 0, 2: 0}
			TN = {0: 0, 1: 0, 2: 0}
			precision = {0: 0, 1: 0, 2: 0}
			recall = {0: 0, 1: 0, 2: 0}
			micro_f1 = {0: 0, 1: 0, 2: 0}

			for bt, batch in enumerate(batch_list):
				triple = dataset[batch[0]]
				hypo, hyper, label = triple
				hypo_ents = sampler.sample(hypo)
				hyper_ents = sampler.sample(hyper)

				if self.model_name in trainable_models:
					if hyperparams['use_probase_text']:
						text_key = 'text_from_dbpedia_probase'
					else:
						text_key = 'text_from_dbpedia'
					hypo_hint = '该实体属于概念"{0}"，判断"{1}"是否被"{2}"包含。'.format(hypo, hypo, hyper) if add_concept_hint else ''
					hypo_texts = [  hypo_hint + e[text_key] for e in hypo_ents]
					hyper_hint = '该实体属于概念"{0}"，判断"{1}"是否包含"{2}"。'.format(hyper, hyper, hypo) if add_concept_hint else ''
					hyper_texts = [  hyper_hint + e[text_key] for e in hyper_ents]
					#pdb.set_trace()
					hypo_inputs = tokenizer(hypo_texts, truncation = True, max_length = max_length, return_tensors='pt', padding=True)
					hyper_inputs = tokenizer(hyper_texts, truncation = True, max_length = max_length, return_tensors='pt', padding=True)

					hypo_inputs.to(device)
					hyper_inputs.to(device)

					#pdb.set_trace()

					hypo_embeddings = model.bert_embed(**hypo_inputs)
					hyper_embeddings = model.bert_embed(**hyper_inputs)

					pred, geoinfo = model(hypo_embeddings, hyper_embeddings, hypo, hyper)
					pred_label = pred.argmax().item()

					total_accuracy += int(pred_label == label)#(pred_label == label).float().mean().item()
					
					loss = criterion(pred.unsqueeze(0), torch.tensor([label]).to(device)) + subclass_loss(geoinfo, label)
					
					total_loss += loss.item()

					if pred_label == label:
						TP[label] += 1
						for i in range(3):
							if i != label:
								TN[i] += 1
					else:
						FP[pred_label] += 1
						FN[label] += 1
						TN[3-label-pred_label] += 1

					optimizer.zero_grad()
					loss.backward()
					#torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=20, norm_type=2)
					optimizer.step()
					#pdb.set_trace()

				count_step += 1
				model.set_alpha(min(0.8, count_step / gradual_steps))

			avg_loss = total_loss / dataset_size
			avg_accuracy = total_accuracy / dataset_size
			print("Train Epoch :{0} Avg Loss:{1:.5f} Accuracy: {2:.5f} ".format(epc, avg_loss, avg_accuracy))

			for label in range(3):
				if TP[label] + FP[label] > 0:
					precision[label] = TP[label] / ( TP[label] + FP[label])
				else:
					precision[label] = float('nan')
				if TP[label] + FN[label] > 0:
					recall[label] = TP[label] / ( TP[label] + FN[label])
				else:
					recall[label] = float('nan')
				if precision[label] + recall[label] > 0:
					micro_f1[label] = (2*precision[label]*recall[label]) / (precision[label] + recall[label])
				else:
					micro_f1[label] = float('nan')
				print("Label {0} Precision:{1:.5f} Recall:{2:.5f} Micro_F1 {3:.5f} TP: {4} FP: {5} FN: {6} TN: {7}".
				format(label, precision[label], recall[label], micro_f1[label], TP[label], FP[label], FN[label], TN[label]))

			if epc % hyperparams['evaluate_every'] == 0:
				self.test(epc)

	def test(self, epc):
		pos_triples = self.data_bundle['test_triples']#[:5]
		concept_entity_info = self.data_bundle['concept_entity_info']
		negative_taxonomy = self.data_bundle['negative_concept_taxonomy']

		model = self.model
		tokenizer = self.tokenizer
		#optimizer = self.optimizer
		device = self.device
		hyperparams = self.hyperparams
		   
		batch_size = hyperparams['batch_size'] 
		epoch = hyperparams['epoch'] 
		model_name = hyperparams['model_name']
		ent_per_con = hyperparams['ent_per_con']
		concepts = self.data_bundle['concept_entity_info'].keys()

		evaluate_threshold = hyperparams['evaluate_threshold']

		criterion = torch.nn.CrossEntropyLoss()

		model.eval()

		sampler = Sampler(concept_entity_info, hyperparams['typicalness'], ent_per_con, 'test')
		
		with torch.no_grad():
			dataset = complete_test_dataset(pos_triples, concepts, negative_taxonomy)
			dataset_size = len(dataset)

			random_map = random.sample(range(dataset_size), dataset_size)
			batch_list = [ random_map[i:i+batch_size] for i in range(0, dataset_size ,batch_size)] 

			total_loss = 0
			total_accuracy = 0
			TP = {0: 0, 1: 0, 2: 0}
			FP = {0: 0, 1: 0, 2: 0}
			FN = {0: 0, 1: 0, 2: 0}
			TN = {0: 0, 1: 0, 2: 0}
			precision = {0: 0, 1: 0, 2: 0}
			recall = {0: 0, 1: 0, 2: 0}
			micro_f1 = {0: 0, 1: 0, 2: 0}

			for bt, batch in enumerate(batch_list):
				triple = dataset[batch[0]]
				hypo, hyper, label = triple
				hypo_ents = sampler.sample(hypo)
				hyper_ents = sampler.sample(hyper)

				if self.model_name in trainable_models:
					if hyperparams['use_probase_text']:
						text_key = 'text_from_dbpedia_probase'
					else:
						text_key = 'text_from_dbpedia'
	
					hypo_texts = [ e[text_key] for e in hypo_ents]
					hyper_texts = [ e[text_key] for e in hyper_ents]

					hypo_inputs = tokenizer(hypo_texts, truncation = True, max_length = max_length, return_tensors='pt', padding=True)
					hyper_inputs = tokenizer(hyper_texts, truncation = True, max_length = max_length, return_tensors='pt', padding=True)

					hypo_inputs.to(device)
					hyper_inputs.to(device)

					hypo_embeddings = model.bert_embed(**hypo_inputs)
					hyper_embeddings = model.bert_embed(**hyper_inputs)

					pred, geoinfo = model(hypo_embeddings, hyper_embeddings, hypo, hyper)
					pred_label = pred.argmax().item()

					total_accuracy += int(pred_label == label)#(pred_label == label).float().mean().item()
					
					loss = criterion(pred.unsqueeze(0), torch.tensor([label]).to(device)) + subclass_loss(geoinfo, label)
					total_loss += loss.item()

					if pred_label == label:
						TP[label] += 1
						for i in range(3):
							if i != label:
								TN[i] += 1
					else:
						FP[pred_label] += 1
						FN[label] += 1
						TN[3-label-pred_label] += 1


			avg_loss = total_loss / dataset_size
			avg_accuracy = total_accuracy / dataset_size
			print("Test Epoch :{0} Avg Loss:{1:.5f} Accuracy: {2:.5f} ".format(epc, avg_loss, avg_accuracy))

			for label in range(3):
				if TP[label] + FP[label] > 0:
					precision[label] = TP[label] / ( TP[label] + FP[label])
				else:
					precision[label] = float('nan')
				if TP[label] + FN[label] > 0:
					recall[label] = TP[label] / ( TP[label] + FN[label])
				else:
					recall[label] = float('nan')
				if precision[label] + recall[label] > 0:
					micro_f1[label] = (2*precision[label]*recall[label]) / (precision[label] + recall[label])
				else:
					micro_f1[label] = float('nan')

				print("Label {0} Precision:{1:.5f} Recall:{2:.5f} Micro_F1 {3:.5f} TP: {4} FP: {5} FN: {6} TN: {7}".
				format(label, precision[label], recall[label], micro_f1[label], TP[label], FP[label], FN[label], TN[label]))

		model.train()

def complete_train_dataset(dataset, concepts):
	negative_dataset = []
	num_triples = len(dataset)
	num_triples_nt = num_triples
	count_nt = 0
	while(count_nt < num_triples_nt):
		hypo, hyper = random.sample(concepts, 2)
		if not (hypo, hyper, 1) in dataset and not (hypo, hyper, 0) in negative_dataset:
			negative_dataset.append((hypo, hyper, 0))
			count_nt += 1

	reverse_dataset = [ (hyper, hypo, 2) for hypo, hyper, _ in dataset]
	dataset_ = dataset + negative_dataset + reverse_dataset
	random.shuffle(dataset_)
	return dataset_

def complete_test_dataset(dataset, concepts, negative_taxonomy):
	
	num_triples = len(dataset)
	num_triples_nt = num_triples
	negative_dataset = negative_taxonomy[:num_triples_nt]

	reverse_dataset = [ (hyper, hypo, 2) for hypo, hyper, _ in dataset]
	dataset_ = dataset + negative_dataset + reverse_dataset
	random.shuffle(dataset_)
	#pdb.set_trace()
	return dataset_