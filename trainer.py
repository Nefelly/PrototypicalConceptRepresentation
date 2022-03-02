import torch
import torch.nn.functional as F
from tqdm import tqdm
import pdb
import random
import time
import math
import os
import pickle
#from sklearn.metrics import roc_auc_score, precision_recall_curve, auc 
from sampler import Sampler
from models import subclass_loss
from utils import clip_grad_norm_,  detect_nan_params
import numpy as np

save_folder = './params/'

add_concept_hint = False#True#False#True# True#False#
trainable_models = ['psn', 'prot_vanilla']

gradual_epochs = 0.001
on_51 = True
if on_51:
	max_length = 512
	sample_limit = 1000000000
else:
	max_length = 400
	sample_limit = 1000000000

distance_margin = 9#0#Prototypical Network 9 # Following KEPLER‘’
distance_eps = 1e-10

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
		self.identifier = hyperparams['identifier']
		self.hyperparams = hyperparams
		self.save_folder = save_folder
		self.load_epoch = hyperparams['load_epoch']


		self.concepts = concepts = data_bundle['concepts']
		

		concept_instance_info = data_bundle['concept_instance_info']

		self.con2id = { con: ic for ic, con in enumerate(sorted(concepts))}
		self.id2con = { ic: con for ic, con in enumerate(sorted(concepts))}
		

		self.instance_info = data_bundle['instance_info']
		self.concept_info = data_bundle.get('concept_info', None)
		self.instances = instances = self.instance_info.keys()#data_bundle['instances']
		self.ins2id = { ins: ic for ic, ins in enumerate(sorted(instances))}
		self.id2ins = { ic: ins for ic, ins in enumerate(sorted(instances))}

		additional_info = ''
		if hyperparams['freeze_plm']:
			additional_info = additional_info + 'freeze_'
		if hyperparams['train_MLM']:
			additional_info = additional_info + 'mlm_'
		if hyperparams['distance_metric']:
			additional_info = additional_info + 'dist_'
		if hyperparams['separate_classifier']:
			additional_info = additional_info + 'sep_'
		if hyperparams['con_desc']:
			additional_info = additional_info + 'condesc_'
		global add_concept_hint 
		if add_concept_hint == False:
			additional_info = additional_info + 'no_ment_'
		additional_info = additional_info + 'eta{0}_'.format(hyperparams['ent_per_con'])

		self.param_path_template = self.save_folder + additional_info + self.model_name + '_' + self.identifier + '_' + self.hyperparams['plm'] +'_'+self.hyperparams['variant']+  '_epc_{0}_metric_{1}'  + '.pt'
		self.history_path = self.save_folder + additional_info + self.model_name + '_' + self.identifier + '_' + self.hyperparams['plm'] +'_'+self.hyperparams['variant']+  '_history_{0}'  + '.pkl'
		
		#pdb.set_trace()
		if hyperparams['freeze_plm']:
			self.initialize_embeddings()

		load_path = hyperparams['load_path']
		if load_path == None and self.load_epoch >= 0:
			load_path = self.param_path_template.format(self.load_epoch, hyperparams['load_metric'])

			history_path = self.history_path.format(self.load_epoch)
			if os.path.exists(history_path):
				with open(history_path, 'rb') as fil:
					self.history_value = pickle.load(fil)


		#pdb.set_trace()
		if ((load_path != None) and (not load_path.startswith(save_folder))):
			load_path = save_folder + load_path
		
		if self.model_name != 'psn':
			add_concept_hint = False

		if (self.model_name in trainable_models) and load_path!=None:
			if os.path.exists(load_path):
				model.load_state_dict(torch.load(load_path), strict=False)
				print('Parameters loaded from {0}.'.format(load_path))
			else:
				print('Parameters {0} Not Found'.format(load_path))



		self.best_metric = {'subclass_acc': 0, 'subclass_f1': 0, 'instance_acc': 0, 'instance_f1': 0,
			'raw_instance_mrr': 0, 'raw_instance_hits1': 0, 'raw_instance_hits3': 0, 'raw_instance_hits10': 0,
			'fil_instance_mrr': 0, 'fil_instance_hits1': 0, 'fil_instance_hits3': 0, 'fil_instance_hits10': 0,
			'raw_subclass_mrr': 0, 'raw_subclass_hits1': 0, 'raw_subclass_hits3': 0, 'raw_subclass_hits10': 0,
			'fil_subclass_mrr': 0, 'fil_subclass_hits1': 0, 'fil_subclass_hits3': 0, 'fil_subclass_hits10': 0,}

		self.best_epoch = {'subclass_acc': -1, 'subclass_f1': -1, 'instance_acc': -1, 'instance_f1': -1,
			'raw_instance_mrr': -1, 'raw_instance_hits1': -1, 'raw_instance_hits3': -1, 'raw_instance_hits10': -1,
			'fil_instance_mrr': -1, 'fil_instance_hits1': -1, 'fil_instance_hits3': -1, 'fil_instance_hits10': -1,
			'raw_subclass_mrr': -1, 'raw_subclass_hits1': -1, 'raw_subclass_hits3': -1, 'raw_subclass_hits10': -1,
			'fil_subclass_mrr': -1, 'fil_subclass_hits1': -1, 'fil_subclass_hits3': -1, 'fil_subclass_hits10': -1}

		self.history_value = {'subclass_acc': [], 'subclass_f1': [], 'instance_acc': [], 'instance_f1': [],
			'raw_instance_mrr': [], 'raw_instance_hits1': [], 'raw_instance_hits3': [], 'raw_instance_hits10': [],
			'fil_instance_mrr': [], 'fil_instance_hits1': [], 'fil_instance_hits3': [], 'fil_instance_hits10': [],
			'raw_subclass_mrr': [], 'raw_subclass_hits1': [], 'raw_subclass_hits3': [], 'raw_subclass_hits10': [],
			'fil_subclass_mrr': [], 'fil_subclass_hits1': [], 'fil_subclass_hits3': [], 'fil_subclass_hits10': []}

		import signal
		signal.signal(signal.SIGINT, self.debug_signal_handler)


	def initialize_embeddings(self):
		device = self.device
		model = self.model

		instance_embeddings = self.get_initial_embeddings()
		#with open('save_embeddings.pkl', 'rb') as fil:
		#    save_embeddings = pickle.load(fil)
		#    instance_embeddings = save_embeddings['instance_embeddings']

		model.instance_embeddings.weight = torch.nn.Parameter(instance_embeddings.clone().detach()).to(device)

	def get_initial_embeddings(self):
		concepts = self.concepts
		instances = self.instances
		id2con = self.id2con
		id2ins = self.id2ins
		dimension = self.model.prototype_size
		device = self.device
		model = self.model
		tokenizer = self.tokenizer

		batch_size = 128

		instance_info = self.instance_info

		instance_embeddings = torch.zeros(len(instances), dimension).float().to(device)
		concept_hint_embeddings = torch.zeros(len(concepts), dimension).float().to(device)

		hyperparams = self.hyperparams
		if hyperparams['language'] == 'zh':
			text_key = 'text'
		else:
			text_key = 'text_from_wikipedia'

		model.eval()
		with torch.no_grad():
			# instance embeddings
			num_instances = len(instances)
			random_map = [i for i in range(num_instances)]
			batch_list = [ random_map[i:i+batch_size] for i in range(0, num_instances ,batch_size)] 
			
			for batch in batch_list:
				insts = [ id2ins[i] for i in batch]
				texts = [ instance_info[ins][text_key] for ins in insts]
				inputs = tokenizer(texts, truncation = True, max_length = max_length, return_tensors='pt', padding=True )
				inputs.to(device)
				embeddings = model.bert_embed(**inputs)
				instance_embeddings[batch] = embeddings

			

		save_embeddings = {
			'instance_embeddings': instance_embeddings
		}

		with open('save_embeddings.pkl', 'wb') as fil:
			pickle.dump(save_embeddings, fil)

		model.train()
		return instance_embeddings


	def run(self):
		if self.model_name in trainable_models:
			self.train()
			try:
				self.train()
			except:
				print('Best Epoch {0} micro_f1 {1}'.format(self.best_epoch, self.best_metric))
				pdb.set_trace()
		else:
			pass
			

	def train(self):
   
		pos_triples_sub = self.data_bundle['subclass_train'][:sample_limit]
		pos_triples_ins = self.data_bundle['instance_train'][:sample_limit]

		pos_triples = pos_triples_sub + pos_triples_ins

		print('Train triples: Subclass {0} Instance {1} '.format(len(pos_triples_sub), len(pos_triples_ins)))

		concept_instance_info = self.data_bundle['concept_instance_info']
		model = self.model
		tokenizer = self.tokenizer
		optimizer = self.optimizer
		device = self.device
		hyperparams = self.hyperparams
		   
		batch_size = hyperparams['batch_size'] 
		epoch = hyperparams['epoch'] 
		model_name = hyperparams['model_name']
		ent_per_con = hyperparams['ent_per_con']
		
		fixed_num_insts = False
		if hyperparams['language'] == 'zh':
			text_key = 'text'
			single_hint_template = '该物属于概念"{0}"。'

		else:
			text_key = 'text_from_wikipedia'
			single_hint_template = 'This item belongs to concept "{0}". '


		concepts = self.data_bundle['concept_instance_info'].keys()

		evaluate_threshold = hyperparams['evaluate_threshold']

		criterion = torch.nn.CrossEntropyLoss()

		model.train()

		instance_info = self.instance_info
		instances = instance_info.keys()
		concept_info = self.concept_info

		sampler = Sampler(concept_instance_info, self.instance_info, hyperparams['typicalness'], ent_per_con, 'train', fixed_num_insts)
		
		count_step = 0

		if self.load_epoch>=10:
			epc = self.load_epoch
			if hyperparams['variant'] in ['default', 'hybrid']:
				self.test_subclass(epc, valid = True)
				if hyperparams['train_instance_of']:
					self.test_instance(epc, valid = True)
			if hyperparams['variant'] in ['selfatt', 'hybrid']:
				self.test_subclass(epc, selfatt = True, valid = True)
				if hyperparams['train_instance_of']:
					self.test_instance(epc, selfatt = True, valid = True)
				self.link_prediction(epc)

			with open(self.history_path, 'wb') as fil:
				pickle.dump(self.history_value, fil)

		for epc in range(self.load_epoch +1, epoch):

			dataset_sub = complete_train_dataset(pos_triples_sub, concepts, hyperparams['add_reverse_label'], concept_instance_info, mode = 'subclass')
			
			if hyperparams['train_instance_of']:
				dataset_ins = complete_train_dataset(pos_triples_ins, concepts, hyperparams['add_reverse_label'], concept_instance_info, instances = instances, mode = 'instance', num_pos = len(pos_triples_sub) )
				dataset = dataset_sub + dataset_ins
			else:
				dataset = dataset_sub

			dataset_size = len(dataset)
			
			if hyperparams['distance_metric']:
				avg_positive_distance = 0 
				count_positive = 0

			random_map = random.sample(range(dataset_size), dataset_size)
			batch_list = [ random_map[i:i+batch_size] for i in range(0, dataset_size ,batch_size)] 

			total_loss = {'subclass': 0, 'instance': 0, 'subclass_selfatt': 0, 'instance_selfatt': 0}
			if hyperparams['train_MLM']:
				total_loss['MLM'] = 0

			total_accuracy = {'subclass': 0, 'instance': 0, 'subclass_selfatt': 0, 'instance_selfatt': 0}
			TP = {'subclass': {0: 0, 1: 0, 2: 0}, 'instance': {0: 0, 1: 0, 2: 0}, 'subclass_selfatt': {0: 0, 1: 0, 2: 0}, 'instance_selfatt': {0: 0, 1: 0, 2: 0}} 
			FP = {'subclass': {0: 0, 1: 0, 2: 0}, 'instance': {0: 0, 1: 0, 2: 0}, 'subclass_selfatt': {0: 0, 1: 0, 2: 0}, 'instance_selfatt': {0: 0, 1: 0, 2: 0}} 
			FN = {'subclass': {0: 0, 1: 0, 2: 0}, 'instance': {0: 0, 1: 0, 2: 0}, 'subclass_selfatt': {0: 0, 1: 0, 2: 0}, 'instance_selfatt': {0: 0, 1: 0, 2: 0}} 
			TN = {'subclass': {0: 0, 1: 0, 2: 0}, 'instance': {0: 0, 1: 0, 2: 0}, 'subclass_selfatt': {0: 0, 1: 0, 2: 0}, 'instance_selfatt': {0: 0, 1: 0, 2: 0}} 
			precision = {'subclass': {0: 0, 1: 0, 2: 0}, 'instance': {0: 0, 1: 0, 2: 0}, 'subclass_selfatt': {0: 0, 1: 0, 2: 0}, 'instance_selfatt': {0: 0, 1: 0, 2: 0}} 
			recall = {'subclass': {0: 0, 1: 0, 2: 0}, 'instance': {0: 0, 1: 0, 2: 0}, 'subclass_selfatt': {0: 0, 1: 0, 2: 0}, 'instance_selfatt': {0: 0, 1: 0, 2: 0}} 
			micro_f1 = {'subclass': {0: 0, 1: 0, 2: 0}, 'instance': {0: 0, 1: 0, 2: 0}, 'subclass_selfatt': {0: 0, 1: 0, 2: 0}, 'instance_selfatt': {0: 0, 1: 0, 2: 0}} 

			time_1 = time.time()
			count_subclass = 0
			count_subclass_selfatt = 0
			count_instance = 0
			count_instance_selfatt = 0

			for bt, batch in tqdm(enumerate(batch_list)):
				triple = dataset[batch[0]]
				hypo, hyper, label = triple

				if hyper not in concepts:
					pdb.set_trace()
				
				hyper_ents = sampler.sample_single(hyper, exclude_ins = hypo)
				if (len(hyper_ents) == 0):
					continue
				if hypo in concepts: # subclass
					rel_type = 'subclass_selfatt'
					count_subclass_selfatt += 1
				else: # 
					rel_type = 'instance_selfatt'
					count_instance_selfatt += 1
				#pdb.set_trace()
				'''
				if hypo in concepts and hyper in concepts:
					hypo_ents, hyper_ents = sampler.sample(hypo, hyper)

					if hyperparams['variant'] == 'selfatt' or (hyperparams['variant'] == 'hybrid' and random.sample(range(10), 1)[0] > 5):
						rel_type = 'subclass_selfatt'
						count_subclass_selfatt += 1
					else:
						rel_type = 'subclass'
						count_subclass += 1
				elif hypo in instances or hyper in instances:
					
					if hypo in instances:
						hyper_ents = sampler.sample_single(hyper, exclude_ins = hypo)
						if (len(hyper_ents) == 0):
							continue
					else:
						hypo_ents = sampler.sample_single(hypo, exclude_ins = hyper)
						if (len(hypo_ents) == 0):
							continue

					if hyperparams['variant'] == 'selfatt' or (hyperparams['variant'] == 'hybrid' and random.sample(range(10), 1)[0] > 5):
						rel_type = 'instance_selfatt'
						count_instance_selfatt += 1
					else:
						rel_type = 'instance'
						count_instance += 1
				else:
					print('Wrong')
				'''

				
				if self.model_name in trainable_models:
					# train subclass of
					'''
					if not hyperparams['con_desc']: # 不使用概念本身的描述，即使用实例的描述
						if not hyperparams['freeze_plm']:
							if rel_type not in  ['subclass_selfatt', 'instance_selfatt']:
								hypo_hint = hypo_hint_template.format(hypo, hypo, hyper) if add_concept_hint else ''
								hyper_hint = hyper_hint_template.format(hyper, hyper, hypo) if add_concept_hint else ''
							else:
								hypo_hint = single_hint_template.format(hypo) if add_concept_hint else ''
								hyper_hint = single_hint_template.format(hyper) if add_concept_hint else ''

							if rel_type in ['subclass', 'subclass_selfatt']:
								hypo_texts = [  hypo_hint + e[text_key] for e in hypo_ents]
								hyper_texts = [  hyper_hint + e[text_key] for e in hyper_ents]
							elif hypo in instances:
								hypo_texts = [  hypo_hint + instance_info[hypo][text_key] ]
								hyper_texts = [  hyper_hint + e[text_key] for e in hyper_ents]
							elif hyper in instances:
								hypo_texts = [  hypo_hint + e[text_key] for e in hypo_ents]
								hyper_texts = [  hyper_hint + instance_info[hyper][text_key]]
						   

							texts = hypo_texts + hyper_texts
							inputs = tokenizer(texts, truncation = True, max_length = max_length, return_tensors='pt', padding=True )
							inputs.to(device)
							embeddings = model.bert_embed(**inputs)
						else:
							pdb.set_trace()
							
							insts = []
							if hypo in instances:
								insts += [hypo]
							else:
								insts += [ e['ins_name'] for e in hypo_ents]
							if hyper in instances:
								insts += [hyper]
							else:
								insts += [ e['ins_name'] for e in hyper_ents]
							insts_idx = torch.tensor([ self.ins2id[ins]  for ins in insts]).to(device)
							embeddings = model.frozen_bert_embed(insts_idx)
							
						
						if rel_type in ['subclass', 'subclass_selfatt']:
							hypo_embeddings = embeddings[:len(hypo_ents)]
							hyper_embeddings = embeddings[len(hypo_ents):]
						else:
							if hypo in instances:
								hypo_embeddings = embeddings[:1]
								hyper_embeddings = embeddings[1:]
							elif hyper in instances:
								hypo_embeddings = embeddings[:-1]
								hyper_embeddings = embeddings[-1:]
					else:
						if not hyperparams['freeze_plm']:
							if hyper in instances:
								hyper_texts = [instance_info[hyper][text_key]]
							else:
								hyper_texts = [concept_info[hyper]]
	
							if hypo in instances:
								hypo_texts = [instance_info[hypo][text_key]]
							else:
								hypo_texts = [concept_info[hypo]]
							texts = hypo_texts + hyper_texts
							inputs = tokenizer(texts, truncation = True, max_length = max_length, return_tensors='pt', padding=True )
							inputs.to(device)
							embeddings = model.bert_embed(**inputs)
							hypo_embeddings = embeddings[:1]
							hyper_embeddings = embeddings[1:]
						else:
							if hyper in instances:
								hyper_idx = torch.tensor([self.ins2id[hyper]]).to(device)
								hyper_embeddings = model.frozen_bert_embed(hyper_idx)
							else:
								hyper_idx = torch.tensor([self.con2id[hyper]]).to(device)
								hyper_embeddings = model.frozen_bert_embed_concepts(hyper_idx)

							if hypo in instances:
								hypo_idx  = torch.tensor([self.ins2id[hypo ]]).to(device)
								hypo_embeddings = model.frozen_bert_embed(hypo_idx)
							else:
								hypo_idx  = torch.tensor([self.con2id[hypo ]]).to(device)
								hypo_embeddings = model.frozen_bert_embed_concepts(hypo_idx)
					'''
					hypo_hint = single_hint_template.format(hypo) if add_concept_hint else ''
					hyper_hint = single_hint_template.format(hyper) if add_concept_hint else ''

					hyper_texts = [  hyper_hint + e[text_key] for e in hyper_ents]

					if hypo in instances:
						hypo_texts = [instance_info[hypo][text_key]]
					else:
						hypo_texts = [concept_info[hypo]]
					texts = hypo_texts + hyper_texts
					inputs = tokenizer(texts, truncation = True, max_length = max_length, return_tensors='pt', padding=True )
					inputs.to(device)
					embeddings = model.bert_embed(**inputs)
					hypo_embeddings = embeddings[:1]
					hyper_embeddings = embeddings[1:]
						

					if rel_type not in  ['subclass_selfatt', 'instance_selfatt']:
						res = model(hypo_embeddings, hyper_embeddings, hypo, hyper, mode = rel_type)
						
					else:
						try:
							hypo_res = model(hypo_embeddings, hypo_embeddings, hypo, hypo, mode = rel_type)
							hypo_prototype = hypo_res['prototype']

							hyper_res = model(hyper_embeddings, hyper_embeddings, hyper, hyper, mode = rel_type)
							hyper_prototype = hyper_res['prototype']
						except:
							pdb.set_trace()

						res = model.judge(hypo_prototype, hyper_prototype, mode = rel_type)
					
					if not hyperparams['distance_metric']:
						subpred = res['sub_pred']

						subpred_label = subpred.argmax().item()

						total_accuracy[rel_type] += int(subpred_label == label)
						loss = criterion(subpred.unsqueeze(0), torch.tensor([label]).to(device)) 
					

						if subpred_label == label:
							TP[rel_type][label] += 1
							for i in range(3):
								if i != label:
									TN[rel_type][i] += 1
						else:
							FP[rel_type][subpred_label] += 1
							FN[rel_type][label] += 1
							TN[rel_type][3-label-subpred_label] += 1

					else:
						distance = res['distance']
						#print('Label {0} Dis {1:.5f}'.format(label, distance) )
						avg_positive_distance += distance
						count_positive += 1
						if label == 1:
							loss = -(torch.sigmoid((distance_margin - distance))).log()
						else:
							loss = -(torch.sigmoid((distance - distance_margin))).log()
						#print('Loss ', loss)
						

					total_loss[rel_type] += loss.item()


					if hyperparams['train_MLM']:

						mlm_loss = model.train_MLM(max_length, tokenizer, texts)
						loss += mlm_loss
						#print('MLM_loss',mlm_loss)
						total_loss['MLM'] +=  mlm_loss

					optimizer.zero_grad()
					loss.backward()
					#pdb.set_trace()
					optimizer.step()
				 
				count_step += 1


			time_2 = time.time()


			avg_loss = { 'subclass': total_loss['subclass'] / count_subclass if count_subclass > 0 else 0, 
				'instance': total_loss['instance'] / count_instance if count_instance > 0 else 0, 
				'subclass_selfatt': total_loss['subclass_selfatt'] / count_subclass_selfatt if count_subclass_selfatt > 0 else 0,
				'instance_selfatt': total_loss['instance_selfatt'] / count_instance_selfatt if count_instance_selfatt > 0 else 0}

			avg_accuracy = { 'subclass': total_accuracy['subclass'] / count_subclass if count_subclass > 0 else 0, 
				'instance': total_accuracy['instance'] / count_instance if count_instance > 0 else 0, 
				'subclass_selfatt': total_accuracy['subclass_selfatt'] / count_subclass_selfatt if count_subclass_selfatt > 0 else 0, 
				'instance_selfatt': total_accuracy['instance_selfatt'] / count_instance_selfatt if count_instance_selfatt > 0 else 0}

			print("Train Epoch :{0} Time: {1:.5f} Avg Time: {2:.5f} {3} {4}".format(epc, time_2 - time_1, (time_2 - time_1) / dataset_size, 
				('MLM loss ' + str(total_loss['MLM'])) if hyperparams['train_MLM'] else '',
				('Avg PosDist ' + str(avg_positive_distance.item() / count_positive)) if hyperparams['distance_metric'] else ''))

			rel_types = []
			if hyperparams['variant'] in ['default', 'hybrid']:
				rel_types.append('subclass')
				if hyperparams['train_instance_of']:
					rel_types.append('instance')
			if hyperparams['variant'] in ['selfatt', 'hybrid']:
				rel_types.append('subclass_selfatt')
				if hyperparams['train_instance_of']:
					rel_types.append('instance_selfatt')

			'''
			for rel_type in rel_types:
				print("{0} Avg Loss:{1:.5f} Accuracy: {2:.5f}".format(rel_type, avg_loss[rel_type], avg_accuracy[rel_type]))
				for label in range(hyperparams['num_labels']):
					if TP[rel_type][label] + FP[rel_type][label] > 0:
						precision[rel_type][label] = TP[rel_type][label] / ( TP[rel_type][label] + FP[rel_type][label])
					else:
						precision[rel_type][label] = float('nan')
					if TP[rel_type][label] + FN[rel_type][label] > 0:
						recall[rel_type][label] = TP[rel_type][label] / ( TP[rel_type][label] + FN[rel_type][label])
					else:
						recall[rel_type][label] = float('nan')

					if precision[rel_type][label] + recall[rel_type][label] > 0:
						micro_f1[rel_type][label] = (2*precision[rel_type][label]*recall[rel_type][label]) / (precision[rel_type][label] + recall[rel_type][label])
					else:
						micro_f1[rel_type][label] = float('nan')
					print("Label {0} Precision:{1:.5f} Recall:{2:.5f} Micro_F1 {3:.5f} TP: {4} FP: {5} FN: {6} TN: {7}".
					format(label, precision[rel_type][label], recall[rel_type][label], micro_f1[rel_type][label], TP[rel_type][label], FP[rel_type][label], FN[rel_type][label], TN[rel_type][label]))
			'''
			if epc % hyperparams['evaluate_every'] == 0:
			
				if hyperparams['variant'] in ['default', 'hybrid']:
					self.test_subclass(epc, valid = True)
					if hyperparams['train_instance_of']:
						self.test_instance(epc, valid = True)
				if hyperparams['variant'] in ['selfatt', 'hybrid']:
					self.test_subclass(epc, selfatt = True, valid = True)
					if hyperparams['train_instance_of']:
						self.test_instance(epc, selfatt = True, valid = True)
					self.link_prediction(epc)

				history_path = self.history_path.format(epc)
				with open(history_path, 'wb') as fil:
					pickle.dump(self.history_value, fil)

				last_history_path = self.history_path.format(epc-1)
				if os.path.exists(last_history_path):
					os.remove(last_history_path)
				




	def test_subclass(self, epc, selfatt = False, valid=True):
		concept_instance_info = self.data_bundle['concept_instance_info']
		negative_triples = self.data_bundle['negative_isSubclassOf_triples']

		model = self.model
		tokenizer = self.tokenizer

		device = self.device
		hyperparams = self.hyperparams
		   
		batch_size = hyperparams['batch_size'] 
		epoch = hyperparams['epoch'] 
		model_name = hyperparams['model_name']
		ent_per_con = hyperparams['ent_per_con']

		fixed_num_insts = False
		if hyperparams['language'] == 'zh':
			text_key = 'text'
			single_hint_template = '该物属于概念"{0}"。'

		else:
			text_key = 'text_from_wikipedia'
			single_hint_template = 'This item belongs to concept "{0}". '

		concepts = self.data_bundle['concept_instance_info'].keys()
		evaluate_threshold = hyperparams['evaluate_threshold']

		criterion = torch.nn.CrossEntropyLoss()

		model.eval()

		sampler = Sampler(concept_instance_info, self.instance_info, hyperparams['typicalness'], ent_per_con, 'test', fixed_num_insts)
		concept_info = self.concept_info

		instance_info = self.instance_info
		instances = instance_info.keys()

		with torch.no_grad():
			if valid:

				pos_triples = self.data_bundle['subclass_valid'][:sample_limit]
				dataset = complete_test_dataset(pos_triples, negative_triples, hyperparams['add_reverse_label'], valid, concept_instance_info)
			else:
				if hyperparams['distance_metric']:

					pos_triples_valid = self.data_bundle['subclass_valid'][:sample_limit]
					dataset_valid = complete_test_dataset(pos_triples_valid, negative_triples, hyperparams['add_reverse_label'], True, concept_instance_info)
					pos_triples_test = self.data_bundle['subclass_test'][:sample_limit] #[:self.num_train_triples]
					dataset_test = complete_test_dataset(pos_triples_test, negative_triples, hyperparams['add_reverse_label'], False, concept_instance_info)
					num_valid = len(dataset_valid)
					dataset = dataset_valid + dataset_test
				else:
					pos_triples = self.data_bundle['subclass_test'][:sample_limit]
					dataset = complete_test_dataset(pos_triples, negative_triples, hyperparams['add_reverse_label'], valid, concept_instance_info)



			dataset_size = len(dataset)

			if hyperparams['distance_metric']:
				avg_positive_distance = 0 
				count_positive = 0

			random_map = [i for i in range(dataset_size)]

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

			#pair_accuracy = {}
			#pair_cnt = {}

			time_1 = time.time()

			scores = []
			labels = []

			for bt, batch in enumerate(batch_list):

				triple = dataset[batch[0]]
				hypo, hyper, label = triple
				#hypo_ents, hyper_ents = sampler.sample(hypo, hyper)

				
				'''
				hypo, hyper = 'sport', 'game'
				hypo_ent_names = ['tennis', 'angling', 'basketball', 'yoga']
				hyper_ent_names = ['football', 'angry birds', 'boxing', 'mahjong']
				hypo_ents =  [ self.instance_info[ent] for ent in hypo_ent_names]
				hyper_ents =  [ self.instance_info[ent] for ent in hyper_ent_names]
				'''

				hyper_ents = sampler.sample_single(hyper, exclude_ins = hypo)
				if (len(hyper_ents) == 0):
					pdb.set_trace()
					continue
				if hypo in concepts: # subclass
					rel_type = 'subclass_selfatt'
					#count_subclass_selfatt += 1
				else: # 
					pdb.set_trace()
				
				if self.model_name in trainable_models:
					'''
					if not hyperparams['con_desc']:
						if not hyperparams['freeze_plm']:
							if selfatt == False:
								hypo_hint = hypo_hint_template.format(hypo, hypo, hyper) if add_concept_hint else ''
								hyper_hint = hyper_hint_template.format(hyper, hyper, hypo) if add_concept_hint else ''
							else:
								hypo_hint = single_hint_template.format(hypo) if add_concept_hint else ''
								hyper_hint = single_hint_template.format(hyper) if add_concept_hint else ''
	
							hypo_texts = [  hypo_hint + e[text_key] for e in hypo_ents]
							hyper_texts = [  hyper_hint + e[text_key] for e in hyper_ents]
	
							texts = hypo_texts + hyper_texts

							inputs = tokenizer(texts, truncation = True, max_length = max_length, return_tensors='pt', padding=True )
							inputs.to(device)
							embeddings = model.bert_embed(**inputs)

						else:
							insts = []
							insts += [ e['ins_name'] for e in hypo_ents]
							insts += [ e['ins_name'] for e in hyper_ents]
							insts_idx = torch.tensor([ self.ins2id[ins]  for ins in insts]).to(device)
							embeddings = model.frozen_bert_embed(insts_idx)
						

						hypo_embeddings = embeddings[:len(hypo_ents)]
						hyper_embeddings = embeddings[len(hypo_ents):]
					else:
						if not hyperparams['freeze_plm']:
							hyper_texts = [concept_info[hyper]]
							hypo_texts = [concept_info[hypo]]
							texts = hypo_texts + hyper_texts
							inputs = tokenizer(texts, truncation = True, max_length = max_length, return_tensors='pt', padding=True )
							inputs.to(device)
							embeddings = model.bert_embed(**inputs)
							hypo_embeddings = embeddings[:1]
							hyper_embeddings = embeddings[1:]
						else:
							hyper_idx = torch.tensor([self.con2id[hyper]]).to(device)
							hyper_embeddings = model.frozen_bert_embed_concepts(hyper_idx)

							hypo_idx  = torch.tensor([self.con2id[hypo ]]).to(device)
							hypo_embeddings = model.frozen_bert_embed_concepts(hypo_idx)
					'''
					hypo_hint = single_hint_template.format(hypo) if add_concept_hint else ''
					hyper_hint = single_hint_template.format(hyper) if add_concept_hint else ''

					hyper_texts = [  hyper_hint + e[text_key] for e in hyper_ents]

					if hypo in instances:
						hypo_texts = [instance_info[hypo][text_key]]
					else:
						hypo_texts = [concept_info[hypo]]
					texts = hypo_texts + hyper_texts
					inputs = tokenizer(texts, truncation = True, max_length = max_length, return_tensors='pt', padding=True )
					inputs.to(device)
					embeddings = model.bert_embed(**inputs)
					hypo_embeddings = embeddings[:1]
					hyper_embeddings = embeddings[1:]


					if selfatt == False:
						res = model(hypo_embeddings, hyper_embeddings, hypo, hyper, mode = 'subclass')

					else:
						hypo_res = model(hypo_embeddings, hypo_embeddings, hypo, hypo, mode = 'subclass_selfatt')
						hypo_prototype = hypo_res['prototype']

						hyper_res = model(hyper_embeddings, hyper_embeddings, hyper, hyper, mode = 'subclass_selfatt')
						hyper_prototype = hyper_res['prototype']

						res = model.judge(hypo_prototype, hyper_prototype, mode = 'subclass_selfatt')
				
					
					if not hyperparams['distance_metric']:
						subpred = res['sub_pred']
						subpred_label = subpred.argmax().item()

						total_accuracy += int(subpred_label == label)#(pred_label == label).float().mean().item()
						
						#pdb.set_trace()
						
						loss = criterion(subpred.unsqueeze(0), torch.tensor([label]).to(device)) #+ subclass_loss(geoinfo, label)
						total_loss += loss.item()

						if subpred_label == label:
							TP[label] += 1
							for i in range(3):
								if i != label:
									TN[i] += 1
						else:
							#pdb.set_trace()
							FP[subpred_label] += 1
							FN[label] += 1
							TN[3-label-subpred_label] += 1

					else:
						distance = res['distance']
						avg_positive_distance += distance
						count_positive += 1

						labels.append(label)
						scores.append(distance.item())

						if (len(labels) != len(scores)):
							pdb.set_trace()

					

					time_2 = time.time()
			

			if hyperparams['distance_metric']:
				if valid:
					# At valid mode, use half data to get a threshold, and another half to calculate metrics         
					to_get_thereshold_scores = [ v for i, v in enumerate(scores) if i % 2 == 0 ]
					to_get_thereshold_labels = [ v for i, v in enumerate(labels) if i % 2 == 0 ]

					to_get_metric_scores = [ v for i, v in enumerate(scores) if i % 2 == 1 ]
					to_get_metric_labels = [ v for i, v in enumerate(labels) if i % 2 == 1 ]

				else:
					# At test mode, use datum in valid set to get threshold
					split_idx = num_valid

					to_get_thereshold_scores = scores[:split_idx]
					to_get_thereshold_labels = labels[:split_idx]

					to_get_metric_scores = scores[split_idx:]
					to_get_metric_labels = labels[split_idx:]

				#pdb.set_trace()
				threshold, res_max = self.get_best_threshold(to_get_thereshold_scores, to_get_thereshold_labels)
				avg_accuracy, TP[1], FP[1], TN[1], FN[1] = self.get_metrics(threshold, to_get_metric_scores, to_get_metric_labels)
				
				avg_positive_distance = avg_positive_distance / count_positive
				print("Test Epoch (Subclass) :{0} Avg Positive Distance {1:.5f} Accuracy: {2:.5f} Time: {3:.5f} SelfAtt: {4} Valid: {5}".
					format(epc, avg_positive_distance, avg_accuracy, time_2 - time_1, selfatt, valid))

			else:
				avg_loss = total_loss / dataset_size
				avg_accuracy = total_accuracy / dataset_size
				print("Test Epoch (Subclass) :{0} Avg Loss:{1:.5f} Accuracy: {2:.5f} Time: {3:.5f} SelfAtt: {4}".format(epc, avg_loss, avg_accuracy, time_2 - time_1, selfatt))

				#for pr in pair_accuracy.keys():
				#    pair_accuracy[pr] /= pair_cnt[pr]

				#pdb.set_trace()


			
			for label in [1]: #range(hyperparams['num_labels']):
				#pdb.set_trace()
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

			if valid:
				self.update_metric(epc, 'subclass_f1', micro_f1[1])
				self.save_model(epc, 'subclass_acc', avg_accuracy)

		model.train()

	def test_instance(self, epc, selfatt = False, valid = True):
		concept_instance_info = self.data_bundle['concept_instance_info']
		negative_triples = self.data_bundle['negative_isInstanceOf_triples']

		model = self.model
		tokenizer = self.tokenizer
  
		device = self.device
		hyperparams = self.hyperparams
		   
		batch_size = hyperparams['batch_size'] 
		epoch = hyperparams['epoch'] 
		model_name = hyperparams['model_name']
		ent_per_con = hyperparams['ent_per_con']

		fixed_num_insts = False
		if hyperparams['language'] == 'zh':
			text_key = 'text'
			single_hint_template = '该物属于概念"{0}"。'

		else:
			text_key = 'text_from_wikipedia'
			single_hint_template = 'This item belongs to concept "{0}". ' 

		concepts = self.data_bundle['concept_instance_info'].keys()
		evaluate_threshold = hyperparams['evaluate_threshold']

		criterion = torch.nn.CrossEntropyLoss()

		model.eval()

		sampler = Sampler(concept_instance_info, self.instance_info, hyperparams['typicalness'], ent_per_con, 'test', fixed_num_insts)
		

		instance_info = self.instance_info

		instances = instance_info.keys()
		concept_info = self.concept_info

		with torch.no_grad():

			if valid:
				pos_triples = self.data_bundle['instance_valid'][:sample_limit]
				dataset = complete_test_dataset(pos_triples, negative_triples, hyperparams['add_reverse_label'], valid, concept_instance_info, mode = 'instance')
			else:
				if hyperparams['distance_metric']:
					pos_triples_valid = self.data_bundle['instance_valid'][:sample_limit]
					dataset_valid = complete_test_dataset(pos_triples_valid, negative_triples, hyperparams['add_reverse_label'], True, concept_instance_info, mode = 'instance')
					pos_triples_test = self.data_bundle['instance_test'][:sample_limit] #[:self.num_train_triples]
					dataset_test = complete_test_dataset(pos_triples_test, negative_triples, hyperparams['add_reverse_label'], False, concept_instance_info, mode = 'instance')
					num_valid = len(dataset_valid)
					dataset = dataset_valid + dataset_test
				else:
					pos_triples = self.data_bundle['instance_test'][:sample_limit]
					dataset = complete_test_dataset(pos_triples, negative_triples, hyperparams['add_reverse_label'], valid, concept_instance_info, mode = 'instance')

			dataset_size = len(dataset)
			
			if hyperparams['distance_metric']:
				avg_positive_distance = 0 
				count_positive = 0

			random_map = [i for i in range(dataset_size)]

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

			time_1 = time.time()

			scores = []
			labels = []

			for bt, batch in enumerate(batch_list):
				triple = dataset[batch[0]]
				hypo, hyper, label = triple

				if hypo in instances:
					hyper_ents = sampler.sample_single(hyper, exclude_ins = hypo)
				elif hyper in instances:
					pdb.set_trace()
					hypo_ents = sampler.sample_single(hypo, exclude_ins = hyper)
				else:
					pdb.set_trace()

				if self.model_name in trainable_models:
					'''
					if not hyperparams['con_desc']:
						if not hyperparams['freeze_plm']:
							if selfatt == False:
								hypo_hint = hypo_hint_template.format(hypo, hypo, hyper) if add_concept_hint else ''
								hyper_hint = hyper_hint_template.format(hyper, hyper, hypo) if add_concept_hint else ''
							else:
								hypo_hint = single_hint_template.format(hypo) if add_concept_hint else ''
								hyper_hint = single_hint_template.format(hyper) if add_concept_hint else ''


							if hypo in instances:
								hypo_texts = [  hypo_hint + instance_info[hypo][text_key] ]
								hyper_texts = [  hyper_hint + e[text_key] for e in hyper_ents]
							elif hyper in instances:
								hypo_texts = [  hypo_hint + e[text_key] for e in hypo_ents]
								hyper_texts = [  hyper_hint + instance_info[hyper][text_key]]

							texts = hypo_texts + hyper_texts

							inputs = tokenizer(texts, truncation = True, max_length = max_length, return_tensors='pt', padding=True )
							inputs.to(device)
							embeddings = model.bert_embed(**inputs)

						else:
							insts = []
							if hypo in instances:
								insts += [hypo]
							else:
								insts += [ e['ins_name'] for e in hypo_ents]
							if hyper in instances:
								insts += [hyper]
							else:
								insts += [ e['ins_name'] for e in hyper_ents]
							insts_idx = torch.tensor([ self.ins2id[ins]  for ins in insts]).to(device)
						  
							embeddings = model.frozen_bert_embed(insts_idx)

						if hypo in instances:
							hypo_embeddings = embeddings[:1]
							hyper_embeddings = embeddings[1:]
						elif hyper in instances:
							hypo_embeddings = embeddings[:-1]
							hyper_embeddings = embeddings[-1:]
					else:
						if not hyperparams['freeze_plm']:
							if hyper in instances:
								hyper_texts = [instance_info[hyper][text_key]]
							else:
								hyper_texts = [concept_info[hyper]]

							if hypo in instances:
								hypo_texts = [instance_info[hypo][text_key]]
							else:
								hypo_texts = [concept_info[hypo]]
							texts = hypo_texts + hyper_texts
							inputs = tokenizer(texts, truncation = True, max_length = max_length, return_tensors='pt', padding=True )
							inputs.to(device)
							embeddings = model.bert_embed(**inputs)
							hypo_embeddings = embeddings[:1]
							hyper_embeddings = embeddings[1:]
						else:
							if hyper in instances:
								hyper_idx = torch.tensor([self.ins2id[hyper]]).to(device)
								hyper_embeddings = model.frozen_bert_embed(hyper_idx)
							else:
								hyper_idx = torch.tensor([self.con2id[hyper]]).to(device)
								hyper_embeddings = model.frozen_bert_embed_concepts(hyper_idx)

							if hypo in instances:
								hypo_idx  = torch.tensor([self.ins2id[hypo ]]).to(device)
								hypo_embeddings = model.frozen_bert_embed(hypo_idx)
							else:
								hypo_idx  = torch.tensor([self.con2id[hypo ]]).to(device)
								hypo_embeddings = model.frozen_bert_embed_concepts(hypo_idx)
					'''
					hypo_hint = single_hint_template.format(hypo) if add_concept_hint else ''
					hyper_hint = single_hint_template.format(hyper) if add_concept_hint else ''

					hyper_texts = [  hyper_hint + e[text_key] for e in hyper_ents]

					if hypo in instances:
						hypo_texts = [instance_info[hypo][text_key]]
					else:
						pdb.set_trace()
						hypo_texts = [concept_info[hypo]]
					texts = hypo_texts + hyper_texts
					inputs = tokenizer(texts, truncation = True, max_length = max_length, return_tensors='pt', padding=True )
					inputs.to(device)
					embeddings = model.bert_embed(**inputs)
					hypo_embeddings = embeddings[:1]
					hyper_embeddings = embeddings[1:]


					if selfatt == False:
						res = model(hypo_embeddings, hyper_embeddings, hypo, hyper, mode = 'instance')
						
					else:
						hypo_res = model(hypo_embeddings, hypo_embeddings, hypo, hypo, mode = 'instance_selfatt')
						hypo_prototype = hypo_res['prototype']

						hyper_res = model(hyper_embeddings, hyper_embeddings, hyper, hyper, mode = 'instance_selfatt')
						hyper_prototype = hyper_res['prototype']

						res = model.judge(hypo_prototype, hyper_prototype, mode='instance_selfatt')

					if not hyperparams['distance_metric']:
						subpred = res['sub_pred']
						subpred_label = subpred.argmax().item()

						total_accuracy += int(subpred_label == label)
					
						loss = criterion(subpred.unsqueeze(0), torch.tensor([label]).to(device)) 
						total_loss += loss.item()

						if subpred_label == label:
							TP[label] += 1
							for i in range(3):
								if i != label:
									TN[i] += 1
						else:
							FP[subpred_label] += 1
							FN[label] += 1
							TN[3-label-subpred_label] += 1
					else:
						distance = res['distance']
						avg_positive_distance += distance
						count_positive += 1

						labels.append(label)
						scores.append(distance.item())

			time_2 = time.time()

			if hyperparams['distance_metric']:

				if valid:
					# At valid mode, use half data to get a threshold, and another half to calculate metrics         
					to_get_thereshold_scores = [ v for i, v in enumerate(scores) if i % 2 == 0 ]
					to_get_thereshold_labels = [ v for i, v in enumerate(labels) if i % 2 == 0 ]

					to_get_metric_scores = [ v for i, v in enumerate(scores) if i % 2 == 1 ]
					to_get_metric_labels = [ v for i, v in enumerate(labels) if i % 2 == 1 ]

				else:
					# At test mode, use datum in valid set to get threshold
					split_idx = num_valid

					to_get_thereshold_scores = scores[:split_idx]
					to_get_thereshold_labels = labels[:split_idx]

					to_get_metric_scores = scores[split_idx:]
					to_get_metric_labels = labels[split_idx:]


				threshold, res_max = self.get_best_threshold(to_get_thereshold_scores, to_get_thereshold_labels)
				avg_accuracy, TP[1], FP[1], TN[1], FN[1] = self.get_metrics(threshold, to_get_metric_scores, to_get_metric_labels)
				
				avg_positive_distance = avg_positive_distance / count_positive
				print("Test Epoch (Instance) :{0} Avg Positive Distance {1:.5f} Accuracy: {2:.5f} Time: {3:.5f} SelfAtt: {4} Valid: {5}".
					format(epc, avg_positive_distance, avg_accuracy, time_2 - time_1, selfatt, valid))

			else:
				avg_loss = total_loss / dataset_size
				avg_accuracy = total_accuracy / dataset_size
				print("Test Epoch (Instance) :{0} Avg Loss:{1:.5f} Accuracy: {2:.5f} Time: {3:.5f} SelfAtt: {4}".format(epc, avg_loss, avg_accuracy, time_2 - time_1, selfatt))


			for label in [1]:#range(hyperparams['num_labels']):
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

			if valid:
				self.update_metric(epc, 'instance_f1', micro_f1[1])
				self.update_metric(epc, 'instance_acc', avg_accuracy)

		model.train()

	def link_prediction(self, epc, valid=True):

		concept_instance_info = self.data_bundle['concept_instance_info']
		model = self.model
		tokenizer = self.tokenizer
		device = self.device
		hyperparams = self.hyperparams
		
		
		concepts = self.concepts #data_bundle['concept_instance_info'].keys()
		instances = self.instances
		prototype_size = model.prototype_size

		type_constrain = self.hyperparams['type_constrain']
		
		#pdb.set_trace()
		embeddings = self.generate_concept_prototype()

		#with open('embeddings.pkl', 'rb') as fil:
		#	embeddings = pickle.load(fil)
		
		instance_embeddings = embeddings['instance_embeddings'].to(device)
		concept_prototypes = embeddings['concept_prototypes'].to(device)
		concept_embeddings = embeddings['concept_embeddings'].to(device)

		#batch_size = 16
		id2con = self.id2con
		con2id = self.con2id

		id2ins = self.id2ins 
		ins2id = self.ins2id

		if not type_constrain:
			num_concepts = len(concepts)
			#pdb.set_trace()
			id2ins = { (k+num_concepts): v for k, v in id2ins.items()}
			ins2id = { k: (v+num_concepts) for k, v in ins2id.items()}


		Candidates = { 'subclass': { 'head': set(), 'tail': set()},
					   'instance': { 'head': set(), 'tail': set()}
		}

		Groundtruth = { 'subclass': { target: { split : { con: set() for con in concepts } for split in ['all', 'test'] } for target in ['head', 'tail'] },
						'instance': { 'head': { split : { con: set() for con in concepts } for split in ['all', 'test'] }, 
									 'tail': { split : { ins: set() for ins in instances } for split in ['all', 'test'] }}
		}

		if valid:
			isSubclassOf_triples = { 'all': self.data_bundle['isSubclassOf_triples'], 'test': self.data_bundle['subclass_valid'] }
			isInstanceOf_triples = { 'all': self.data_bundle['isInstanceOf_triples'], 'test': self.data_bundle['instance_valid'] }
		else:
			isSubclassOf_triples = { 'all': self.data_bundle['isSubclassOf_triples'], 'test': self.data_bundle['subclass_test'] }
			isInstanceOf_triples = { 'all': self.data_bundle['isInstanceOf_triples'], 'test': self.data_bundle['instance_test'] }
			# 忘了把valid时link prediction的设为valid集合了，那只能在test时用valid集合了


		print('Valid = ', valid, ' Num Test: sub {0} ins {1}'.format(len(isSubclassOf_triples['test']), len(isInstanceOf_triples['test'])))
		
		for split in ['all']:
			for triple in isSubclassOf_triples[split]:
				hypo, hyper, label = triple 
				Candidates['subclass']['head'].add(con2id[hypo])
				Candidates['subclass']['tail'].add(con2id[hyper])

			for triple in isInstanceOf_triples[split]:
				hypo, hyper, label = triple 
				Candidates['instance']['head'].add(ins2id[hypo])
				Candidates['instance']['tail'].add(con2id[hyper])

		Candidates['subclass']['head'] = sorted(list(Candidates['subclass']['head']))
		Candidates['subclass']['tail'] = sorted(list(Candidates['subclass']['tail']))
		Candidates['instance']['head'] = sorted(list(Candidates['instance']['head']))
		Candidates['instance']['tail'] = sorted(list(Candidates['instance']['tail']))


		#pdb.set_trace()
		for split in ['all', 'test']:
			for triple in isSubclassOf_triples[split]:
				hypo, hyper, label = triple 
				Groundtruth['subclass']['tail'][split][hypo].add(hyper)
				Groundtruth['subclass']['head'][split][hyper].add(hypo)

			for triple in isInstanceOf_triples[split]:
				hypo, hyper, label = triple 
				Groundtruth['instance']['tail'][split][hypo].add(hyper)
				Groundtruth['instance']['head'][split][hyper].add(hypo)

		#for rel in ['subclass', 'instance']:
		#    for target in ['tail', 'head']:
		#        for k in Groundtruth[rel][target]['all'].keys():
		#            assert len(Groundtruth[rel][target]['test'][k].difference(Groundtruth[rel][target]['all'][k])) == 0

		
		model.eval()

		ks = [1, 3, 10]
		MR = { 
				setting:
					{target:
						{rel: 0 for rel in ['subclass', 'instance']} 
					for target in ['head', 'tail']} 
				for setting in ['raw', 'filter']
			} 

  
		MRR = { 
				setting:
					{target:
						{rel: 0 for rel in ['subclass', 'instance']} 
					for target in ['head', 'tail']} 
				for setting in ['raw', 'filter']
			}
		hits = { 
				setting:
					{target:
						{rel: {k: 0 for k in ks} for rel in ['subclass', 'instance']} 
					for target in ['head', 'tail']} 
				for setting in ['raw', 'filter']
			}

		#pair_mrr = {}
		#pair_cnt = {}

		#pdb.set_trace()
		with torch.no_grad():
			# link prediction for subclass
			for setting in ['raw', 'filter']:
				for rel in ['subclass', 'instance']:
					for target in ['head', 'tail']:
						if rel == 'subclass':
							givens = concepts 
							targets = concepts 
							given_type = target_type = 'concept'
						else:
							if target == 'head':
								givens = concepts
								#targets = instances 
								given_type = 'concept'
								target_type ='instance'
							else:
								givens = instances 
								#targets = concepts
								given_type = 'instance'
								target_type = 'concept'


						count_triples = 0
						for giv in givens:        
							testees = Groundtruth[rel][target]['test'][giv]
							groundtruth = Groundtruth[rel][target]['all'][giv]#.union(Groundtruth[rel][target]['test'][giv])
							#assert(groundtruth == Groundtruth[rel][target]['all'][giv])
							if not type_constrain:
								groundtruth = Groundtruth['subclass'][target]['all'].get(giv,set()).union(Groundtruth['instance'][target]['all'].get(giv, set()))
							

							if len(testees) > 0:
								count_triples += len(testees)
								if given_type == 'concept':
									igiv = con2id[giv]
									if target == 'head': # so given is tail(hyper)
										prototype = concept_prototypes[igiv]
									else: # given is head(hypo)
										prototype = concept_embeddings[igiv]
								else:
									igiv = ins2id[giv]
									if type_constrain:
										prototype = instance_embeddings[igiv]
									else:
										prototype = instance_embeddings[igiv-num_concepts]

								if type_constrain:
									
									if target_type == 'concept':
										candidate_idxs = Candidates[rel][target]#[ con2id[c] for c in Candidates[rel][target]]
										if target == 'tail':
											all_candidates = concept_prototypes[candidate_idxs]
										else:
											all_candidates = concept_embeddings[candidate_idxs]

										target_size = len(candidate_idxs)
									elif target_type == 'instance':
										candidate_idxs = Candidates[rel][target]#[ ins2id[c] for c in Candidates[rel][target]]
										all_candidates = instance_embeddings[candidate_idxs]
										target_size = len(candidate_idxs)
									
									candidate_maps = {c:i for i, c in enumerate(candidate_idxs)}
									candidate_maps_reverse = { i:c for i, c in enumerate(candidate_idxs)}
									'''
									if target_type == 'concept':
										all_candidates = concept_prototypes
										target_size = len(concepts)
									elif target_type == 'instance':
										all_candidates = instance_embeddings
										target_size = len(instances)
									'''
									
								else:
									pdb.set_trace()
									if target == 'tail':
										all_candidates = torch.cat([concept_prototypes, instance_embeddings], dim=0)
									else:
										all_candidates = torch.cat([concept_embeddings, instance_embeddings], dim=0)
									target_size = len(concepts) + len(instances)

								#if target == 'head':
								#    feature_tensors = [prototype.expand(len(concepts), prototype_size), concept_prototypes, prototype - concept_prototypes, prototype * concept_prototypes]
								#else:
								#    feature_tensors = [concept_prototypes, prototype.expand(len(concepts), prototype_size), concept_prototypes - prototype, concept_prototypes * prototype]
								
								if not hyperparams['distance_metric']:
									pdb.set_trace()
									if target == 'head':
										feature_tensors = [prototype.expand(target_size, prototype_size), all_candidates, prototype - all_candidates, prototype * all_candidates]
									else:
										feature_tensors = [all_candidates, prototype.expand(target_size, prototype_size), all_candidates - prototype, all_candidates * prototype]

									#pdb.set_trace()
									if self.model_name == 'psn':
										if rel == 'instance' and hyperparams['separate_classifier']:
											sub_pred = model.ins_selfatt_classifier(torch.cat(feature_tensors, dim=-1)).squeeze()
										else:
											sub_pred = model.sub_selfatt_classifier(torch.cat(feature_tensors, dim=-1)).squeeze()

									else:
									
										if rel == 'instance' and hyperparams['separate_classifier']:
											sub_pred = model.ins_classifier(torch.cat(feature_tensors, dim=-1)).squeeze()
										else:
											sub_pred = model.sub_classifier(torch.cat(feature_tensors, dim=-1)).squeeze()

									score = sub_pred.softmax(dim=-1)[:,1]
									tops = score.argsort(descending=True).tolist()
								else:
									
									if rel == 'subclass':
										rel_embedding = 0#model.isA_embedding(torch.tensor(0).to(device))
									else:
										rel_embedding = 0#model.isA_embedding(torch.tensor(1).to(device))
									if target == 'head':
										feature_tensors = all_candidates + rel_embedding - prototype
									else:
										feature_tensors = prototype + rel_embedding - all_candidates
									distance = torch.norm(feature_tensors, p=2, dim=-1) #feature_tensors.abs().sum(dim=1)
									#pdb.set_trace()
									score = -distance 
									tops = score.argsort(descending=True).tolist()

								# e.g. tops = [2470, 2606,  954,  ..., 2566, 1346,  262], 即第2470个概念分数最高，第2606个概念分数其次 
								
								countt = 0 
								for testee in testees:
									if target_type == 'concept':
										itestee = con2id[testee]
									else:
										itestee = ins2id[testee]

									if type_constrain:
										itestee = candidate_maps[itestee]

									if setting == 'raw':
										#ranks = {ic: r for r, ic in enumerate(tops)}
										# 可以得到 {2470: 0, 2606: 1 ...}，即可以根据icon得到排名
										#rank = ranks[icand]
										#rank = tops.index(itestee) + 1 
										tops_ = tops

									else:
										other_groundtruth = groundtruth.difference(set([testee])) 
										if target_type == 'concept':
											other_groundtruth_idx = set([ (con2id[g] if g in con2id else ins2id[g]) for g in other_groundtruth])
										else:
											other_groundtruth_idx = set([ (ins2id[g] if g in ins2id else con2id[g]) for g in other_groundtruth])
										# Filter out other groundtruth
										if type_constrain:
											other_groundtruth_idx = set([candidate_maps[i] for i in other_groundtruth_idx])
										tops_ = [ t for t in tops if (not t in other_groundtruth_idx)]

										if target_type == 'concept':
											tops_names = [ id2con[candidate_maps_reverse[t]] for t in tops_]
										else:
											tops_names = [ id2ins[candidate_maps_reverse[t]] for t in tops_]
										
										#if target == 'tail' and countt == 0:
										#    print('Rel: {0} Giv: {1} Target: {2} Tops10 {3}'.format(rel, giv, target, tops_names[:10]))
										#    pdb.set_trace()
										#else:
										#    countt += 1

									'''
									if type_constrain:
										if target_type == 'concept':
											candidate_idxs = [ con2id[c] for c in Candidates[rel][target]]                                        
										elif target_type == 'instance':
											candidate_idxs = [ ins2id[c] for c in Candidates[rel][target]]
										tops_ = [ t for t in tops_ if (t in candidate_idxs)]
									'''

									rank = tops_.index(itestee) + 1

									MRR[setting][target][rel] += 1/rank 
									MR[setting][target][rel] += rank 
		
									for k in ks:
										if rank <= k:
											hits[setting][target][rel][k] += 1 

									#if setting == 'filter' and rel == 'subclass':
									#    #pdb.set_trace()
									#    if target == 'head':
									#        n_hypo = len(concept_instance_info[testee])
									#        n_hyper = len(concept_instance_info[giv]) 
									#    else:
									#        n_hyper = len(concept_instance_info[testee])
									#        n_hypo = len(concept_instance_info[giv]) 


									#    pair_mrr[(n_hypo, n_hyper)] = pair_mrr.get((n_hypo, n_hyper), 0) + 1 / rank
									#    pair_cnt[(n_hypo, n_hyper)] = pair_cnt.get((n_hypo, n_hyper), 0) + 1


						if rel == 'subclass':
							total_triplets = len(isSubclassOf_triples['test'])
						else:
							total_triplets = len(isInstanceOf_triples['test'])

						assert(count_triples == total_triplets)

						MR[setting][target][rel] /= total_triplets
						MRR[setting][target][rel] /= total_triplets
						for k in ks:
							hits[setting][target][rel][k] /= total_triplets

						print('MR {0:.5f} MRR {1:.5f} hits 1 {2:.5f} 3 {3:.5f} 10 {4:.5f}, Setting: {5} Target: {6} Rel: {7} Constrain {8} '.format(
							MR[setting][target][rel], MRR[setting][target][rel], hits[setting][target][rel][1], hits[setting][target][rel][3], hits[setting][target][rel][10],
							setting, target, rel, hyperparams['type_constrain']
						))

						#if setting == 'filter' and rel == 'subclass':
						#    for pr in pair_mrr.keys():
						#        pair_mrr[pr] /= pair_cnt[pr]
						#    pdb.set_trace()

		raw_subclass_mrr = (MRR['raw']['head']['subclass'] + MRR['raw']['tail']['subclass']) / 2
		raw_subclass_hits1 = (hits['raw']['head']['subclass'][1] + hits['raw']['tail']['subclass'][1]) / 2
		raw_subclass_hits3 = (hits['raw']['head']['subclass'][3] + hits['raw']['tail']['subclass'][3]) / 2
		raw_subclass_hits10 = (hits['raw']['head']['subclass'][10] + hits['raw']['tail']['subclass'][10]) / 2


		fil_subclass_mrr = (MRR['filter']['head']['subclass'] + MRR['filter']['tail']['subclass']) / 2
		fil_subclass_hits1 = (hits['filter']['head']['subclass'][1] + hits['filter']['tail']['subclass'][1]) / 2
		fil_subclass_hits3 = (hits['filter']['head']['subclass'][3] + hits['filter']['tail']['subclass'][3]) / 2
		fil_subclass_hits10 = (hits['filter']['head']['subclass'][10] + hits['filter']['tail']['subclass'][10]) / 2

		raw_instance_mrr = (MRR['raw']['head']['instance'] + MRR['raw']['tail']['instance']) / 2
		raw_instance_hits1 = (hits['raw']['head']['instance'][1] + hits['raw']['tail']['instance'][1]) / 2
		raw_instance_hits3 = (hits['raw']['head']['instance'][3] + hits['raw']['tail']['instance'][3]) / 2
		raw_instance_hits10 = (hits['raw']['head']['instance'][10] + hits['raw']['tail']['instance'][10]) / 2


		fil_instance_mrr = (MRR['filter']['head']['instance'] + MRR['filter']['tail']['instance']) / 2
		fil_instance_hits1 = (hits['filter']['head']['instance'][1] + hits['filter']['tail']['instance'][1]) / 2
		fil_instance_hits3 = (hits['filter']['head']['instance'][3] + hits['filter']['tail']['instance'][3]) / 2
		fil_instance_hits10 = (hits['filter']['head']['instance'][10] + hits['filter']['tail']['instance'][10]) / 2
		
		self.update_metric(epc, 'raw_subclass_mrr', raw_subclass_mrr)
		self.update_metric(epc, 'raw_subclass_hits1', raw_subclass_hits1)
		self.update_metric(epc, 'raw_subclass_hits3', raw_subclass_hits3)
		self.update_metric(epc, 'raw_subclass_hits10', raw_subclass_hits10)

		#self.update_metric(epc, 'fil_subclass_mrr', fil_subclass_mrr)
		self.save_model(epc, 'fil_subclass_mrr', fil_subclass_mrr)
		#self.update_metric(epc, 'fil_subclass_hits1', fil_subclass_hits1)
		self.save_model(epc, 'fil_subclass_hits1', fil_subclass_hits1)
		self.update_metric(epc, 'fil_subclass_hits3', fil_subclass_hits3)
		#self.update_metric(epc, 'fil_subclass_hits10', fil_subclass_hits10)
		self.save_model(epc, 'fil_subclass_hits10', fil_subclass_hits10)

		self.update_metric(epc, 'raw_instance_mrr', raw_instance_mrr)
		self.update_metric(epc, 'raw_instance_hits1', raw_instance_hits1)
		self.update_metric(epc, 'raw_instance_hits3', raw_instance_hits3)
		self.update_metric(epc, 'raw_instance_hits10', raw_instance_hits10)

		self.update_metric(epc, 'fil_instance_mrr', fil_instance_mrr)
		self.update_metric(epc, 'fil_instance_hits1', fil_instance_hits1)
		#self.save_model(epc, 'fil_instance_hits1', fil_instance_hits1)
		self.update_metric(epc, 'fil_instance_hits3', fil_instance_hits3)
		self.update_metric(epc, 'fil_instance_hits10', fil_instance_hits10)
		#self.save_model(epc, 'fil_instance_hits10', fil_instance_hits10)

		if not valid:
			print('& {0:.2f} / {1:.2f} & {2:.2f} / {3:.2f} & {4:.2f} / {5:.2f} & {6:.2f} / {7:.2f} & {8:.2f} / {9:.2f} & {10:.2f} / {11:.2f} & {12:.2f} / {13:.2f} & {14:.2f} / {15:.2f} '.format(
				raw_subclass_mrr*100, fil_subclass_mrr*100,
				raw_subclass_hits1*100, fil_subclass_hits1*100,
				raw_subclass_hits3*100, fil_subclass_hits3*100,
				raw_subclass_hits10*100, fil_subclass_hits10*100, 
				raw_instance_mrr*100, fil_instance_mrr*100,
				raw_instance_hits1*100, fil_instance_hits1*100,
				raw_instance_hits3*100, fil_instance_hits3*100,
				raw_instance_hits10*100, fil_instance_hits10*100, 
				))

		model.train()

	def generate_concept_prototype(self):

		concept_instance_info = self.data_bundle['concept_instance_info']
		model = self.model
		tokenizer = self.tokenizer
		device = self.device
		hyperparams = self.hyperparams

		model_name = hyperparams['model_name']
		ent_per_con = hyperparams['ent_per_con']

		fixed_num_insts = False
		if hyperparams['language'] == 'zh':
			text_key = 'text'
			single_hint_template = '该物属于概念"{0}"。'

		else:
			text_key = 'text_from_wikipedia'
			single_hint_template = 'This item belongs to concept "{0}". '

		concepts = self.concepts #data_bundle['concept_instance_info'].keys()
		instances = self.instances 
		prototype_size = model.prototype_size
		model.eval()
		sampler = Sampler(concept_instance_info, self.instance_info, hyperparams['typicalness'], ent_per_con, 'test', fixed_num_insts)
		concept_info = self.concept_info

		concept_prototypes = torch.zeros(len(concepts), prototype_size).float()
		concept_embeddings = torch.zeros(len(concepts), prototype_size).float() # For Prototypical Network
		instance_embeddings = torch.zeros(len(instances), prototype_size).float()

		batch_size = 128
		num_instances = len(instances)
		num_concepts = len(concepts)


		with torch.no_grad():
			# Embedding concepts 
			random_map = [i for i in range(num_concepts)]
			batch_list = [ random_map[i:i+batch_size] for i in range(0, num_concepts ,batch_size)] 

			#pdb.set_trace()
			for batch in batch_list:
				cons = [ self.id2con[i] for i in batch]
				if not hyperparams['freeze_plm']:
					texts = [ (single_hint_template.format(con) if add_concept_hint else '') + self.concept_info[con] for con in cons]

					inputs = tokenizer(texts, truncation = True, max_length = max_length, return_tensors='pt', padding=True )
					inputs.to(device)
					embeddings = model.bert_embed(**inputs)

					# At test time, such operation is not needed . Because It gets identical results.
					#embeddings_ = torch.zeros(embeddings.shape)
					#for ie, embedding in enumerate(embeddings):
					#    pdb.set_trace()
					#    emb = embedding.unsqueeze(0)
					#    res = model(emb, emb, mode = 'instance_selfatt')
					#    embeddings_[ie] = res['prototype'].cpu()

				else:
					pdb.set_trace()

				concept_embeddings[batch] = embeddings.cpu()


			random_map = [i for i in range(num_instances)]
			batch_list = [ random_map[i:i+batch_size] for i in range(0, num_instances ,batch_size)] 

			for batch in batch_list:
				insts = [ self.id2ins[i] for i in batch]
				if not hyperparams['freeze_plm']:
					texts = [ (single_hint_template.format(ins) if add_concept_hint else '') + self.instance_info[ins][text_key] for ins in insts]

					inputs = tokenizer(texts, truncation = True, max_length = max_length, return_tensors='pt', padding=True )
					inputs.to(device)
					embeddings = model.bert_embed(**inputs)

					# At test time, such operation is not needed . Because It gets identical results.
					#embeddings_ = torch.zeros(embeddings.shape)
					#for ie, embedding in enumerate(embeddings):
					#    pdb.set_trace()
					#    emb = embedding.unsqueeze(0)
					#    res = model(emb, emb, mode = 'instance_selfatt')
					#    embeddings_[ie] = res['prototype'].cpu()

				else:
					#pdb.set_trace()
					insts_idx = torch.tensor([ self.ins2id[ins]  for ins in insts]).to(device)
					embeddings = model.frozen_bert_embed(insts_idx)

				instance_embeddings[batch] = embeddings.cpu()
			

			for icon in range(num_concepts):
				# For every concept con, 
				con = self.id2con[icon]

				'''
				# case study
				con = 'bird'
				avg_family_resemblance = { ins: 0 for ins in concept_instance_info['bird'] }
				cnt_instance = { ins: 0 for ins in concept_instance_info['bird'] }  
				'''

				if not hyperparams['con_desc']:
					hyper = hypo = con
					insts = concept_instance_info[con]

					if hyperparams['model_name'] == 'psn':
						embed_times = max(math.floor((math.log(len(insts)) / math.log(2))) - 1, 1)
					else:
						embed_times = 1

					#embed_times = 10000 # case study
					
					sum_prot = torch.zeros(prototype_size).float().to(device)
					hint = single_hint_template.format(con) if add_concept_hint else ''
					rel_type = 'subclass_selfatt'
					for t in range(embed_times):
						insts = sampler.sample_single(con)
						
						if not hyperparams['freeze_plm']:
							texts = [  hint + e[text_key] for e in insts]
							inputs = tokenizer(texts, truncation = True, max_length = max_length, return_tensors='pt', padding=True )
							inputs.to(device)
							embeddings = model.bert_embed(**inputs)
						else:

							insts_idx = torch.tensor([ self.ins2id[ins['ins_name']]  for ins in insts]).to(device)
							embeddings = model.frozen_bert_embed(insts_idx)

						res = model(embeddings, embeddings, con, con, mode = rel_type)

						'''
						# case study 
						insts_name = [ e['ins_name'] for e in insts]
						for ii, inst in enumerate(insts_name):
							avg_family_resemblance[inst] += res['family_resemblance'][ii]
							cnt_instance[inst] += 1
						'''

						prototype = res['prototype']
						sum_prot += prototype 

					prototype = sum_prot / embed_times

					'''
					for inst in avg_family_resemblance:
						if cnt_instance[inst]>0:
							avg_family_resemblance[inst] /= cnt_instance[inst]
						else:
							pdb.set_trace()
					pdb.set_trace()
					'''

				else:
					if not hyperparams['freeze_plm']:
						texts = [self.concept_info[con]]  
						# 注意要去掉concept hint ? 
						inputs = tokenizer(texts, truncation = True, max_length = max_length, return_tensors='pt', padding=True )
						inputs.to(device)
						embedding = model.bert_embed(**inputs)
						prototype = embedding
					else:
						con_idx = torch.tensor([icon]).to(device)
						prototype = model.frozen_bert_embed_concepts(con_idx)


				#concept_prototypes[self.con2id[con]] = prototype.cpu()
				concept_prototypes[icon] = prototype.cpu()            
		
		embeddings = {
			'instance_embeddings': instance_embeddings,
			'concept_prototypes': concept_prototypes,
			'concept_embeddings': concept_embeddings,
			'id2con': self.id2con,
			'id2ins': self.id2ins
		}
		
		model.train()
		with open('embeddings.pkl', 'wb') as fil:
			pickle.dump(embeddings, fil)

		#pdb.set_trace()
		return embeddings

	def get_best_threshold(self, scores, labels):
		scores = np.array(scores)
		labels = np.array(labels)

		res = np.concatenate([labels.reshape(-1,1), scores.reshape(-1,1)], axis = -1)
		order = np.argsort(scores)
		res = res[order]

		total_all = (float)(len(scores))
		total_current = 0.0
		total_true = np.sum(labels)
		total_false = total_all - total_true

		res_max = 0.0
		threshold = None

		for index, [ans, score] in enumerate(res):
			# if we use this score as a threshold, 
			# triples having score <= this score would be regarded as positive
			# total_current = TP current
			if ans == 1:
				total_current += 1.0
			res_current = (2 * total_current + total_false - index - 1) / total_all
			if res_current > res_max:
				res_max = res_current
				threshold = score

		return threshold, res_max

	def get_metrics(self, threshold, scores, labels):
		scores = np.array(scores)
		labels = np.array(labels)

		res = np.concatenate([labels.reshape(-1,1), scores.reshape(-1,1)], axis = -1)
		order = np.argsort(scores)
		res = res[order]

		total_all = len(scores)
		total_true = np.sum(labels)
		total_false = total_all - total_true

		TP = 0
		FP = 0
		TN = 0
		FN = 0
		for index, [ans, score] in enumerate(res):
			if score <= threshold:
				if ans == 1:
					TP += 1
				else:
					FP += 1
			else:
				FN = total_true - TP
				TN = total_false - FP 
				break

		accuracy = (TP + TN) / total_all 
		'''
		if TP + FP > 0:
			precision = TP / ( TP + FP)
		else:
			precision = float('nan')
		if TP + FN > 0:
			recall = TP / ( TP + FN)
		else:
			recall[label] = float('nan')
		if precision + recall > 0:
			micro_f1 = (2*precision*recall) / (precision + recall)
		else:
			micro_f1 = float('nan')
		'''

		return accuracy, TP, FP, TN, FN #precision, recall, micro_f1

		

	def update_metric(self, epc, name, score):
		self.history_value[name].append(score)
		if score > self.best_metric[name]:
			self.best_metric[name] = score
			self.best_epoch[name] = epc
			print('! Metric {0} Updated as: {1:.2f}'.format(name, score*100))
			return True
		else:
			return False

	def save_model(self, epc, metric, metric_val):
		save_path = self.param_path_template.format(epc, metric)
		last_path = self.param_path_template.format(self.best_epoch[metric], metric)
		#pdb.set_trace()
		if self.update_metric(epc, metric, metric_val):
			if os.path.exists(last_path) and save_path != last_path and epc > self.best_epoch[metric]:
				os.remove(last_path)
			
			torch.save(self.model.state_dict(), save_path)
			print('Parameters saved into ', save_path)

	def test_triple_classification(self, epc=-1, valid = True):
		hyperparams = self.hyperparams
		print('Triple Classification: Valid = ', valid)
		print(hyperparams)
		if hyperparams['variant'] in ['default', 'hybrid']:
			self.test_subclass(epc, valid = valid)
			if hyperparams['train_instance_of']:
				self.test_instance(epc, valid = valid)
		if hyperparams['variant'] in ['selfatt', 'hybrid']:
			self.test_subclass(epc, selfatt = True, valid = valid)
			if hyperparams['train_instance_of']:
				self.test_instance(epc, selfatt = True, valid = valid)
			#self.link_prediction(epc)

	def debug_signal_handler(self, signal, frame):
		pdb.set_trace()

def complete_train_dataset(dataset, concepts, add_reverse_label, concept_instance_info = None, instances = None, mode = 'subclass', num_pos=-1):
	orig_dataset = set(dataset)

	if mode == 'instance':
		dataset = [d for d in dataset if len(concept_instance_info[d[1]]) > 2]
		# 保证做的instance of里，我去掉这个instance本身，还能找到至少2个support的实例

	if num_pos != -1:
		dataset = random.sample(dataset, min(num_pos,len(dataset)) )

	negative_dataset = set()
	num_triples = len(dataset)
	num_triples_nt = num_triples
	count_nt = 0

	#pdb.set_trace()
	while(count_nt < num_triples_nt):
		if mode == 'subclass':
			hypo, hyper = random.sample(concepts, 2)
		elif mode == 'instance':
			hypo = random.sample(instances, 1)[0]
			hyper = random.sample(concepts, 1)[0]
		else:
			pdb.set_trace()

		if not (hypo, hyper, 1) in orig_dataset and not (hypo, hyper, 0) in negative_dataset:
			negative_dataset.add((hypo, hyper, 0))
			count_nt += 1

	negative_dataset = list(negative_dataset)

	dataset_ = dataset + negative_dataset
	if add_reverse_label:
		reverse_dataset = [ (hyper, hypo, 2) for hypo, hyper, _ in dataset]
		dataset_ = dataset_ + reverse_dataset
	random.shuffle(dataset_)
	return dataset_

def complete_test_dataset(dataset, negative_triples, add_reverse_label, valid, concept_instance_info = None, mode = 'subclass' ):
	#pdb.set_trace()
	if mode == 'instance':
		dataset = [d for d in dataset if len(concept_instance_info[d[1]]) > 2]

	num_triples = len(dataset)
	num_triples_nt = num_triples
	if valid:
		negative_dataset = negative_triples[:num_triples_nt]
	else:
		negative_dataset = negative_triples[-num_triples_nt:]

	dataset_ = dataset + negative_dataset
	if add_reverse_label:
		reverse_dataset = [ (hyper, hypo, 2) for hypo, hyper, _ in dataset]
		dataset_ = dataset_ + reverse_dataset
	#random.shuffle(dataset_)
	#pdb.set_trace()
	return dataset_