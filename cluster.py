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
from utils import clip_grad_norm_,  detect_nan_params
import numpy as np


def my_cluster(instance_embeddings, concept_prototypes, n_ins_of_con):
	n_ins = instance_embeddings.shape[0]
	n_con = concept_prototypes.shape[0]

	cluster_preds = torch.zeros(n_ins, n_con)
	with torch.no_grad():
		for i_con, con in enumerate(concept_prototypes):
			dist_to_ins = (con - instance_embeddings).square().sum(dim=-1)
			closet = dist_to_ins.argsort(descending=False)[:n_ins_of_con[i_con]]
			cluster_preds[closet, i_con] = 1

	return cluster_preds

def k_means_cluster(instance_embeddings, concept_prototypes):
	from sklearn.cluster import KMeans
	n_ins = instance_embeddings.shape[0]
	n_con = concept_prototypes.shape[0]

	kmeans = KMeans(n_clusters=n_con, init=concept_prototypes.cpu()).fit(instance_embeddings.cpu())

	cluster_preds = kmeans.labels_
	return cluster_preds

def calc_ARI(cluster_label_table, n_ins):
	n_cluster, n_label = cluster_label_table.shape
	sigma_nijC2 = 0
	sigma_aiC2 = 0
	sigma_bjC2 = 0 

	nC2 = n_ins * (n_ins - 1) / 2

	for i in range(n_cluster):
		ai = cluster_label_table[i].sum()
		#pdb.set_trace()
		for j in range(n_label):
			nij = cluster_label_table[i, j]
			sigma_nijC2 += nij * (nij - 1) / 2
		sigma_aiC2 += ai * (ai - 1) / 2


	for j in range(n_label):
		bj = cluster_label_table[:, j].sum()
		sigma_bjC2 += bj * (bj - 1) / 2


	index = sigma_nijC2 
	expected_index = sigma_aiC2 * sigma_bjC2 / nC2
	max_index = (sigma_aiC2 + sigma_bjC2) / 2

	ARI = (index - expected_index) / (max_index - expected_index)
	pdb.set_trace()
	return ARI

def calc_RI(cluster_preds, concept_labels, n_con):
	n_clus = n_con 

	ins_of_clusters = { i:[] for i in range(n_clus)}
	for i_ins, clus in enumerate(cluster_preds):
		ins_of_clusters[clus].append(i_ins)
	#pdb.set_trace()
	cons_of_ins = { k: set(v.nonzero().squeeze(1).tolist()) for k, v in enumerate(concept_labels)}

	soft_TP = 0
	soft_FP = 0
	soft_TN = 0 
	soft_FN = 0

	hard_TP = 0
	hard_FP = 0
	hard_TN = 0 
	hard_FN = 0

	with torch.no_grad():
		for i_clus, insts in ins_of_clusters.items():
			n_ins = len(insts)
			for i_x in range(n_ins):
				x = insts[i_x]
				x_concepts = cons_of_ins[x] #concept_labels[x]
				for i_y in range(i_x+1, n_ins):
					y = insts[i_y]
					y_concepts = cons_of_ins[y] #concept_labels[y]
					inter = (x_concepts.intersection(y_concepts)) 
					sim = len(inter) / math.sqrt(len(x_concepts) * len(y_concepts))
					positive = len(inter) > 0 #((x_concepts * y_concepts).sum() > 0).item()
					if (positive):
						hard_TP += 1
					else:
						hard_FP += 1
					soft_TP += sim 
					soft_FP += (1-sim) 
	
		for i_clus in tqdm(range(n_clus)):
			i_insts = ins_of_clusters[i_clus]
			n_i_insts = len(i_insts) 

			for j_clus in range(i_clus+1, n_clus):
				j_insts = ins_of_clusters[j_clus]
				n_j_insts = len(j_insts)
	
				for i_x_ins in i_insts:
					for j_y_ins in j_insts:
						x_concepts = cons_of_ins[i_x_ins] #concept_labels[x]
						y_concepts = cons_of_ins[j_y_ins] #concept_labels[y]
						inter = (x_concepts.intersection(y_concepts)) 
						sim = len(inter) / math.sqrt(len(x_concepts) * len(y_concepts))
						positive = len(inter) > 0 #((x_concepts * y_concepts).sum() > 0).item()
						if (positive):
							hard_FN += 1
						else:
							hard_TN += 1 
						soft_FN += sim 
						soft_TN += (1-sim)
		
	print('Soft : TP {} FN {} FP {} TN {}'.format(soft_TP, soft_FN, soft_FP, soft_TN))
	print('Hard : TP {} FN {} FP {} TN {}'.format(hard_TP, hard_FN, hard_FP, hard_TN))

	RI = (hard_TP + hard_TN) / (hard_TP + hard_FN + hard_FP + hard_TN)
	
	return RI 


def calc_NMI(cluster_label_table, cluster_preds, concept_labels, n_ins):
	n_cluster, n_label = cluster_label_table.shape

	n_ins_of_clus = torch.zeros(n_cluster) #{ i:0 for i in range(n_cluster)}
	n_ins_of_label = torch.zeros(n_label) #{ i:0 for i in range(n_label)}

	for pred in cluster_preds: 
		n_ins_of_clus[pred] += 1 

	for labels in concept_labels:
		labels = labels.nonzero().squeeze(1).tolist()
		for label in labels:
			n_ins_of_label[label] += 1

	
	p_cluster = n_ins_of_clus / n_ins 
	p_label = n_ins_of_label / n_ins

	entropy_cluster = -(p_cluster * p_cluster.log()).mean()
	entropy_label = -(p_label * p_label.log()).mean()

	MI = 0
	for i in range(n_cluster):
		for j in range(n_label):
			pij = cluster_label_table[i, j] / n_ins 
			if pij > 0:
				MI += pij * (pij / (p_cluster[i] * p_label[j] )).log()

	print('MI ', MI)

	NMI = MI / ((entropy_cluster + entropy_label) / 2)
	return NMI


if __name__ == '__main__':
	device = torch.device('cuda')

	with open('embeddings_psnsa.pkl', 'rb') as fil:
		embeddings = pickle.load(fil)
	instance_embeddings = embeddings['instance_embeddings'].to(device)
	concept_prototypes = embeddings['concept_prototypes'].to(device)
	ins_of_con_embeddings = embeddings['ins_of_con_embeddings']
	id2con = embeddings['id2con']
	id2ins = embeddings['id2ins']
	con2id = { v:k for k, v in id2con.items()}
	ins2id = { v:k for k, v in id2ins.items()}



	n_cluster = len(id2con.keys())
	n_ins = len(id2ins.keys())
	n_con = len(id2con.keys())

	data_path = '/mnt/data/nefeli/data/v2_en/{0}_en_dataset.pkl'.format('16_condesc') 

	with open(data_path, 'rb') as fil:
		data_bundle = pickle.load(fil)

	concept_instance_info = data_bundle['concept_instance_info']

	concept_labels = torch.zeros(n_ins, n_con) 

	n_ins_of_con = {}

	for con, insts in concept_instance_info.items():
		i_con = con2id[con]
		for ins in insts:
			i_ins = ins2id[ins]
			concept_labels[i_ins, i_con] = 1 
		n_ins_of_con[i_con] = len(insts)
		#pdb.set_trace()

	cluster_preds = k_means_cluster(instance_embeddings, concept_prototypes)
	
	

	# for calculate ARI 
	#cluster_preds = my_cluster(instance_embeddings, concept_prototypes, n_ins_of_con)
	cluster_label_table = torch.zeros(n_cluster, n_con)
	for i_ins, cluster_pred in enumerate(cluster_preds):
		concept_label = concept_labels[i_ins]
		if not isinstance(cluster_pred, np.int32):
			belong_cluster = cluster_pred.nonzero().squeeze(1)
		else:
			belong_cluster = [cluster_pred]
		belong_concept = concept_label.nonzero().squeeze(1)
		#pdb.set_trace()
		for clus in belong_cluster:
			for con in belong_concept:
				cluster_label_table[clus, con] += 1
		#cluster_label_table[belong_cluster,  belong_concept] += 1

	'''
	from sklearn import metrics
	labels_true = [0, 0, 0, 1, 1, 1]
	labels_pred = [0, 0, 1, 1, 2, 2]
	score = metrics.adjusted_rand_score(labels_true, labels_pred)
	print('sklearn ', score)

	a_cluster_label_table = torch.zeros(3, 2)
	for i_ins, belong_cluster in enumerate(labels_pred):
		belong_concept = labels_true[i_ins]
		a_cluster_label_table[belong_cluster, belong_concept] += 1

	ARI = calc_ARI(a_cluster_label_table, 6)
	print('my ', ARI)
	'''

	#ARI = calc_ARI(cluster_label_table, n_ins)
	NMI = calc_NMI(cluster_label_table, cluster_preds, concept_labels, n_ins)
	print('NMI ', NMI)

	# for K-means & RI 
	RI = calc_RI(cluster_preds, concept_labels, n_con)
	print('RI ', RI)

	#print('ARI ', ARI)

