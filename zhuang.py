import os
import argparse
from tqdm import tqdm
import torch
import pdb
import random
import pickle
import numpy as np
import csv
import re 

data_path = '/home/nefeli/data/v2_en/{0}_en_dataset.pkl'.format("16_condesc")
with open(data_path, 'rb') as fil:
	data_bundle = pickle.load(fil)

#pdb.set_trace()
#con2id = data_bundle['con2id']
#ins2id = data_bundle['ins2id']
ins2id = { ins: ic for ic, ins in enumerate(sorted(data_bundle['instances']))}
con2id = { con: ic for ic, con in enumerate(sorted(data_bundle['concepts']))}

#with open('embeddings.pkl', 'rb') as fil:
#	yy = pickle.load(fil)
#pdb.set_trace()
concepts = data_bundle['concepts']
instances = data_bundle['instances']

ignore = ['a', 'the', 'will', 'new']
concepts=concepts.difference(set(ignore))
instances=instances.difference(set(ignore))

cnt = { e: 0 for e in (concepts.union(instances))}

def search_concepts_instances(text, concepts, instances):
	cons = []
	inss = []
	for con in concepts:
		if con in text.lower() and re.search(r"\b{}\b".format(con), text.lower().strip()):#con in text:
			cons.append(con2id[con])
			cnt[con] += 1
			#if len(con.split(' ')) >= 2:
			#	pdb.set_trace()
	for ins in instances:
		if ins in text.lower() and re.search(r"\b{}\b".format(ins), text.lower().strip()):#ins in text:
			inss.append(ins2id[ins])
			cnt[ins] += 1
	
	return cons, inss
	#if (len(cons)+len(inss) > 0):
	#	pdb.set_trace() 

train_texts = []
test_texts = []
count = 0
with open('./ag_news_csv/train.csv') as f:
	f_train = csv.reader(f)
	for row in tqdm(f_train):
		count += 1
		text = row[1]+'\t'+row[2]
		cons, inss = search_concepts_instances(text, concepts, instances)
		if (len(cons)+len(inss) > 0):
			sample = {'text': text, 'concepts': cons, 'instances': inss, 'label': row[0], 'title':row[1], 'content':row[2]}
			train_texts.append(sample)
		
		if count == 1000:
			aa = sorted(cnt.items(), key=lambda x:x[1], reverse=True)
			pdb.set_trace()

with open('./ag_news_csv/test.csv') as f:
	f_test = csv.reader(f)
	for row in f_test:
		text = row[1]+'\t'+row[2]
		cons, inss = search_concepts_instances(text, concepts, instances)
		if (len(cons)+len(inss) > 0):
			sample = {'text': text, 'concepts': cons, 'instances': inss, 'label': row[0], 'title':row[1], 'content':row[2]}
			test_texts.append(sample)


data_bundle = {'train': train_texts, 'test': test_texts}
with open('matched_agnews.pkl', 'wb') as fil:
	pickle.dump(data_bundle, fil)

pdb.set_trace()

aa = sorted(cnt.items(), key=lambda x:x[1], reverse=True)
pdb.set_trace()

## 没写好，多词的，不好搞...