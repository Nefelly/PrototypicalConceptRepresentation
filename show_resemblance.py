import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pickle
import pdb


with open('bird_family_resemblance.pkl', 'rb') as fil:
	fr = pickle.load(fil)

with open('16_condesc_en_dataset.pkl', 'rb') as fil:
	db = pickle.load(fil)
	instance_info = db['instance_info']


for ins, score in fr:
	print('{0}\t{1:.3f}\t{2}'.format( ins, score*100, instance_info[ins]['text_from_wikipedia'][:200]))
pdb.set_trace()