import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pdb

def visualize_attention(att, cmap='Greys', rows=None, columns=None):
	
	att = att.cpu().numpy()

	try:
		df = pd.DataFrame(att, index = rows, columns=columns)
	except:
		pdb.set_trace()
	f,ax = plt.subplots()
	ax.set_xticklabels(df, rotation='horizontal')

	sns.heatmap(df, annot=False, cmap=cmap, square =True)
	if rows:
		label_y = ax.get_yticklabels()
		plt.setp(label_y , rotation = 90, va='center')

	label_x = ax.get_xticklabels()
	plt.setp(label_x , rotation = 0)
	plt.show()

#Hint
att = torch.tensor([[0.571245, 0.461286, 0.540925, 0.517595],
        [0.511843, 0.361664, 0.470620, 0.428832],
        [0.528216, 0.395673, 0.490528, 0.459062],
        [0.420098, 0.251036, 0.388258, 0.319061]], device='cuda:0')

#No hint
att = torch.tensor([[0.406745, 0.272499, 0.330201, 0.422996],
        [0.199912, 0.177528, 0.140920, 0.221043],
        [0.416689, 0.214321, 0.266601, 0.308860],
        [0.217310, 0.064113, 0.218197, 0.181564]], device='cuda:0')

rows = ['tennis', 'angling', 'basketball', 'yoga']
columns = ['football', 'angry birds', 'boxing', 'mahjong']

hypo_side_att = torch.softmax(att, dim = -1) # hypo对hyper的attention
hyper_side_att = torch.softmax(att, dim = 0) # hyper对hypo的attention

hyper_typicalness = hypo_side_att.sum(dim = 0)
hyper_typicalness = hyper_typicalness / sum(hyper_typicalness)

hypo_typicalness = hyper_side_att.sum(dim = -1) 
hypo_typicalness = hypo_typicalness / sum(hypo_typicalness)



pdb.set_trace()
visualize_attention(att, 'Greys', rows, columns)
visualize_attention(hypo_side_att, "YlGnBu", rows, columns)
visualize_attention(hyper_side_att, "Greens", rows, columns)
visualize_attention(hypo_typicalness.unsqueeze(0), "YlGnBu", columns=rows)
visualize_attention(hyper_typicalness.unsqueeze(0), "Greens", columns=columns)
