import pickle
import pdb

with open('/mnt/data/nefeli/data/v2_en/{0}_en_dataset.pkl'.format('16'), 'rb') as fil:
	#pdb.set_trace()
	db16 = pickle.load(fil)

with open('/mnt/data/nefeli/data/v2_en/{0}_en_dataset.pkl'.format('16_condesc'), 'rb') as fil:
	db16c = pickle.load(fil)

con = db16['concepts'].difference(db16c['concepts'])
instance_info = db16['concept_instance_info']
con_ins = {c: len(instance_info[c]) for c in con}
sort_con = sorted(list(con_ins.items()), key = lambda x:x[1], reverse = True)
pdb.set_trace()