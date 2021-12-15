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

data_path = '/mnt/data/nefeli/data/v2_en/{0}_en_dataset_53.pkl'.format("16_condesc") 
# 用53上的新数据集来做，不然test、train分割不一样，138上test里的数据大多数在53的train里

with open(data_path, 'rb') as fil:
    data_bundle = pickle.load(fil)

concepts = data_bundle['concepts']
instances = data_bundle['instances']

con2id = { con: ic for ic, con in enumerate(sorted(concepts))}
id2con = { ic: con for ic, con in enumerate(sorted(concepts))}
ins2id = { ins: ic for ic, ins in enumerate(sorted(instances))}
id2ins = { ic: ins for ic, ins in enumerate(sorted(instances))}

allscores_sub_path = ''
allscores_ins_path = ''
allscores_sub = {}
allscores_ins = {}

def link_prediction(valid=True):

        global ins2id
        global id2ins
        concept_instance_info = data_bundle['concept_instance_info']
        #model = self.model
        #tokenizer = self.tokenizer
        device = torch.device('cuda')
        #hyperparams = self.hyperparams
        
        
        #prototype_size = model.prototype_size

        type_constrain = True#self.hyperparams['type_constrain']
        
        #pdb.set_trace()
        #embeddings = self.generate_concept_prototype()

        #with open('embeddings.pkl', 'rb') as fil:
        #    embeddings = pickle.load(fil)

        #instance_embeddings = embeddings['instance_embeddings'].to(device)
        #concept_prototypes = embeddings['concept_prototypes'].to(device)

        #batch_size = 16

        #id2con = self.id2con
        #con2id = self.con2id

        #id2ins = self.id2ins 
        #ins2id = self.ins2id

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
            isSubclassOf_triples = { 'all': data_bundle['isSubclassOf_triples'], 'test': data_bundle['subclass_valid'] }
            isInstanceOf_triples = { 'all': data_bundle['isInstanceOf_triples'], 'test': data_bundle['instance_valid'] }
        else:
            isSubclassOf_triples = { 'all': data_bundle['isSubclassOf_triples'], 'test': data_bundle['subclass_test'] }
            isInstanceOf_triples = { 'all': data_bundle['isInstanceOf_triples'], 'test': data_bundle['instance_test'] }
            # 忘了把valid时link prediction的设为valid集合了，那只能在test时用valid集合了

        #pdb.set_trace()

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


        for split in ['all', 'test']:
            for triple in isSubclassOf_triples[split]:
                hypo, hyper, label = triple 
                Groundtruth['subclass']['tail'][split][hypo].add(hyper)
                Groundtruth['subclass']['head'][split][hyper].add(hypo)

            for triple in isInstanceOf_triples[split]:
                hypo, hyper, label = triple 

                Groundtruth['instance']['tail'][split][hypo].add(hyper)
                Groundtruth['instance']['head'][split][hyper].add(hypo)

        #pdb.set_trace()

        #for rel in ['subclass', 'instance']:
        #    for target in ['tail', 'head']:
        #        for k in Groundtruth[rel][target]['all'].keys():
        #            assert len(Groundtruth[rel][target]['test'][k].difference(Groundtruth[rel][target]['all'][k])) == 0

    

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
                                    #prototype = concept_prototypes[igiv]
                                else:
                                    igiv = ins2id[giv]
                                    #if type_constrain:
                                    #    prototype = instance_embeddings[igiv]
                                    #else:
                                    #    prototype = instance_embeddings[igiv-num_concepts]

                                if type_constrain:
                                    if target_type == 'concept':
                                        candidate_idxs = Candidates[rel][target]#[ con2id[c] for c in Candidates[rel][target]]
                                        #all_candidates = concept_prototypes[candidate_idxs]
                                        target_size = len(candidate_idxs)
                                    elif target_type == 'instance':
                                        candidate_idxs = Candidates[rel][target]#[ ins2id[c] for c in Candidates[rel][target]]
                                        #all_candidates = instance_embeddings[candidate_idxs]
                                        target_size = len(candidate_idxs)
                                    
                                    candidate_maps = {c:i for i, c in enumerate(candidate_idxs)}
                                    #pdb.set_trace()
                                    '''
                                    if target_type == 'concept':
                                        all_candidates = concept_prototypes
                                        target_size = len(concepts)
                                    elif target_type == 'instance':
                                        all_candidates = instance_embeddings
                                        target_size = len(instances)
                                    '''
                                    
                                else:
                                    #pdb.set_trace()
                                    #all_candidates = torch.cat([concept_prototypes, instance_embeddings], dim=0)
                                    target_size = len(concepts) + len(instances)

                                
                                if target == 'head':
                                    if rel == 'subclass':
                                        score = [allscores_sub[(cand, igiv)] for cand in candidate_idxs]
                                    elif rel == 'instance':
                                        score = [allscores_ins[(cand, igiv)] for cand in candidate_idxs]
                                else:
                                    if rel == 'subclass':
                                        score = [allscores_sub[(igiv, cand)] for cand in candidate_idxs]
                                    elif rel == 'instance':
                                        score = [allscores_ins[(igiv, cand)] for cand in candidate_idxs]   

                                score = torch.tensor(score)
                                tops = score.argsort(descending=False).tolist()
                                #pdb.set_trace()
                                '''
                                if not hyperparams['distance_metric']:
                                    if target == 'head':
                                        feature_tensors = [prototype.expand(target_size, prototype_size), all_candidates, prototype - all_candidates, prototype * all_candidates]
                                    else:
                                        feature_tensors = [all_candidates, prototype.expand(target_size, prototype_size), all_candidates - prototype, all_candidates * prototype]

                                    #pdb.set_trace()
                                    if model_name == 'psn':
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
                                        rel_embedding = model.isA_embedding(torch.tensor(0).to(device))
                                    else:
                                        rel_embedding = model.isA_embedding(torch.tensor(1).to(device))
                                    if target == 'head':
                                        feature_tensors = all_candidates + rel_embedding - prototype
                                    else:
                                        feature_tensors = prototype + rel_embedding - all_candidates
                                    distance = torch.norm(feature_tensors, p=1, dim=-1) #feature_tensors.abs().sum(dim=1)
                                    #pdb.set_trace()
                                    score = -distance 
                                    tops = score.argsort(descending=True).tolist()
                                '''

                                # e.g. tops = [2470, 2606,  954,  ..., 2566, 1346,  262], 即第2470个概念分数最高，第2606个概念分数其次 
                                #pdb.set_trace()
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
                            setting, target, rel, type_constrain
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

        return fil_subclass_mrr


best_subclass_mrr = 0 
best_epoch = 0

epcs = [820, 840, 860, 880, 900, 920, 940, 960, 980]#[900]#[1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
#[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300]
for epc in epcs:
    print("Epoch ",epc)
    allscores_sub_path = '/mnt/data/nefeli/reproduction/transc/src/allscores_sub_{}.txt'.format(epc)
    allscores_ins_path = '/mnt/data/nefeli/reproduction/transc/src/allscores_ins_{}.txt'.format(epc)
    allscores_sub = {}
    allscores_ins = {}

    with open(allscores_sub_path, 'r') as fil:
        for line in fil.readlines():
            line = line.strip('\n')
            h, t, score = line.split('\t')
            allscores_sub[(int(h), int(t))] = float(score)

    with open(allscores_ins_path, 'r') as fil:
        for line in fil.readlines():
            line = line.strip('\n')
            h, t, score = line.split('\t')
            allscores_ins[(int(h), int(t))] = float(score)

    fil_subclass_mrr = link_prediction()
    if best_subclass_mrr < fil_subclass_mrr:
        best_subclass_mrr = fil_subclass_mrr
        best_epoch = epc 
        print('New best mrr at epoch ', epc)


epc = best_epoch
print('Reload Best epc ', epc)
if 1:
    allscores_sub_path = '/mnt/data/nefeli/reproduction/transc/src/allscores_sub_{}.txt'.format(epc)
    allscores_ins_path = '/mnt/data/nefeli/reproduction/transc/src/allscores_ins_{}.txt'.format(epc)
    allscores_sub = {}
    allscores_ins = {}

    with open(allscores_sub_path, 'r') as fil:
        for line in fil.readlines():
            line = line.strip('\n')
            h, t, score = line.split('\t')
            allscores_sub[(int(h), int(t))] = float(score)

    with open(allscores_ins_path, 'r') as fil:
        for line in fil.readlines():
            line = line.strip('\n')
            h, t, score = line.split('\t')
            allscores_ins[(int(h), int(t))] = float(score)

    # Run test
    fil_subclass_mrr = link_prediction(valid=False)