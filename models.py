import math
import os
import pdb
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import numpy as np
#import pandas as pd
import pickle
import random
#from sklearn.metrics import roc_curve

include_radius = False#True#False#True #False
include_distance = False#True#False#True
learn_prototype = False



class ProtSiam(nn.Module):
    def __init__(self, bert, ent_per_con, concepts, train_instance_of = False, freeze_plm = False, num_concepts = 0, num_instances = 0, 
        num_labels = 3, separate_classifier = False, train_MLM = False, distance_metric = False, bertMLMcls = None, use_cls_token=False):
        super().__init__()

        self.hidden_dim = bert.config.hidden_size
        self.prototype_size = bert.config.hidden_size
        self.config = bert.config
        self.train_instance_of = train_instance_of
        self.freeze_plm = freeze_plm
        self.use_cls_token = use_cls_token
        self.separate_classifier = separate_classifier
        self.train_MLM_ = train_MLM
        self.bertMLMcls = bertMLMcls
        self.distance_metric = distance_metric

        self.bert = bert
        #self.threshold = nn.Parameter(torch.tensor(10.0), requires_grad=True) # 设成10，因为欧几里得距离一般在1e1 - 1e2，设太小一开始跳不出循环
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

        self.activation = nn.Tanh()
        #self.classifier = nn.Linear(self.config.hidden_size * num_sample_hypo, num_labels)

        additional_dims = 0
        if include_distance:
            additional_dims += 1
        if include_radius:
            additional_dims += 2

        self.con_to_idx = { con: ic for ic, con in enumerate(concepts)}
        self.alpha_prototype = 0
        self.learned_prototypes = nn.Parameter(torch.randn(len(concepts), self.prototype_size), requires_grad=learn_prototype)
        
        self.sub_classifier = nn.Linear(self.config.hidden_size * 4 + additional_dims, num_labels) #self.config.hidden_size * 4 +
        #self.ins_classifier = nn.Linear(self.config.hidden_size * 4 + additional_dims, num_labels)
        self.sub_selfatt_classifier = nn.Linear(self.config.hidden_size * 4, num_labels)
        self.ins_selfatt_classifier = nn.Linear(self.config.hidden_size * 4, num_labels)

        if self.train_instance_of:
            self.ins_classifier = nn.Linear(self.config.hidden_size * 4 + additional_dims, num_labels)
        
        #self.learn_prototype = learn_prototype
        #self.prototypes = nn.Parameter(torch.randn(num_labels, num_prototypes, self.prototype_size), requires_grad=learn_prototype)
        #self.frozen_prototypes = self.prototypes.detach().clone()#
        #self.count_prototype_samples = [0 for i in range(num_labels)]
        self.count = 0

        self.dense = nn.Linear(bert.config.hidden_size, bert.config.hidden_size)

        self.instance_embeddings = nn.Embedding(num_instances, self.prototype_size)
        
        if freeze_plm:
            self.instance_embeddings = nn.Embedding(num_instances, self.prototype_size)
            self.concept_embeddings = nn.Embedding(num_concepts, self.prototype_size)
        
        if self.distance_metric:
            self.isA_embedding = nn.Embedding(2, self.prototype_size)#, max_norm = 10, norm_type = 1)#Parameter(torch.randn(1, self.prototype_size), requires_grad=True)
            self.isA_embedding.weight.data.uniform_(-0.5, 0.5)
            #pdb.set_trace()
            #if self.train_instance_of:
            #    self.isA_embedding[1] = nn.Embedding(1, self.prototype_size)#Parameter(torch.randn(1, self.prototype_size), requires_grad=True)


    def forward(
        self,
        hypo_embeddings, hyper_embeddings, hypo=None, hyper=None, mode = 'subclass'
    ):  
        # mode -> 'subclass' 单独train subclass; 
        #         'subclass_instance', train subclass，且同时计算每个instance和对方concept的关系
        #         'instance' 单独train instance
        device = hypo_embeddings.device
        #pdb.set_trace()

        hypo_embeddings_ = self.dropout(hypo_embeddings)
        hyper_embeddings_ = self.dropout(hyper_embeddings)
        
        num_sample_hyper = hyper_embeddings_.shape[0]
        num_sample_hypo = hypo_embeddings_.shape[0]

        att = hypo_embeddings_.unsqueeze(1).expand(-1, num_sample_hyper, -1) * hyper_embeddings_.unsqueeze(0).expand(num_sample_hypo, -1, -1)
        att = self.activation(att) / math.sqrt(self.hidden_dim)
        att = att.sum(dim = -1)


        hypo_side_att = torch.softmax(att, dim = -1) # hypo对hyper的attention
        hyper_side_att = torch.softmax(att, dim = 0) # hyper对hypo的attention

        hyper_typicalness = hypo_side_att.sum(dim = 0)  # 对于一个hyper_ent，每个hypo_ent对它的注意力之和，作为这个hyper_ent的典型度
        

        assert sum(hyper_typicalness) != 0

        hyper_typicalness = hyper_typicalness / sum(hyper_typicalness)
        # 同理，对于一个hypo_ent，用每个hyper_ent的注意力之和，作为hypo_ent的典型度
        # hypo_typicalness = hyper_side_att.sum(dim = -1) 
        # 但我不想直接求和，我想用hyper_typicalness，即hyper实体的典型度来加权求和
        # 这里可以多尝试一下 mark
        # hypo_typicalness = (hyper_typicalness * hyper_side_att).sum(dim = -1)
        hypo_typicalness = hyper_side_att.sum(dim = -1) 
        assert sum(hypo_typicalness) != 0
        hypo_typicalness = hypo_typicalness / sum(hypo_typicalness)


        attended_hyper_prototypes = torch.matmul(hypo_side_att, hyper_embeddings_) # 每个hypo_ent里，hyper_con的prototype
        hyper_prototype = (attended_hyper_prototypes * hypo_typicalness.unsqueeze(1)).sum(dim=0)

        attended_hypo_prototypes = torch.matmul(hyper_side_att.transpose(0, 1), hypo_embeddings_) # 每个hyper_ent里，hypo_con的prototype
        hypo_prototype = (attended_hypo_prototypes * hyper_typicalness.unsqueeze(1)).sum(dim=0)

        if mode in ['subclass_selfatt', 'instance_selfatt']:
            res = {'prototype': (hyper_prototype + hypo_prototype) /2, 
                'family_resemblance': (hypo_typicalness + hyper_typicalness)/2 }
            return res

        feature_tensors = []

        geoinfo = {}

        if include_radius:
            hypo_shift = (hypo_embeddings_ - hypo_prototype)
            hypo_distance = hypo_shift.square().sum(dim = 1).sqrt()

            if hypo_distance.sum().item() == 0:
                hypo_distance = hypo_distance.detach()

            hypo_radius = (hypo_distance * hypo_typicalness).sum()

            hyper_shift = (hyper_embeddings_ - hyper_prototype)
            hyper_distance = hyper_shift.square().sum(dim = 1).sqrt()
            if hyper_distance.sum().item() == 0:
                hyper_distance = hyper_distance.detach()
            hyper_radius = (hyper_distance * hyper_typicalness).sum()

            feature_tensors += [ hyper_radius.unsqueeze(0), hypo_radius.unsqueeze(0)] # 之前的顺序是先hypo_radius.unsqueeze(0)..这个不会影响吧？
            geoinfo['hypo_radius'] = hypo_radius
            geoinfo['hyper_radius'] = hyper_radius


        if learn_prototype:
            if self.alpha_prototype < 1:
                hypo_prototype = (1-self.alpha_prototype)*hypo_prototype + self.alpha_prototype*self.learned_prototypes[self.con_to_idx[hypo]]
                hyper_prototype = (1-self.alpha_prototype)*hyper_prototype + self.alpha_prototype*self.learned_prototypes[self.con_to_idx[hyper]]
            else:
                hypo_prototype = self.learned_prototypes[self.con_to_idx[hypo]]
                hyper_prototype = self.learned_prototypes[self.con_to_idx[hyper]]

        euclidean_distance = (hypo_prototype - hyper_prototype).square().sum(dim=-1).sqrt()
        #pred = torch.sigmoid(self.threshold - euclidean_distance)
        if include_distance:
            feature_tensors.append(euclidean_distance.unsqueeze(0))
            geoinfo['euclidean_distance'] = euclidean_distance

        
        
        self.count += 1
        
        if not self.distance_metric:
            #pdb.set_trace()
            feature_tensors += [hyper_prototype, hypo_prototype, hyper_prototype - hypo_prototype, hyper_prototype * hypo_prototype]
            if mode == 'instance' and self.separate_classifier:
                #sub_pred = self.ins_classifier(torch.cat(feature_tensors)).squeeze() #回来再改这个问题! 要将距离计算也加上
                sub_pred = self.ins_classifier(torch.cat(feature_tensors)).squeeze()
                res = {'sub_pred': sub_pred}
            else:
                sub_pred = self.sub_classifier(torch.cat(feature_tensors)).squeeze()
                res = {'sub_pred': sub_pred}
        else:
            if mode == 'instance':
                rel_embedding = self.isA_embedding(torch.tensor(1).to(device))
            else:
                rel_embedding = self.isA_embedding(torch.tensor(0).to(device))
            offset = (hypo_prototype + rel_embedding - hyper_prototype)
            
            distance = torch.norm(offset, p=1, dim = -1) #offset.abs().sum() # L1距离
            res = {'distance': distance}

        #torch.set_printoptions(precision=6)
        #pdb.set_trace()

        return res


    def judge(self, hypo_prototype, hyper_prototype, mode):
        if not self.distance_metric:
            feature_tensors = [hyper_prototype, hypo_prototype, hyper_prototype - hypo_prototype, hyper_prototype * hypo_prototype]
            if mode == 'instance_selfatt' and self.separate_classifier:
                sub_pred = self.ins_selfatt_classifier(torch.cat(feature_tensors)).squeeze()
            else:
                sub_pred = self.sub_selfatt_classifier(torch.cat(feature_tensors)).squeeze()
            res = {'sub_pred': sub_pred}
        else:
            if mode == 'instance_selfatt':
                rel_embedding = self.isA_embedding(torch.tensor(1).to(self.bert.device))
            else:
                rel_embedding = self.isA_embedding(torch.tensor(0).to(self.bert.device))
            #offset = (hypo_prototype + rel_embedding - hyper_prototype)
            offset = (hypo_prototype - hyper_prototype) # prototypical network
            #pdb.set_trace()

            distance = torch.norm(offset, p=2, dim = -1) # offset.abs().sum()
            res = {'distance': distance}

        return res

    def bert_embed(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):  
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
        )

        if self.use_cls_token:
            pooled_output = outputs[0][:, 0, :]
        else:
            pooled_output = outputs[1]

        return pooled_output

    def frozen_bert_embed(self, insts_idxs):
        #pdb.set_trace()
        insts_embeddings = self.instance_embeddings(insts_idxs)
        pooled_output = self.dense(insts_embeddings)
        #pooled_output = self.activation(insts_embeddings)
        pooled_output = self.activation(pooled_output)

        return pooled_output

    def frozen_bert_embed_concepts(self, cons_idxs):
        cons_embeddings = self.concept_embeddings(cons_idxs)
        pooled_output = self.dense(cons_embeddings)
        pooled_output = self.activation(pooled_output)

        return pooled_output

    def set_alpha(self, alpha):
        self.alpha_prototype = alpha

    def train_MLM(self, max_length, tokenizer, texts):

        
        predict_ratio = 0.15
        mask_ratio = 0.8
        random_ratio = 0.1
        unchanged_ratio = 0.1

        vocab = tokenizer.vocab.keys()
        vocab_size = len(vocab)

        predict_positions_list = []
        predict_labels_list = []

        device = self.bert.device

        inputs = tokenizer(texts, max_length = max_length, truncation=True, return_tensors='pt', padding=True)

        for it, text in enumerate(texts):
            #tokens = tokenizer.convert_ids_to_tokens(tokenizer(text, max_length = max_length, truncation=True)['input_ids'])
            tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][it].tolist())
            #pdb.set_trace()
            num_tokens = tokens.index('[SEP]')+1#len(tokens)
            num_predicts = int((num_tokens-2) * predict_ratio) # 去掉CLS和SEP
            predict_positions = sorted(random.sample(range(1, num_tokens-2), num_predicts))
            predict_positions_list.append(predict_positions)

            aa = [tokens[p] for p in predict_positions]

            
            predict_labels = tokenizer.convert_tokens_to_ids([tokens[p] for p in predict_positions])
            if 0 in predict_labels:
                pdb.set_trace()
            predict_labels_list.append(predict_labels)


            mask_positions = random.sample(predict_positions, int(len(predict_positions) * mask_ratio))
            remaining = set(predict_positions).difference(mask_positions)
            random_positions = random.sample(remaining, int(len(remaining) * (random_ratio / (random_ratio + unchanged_ratio)) ))
            unchanged_positions = set(remaining).difference(random_positions)
            
            for mask_position in mask_positions:
                tokens[mask_position] = '[MASK]'
            for random_position in random_positions:
                random_word = random.sample(vocab, 1)[0]
                tokens[random_position] = random_word

            inputs['input_ids'][it] = torch.tensor(tokenizer.convert_tokens_to_ids(tokens))
        #pdb.set_trace()
        inputs.to(device)

        outputs = self.bert(**inputs)
        sequence_output = outputs[0]
        prediction_scores_3d = self.bertMLMcls(sequence_output)

        all_mlm_loss = 0
        total_num_predicts = 0
        for it in range(len(texts)):
            prediction_scores = prediction_scores_3d[it]
            predict_positions = predict_positions_list[it]
            predict_position_scores = prediction_scores[predict_positions]
            predict_labels = predict_labels_list[it]

            num_predicts = predict_position_scores.shape[0]
            total_num_predicts += num_predicts

            loss_fct = CrossEntropyLoss()
            #pdb.set_trace()
            masked_lm_loss = loss_fct(predict_position_scores, torch.tensor(predict_labels).to(device))
            all_mlm_loss += masked_lm_loss * num_predicts
            
        all_mlm_loss /= total_num_predicts
        
        return all_mlm_loss



class ProtVanilla(nn.Module):
    def __init__(self, bert, ent_per_con, train_instance_of = False, freeze_plm = False, num_concepts = 0, num_instances = 0,
         num_labels = 3, separate_classifier = False, 
         train_MLM = False, distance_metric = False, bertMLMcls = None, use_cls_token=False):
        super().__init__()

        self.hidden_dim = bert.config.hidden_size
        self.config = bert.config
        self.bert = bert
        self.use_cls_token = use_cls_token
        self.separate_classifier = separate_classifier
        self.train_MLM_ = train_MLM
        self.bertMLMcls = bertMLMcls
        self.distance_metric = distance_metric
        
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

        self.dense = nn.Linear(bert.config.hidden_size, bert.config.hidden_size)
        self.activation = nn.Tanh()

        #pdb.set_trace()
        self.sub_classifier = nn.Linear(self.config.hidden_size * 4, num_labels)
        

        self.prototype_size = self.hidden_dim

        self.freeze_plm = freeze_plm

        if freeze_plm:
            self.instance_embeddings = nn.Embedding(num_instances, self.prototype_size)
            self.concept_embeddings = nn.Embedding(num_concepts, self.prototype_size)

        self.train_instance_of = train_instance_of
        if train_instance_of:
            self.ins_classifier = nn.Linear(self.config.hidden_size * 4, num_labels)

        if self.distance_metric:
            self.isA_embedding = nn.Embedding(2, self.prototype_size)#Parameter(torch.randn(1, self.prototype_size), requires_grad=True)
            self.isA_embedding.weight.data.uniform_(-0.5, 0.5)


    def forward(
        self,
        hypo_embeddings, hyper_embeddings, hypo=None, hyper=None, mode = 'subclass'
    ):  
        device = hypo_embeddings.device

        hypo_embeddings_ = self.dropout(hypo_embeddings)
        hyper_embeddings_ = self.dropout(hyper_embeddings)

        hypo_prototype = hypo_embeddings_.mean( dim = 0)
        hyper_prototype = hyper_embeddings_.mean( dim = 0)


        if mode in ['subclass_selfatt', 'instance_selfatt']:
            res = {'prototype': hyper_prototype } #(hyper_prototype + hypo_prototype) /2}
            return res
        
        if not self.distance_metric:
            if mode == 'instance' and self.separate_classifier:
                pred = self.ins_classifier(torch.cat([hyper_prototype, hypo_prototype, hyper_prototype - hypo_prototype, hyper_prototype * hypo_prototype]).reshape(-1)).squeeze()
            else:
                pred = self.sub_classifier(torch.cat([hyper_prototype, hypo_prototype, hyper_prototype - hypo_prototype, hyper_prototype * hypo_prototype]).reshape(-1)).squeeze()
            
            res = {'sub_pred': pred}
            
        else:
            if mode == 'instance':
                rel_embedding = self.isA_embedding(torch.tensor(1).to(device))
            else:
                rel_embedding = self.isA_embedding(torch.tensor(0).to(device))
            offset = (hypo_prototype + rel_embedding - hyper_prototype)
            distance = torch.norm(offset, p=1, dim=1)#offset.abs().sum() # L1距离
            res = {'distance': distance}

        return res

    def judge(self, hypo_prototype, hyper_prototype, mode):
        if not self.distance_metric:
            feature_tensors = [hyper_prototype, hypo_prototype, hyper_prototype - hypo_prototype, hyper_prototype * hypo_prototype]
            if mode == 'instance_selfatt' and self.separate_classifier:
                sub_pred = self.ins_classifier(torch.cat(feature_tensors)).squeeze()
            else:
                sub_pred = self.sub_classifier(torch.cat(feature_tensors)).squeeze()
            res = {'sub_pred': sub_pred}
        else:
            if mode == 'instance_selfatt':
                rel_embedding = self.isA_embedding(torch.tensor(1).to(self.bert.device))
            else:
                rel_embedding = self.isA_embedding(torch.tensor(0).to(self.bert.device))
            offset = (hypo_prototype + rel_embedding - hyper_prototype)

            distance = torch.norm(offset, p=1, dim = -1) # offset.abs().sum()
            res = {'distance': distance}

        return res

    def bert_embed(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):  

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
        )

        if self.use_cls_token:
            pooled_output = outputs[0][:, 0, :]
        else:
            pooled_output = outputs[1]

        return pooled_output

    def frozen_bert_embed(self, insts_idxs):
        #pdb.set_trace()
        insts_embeddings = self.instance_embeddings(insts_idxs)
        pooled_output = self.dense(insts_embeddings)
        #pooled_output = self.activation(insts_embeddings)
        pooled_output = self.activation(pooled_output)

        return pooled_output

    def frozen_bert_embed_concepts(self, cons_idxs):
        cons_embeddings = self.concept_embeddings(cons_idxs)
        pooled_output = self.dense(cons_embeddings)
        pooled_output = self.activation(pooled_output)

        return pooled_output

    def set_alpha(self, alpha):
        pass
    def train_MLM(self, max_length, tokenizer, texts):

        predict_ratio = 0.15
        mask_ratio = 0.8
        random_ratio = 0.1
        unchanged_ratio = 0.1

        vocab = tokenizer.vocab.keys()
        vocab_size = len(vocab)

        predict_positions_list = []
        predict_labels_list = []

        device = self.bert.device

        inputs = tokenizer(texts, max_length = max_length, truncation=True, return_tensors='pt', padding=True)

        for it, text in enumerate(texts):
            #tokens = tokenizer.convert_ids_to_tokens(tokenizer(text, max_length = max_length, truncation=True)['input_ids'])
            tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][it].tolist())
            #pdb.set_trace()
            num_tokens = tokens.index('[SEP]')+1#len(tokens)
            num_predicts = int((num_tokens-2) * predict_ratio) # 去掉CLS和SEP
            predict_positions = sorted(random.sample(range(1, num_tokens-2), num_predicts))
            predict_positions_list.append(predict_positions)

            aa = [tokens[p] for p in predict_positions]

            
            predict_labels = tokenizer.convert_tokens_to_ids([tokens[p] for p in predict_positions])
            if 0 in predict_labels:
                pdb.set_trace()
            predict_labels_list.append(predict_labels)


            mask_positions = random.sample(predict_positions, int(len(predict_positions) * mask_ratio))
            remaining = set(predict_positions).difference(mask_positions)
            random_positions = random.sample(remaining, int(len(remaining) * (random_ratio / (random_ratio + unchanged_ratio)) ))
            unchanged_positions = set(remaining).difference(random_positions)
            
            for mask_position in mask_positions:
                tokens[mask_position] = '[MASK]'
            for random_position in random_positions:
                random_word = random.sample(vocab, 1)[0]
                tokens[random_position] = random_word

            inputs['input_ids'][it] = torch.tensor(tokenizer.convert_tokens_to_ids(tokens))
        #pdb.set_trace()
        inputs.to(device)

        outputs = self.bert(**inputs)
        sequence_output = outputs[0]
        prediction_scores_3d = self.bertMLMcls(sequence_output)

        all_mlm_loss = 0
        total_num_predicts = 0
        for it in range(len(texts)):
            prediction_scores = prediction_scores_3d[it]
            predict_positions = predict_positions_list[it]
            predict_position_scores = prediction_scores[predict_positions]
            predict_labels = predict_labels_list[it]


            num_predicts = predict_position_scores.shape[0]
            if num_predicts == 0:
                continue

            total_num_predicts += num_predicts

            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(predict_position_scores, torch.tensor(predict_labels).to(device))

            all_mlm_loss += masked_lm_loss * num_predicts
        
        if total_num_predicts != 0:
            all_mlm_loss /= total_num_predicts
        else:
            pdb.set_trace()
            
        return all_mlm_loss

def subclass_loss(geoinfo, label):
    euclidean_distance, hypo_radius, hyper_radius = geoinfo['euclidean_distance'], geoinfo['hypo_radius'], geoinfo['hyper_radius']
    if label == 0:
        loss = max((hyper_radius - hypo_radius).abs() - euclidean_distance, torch.tensor(0).float().to(hypo_radius.device))
        # 如果相容，那么 (hyper_radius - hypo_radius).abs() - euclidean_distance > 0
        # 那么 不相容，要求 (hyper_radius - hypo_radius).abs() - euclidean_distance < 0，越小越好
        # 如果loss很大，说明euclidean_distance很小，不好！
    elif label == 1:
        # hypo 属于 hyper，希望hypo被hyper包含，
        # 即，希望 euclidean_distance + hypo_radius < hyper_radius，且hypo_radius < hyper_radius
        # 这已经蕴含了 hypo_radius < hyper_radius
        loss = max(euclidean_distance + hypo_radius - hyper_radius, torch.tensor(0).float().to(hypo_radius.device))
    else:
        loss = max(euclidean_distance + hyper_radius - hypo_radius, torch.tensor(0).float().to(hypo_radius.device))
    
    return loss

