import math
import os
import pdb
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import roc_curve

include_radius = True
include_distance = True
learn_prototype = False

class ProtProtModel(nn.Module):
    def __init__(self, bert, ent_per_con, concepts, num_labels = 3):
        super().__init__()
        self.ent_per_con = ent_per_con
        self.num_sample_hypo = ent_per_con
        self.num_sample_hyper = ent_per_con
        self.num_labels = 3

        self.hidden_dim = bert.config.hidden_size
        self.prototype_size = bert.config.hidden_size
        self.config = bert.config

        self.bert = bert
        #self.threshold = nn.Parameter(torch.tensor(10.0), requires_grad=True) # 设成10，因为欧几里得距离一般在1e1 - 1e2，设太小一开始跳不出循环
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

        self.activation = nn.Tanh()
        #self.classifier = nn.Linear(self.config.hidden_size * num_sample_hypo, num_labels)

        addtional_dims = 0
        if include_distance:
            addtional_dims += 1
        if include_radius:
            addtional_dims += 2

        #pdb.set_trace()
        self.con_to_idx = { con: ic for ic, con in enumerate(concepts)}
        self.alpha_prototype = 0
        self.learned_prototypes = nn.Parameter(torch.randn(len(concepts), self.prototype_size), requires_grad=learn_prototype)
        self.classifier = nn.Linear(self.config.hidden_size * 4 + addtional_dims, num_labels) #self.config.hidden_size * 4 +
        #pdb.set_trace()
        # 距离无法处理对称性 但分类器可以

        self.prototype_size = 768
        #self.learn_prototype = learn_prototype

        #self.prototypes = nn.Parameter(torch.randn(num_labels, num_prototypes, self.prototype_size), requires_grad=learn_prototype)
        #self.frozen_prototypes = self.prototypes.detach().clone()
        #self.count_prototype_samples = [0 for i in range(num_labels)]


    def forward(
        self,
        hypo_embeddings, hyper_embeddings, hypo, hyper
    ):  
        hypo_embeddings_ = self.dropout(hypo_embeddings)
        hyper_embeddings_ = self.dropout(hyper_embeddings)

        att = hypo_embeddings_.unsqueeze(1).expand(-1, self.num_sample_hyper, -1) * hyper_embeddings_.unsqueeze(0).expand(self.num_sample_hypo, -1, -1)
        att = self.activation(att) / math.sqrt(self.hidden_dim)
        att = att.sum(dim = -1)
        #pdb.set_trace()
        hypo_side_att = torch.softmax(att, dim = -1) # hypo对hyper的attention
        hyper_side_att = torch.softmax(att, dim = 0) # hyper对hypo的attention

        hyper_typicalness = hypo_side_att.sum(dim = 0)  # 对于一个hyper_ent，每个hypo_ent对它的注意力之和，作为这个hyper_ent的典型度
        hyper_typicalness = hyper_typicalness / sum(hyper_typicalness)
        # 同理，对于一个hypo_ent，用每个hyper_ent的注意力之和，作为hypo_ent的典型度
        # hypo_typicalness = hyper_side_att.sum(dim = -1) 
        # 但我不想直接求和，我想用hyper_typicalness，即hyper实体的典型度来加权求和
        # 这里可以多尝试一下 mark
        # hypo_typicalness = (hyper_typicalness * hyper_side_att).sum(dim = -1)
        hypo_typicalness = hyper_side_att.sum(dim = -1) 
        hypo_typicalness = hypo_typicalness / sum(hypo_typicalness)

        attended_hyper_prototypes = torch.matmul(hypo_side_att, hyper_embeddings_) # 每个hypo_ent里，hyper_con的prototype
        hyper_prototype = (attended_hyper_prototypes * hypo_typicalness.unsqueeze(1)).sum(dim=0)

        attended_hypo_prototypes = torch.matmul(hyper_side_att.transpose(0, 1), hypo_embeddings_) # 每个hyper_ent里，hypo_con的prototype
        hypo_prototype = (attended_hypo_prototypes * hyper_typicalness.unsqueeze(1)).sum(dim=0)

        feature_tensors = []

        geoinfo = {}

        if include_radius:
            hypo_shift = (hypo_embeddings_ - hypo_prototype)
            hypo_distance = hypo_shift.square().sum(dim = 1).sqrt()
            hypo_radius = (hypo_distance * hypo_typicalness).sum()

            hyper_shift = (hyper_embeddings_ - hyper_prototype)
            hyper_distance = hyper_shift.square().sum(dim = 1).sqrt()
            hyper_radius = (hyper_distance * hyper_typicalness).sum()

            feature_tensors += [hypo_radius.unsqueeze(0), hyper_radius.unsqueeze(0)]
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

        feature_tensors += [hyper_prototype, hypo_prototype, hyper_prototype - hypo_prototype, hyper_prototype * hypo_prototype]

        pred = self.classifier(torch.cat(feature_tensors)).squeeze()
        if math.isnan(pred.sum()) :
            pdb.set_trace()
        #pred = torch.softmax(pred, dim = 0)
        
        

        return pred, geoinfo

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


        pooled_output = outputs[1]

        return pooled_output

    def set_alpha(self, alpha):
        self.alpha_prototype = alpha


class Bert_Classifier(nn.Module):
    def __init__(self, bert, ent_per_con, num_labels = 3):
        super().__init__()
        self.ent_per_con = ent_per_con
        self.num_sample_hypo = ent_per_con
        self.num_sample_hyper = ent_per_con
        self.num_labels = 3

        self.hidden_dim = bert.config.hidden_size
        self.config = bert.config

        self.bert = bert
        
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

        self.activation = nn.Tanh()

        self.classifier = nn.Linear(self.config.hidden_size * 8, num_labels)

        self.prototype_size = 768


    def forward(
        self,
        hypo_embeddings, hyper_embeddings, hypo, hyper
    ):  
        hypo_embeddings_ = self.dropout(hypo_embeddings)
        hyper_embeddings_ = self.dropout(hyper_embeddings)
        #pdb.set_trace()
        pred = self.classifier(torch.cat([hypo_embeddings_, hyper_embeddings_]).reshape(-1)).squeeze()
        #pdb.set_trace()
        return pred

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


        pooled_output = outputs[1]

        return pooled_output

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