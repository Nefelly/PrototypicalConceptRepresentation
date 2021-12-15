    def generate_concept_prototype(self):

        concept_instance_info = self.data_bundle['concept_instance_info']
        model = self.model
        tokenizer = self.tokenizer
        device = self.device
        hyperparams = self.hyperparams

        model_name = hyperparams['model_name']
        ent_per_con = hyperparams['ent_per_con']

        if hyperparams['language'] == 'cn':
            fixed_num_insts = True
            if hyperparams['use_probase_text']:
                text_key = 'text_from_dbpedia_probase'
            else:
                text_key = 'text_from_dbpedia'
            hypo_hint_template = '该物属于概念"{0}"，判断"{1}"是否被"{2}"包含。'
            single_hint_template = '该物属于概念"{0}"。'

        else:
            fixed_num_insts = False 
            text_key = 'text_from_wikipedia'
            hypo_hint_template = 'This item belongs to concept "{0}". Please judge whether concept "{1}" is contained by concept "{2}". '
            hyper_hint_template = 'This item belongs to concept "{0}". Please judge whether concept "{1}" contain concept "{2}". '
            single_hint_template = 'This item belongs to concept "{0}". '

        concepts = self.concepts #data_bundle['concept_instance_info'].keys()
        instances = self.instances 
        prototype_size = model.prototype_size
        model.eval()
        sampler = Sampler(concept_instance_info, self.instance_info, hyperparams['typicalness'], ent_per_con, 'test', fixed_num_insts)
        concept_info = self.concept_info

        concept_prototypes = torch.zeros(len(concepts), prototype_size).float()
        instance_embeddings = torch.zeros(len(instances), prototype_size).float()

        batch_size = 128
        num_instances = len(instances)
        num_concepts = len(concepts)


        with torch.no_grad():
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
                if not hyperparams['con_desc']:
                    hyper = hypo = con
                    insts = concept_instance_info[con]
                    embed_times = max(math.floor((math.log(len(insts)) / math.log(2))) - 1, 1)
                    #print('Concept {0} Num of Instances: {1} Embed Times: {2}'.format(con, len(insts), embed_times))

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
                        prototype = res['prototype']
                        sum_prot += prototype 

                    prototype = sum_prot / embed_times
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
            'id2con': self.id2con,
            'id2ins': self.id2ins
        }
        model.train()
        with open('embeddings.pkl', 'wb') as fil:
            pickle.dump(embeddings, fil)

        return embeddings