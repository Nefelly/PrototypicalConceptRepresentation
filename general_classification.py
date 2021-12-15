import numpy as np

def run_triple_classification(self, epc, valid=True, threshold = None, for_threshold=False):
        self.lib.initTest()
        
        self.data_loader.set_sampling_mode('classification')
        
        #training_range = tqdm(self.data_loader)
        
        #for index, [pos_ins, neg_ins] in enumerate(self.data_loader):
        #    pdb.set_trace()

        if 1:
            score = []
            ans = []
            if self.test_rel == 'subclass':
                if valid:
                    pos_set = self.db['subclass_valid']
                    neg_set = self.db['negative_isSubclassOf_triples'][:len(pos_set)]
                else:
                    pos_set = self.db['subclass_test']
                    neg_set = self.db['negative_isSubclassOf_triples'][len(pos_set):(len(pos_set)*2)]
                rel = 0
            else:
                if valid:
                    pos_set = self.db['instance_valid']
                    neg_set = self.db['negative_isInstanceOf_triples'][:len(pos_set)]
                else:
                    pos_set = self.db['instance_test']
                    neg_set = self.db['negative_isInstanceOf_triples'][len(pos_set):(len(pos_set)*2)]
                rel = 1

            pos_head = np.array([ self.ent2id[i[0]] for i in pos_set], dtype = 'int64')
            pos_tail = np.array([ self.ent2id[i[1]] for i in pos_set], dtype = 'int64')
            pos_rel = np.array([ rel for i in pos_set], dtype = 'int64')
            pos_order = pos_head.argsort()
            pos_head = pos_head[pos_order]
            pos_tail = pos_tail[pos_order]

            neg_head = np.array([ self.ent2id[i[0]] for i in neg_set], dtype = 'int64')
            neg_tail = np.array([ self.ent2id[i[1]] for i in neg_set], dtype = 'int64')
            neg_rel = np.array([ rel for i in neg_set], dtype = 'int64')
            neg_order = neg_head.argsort()
            neg_head = neg_head[neg_order]
            neg_tail = neg_tail[neg_order]

            pos_ins = {'batch_h': pos_head, 'batch_t': pos_tail, 'batch_r': pos_rel, 'mode': 'normal'}
            neg_ins = {'batch_h': neg_head, 'batch_t': neg_tail, 'batch_r': neg_rel, 'mode': 'normal'}

            
            res_pos = self.test_one_step(pos_ins)
            ans = ans + [1 for i in range(len(res_pos))]
            score.append(res_pos)
            
            res_neg = self.test_one_step(neg_ins)
            ans = ans + [0 for i in range(len(res_neg))]
            score.append(res_neg)
            assert(len(res_pos) == len(res_neg))
            print('Epoch {0} len pos len neg {1} {2}'.format(epc, len(res_pos), len(res_neg)))

            score = np.concatenate(score, axis = -1)
            ans = np.array(ans)

            if valid:
                if for_threshold == False:
                    split_idx1 = int(ans.shape[0] * 1 / 4)
                    split_idx2 = int(ans.shape[0] * 2 / 4)
                    split_idx3 = int(ans.shape[0] * 3 / 4)

                    valid_score = np.concatenate((score[:split_idx1],score[split_idx2:split_idx3]), axis=0)
                    valid_ans = np.concatenate((ans[:split_idx1], ans[split_idx2:split_idx3]), axis=0)
        
                    test_score = np.concatenate((score[split_idx1:split_idx2] , score[split_idx3:]), axis=0)
                    test_ans = np.concatenate((ans[split_idx1:split_idx2],ans[split_idx3:]), axis=0)
                else:
                    valid_score = score 
                    valid_ans = ans
            else:
                test_score = score 
                test_ans = ans
            
        if for_threshold == True:
            threshold, _ = self.get_best_threshold(valid_score, valid_ans)
            print('Return threshold ',threshold)
            return threshold
        

        if threshold == None:
            if valid:
                threshold, _ = self.get_best_threshold(valid_score, valid_ans)
            else:
                threshold = self.run_triple_classification(epc, valid=True, threshold = None, for_threshold=True)
        
        #if valid == False:
        #    interested_idx = -1
        #    for i in range(len(pos_set)):
        #        if pos_ins['batch_h'][i] == self.ent2id['sport'] and pos_ins['batch_t'][i] == self.ent2id['game']:
        #           interested_idx = i
        #           break 
        #    interested_score = res_pos[interested_idx]
        #    print('{0} {1} {2}'.format(interested_idx, interested_score, threshold))
        #    pdb.set_trace()

        res = np.concatenate([test_ans.reshape(-1,1), test_score.reshape(-1,1)], axis = -1)
        order = np.argsort(test_score)
        res = res[order]

        #pdb.set_trace()
        total_all = (float)(len(test_score))
        total_current = 0.0
        total_true = np.sum(test_ans)
        total_false = total_all - total_true

        acc = 0
        f1 = 0
        for index, [ans, score] in enumerate(res):
            if score >= threshold:
                P = index 
                N = total_all - index 
                TP = total_current
                FP = P - TP 
                FN = total_true - total_current
                TN = N - FN 

                acc_ = (2 * total_current + total_false - index) / total_all
                acc = (TP+TN)/(TP+TN+FP+FN)

                precision = TP / (TP + FP)
                recall = TP / (TP + FN)
                f1 = 2*precision*recall / (precision + recall)

                assert(acc_ == acc)
                break
            elif ans == 1:
                total_current += 1.0
        
        print('Valid={0} Epoch {1} Acc {2} threshold {3}'.format(valid ,epc, acc, threshold))
        return acc, f1, threshold