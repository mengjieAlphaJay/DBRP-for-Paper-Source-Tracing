import random
random.seed(42)
import os
import xml.etree.ElementTree as ET
import json
import re
import bert_score
from rouge_score import rouge_scorer
import pandas as pd
from torchmetrics.text.bert import BERTScore
from txtai import Embeddings

PTS_dir = "dataset/PTS-2"
bertscore = BERTScore()

with open(os.path.join(PTS_dir,"test_similarity.json"), 'r') as file:
    simi_test = json.load(file)

with open(os.path.join(PTS_dir,"data_all.json"), 'r') as file:
    data_all = json.load(file)

sim_mat = pd.read_csv(os.path.join(PTS_dir,"similarity_matrix.csv"))
sim_mat = sim_mat.rename(columns={'Unnamed: 0': 'id_'})

def subtract_lists(list_a, list_b):
    return [item for item in list_a if item not in list_b]




class PTSData:
    def __init__(self):
        self.embeddings = Embeddings(path="allenai/scibert_scivocab_uncased")
        self.data_all = data_all
        with open(os.path.join(PTS_dir,"test20.json"), 'r') as file:
            data_test = json.load(file)
        with open(os.path.join(PTS_dir,"train80.json"), 'r') as file:
            data_train = json.load(file)
        with open(os.path.join(PTS_dir,"similarity.json"), 'r') as file:
            simi_test = json.load(file)
        self.simi_test = simi_test
        self.node_index_list = [dp['_id'] for dp in data_test]
        self.main_papers = {}
        self.data_train = data_train
        self.sim_dict = {}
        for k in self.simi_test.keys():
            simi_list = self.simi_test[k]['refs_trace']+self.simi_test[k]['references']
            self.sim_dict[k] = {item['ref_id']: item for item in simi_list}

        for paper in data_train:
            Apaper = Paper(paper,'train')
            self.main_papers[paper['_id']] = Apaper
        for paper in data_test:
            Apaper = Paper(paper,'test')
            self.main_papers[paper['_id']] = Apaper
        self.data_train_ids = [paper['_id'] for paper in data_train]
        self.data_test_ids = [paper['_id'] for paper in data_test]

    def get_type(self,mid,rid):
        if rid in self.main_papers[mid].trace:
            return 'trace'
        elif rid in subtract_lists(self.main_papers[mid].ref,self.main_papers[mid].trace):
            return 'no_trace'
        else:
            return 'error'
    def get_paperinfo(self,rid,mid=""):
        if mid!="":
            sim = 'none'#self.sim_dict[mid][rid]['similarity']
        else:
            sim = 'none'
        return {'title':self.data_all[rid]['title'],'abs':self.data_all[rid]['abstract'],'cite':self.data_all[rid]['n_citation'],'sim':sim}

    def find_candidate(self,mid):
        #ref_paper = ptsdata.get_paperinfo(pid) [title] [abs]
        sorted_df = sim_mat.sort_values(by=mid, ascending=False).head(15)[['id_',mid]]
        sorted_df2 = sorted_df[sorted_df['id_'].isin(self.data_train_ids)]
        #print("sorted_df2",sorted_df2)
        candidates = []
        candi_id = []
        maintxt = self.main_papers[mid].title+self.main_papers[mid].abs
        for j in range(len(sorted_df2)):
            pid = sorted_df2['id_'].iloc[j]
            tit = self.main_papers[pid].title
            abs = self.main_papers[pid].abs
            tiabs = tit+abs
            candidates.append(tiabs)
            candi_id.append(pid)
        ### Select using ROUGE
        # 计算 BERTScore
        P, R, F1 = bert_score.score(candidates, [maintxt]*len(candidates), lang="en", verbose=True)
        #print("R",R)
        #R = bertscore(candidates, [maintxt]*len(candidates))['recall']
        max_idx = R.argmax()
        best_candi = candi_id[max_idx]
        return best_candi


    

    def find_max_similarity(self,rpapers,paper,k):
        self.embeddings.index(rpapers)
        sort_ind = self.embeddings.search(paper, k)
        return [sort_ind[i][0] for i in range(k)]

    def find_max_rouge(self,rpapers,paper):
        scorer = rouge_scorer.RougeScorer(['rouge2'], use_stemmer=True)
        rougeli = [scorer.score(rp,paper)['rouge2'].recall for rp in rpapers]
        #print("rougeli",rougeli)
        sorted_indices = sorted(range(len(rougeli)), key=lambda k: rougeli[k],reverse=True)
        return sorted_indices

    def find_ref_of_candi(self,best_candi):
        best_mainpa = self.main_papers[best_candi]
        main_txt = best_mainpa.title+best_mainpa.abs
        rrids = []
        rpas = []
        #print("best_mainpa",best_candi)
        for refid in best_mainpa.ref:
            rpaper = self.get_paperinfo(refid)
            rpas.append(rpaper['title']+rpaper['abs'])
            rrids.append(refid)
        sortindex = self.find_max_similarity(rpas,main_txt,5)
        sortind = [rrids[i] for i in sortindex]
        return sortind

    def find_examples(self,best_candi,rid):
        def replace_example(mid,rid,occup_list):
            sim = self.sim_dict[mid][rid]['similarity']

            sorted_refs = sorted(self.sim_dict[mid].items(), key=lambda item: item[1]['similarity'], reverse=True)
            for ref_id in sorted_refs:
                if ref_id != rid and ref_id not in occup_list:
                    return ref_id
            return rid

        #print("best_candi",best_candi)
        main_ref= self.get_paperinfo(rid)
        main_ref_txt = main_ref['title']+main_ref['abs']
        rrids = []
        rpas = []
        best_mainpa = self.main_papers[best_candi]
        for refid in best_mainpa.ref:
            rpaper = self.get_paperinfo(refid)
            rpas.append(rpaper['title']+rpaper['abs'])
            rrids.append(refid)
        sortindex = self.find_max_rouge(rpas,main_ref_txt)
        candi_ref = [self.get_paperinfo(iid) for iid in best_mainpa.ref_wo_trace]
        sortindex_candi = self.find_max_rouge([rp['title']+rp['abs'] for rp in candi_ref],best_mainpa.title+best_mainpa.abs)
        ### Select Ref using rouge
        best_ref = rrids[sortindex[0]]
        if best_ref not in best_mainpa.trace:
            next_ref = best_mainpa.trace[0]
        else:
            next_ref = rrids[sortindex[1]]
        return [best_ref,next_ref]

class Paper:
    def __init__(self,paper_info,sets):
        ### for main paper
        self.id = paper_info['_id']
        self.title = paper_info['title']
        self.abs = data_all[self.id]['abstract']
        self.ref = paper_info['references'] ### ref is no-traceref
        self.trace = [dp['_id'] for dp in paper_info['refs_trace']]
        self.ref_wo_trace = subtract_lists(self.ref,self.trace)
        self.set = sets
        if sets == 'test':

            selected_ref = self.select_references(simi_test[self.id])
            self.pair_test = self.match_pair3(selected_ref,self.trace)
            self.no_trace_ref = subtract_lists(selected_ref,self.trace)
    def sample_test_ref(self):
        random.seed(42)
        return random.sample(self.ref_wo_trace,6-len(self.trace))


    def pair_elements(self,trace_list, ref_list):
        paired_list = []  # 初始化结果列表
        trace_index, ref_index = 0, 0  # 

        while trace_index < len(trace_list) and ref_index < len(ref_list):
            paired_list.append([trace_list[trace_index], ref_list[ref_index]])
            trace_index += 1
            ref_index += 1

        remaining = trace_list[trace_index:] + ref_list[ref_index:]
        while len(remaining) > 0:
            if len(remaining) >= 2:
                paired_list.append(remaining[:2])
                remaining = remaining[2:]
            else:
                paired_list.append(remaining)
                break

        while len(paired_list) > 3:
            last = paired_list.pop()
            paired_list[-1].extend(last)

        return paired_list

    def select_references(self,data):
        random.seed(42)
        selected_refs = [trace['ref_id'] for trace in data['refs_trace']]
        top_references = sorted(subtract_lists(data['references'],data['refs_trace']), key=lambda x: x['similarity'], reverse=True)[:10]
        #print("top_references......",top_references)
        num_to_select = min(6 - len(selected_refs), 2)  # 需要确保总数不超过6条
        if len(selected_refs) + num_to_select <= 3:
            num_to_select = 3
        random_top_refs = random.sample(top_references, num_to_select)
        selected_refs.extend([ref['ref_id'] for ref in random_top_refs])
        remaining_refs = [ref for ref in data['references'] if ref['ref_id'] not in selected_refs]

        if len(selected_refs) < 6:
            needed_refs = 6 - len(selected_refs)
            random_remaining_refs = random.sample(remaining_refs, needed_refs)
            selected_refs.extend([ref['ref_id'] for ref in random_remaining_refs])
        return selected_refs

    def match_pair3(self,selected_refs,refs_trace_ids):
        random.seed(42)
        trace_refs = [ref for ref in selected_refs if ref in refs_trace_ids]
        non_trace_refs = [ref for ref in selected_refs if ref not in refs_trace_ids]

        random.shuffle(non_trace_refs)

        half_size = len(trace_refs) // 2
        group1 = trace_refs[:half_size]
        group2 = trace_refs[half_size:]
        l1remain = 3-len(group1)
        group1.extend(non_trace_refs[:l1remain])
        #print("///",non_trace_refs[3-len(group1):6-len(group1)-len(group2)])
        group2.extend(non_trace_refs[l1remain:])
        
        return [group1, group2]