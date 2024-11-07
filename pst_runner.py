import itertools
from prompthub import PST_proximity
import json
import numpy as np
import random
from LLM import Inferencer
Olla = Inferencer()

A1 = "GOAL"
A2 = "METHOD"
A3 = "IDEA"
A4 = "THEORY/EXPER"



class PSTRunner:
    def __init__(self,agent,BRD_file):
        PST_dir = "dataset/PST/"
        self.pster = PST_proximity(agent)
        self.agent = agent
        self.llmres_exm = agent.loadjson(PST_dir+BRD_file)
        self.pstdata = self.agent.pstdata
        self.mid_list = self.pstdata.node_index_list
        self.LLM = Olla


    def run_pst_aspect(self,llmres_base1,model,aspdict,mode,save,idmaps,end):
        ## aspdict = {"A1":[], "A2":[], "A3":[], "A4":[]}
        fname = mode
        print("==================================================")
        for i,id_ in enumerate(self.mid_list[:end]):
            result = []
            print(id_,"**************************")
            pair_test = self.pstdata.main_papers[id_].pair_test
            for input_id in pair_test:
                print("'''''''''''''''")
                self.pster.set_idx(id_,input_id,examples = self.llmres_exm[id_])
                labelslist = self.pster.prompt['ref_labels']
                if mode == "dot":
                    self.pster.set_aspect({A2:"",A3:"",A4:""})
                    message = self.pster.COT_DCOM_promting_base()
                print(message[0]['content'],message[1]['content'])
                response = ""#Olla.LLMsInfer(message,model,"no_str",False)
                scores = ""#self.pstdata.extract_anser_cot(response)
                result.append([input_id,message,response,scores,labelslist])
            llmres_base1[id_] = result
        if save:
            with open("main_experiment/pst/llmres_"+fname+".json","w") as f:
                json.dump(llmres_base1,f)
        return llmres_base1
