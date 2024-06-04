import os
import torch
import argparse
import json
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModelForSeq2SeqLM
from load_data import CoTLoader, InterventionData
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
from functools import partial
from utils import get_prompter

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(17)

class Model():
    def __init__(self, model_name):
        self.is_gpt2 = model_name.startswith('gpt2')
        self.is_gptj = model_name.startswith('gpt-j')
        self.is_baichuan = model_name.startswith('Baichuan')
        self.is_neox = model_name.startswith('gpt-neox')
        self.is_gptneo = model_name.startswith('gpt-neo')
        self.is_opt = model_name.startswith('opt')
        self.is_llama = model_name.startswith('Llama') 
        self.is_flan = model_name.startswith('flan-t5')
        self.is_pythia = model_name.startswith('pythia')

        model_path = f'./model/{model_name}'
        if self.is_llama or self.is_baichuan:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True, device_map='auto') 
            self.model.eval()
        # store model details
        self.num_layers = self.model.config.num_hidden_layers
        self.num_neurons = self.model.config.hidden_size
        self.num_heads = self.model.config.num_attention_heads
        self.device = self.model.device
        
        if self.is_gpt2 or self.is_gptj:
            self.get_attention_layer = lambda layer: self.model.transformer.h[layer].attn
            self.word_emb_layer = self.model.transformer.wte
            self.get_neuron_layer = lambda layer: self.model.transformer.h[layer].mlp
        elif self.is_neox:
            self.get_attention_layer = lambda layer: self.model.gpt_neox.layers[layer].attention
            self.word_emb_layer = self.model.gpt_neox.embed_in
            self.get_neuron_layer = lambda layer: self.model.gpt_neox.layers[layer].mlp
        elif self.is_flan:
            self.get_attention_layer = lambda layer: (self.model.encoder.block + self.model.decoder.block)[layer].layer[
                0]
            self.word_emb_layer = self.model.encoder.embed_tokens
            self.get_neuron_layer = lambda layer: (self.model.encoder.block + self.model.decoder.block)[layer].layer[
                1 if layer < len(self.model.encoder.block) else 2]
        elif self.is_pythia:
            self.get_attention_layer = lambda layer: self.model.gpt_neox.layers[layer].attention
            self.word_emb_layer = self.model.gpt_neox.embed_in
            self.get_neuron_layer = lambda layer: self.model.gpt_neox.layers[layer].mlp
        elif self.is_llama or self.is_baichuan:
            self.get_attention_layer = lambda layer: self.model.model.layers[layer].self_attn
            self.word_emb_layer = self.model.model.embed_tokens
            self.get_neuron_layer = lambda layer: self.model.model.layers[layer].mlp
        else:
            raise Exception(f'Model not supported')


    def get_representations(self, context, position=-1, is_attention=False):
            # Hook for saving the representation
        def extract_representation_hook(module,
                                        input,
                                        output,
                                        position,
                                        representations,
                                        layer):
            representations[layer] = torch.flatten(output[(0, position)].mean(axis=0)).to('cpu').numpy()


        def extract_representation_hook_attn(module,
                                            input,
                                            output,
                                            position,
                                            representations,
                                            layer):
            
            representations[layer] = torch.flatten(output[0][0, position].mean(axis=0)).to('cpu').numpy()

        handles = []
        representation = {}
        with torch.no_grad():
            # construct all the hooks
            # word embeddings will be layer -1
            if not is_attention:
                handles.append(self.word_emb_layer.register_forward_hook(
                    partial(extract_representation_hook,
                            position=position,
                            representations=representation,
                            layer=-1)))
            # hidden layers
            for layer_n in range(self.num_layers):
                if is_attention:
                    handles.append(self.get_attention_layer(layer_n).register_forward_hook(
                        partial(extract_representation_hook_attn,
                                position=position,
                                representations=representation,
                                layer=layer_n)))
                else:
                    handles.append(self.get_neuron_layer(layer_n).register_forward_hook(
                        partial(extract_representation_hook,
                                position=position,
                                representations=representation,
                                layer=layer_n)))
            if self.is_flan:
                self.model(context.to(self.device), decoder_input_ids=torch.tensor([[0]]).to(self.device))
            else:
                self.model(context.to(self.device))
            for h in handles:
                h.remove()
        return representation
    
    
    def cal_rep(self, intervention):
        with torch.no_grad():
            context = intervention.cot_input_ids
            positions = intervention.cot_intervention_idx
            reps = {}
            for i, position in positions.items():
                rep = self.get_representations(context, position=position, is_attention=attn)
                reps[i] = rep
                
            return reps
                            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='Llama-2-13b-chat-hf')
    parser.add_argument('--dataset', type=str, default='wino')
    parser.add_argument('--datalength', type=int, default=2000)
    parser.add_argument('--attn', action='store_true')

    args = parser.parse_args()
    model_name = args.model
    dataset = args.dataset
    datalength = args.datalength
    attn = args.attn
    ## Path 
    model_path = f'./model/{model_name}'
    cot_file_path  = f'./result/{dataset}/{model_name}_cot_answer_2000.json'
    base_file_path = f'./result/{dataset}/{model_name}_direct_answer_2000.json'
    result_path = f'./result/{dataset}/{model_name}-{attn}-{datalength}_rep_std.json'

    ## Load Model
    
    model = Model(model_name=model_name)
    if model_name.startswith('Baichuan'):
        tokenizer = AutoTokenizer.from_pretrained(model_path,
                revision="v2.0",
                use_fast=False,
                trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)  
    cot_prompter = get_prompter(model_name=model_name, dataset=dataset, task='cot_answer')
    index = range(datalength)
    dataloader = CoTLoader()
    data, index = dataloader.load_data(cot_file=cot_file_path, base_file=base_file_path, index=index)
    
    inter_dic = {1:'stem', 2:'option', 3:'cot', 4:'last'}
    reps = {}

    for key in inter_dic.keys():
        reps[key] = {k:[] for k in range(model.num_layers)}
    for msg in tqdm(data):
        if model_name.startswith('Baichuan'):
            inter_data = InterventionData(msg, tokenizer, cot_prompter, model.model)
        else:
            inter_data = InterventionData(msg, tokenizer, cot_prompter)
        rep = model.cal_rep(inter_data)
        for key in inter_dic.keys():
            for k in range(model.num_layers):
                reps[key][k].append(rep[key][k])
    results = {}
    for key, rep_layer_dic in reps.items():
        results[key] = {}
        for k, rep in rep_layer_dic.items():
            # print(rep)
            rep = np.array(rep,dtype=np.float32)
            # print(rep)
            results[key][k] = float(np.std(rep))
            # print( results[key][k])
    
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=4)
            
    
    