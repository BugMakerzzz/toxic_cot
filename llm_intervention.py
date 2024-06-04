import os
import torch
import argparse
import json
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm
from transformers import AutoTokenizer
from load_data import CoTLoader, InterventionData
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
from metrics import draw_plot, draw_heat, draw_line_plot
from intervention_model import Model
from utils import get_prompter
random.seed(17)

## argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='Llama-2-13b-chat-hf')
parser.add_argument('--dataset', type=str, default='wino')
parser.add_argument('--mode', type=str, default='C2W')
parser.add_argument('--cnt', type=int, default=10)
parser.add_argument('--exp', type=str, default='mlp')
parser.add_argument('--avg', action='store_true')

args = parser.parse_args()
model_name = args.model
dataset = args.dataset
mode = args.mode
cnt = args.cnt
exp = args.exp
avg = args.avg
## Path 
model_path = f'./model/{model_name}'
if model_name.startswith('Llama'):
    cot_file_path  = f'./result/{dataset}/{model_name}_cot_answer_dev_1000.json'
    base_file_path = f'./result/{dataset}/{model_name}_direct_answer_dev_1000.json'
    full_cot_path = f'./result/{dataset}/{model_name}_cot_dev_1000.json'
    mlp_avg_rep_path = f'./result/{dataset}/attn-False_cnt-100_rep_std.json' 
    attn_avg_rep_path = f'./result/{dataset}/attn-True_cnt-100_rep_std.json'    
else:
    cot_file_path  = f'./result/{dataset}/{model_name}_cot_answer_2000.json'
    base_file_path = f'./result/{dataset}/{model_name}_direct_answer_2000.json'
    full_cot_path = f'./result/{dataset}/{model_name}_cot_dev_500.json'
    mlp_avg_rep_path = f'./result/{dataset}/{model_name}-False-2000_rep_std.json' 
    attn_avg_rep_path = f'./result/{dataset}/{model_name}-True-100_rep_std.json' 
result_path = f'./result/{dataset}/fig/{exp}_inter/{model_name}_{mode}_{cnt}'

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(17)
## Load Model

if model_name.startswith('Baichuan'):
    tokenizer = AutoTokenizer.from_pretrained(model_path,
                revision="v2.0",
                use_fast=False,
                trust_remote_code=True)
else:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)  
    cot_prompter = get_prompter(model_name=model_name, dataset=dataset, task='cot_answer')
    
cot_prompter = get_prompter(model_name=model_name, dataset=dataset, task='cot_answer')

## Load Data
index = None 
if model_name.startswith('Llama'):
    if mode == 'C2W':
        if dataset == 'csqa':
            # index = [41,49,158,161,174,219,244,276,283,286,297,386,394,402,413,424,431,441,443,457][:cnt]
            index = [36,331,379,395,521,525,527,599,654,826,893,913,998]
        elif dataset == 'wino':
            index = [40,47,73,175,180,185,197,232,255,266,274,306,316,327,333,409,423,427,433,444,454,481,493]
        #  index = [7, 15, 50, 53, 84, 97, 108, 119, 121, 132, 201, 207, 209, 235, 253][:cnt]
else:
    if mode == 'C2W':
        if dataset == 'csqa':
            index = [86,221,263,279,280,342,352,395,399,408,471,545,599,761,857,877,913]
        elif dataset == 'wino':
            # index = [28,53,90,93,97,102,145,148,158,183,185,201,261,316,327,348,366,393,429,437,453,465,506,584,642,658,661,678,696,710,732,734,755,756,771,805,843,882]
            index = [28,90,97,158,183,185,201,261,316,327,348,393,437,453,465,506,584,661,678,696,732,734,755,771,805,843]

    
dataloader = CoTLoader()
data, index = dataloader.load_data(cot_file=cot_file_path, base_file=base_file_path, mode=mode, cnt=cnt, index=index)

inter_data_list = []
model = Model(model_name=model_name)
for msg in data:
    
    if model_name.startswith('Baichuan'):
        inter_data_list.append(InterventionData(msg, tokenizer, cot_prompter, model.model))
    else:
        inter_data_list.append(InterventionData(msg, tokenizer, cot_prompter))


if exp == 'mlp':
    with open(mlp_avg_rep_path, 'r') as f:
        reps = json.load(f) 
    results = model.intervention_experiment(inter_data_list, reps)
else:
    with open(attn_avg_rep_path, 'r') as f:
        reps = json.load(f) 
    results = model.attention_experiment(inter_data_list, reps)
    
x_range = range(1, 41)
    
inter_dic = {1:'context', 2:'option', 3:'cot', 4:'last'}  
scores = np.zeros(shape=(len(inter_dic.keys()),len(x_range),len(index)))
labels = []

for i, result in results.items():   
    labels = []
    values = []
    for idx, score in result.items():
        label = inter_dic[idx]
        values.append(score.squeeze().numpy())
        labels.append(label)
        # print(score)
    if avg:
        values = np.array(values)
        scores[:,:,i] = values
    else:
        draw_heat(labels, values, result_path+'.pdf')
if avg:
    scores = np.mean(scores,axis=-1)
    if exp == 'attn':
        if model_name.startswith('Baichuan'):
            vmax = 0.2
        else:
            vmax = 0.5
    else:
        if model_name.startswith('Baichuan'):
            vmax = 0.2
        else:
            vmax = 0.4
     
    draw_heat(labels, scores, result_path+'.pdf', exp=exp, vmax=vmax)