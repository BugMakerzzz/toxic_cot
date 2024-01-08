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
from prompts.wrap_prompt import LlamaPrompter
from load_data import CoTLoader, InterventionData
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
from metrics import draw_plot, draw_heat, draw_line_plot, draw_attr_heat
from intervention_model import Model
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
cot_file_path  = f'./result/{dataset}/{model_name}_cot_answer_dev_1000.json'
base_file_path = f'./result/{dataset}/{model_name}_direct_answer_dev_1000.json'
full_cot_path = f'./result/{dataset}/{model_name}_cot_dev_1000.json'
result_path = f'./result/{dataset}/fig/{model_name}_{mode}_'
mlp_avg_rep_path = f'./result/{dataset}/attn-False_cnt-100_rep_std.json' 
attn_avg_rep_path = f'./result/{dataset}/attn-True_cnt-100_rep_std.json' 

## Load Model
model = Model(model_name=model_name)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)  
cot_prompter = LlamaPrompter(dataset=dataset, task='cot_answer')

## Load Data
index = None 
if mode == 'C2W':
    if dataset == 'csqa':
        index = [41,49,158,161,174,219,244,276,283,286,297,386,394,402,413,424,431,441,443,457][:cnt]
        # index = [108,119,121,132,201]
    elif dataset == 'wino':
        index = [40,47,73,175,180,185,197,232,255,266,274,306,316,327,333,409,423,427,433,444,454,481][:cnt]
        #  index = [7, 15, 50, 53, 84, 97, 108, 119, 121, 132, 201, 207, 209, 235, 253][:cnt]
        
dataloader = CoTLoader()
data, index = dataloader.load_data(cot_file=cot_file_path, base_file=base_file_path, mode=mode, cnt=cnt, index=index)
inter_data_list = []
for msg in data:
    if mode == 'W2C':
        msg['pred'] = msg['label']
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
    
fold_path = result_path + f'_{exp}-inter/'
if not os.path.exists(fold_path):
    os.mkdir(fold_path)
    
    
inter_dic = {1:'stem', 2:'option', 3:'cot', 4:'last'}  
scores = np.zeros(shape=(len(inter_dic.keys()),len(x_range),cnt))
labels = []

for i, result in results.items():   
    labels = []
    values = []
    for idx, score in result.items():
        label = inter_dic[idx]
        values.append(score.squeeze().numpy())
        labels.append(label)
        # print(score)
    path = os.path.join(fold_path, f'idx-{index[i]}.png')
    if avg:
        values = np.array(values)
        scores[:,:,i] = values
    else:
        draw_heat(x_range, labels, values, path)
if avg:
    scores = np.mean(scores,axis=-1)
    path = os.path.join(fold_path, f'cnt-{cnt}.png')
    draw_heat(x_range, labels, scores, path)