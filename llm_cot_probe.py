import os
import torch
import copy
import re
import argparse
import json
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModelForSeq2SeqLM
from prompts.wrap_prompt import LlamaPrompter
from load_data import DataLoader
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
from metrics import draw_plot,draw_heat
random.seed(17)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='Llama-2-13b-chat-hf')
parser.add_argument('--dataset', type=str, default='wino')
parser.add_argument('--mode', type=str, default='C2W')
parser.add_argument('--diff_cot', type=str, default=None)
parser.add_argument('--diff_logits', type=str, default=None)
parser.add_argument('--avg', type=str, default=None)
parser.add_argument('--cnt', type=int, default=50)
parser.add_argument('--reg', action='store_true')
parser.add_argument('--split', action='store_true')
parser.add_argument('--direct', action='store_true')
args = parser.parse_args()


model_name = args.model
dataset = args.dataset
mode = args.mode
diff_logits = args.diff_logits
diff_cot = args.diff_cot
avg = args.avg
cnt = args.cnt
reg = args.reg
direct = args.direct
split = args.split
model_path = f'./model/{model_name}'
cot_file_path  = f'./result/{dataset}/{model_name}_cot_answer_dev_1000.json'
base_file_path = f'./result/{dataset}/{model_name}_direct_answer_dev_1000.json'
result_path = f'./result/{dataset}/fig/{model_name}_{mode}_{diff_logits}-dl_{diff_cot}-dc_{cnt}-cnt_{reg}-reg_{split}-split_{direct}-direct'
full_cot_path = f'./result/{dataset}/{model_name}_cot_dev_1000.json'

config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16, trust_remote_code=True)
    # model = AutoModelForSeq2SeqLM.from_config(config, torch_dtype=torch.float16, trust_remote_code=True)
no_split_modules = model._no_split_modules
model = load_checkpoint_and_dispatch(
    model, model_path, device_map="auto", no_split_module_classes=no_split_modules
)
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)  
cot_prompter = LlamaPrompter(dataset=dataset, task='cot_answer')
base_prompter = LlamaPrompter(dataset=dataset, task='direct_answer')


def collect_tf_data(dataloader):
    cor_data = []
    wr_data = []
    for data in dataloader:
        if data['pred'] == 'None':
            continue
        if data['cor_flag']:
            cor_data.append(data)
        else:
            wr_data.append(data)
    return cor_data, wr_data

def merge_data(mode):
    with open(cot_file_path, 'r') as f:
        full_cot_data = json.load(f)[:-1]
    cot_cor_data, cot_wr_data = collect_tf_data(full_cot_data)
    with open(base_file_path, 'r') as f:
        full_base_data = json.load(f)[:-1]
    base_cor_data, base_wr_data = collect_tf_data(full_base_data)
    if mode == 'W2C':
        base_data = base_wr_data
        cot_data = cot_cor_data
    elif mode == 'C2W':
        base_data = base_cor_data
        cot_data = cot_wr_data  
    elif mode == 'W2W':
        base_data = base_wr_data
        cot_data = cot_wr_data  
    elif mode == 'C2C':
        base_data = base_cor_data
        cot_data = cot_cor_data
    else:
        return  
    results = []
    question_set = []
    for data in base_data:
        question_set.append(data['question'])
    for data in cot_data:
        if data['question'] in question_set:
            results.append(full_cot_data.index(data))
    return results


def cal_logits(question_text, pred_text, layers, diff_logits):
    input_text = question_text + pred_text
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
    prefix_ids = tokenizer(question_text, return_tensors="pt").input_ids.to(model.device)
    continue_ids = input_ids[0, prefix_ids.shape[-1]:].cpu().numpy()
    dict_outputs, outputs = model(
        input_ids=input_ids,
        return_dict=True,
        output_attentions=False,
        output_hidden_states=False,
        early_exit_layers=layers,
    )
    
    scores = []
    for layer in layers:
        logits = dict_outputs[layer][0, prefix_ids.shape[-1] - 1: -1, :]
        logits = logits.log_softmax(dim=-1)
        if diff_logits and diff_logits != 'contrast':
            if layer == 0:
                continue
            if diff_logits == 'base':
                base_logits = dict_outputs[0][0, prefix_ids.shape[-1] - 1: -1, :]
                base_logits = base_logits.log_softmax(dim=-1)
            elif diff_logits == 'adj':
                base_layer = layers[layers.index(layer)-1]
                base_logits = dict_outputs[base_layer][0, prefix_ids.shape[-1] - 1: -1, :]
                base_logits = base_logits.log_softmax(dim=-1)
            logits = logits - base_logits
            logits = logits.log_softmax(dim=-1)
            # print(logits)
            # probs = diff_logits[range(diff_logits.shape[0]), continue_ids].sum().item()
        scores.append(logits.cpu().numpy())  
    torch.cuda.empty_cache()

    return scores, continue_ids


def cal_score(logits_ls, ids_ls, diff_cot, diff_logits):
    scores = []
    for i in range(len(logits_ls)):
        logits = np.array(logits_ls[i])
        ids = ids_ls[i]
        if diff_cot and i == 0:
            continue
        if diff_cot == 'base':
            base_logits = np.array(logits_ls[0])
        elif diff_cot == 'adj':
            base_logits = np.array(logits_ls[i-1])
        elif diff_cot == 'add':
            base_logits = - np.array(logits_ls[0])

        if diff_cot:
            logits = logits - base_logits

        # if diff_logits == 'score':
        #     probs = logits[:,range(logits.shape[1]), ids].sum(axis=-1)
        #     probs = (probs - probs[0])[1:].tolist()
        # else:
        probs = logits[:,range(logits.shape[1]), ids].sum(axis=-1).tolist()
        
        scores.append(probs)
    return scores


def cot_score(question, pred, cot, layers, direct=False, split=False, diff_cot=None, diff_logits=None):
    with torch.no_grad():
        pred_ids = []
        pred_logits = []
        pred_scores = []
        if split:
            cot = cot.split('.')
        else:
            cot = [cot]
        for i in range(len(cot)+1):
            if i == 0:
                cot_question = cot_prompter.wrap_input(question, icl_cnt=5)   
                if not direct:
                    continue
            else:
                cot_question += cot[i-1] 
            # print(target)
            if i == len(cot) or i == 0:
                logits, ids = cal_logits(cot_question + '\nSo the answer is: ', pred, layers, diff_logits)
            else:
                logits, ids = cal_logits(cot_question + '...\nSo the answer is: ', pred, layers, diff_logits)
            pred_logits.append(logits)
            pred_ids.append(ids)
        pred_scores = cal_score(pred_logits, pred_ids, diff_cot, diff_logits)

    return pred_scores



def cot_avg(scores, avg):
    if avg == 'norm':
        max_len = 0
        for score in scores:
            layer_len = len(score[0])
            if len(score) > max_len:
                max_len = len(score)
        avg_scores = np.ma.empty((max_len, layer_len, len(scores))) 
        avg_scores.mask = True
        for i in range(len(scores)):
            score = np.array(scores[i])
            avg_scores[:score.shape[0], :score.shape[1], i] = score            
        scores = avg_scores.mean(axis=-1).tolist()[:5]
    elif avg == 'steps':
        score_dic = {}
        for score in scores:
            if len(score) not in score_dic.keys():
                score_dic[len(score)] = [score]
            else:
                score_dic[len(score)].append(score)
        for i, score in score_dic.items():
            score = np.array(score).mean(axis=0).tolist()
            score_dic[i] = score
        scores = score_dic
    return scores




def probe(case_index, layers):
    with open(cot_file_path, 'r') as f:
        cot_data = json.load(f)[:-1]
    with open(base_file_path, 'r') as f:
        base_data = json.load(f)[:-1]
    if reg:
        with open(full_cot_path, 'r') as f:
            cots = json.load(f)
    data = []
    for i in range(len(cot_data)):
        if i in case_index:
            data.append(cot_data[i])  
    x_range = 0
    pred_scores_ls = []  
    label_scores_ls = []
    for msg in tqdm(data):
        label = msg['label']
        idx = case_index[data.index(msg)]
        if mode == 'W2C':
            pred = base_data[idx]['pred']
        else:
            pred = msg['pred']
        question = msg['question']
        answers = msg['answer']
        if reg:
            steps = cots[idx]['answer']
            if eval(pred)-1 >= len(steps):
                continue
            pred_cot = steps[eval(pred)-1].split('.')[:-1]
            pred_cot = '.'.join(pred_cot) 
            label_cot = steps[eval(label)-1].split('.')[:-1]
            label_cot = '.'.join(label_cot) 
        else:
            steps = answers.split('.')[:-2]
            pred_cot = '.'.join(steps) 
            label_cot = '.'.join(steps) 
        if not steps:
            print(f'Num {idx} question\'s cot is too short!!!')
            continue
        if dataset == 'gsm8k':
            label_option = str(label)
            pred_option = str(pred)
        else:
            label_option = f'({label})'
            pred_option = f'({pred})'
            # options = question.split('\n')[-1].split('(')
            # label_option = ' (' + options[eval(label)]
            # pred_option = ' (' + options[eval(pred)]


        pred_scores = cot_score(question, pred_option, pred_cot, layers, direct, split, diff_cot, diff_logits)
        label_scores = cot_score(question, label_option, label_cot, layers, direct, split, diff_cot, diff_logits)
        if diff_logits == 'contrast':
            pred_sub_scores = cot_score(question, label_option, pred_cot, layers, direct, split, diff_cot, diff_logits)
            label_sub_scores = cot_score(question, pred_option, label_cot, layers, direct, split, diff_cot, diff_logits)
            pred_scores = (np.array(pred_scores) - np.array(pred_sub_scores)).tolist()
            label_scores = (np.array(label_scores) - np.array(label_sub_scores)).tolist()
            
        pred_legends = [f'pred_step_{i+1}' for i in range(len(pred_scores))]
        label_legends = [f'label_step_{i+1}' for i in range(len(label_scores))]
        x_range = layers
        if diff_logits and diff_logits != 'contrast':
            x_range = x_range[1:]
        if avg:
            pred_scores_ls.append(pred_scores)
            label_scores_ls.append(label_scores)
        else:
            fig_path = result_path + f'_{cot_data.index(msg)}.png'
            draw_plot(x_range, pred_scores+label_scores, pred_legends+label_legends, fig_path)
    if avg:
        if avg == 'heat':
            pred_fig_path = result_path + '_heat-pred.png'
            label_fig_path = result_path + '_heat-label.png'
            pred_first_scores = np.zeros(len(layers))
            pred_mid_scores = np.zeros(len(layers))
            pred_last_scores = np.zeros(len(layers))
            label_first_scores = np.zeros(len(layers))
            label_mid_scores = np.zeros(len(layers))
            label_last_scores = np.zeros(len(layers))
            for scores in pred_scores_ls:
                steps = len(scores)
                pred_first_scores += np.array(scores[0])
                pred_last_scores += np.array(scores[-1])
                if steps <= 2:
                    pred_mid_scores += np.array([scores[0], scores[-1]]).mean(axis=0)
                else:
                    pred_mid_scores += np.array(scores[1:-1]).mean(axis=0)
            for scores in label_scores_ls:
                steps = len(scores)
                label_first_scores += np.array(scores[0])
                label_last_scores += np.array(scores[-1])
                if steps <= 2:
                    label_mid_scores += np.array([scores[0], scores[-1]]).mean(axis=0)
                else:
                    label_mid_scores += np.array(scores[1:-1]).mean(axis=0) 
            pred_cnt = len(pred_scores_ls)
            label_cnt = len(label_scores_ls)
            pred_scores_ls = [(pred_first_scores).tolist(), (pred_mid_scores).tolist(), (pred_last_scores).tolist()]
            label_scores_ls = [(label_first_scores).tolist(), (label_mid_scores).tolist(), (label_last_scores).tolist()]
            index = ['First', 'Mid', 'Last']
            draw_heat(x_range, index, pred_scores_ls, pred_fig_path)
            draw_heat(x_range, index, label_scores_ls, label_fig_path)
        else:
            pred_scores = cot_avg(pred_scores_ls, avg)
            label_scores = cot_avg(label_scores_ls, avg)
            if avg == 'norm':
                legends = []
                for i in range(len(pred_scores)):
                    legends.append(f'pred_step_{i+1}')
                for i in range(len(label_scores)):
                    legends.append(f'label_step_{i+1}')
                fig_path = result_path + '_norm-avg.png'
                draw_plot(x_range, pred_scores+label_scores, legends, fig_path)
            elif avg == 'steps':
                fold_path = result_path + '_steps-avg/'
                if not os.path.exists(fold_path):
                    os.mkdir(fold_path)
                for i in sorted(pred_scores.keys()):
                    pred_score = pred_scores[i]
                    label_score = label_scores[i]
                    pred_legend = [f'l{i}_pred_step_{x}' for x in range(1,i+1)]
                    label_legend = [f'l{i}_label_step_{x}' for x in range(1,i+1)]
                    fig_path = os.path.join(fold_path, f'l{i}.png')
                    draw_plot(x_range, pred_score+label_score, pred_legend+label_legend, fig_path)
        

case_index = merge_data(mode)[:cnt]
print(case_index)
layers = range(0,41)

probe(case_index=case_index, layers=layers)
