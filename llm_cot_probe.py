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
from metrics import draw_plot
random.seed(17)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='Llama-2-13b-chat-hf')
parser.add_argument('--dataset', type=str, default='wino')
parser.add_argument('--datalength', type=int, default=1000)
# parser.add_argument('--mode', type=str, default='acc')
# parser.add_argument('--cot_mode', type=str, default=['full', 'none'])
parser.add_argument('--mode', type=str, default='IWP')
parser.add_argument('--diff_cot', type=str, default=None)
parser.add_argument('--diff_logits', type=str, default=None)
parser.add_argument('--avg', type=str, default=None)
parser.add_argument('--cnt', type=int, default=20)
parser.add_argument('--reg', type=bool, default=False)
args = parser.parse_args()


model_name = args.model
dataset = args.dataset
# datalength = args.datalength
# mode = args.mode
# cot_mode = args.cot_mode
mode = args.mode
diff_logits = args.diff_logits
diff_cot = args.diff_cot
avg = args.avg
cnt = args.cnt
reg = args.reg
model_path = f'./model/{model_name}'
cot_file_path  = f'./result/{dataset}/{model_name}_cot_answer_dev_1000.json'
base_file_path = f'./result/{dataset}/{model_name}_direct_answer_dev_1000.json'
result_path = f'./result/{dataset}/fig/{model_name}_{mode}_{diff_logits}-dl_{diff_cot}-dc_{cnt}-cnt_{reg}-reg'
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


def lm_logit(question_text, pred_text, layers, diff_logits):
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
        if diff_logits:
            if layer == 0:
                continue
            if diff_logits == 'base':
                base_logits = dict_outputs[0][0, prefix_ids.shape[-1] - 1: -1, :]
                base_logits = base_logits.log_softmax(dim=-1)
            else:
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


def cal_score(logits_ls, ids_ls, diff_cot, diff_score=True):
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

        probs = logits[:,range(logits.shape[1]), ids].sum(axis=-1)
        if diff_score:
            probs = (probs - probs[0])[1:].tolist()
        scores.append(probs)

    return scores 


def cot_score(question, pred, cot, layers, diff_cot=None, diff_logits=None, merge_cot=True):
    with torch.no_grad():
        pred_ids = []
        pred_logits = []
        pred_scores = []
        if merge_cot:
            cot = ['.'.join(cot)]
        for i in range(len(cot)+1):
            if i == 0:
                cot_question = cot_prompter.wrap_input(question, icl_cnt=5)   
            else:
                cot_question += cot[i-1] + '.'
            logits, ids = lm_logit(cot_question+' So the answer is: ', pred, layers, diff_logits)
            pred_logits.append(logits)
            pred_ids.append(ids)
        pred_scores = cal_score(pred_logits, pred_ids, diff_cot)

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
            pred_steps = steps[eval(pred)-1].split('.')[:-1]
            label_steps = steps[eval(label)-1].split('.')[:-1]
        else:
            steps = answers.split('.')[:-2]
            pred_steps = steps
            label_steps = steps
        if not steps:
            continue
        if dataset == 'gsm8k':
            label_option = str(label)
            pred_option = str(pred)
        else:
            options = question.split('\n')[-1].split('(')
            label_option = ' (' + options[eval(label)]
            pred_option = ' (' + options[eval(pred)]
        # print(pred_steps)
        pred_scores = cot_score(question, pred_option, pred_steps, layers, diff_cot, diff_logits)
        label_scores = cot_score(question, label_option, label_steps, layers, diff_cot, diff_logits)
        pred_legends = [f'pred_step_{i+1}' for i in range(len(pred_scores))]
        label_legends = [f'label_step_{i+1}' for i in range(len(label_scores))]
        x_range = layers
        # if diff_logits:
        x_range = x_range[1:]
        if avg:
            pred_scores_ls.append(pred_scores)
            label_scores_ls.append(label_scores)
        else:
            fig_path = result_path + f'_{cot_data.index(msg)}.png'
            draw_plot(x_range, pred_scores+label_scores, pred_legends+label_legends, fig_path)
    if avg:
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
        

# def probe(cot_mode, cot_case, mode, return_diff):
#     with open(cot_file_path, 'r') as f:
#         cot_data = json.load(f)[:-1]
#     with open(base_file_path, 'r') as f:
#         base_data = json.load(f)[:-1]
#     cor_cot_data, wr_cot_data = collect_tf_data(cot_data)     
#     cor_base_data, wr_base_data = collect_tf_data(base_data)        
#     score_list = []
#     legend_list = []
#     for mode in cot_mode:
#         for case in cot_case:
#             if case == 'w2w':
#                 data = merge_data(wr_cot_data, wr_base_data)
#             elif case == 'w2c':
#                 data = merge_data(cor_cot_data, wr_base_data)
#             elif case == 'c2w':
#                 data = merge_data(wr_cot_data, cor_base_data)
#             else:
#                 data = merge_data(cor_cot_data, cor_base_data) 
#             data, labels = prepare_probe_data(data, mode)
#             scores = cal_layer_score(dataloader=data, mode=mode, return_diff=return_diff)
#             if mode == 'acc':
#                 corrects = [0] * 40
#                 for i in range(len(scores)):
#                     label = labels[i]
#                     score_ls = scores[i]
#                     for j in range(len(score_ls)):
#                         score = score_ls[j]
#                         pred = str(np.argmax(score) + 1)
#                         if pred == label:
#                             corrects[j] += 1
#                 accs = [correct / len(scores) for correct in corrects]
#                 score_list.append(accs)
#                 legend_list.append(f'{mode}_{case}')
#             else:
#                 scores = np.array(scores)
#                 pred_scores = np.mean(scores[:,:,0], axis=0)
#                 label_scores = np.mean(scores[:,:,1], axis=0)
#                 score_list.append(pred_scores)
#                 score_list.append(label_scores)
#                 legend_list.append(f'pred_{mode}_{case}')
#                 legend_list.append(f'label_{mode}_{case}')
#     return score_list, legend_list

case_index = merge_data(mode)[:50]
print(case_index)
if not case_index:
    case_index = [10,11,13,15,16,27,28,40,47,49]
case_index = case_index[:cnt]
layers = [0, 5, 10, 15, 20, 25, 30, 35, 40]

probe(case_index=case_index, layers=layers, load_cot=False)
