import os
import torch
import re
import argparse
import json
import numpy as np
import time 
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.utils import GenerationConfig
from load_data import DataLoader, CoTLoader
from utils import get_prompter, build_chat_input, llama_generate, baichuan_generate, mistral_generate

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='Llama-2-13b-chat-hf')
parser.add_argument('--datalength', type=int, default=2000)
parser.add_argument('--dataset', type=str, default='csqa')
parser.add_argument('--scale', type=int, default=40)
parser.add_argument('--weight', type=float, default=2.0)
parser.add_argument('--num_candidates', type=int, default=4)
parser.add_argument('--res', action='store_true')
parser.add_argument('--test', action='store_true')

args = parser.parse_args()


model_name = args.model
dataset = args.dataset
datalength = args.datalength
scale_factor = args.scale
penalty_weights = args.weight
num_attn_candidates = args.num_candidates
res = args.res
test = args.test

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(17)
if model_name.startswith('Mistral'):
    model_path = f'/mnt/publiccache/huggingface/Mistral-7B-Instruct-v0.2'
elif '70b' in model_name:
    model_path = '/mnt/publiccache/huggingface/Llama-2-70b-chat-hf'
else:
    model_path = f'./model/{model_name}'
result_path = f'./result/{dataset}/{model_name}_res_answer_{datalength}_s{scale_factor}_w{penalty_weights}_c{num_attn_candidates}_r{res}_t{test}.json'
if model_name.startswith('Baichuan'):
    cot_file_path  = f'./result/{dataset}/{model_name}_cot_answer_2000.json'
    base_file_path = f'./result/{dataset}/{model_name}_direct_answer_2000.json'
    tokenizer = AutoTokenizer.from_pretrained(model_path,
        revision="v2.0",
        use_fast=False,
        trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path,
        revision="v2.0",
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True)
   
else:
    cot_file_path  = f'./result/{dataset}/{model_name}_cot_answer_2000.json'
    base_file_path = f'./result/{dataset}/{model_name}_direct_answer_2000.json'
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)  
model.eval()

prompter = get_prompter(model_name=model_name, dataset=dataset, task='cot_answer')
dataloader = DataLoader(dataset=dataset, data_length=datalength)
with open(cot_file_path, 'r') as f:
    cot_data = json.load(f)[:-1]
with open(base_file_path, 'r') as f:
    base_data = json.load(f)[:-1]

indexloader = CoTLoader()
_, index1 = indexloader.load_data(cot_file_path, base_file_path, mode='C2W', cnt=10)
if dataset == 'siqa' and model_name.startswith('Baichuan'):
    _, index2 = indexloader.load_data(cot_file_path, base_file_path, mode='W2C', cnt=11)
else:
    _, index2 = indexloader.load_data(cot_file_path, base_file_path, mode='W2C', cnt=10)
index = index1 + index2

def baichuan_get_key_position(question):
    question_len = len(tokenizer(question, return_tensors="pt").input_ids[0])
    question_msg = prompter.wrap_input(question, icl_cnt=5)
    question_ids = build_chat_input(model, tokenizer, question_msg)
    prompt_len = len(question_ids[0]) - question_len - 1
    stem_len = len(tokenizer(question.split('\n')[0],return_tensors="pt").input_ids[0])
    stem_len = prompt_len + stem_len
    key_position = {'start':prompt_len, 'end':stem_len}
    return key_position

def mistral_get_key_position(question):
    stem = '\n'.join(question.split('\n')[:-1])
    stem_msg = [{"role":"user", "content": stem}]
    question_len = len(tokenizer.apply_chat_template(stem_msg, return_tensors="pt")[0])
    question_msg = prompter.wrap_input(question, icl_cnt=5)
    prompt_len = len(tokenizer(question_msg[:-1], return_tensors="pt")[0])
    stem_len = prompt_len + question_len
    key_position = {'start':prompt_len, 'end':stem_len}
    return key_position

def llama_get_key_position(question):
    input = prompter.wrap_input(question, icl_cnt=5)
    question_len = len(prompter.user_prompt.format(question))
    prompt = input[:-question_len]
    stem = '\n'.join(input.split('\n')[:-1])
    stem_end = len(tokenizer(stem, return_tensors="pt").input_ids[0])
    stem_start = len(tokenizer(prompt, return_tensors="pt").input_ids[0]) - 1
    key_position = {'start':stem_start, 'end':stem_end}
    return key_position

def res_inference(question, **kwargs):
    if not kwargs:
        kwargs = {'scale_factor':scale_factor, 'num_attn_candidates':num_attn_candidates, 'penalty_weights':penalty_weights}
    kwargs['max_new_tokens'] = 200
    kwargs['do_sample'] = False
    kwargs['res_decoding'] = True
    input = prompter.wrap_input(question, icl_cnt=5)
    if model_name.startswith('Llama'):
        key_position = llama_get_key_position(question)
        kwargs['key_position'] = key_position
        # config = GenerationConfig.from_pretrained(model_path, **kwargs)
        result, pred = llama_generate(model, kwargs, tokenizer, input, 'cot_answer')  
    elif model_name.startswith('Mistral'):
        key_position = mistral_get_key_position(question)
        kwargs['key_position'] = key_position
        # config = GenerationConfig.from_pretrained(model_path, **kwargs)
        result, pred = mistral_generate(model, kwargs, tokenizer, input, 'cot_answer')  
    else:
        key_position = baichuan_get_key_position(question)
        kwargs['key_position'] = key_position
        config = GenerationConfig.from_pretrained(model_path, **kwargs)
        model.generation_config = config
        result, pred = baichuan_generate(model, tokenizer, input, 'cot_answer')
 
    match = re.findall(r'[1-5]\)',pred)
    if match:
        pred = match[-1][:-1]
    else:
        pred = 'None'
    
    return result, pred



if test:
    max_acc = 0
    max_index = -1
    idx = 0
    weight_ls = []
    max_results = []
    for scale_factor in [40,50,60,70,80,90]:
        for penalty_weights in [0.5, 1.0, 1.5, 2.0]:
            for num_attn_candidates in range(3, 11):
                config = {'scale_factor':scale_factor, 'num_attn_candidates':num_attn_candidates, 'penalty_weights':penalty_weights}
                weight_ls.append(config)
                print(config)
                correct = 0
                results = []
                dataloader.idx = 0
                cnt = 0
                for data in tqdm(dataloader):
                    question = data['question']
                    label = data['label']
                    cot_msg = cot_data[dataloader.idx - 1]
                    base_msg = base_data[dataloader.idx - 1]
                    if dataloader.idx - 1 not in index and test:
                        continue
                    result, pred = res_inference(question=question, **config)
                    cor_flag = (pred == label)
                    if cor_flag:
                        correct += 1
                    cnt += 1  
                    cot_msg = {'question':question, 'answer':result, 'pred':pred, 'label':label, 'cor_flag':cor_flag}
                    results.append(cot_msg)
                    
                    torch.cuda.empty_cache()

                acc = correct / cnt
                print(f'Acc: {acc}')
                if acc > max_acc:
                    max_acc = acc
                    max_index = idx
                    max_results = results
                    print(f'Acc: {max_acc}')
                    print(f'Config: {weight_ls[max_index]}')
                idx += 1
            
    print(f'Acc: {max_acc}')
    print(f'Config: {weight_ls[max_index]}')
    results = [{'config':weight_ls[max_index]}, {'acc':max_acc}]
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
else:
    correct = 0
    results = []
    dataloader.idx = 0
    cnt = 0
    cost = 0
    for data in tqdm(dataloader):
        start = time.time()
        question = data['question']
        label = data['label']
        cot_msg = cot_data[dataloader.idx - 1]
        base_msg = base_data[dataloader.idx - 1]
        # if cot_msg['pred'] == base_msg['pred']:
        #     res = False
        # else:
        #     res = True
        if cot_msg['pred'] == base_msg['pred'] and res:
            result = cot_msg['answer'] 
            pred = cot_msg['pred']
        else:
            result, pred = res_inference(question=question)
        cor_flag = (pred == label)
        if cor_flag:
            correct += 1
        cnt += 1  
        end = time.time()
        cost += end - start
        cot_msg = {'question':question, 'answer':result, 'pred':pred, 'label':label, 'cor_flag':cor_flag}
        results.append(cot_msg)
        
        torch.cuda.empty_cache()

    acc = correct / cnt
    print(f'Acc: {acc}')
    print(f'Time:{cost/cnt}')
    results.append({'acc':acc})
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4)