import os
import torch
import re
import argparse
import json
import time 
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModelForSeq2SeqLM
from load_data import DataLoader
from utils import llama_generate, baichuan_generate, get_config, get_prompter, chat_generate, mistral_generate


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='Llama-2-13b-chat-hf')
parser.add_argument('--datalength', type=int, default=2)
parser.add_argument('--dataset', type=str, default='siqa')
parser.add_argument('--task', type=str, default='direct_answer')
parser.add_argument('--icl', type=int, default=5)
parser.add_argument('--shuffle', action='store_true')
args = parser.parse_args()


model_name = args.model
dataset = args.dataset
datalength = args.datalength
task = args.task
icl = args.icl
shuffle = args.shuffle

if model_name.startswith('Vicuna'):
    model_path = f'/netcache/huggingface/vicuna-13b'
elif model_name.startswith('Mistral'):
    model_path = f'/mnt/publiccache/huggingface/Mistral-7B-Instruct-v0.2'
else:
    if '70b' in model_name:
        model_path = '/mnt/publiccache/huggingface/Llama-2-70b-chat-hf'
    else:    
        model_path = f'./model/{model_name}'
result_path = f'./result/{dataset}/{model_name}_{task}_{datalength}.json'

if model_name.startswith('Baichuan'):
    tokenizer = AutoTokenizer.from_pretrained(model_path,
        revision="v2.0",
        use_fast=False,
        trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path,
        revision="v2.0",
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True)
    model.eval()
elif model_name.startswith('Llama') or model_name.startswith('Vicuna') or model_name.startswith('Mistral'):
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)  
    model.eval()
    


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     torch.backends.cudnn.deterministic = True
setup_seed(17)



dataloader = DataLoader(dataset=dataset, data_length=datalength, shuffle=shuffle)
prompter = None 


correct = 0
cnt = 0
results = []


def model_generate(question):
    input = prompter.wrap_input(question, icl_cnt=5)
    if task == 'sc':
        config = get_config(model_name=model_name, strategy='beam')
    elif task in ['cons_answer', 'l2m']:
        config = get_config(model_name=model_name, strategy='sample')
    else:
        config = get_config(model_name=model_name, strategy='greedy')
    if model_name.startswith('Baichuan'):
        model.generation_config = config
        return baichuan_generate(model, tokenizer, input, task)
    elif model_name.startswith('Chat'):
        return chat_generate(input, task)    
    elif model_name.startswith('Mistral'):
        return mistral_generate(model, config, tokenizer, input, task)
    else:
        return llama_generate(model, config, tokenizer, input, task)

cost = 0
if task == 'dpr':
    path = f'./{task}_{dataset}_documents.json'
    with open(path, 'r') as f:
        documents = json.load(f)
for data in tqdm(dataloader):
    start = time.time()
    question = data['question']
    label = data['label']
    if task == 'l2m':
        prompter = get_prompter(model_name, dataset, 'l2m_question')
        result, _ = model_generate(question)
        split_result = result.split('\n')
        questions = []
        for q in split_result[1:]:
            if 'Question' in q:
                questions.append(q)
        prompter = get_prompter(model_name, dataset, 'l2m_mid_answer')
        for q in questions:
           question += '\n' + q
           result, _ = model_generate(question)
           question += " " + result.split('\n')[0]
        prompter = get_prompter(model_name, dataset, 'l2m_final_answer')
        result, pred = model_generate(question)
        prompter = get_prompter(model_name, dataset, 'l2m_question')
    elif task == 'sr':
        prompter = get_prompter(model_name, dataset, 'cot_answer')
        result, _ = model_generate(question)
        question += '\nRationale: ' + result
        prompter = get_prompter(model_name, dataset, 'sr_feedback')
        result, _ = model_generate(question)
        question += ' ' + result
        prompter = get_prompter(model_name, dataset, 'sr_answer')
        result, pred = model_generate(question)
    elif task == 'cons_answer':
        prompter = get_prompter(model_name, dataset, 'cons_answer')
        result, pred = model_generate(question)
    elif task == 'direct_answer':
        prompter = get_prompter(model_name, dataset, 'direct_answer')
        result, pred = model_generate(question)
    elif task == 'dpr':   
        if dataset in ['wino','piqa']:
            width = 2 
        elif dataset == 'hella':
            width = 3
        else:
            width = 4
        document = ""
        for i in range(cnt, cnt+width):
            document += documents[i]['ctxs'][0]['text'] + '. '
        question =  document + question  
        prompter = get_prompter(model_name, dataset, 'direct_answer')
        result, pred = model_generate(question)
    elif task == 'bm25':   
        prompter = get_prompter(model_name, dataset, 'cot_answer')
        result, pred = model_generate(question)
    else:   
        prompter = get_prompter(model_name, dataset, 'cot_answer')
        result, pred = model_generate(question)
    if dataset != 'gsm8k':
        match = re.findall(r'[1-5]\)',pred)
        if match:
            pred = match[0][:-1]
        else:
            pred = 'None'
    else:
        output = pred.split('\n')
        output = [line for line in output if len(re.findall('\d+', line)) > 0][-1]
        answer = output.replace(',', '')  # remove middle ',' from numbers like '1,234'
        answer = re.findall('\d+', answer)
        pred = label if label in answer else answer[-1]
        pred = answer.strip()
    cor_flag = (pred == label)
    cnt += 1
    end = time.time()
    cost += end - start 
    if cor_flag:
        correct += 1
    msg = {'question':question, 'answer':result, 'pred':pred, 'label':label, 'cor_flag':cor_flag}
    results.append(msg)
    torch.cuda.empty_cache()
    
results.append({'acc':correct/cnt})
print(f'Acc:{correct/cnt}')
print(f'Time:{cost/cnt}')
with open(result_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=4)