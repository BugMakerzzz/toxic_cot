import torch
import argparse
import json
import numpy as np
import torch.nn.functional as F
import re
import time
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils import llama_generate, baichuan_generate, get_config, get_prompter, chat_generate, mistral_generate
from load_data import DataLoader, CoTLoader
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='Llama-2-13b-chat-hf')
parser.add_argument('--datalength', type=int, default=2)
parser.add_argument('--dataset', type=str, default='csqa')
parser.add_argument('--res', action='store_true')
args = parser.parse_args()

model_name = args.model
dataset = args.dataset
datalength = args.datalength
res = args.res

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     torch.backends.cudnn.deterministic = True
seed = 17
setup_seed(seed)

cot_file_path  = f'./result/{dataset}/{model_name}_cot_answer_2000.json'
base_file_path = f'./result/{dataset}/{model_name}_direct_answer_2000.json'
res_cot_file_path = f'./result/{dataset}/{model_name}_res_.json'
if model_name.startswith('Vicuna'):
    model_path = f'/netcache/huggingface/vicuna-13b'
elif model_name.startswith('Mistral'):
    model_path = f'/mnt/publiccache/huggingface/Mistral-7B-Instruct-v0.2'
    cot_file_path  = f'./result/{dataset}/{model_name}_cot_answer_2000.json'
    base_file_path = f'./result/{dataset}/{model_name}_direct_answer_2000.json'
else:
    if '70b' in model_name:
        model_path = '/mnt/publiccache/huggingface/Llama-2-70b-chat-hf'
    else:    
        model_path = f'./model/{model_name}'

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
result_path = f'./result/{dataset}/{model_name}_rt_result_{datalength}_r{res}.json'

dataloader = DataLoader(dataset=dataset, data_length=datalength)
with open(cot_file_path, 'r') as f:
    cot_data = json.load(f)
    f.close()
with open(base_file_path, 'r') as f:
    base_data = json.load(f)
    f.close()
# if res:
#     with open(res_cot_file_path, 'r') as f:
#         res_cots = json.load(f)
#         f.close()


def lm_logit(input_text):
    with torch.no_grad():
        if model_name.startswith('Mistral'):
            inputs = tokenizer.apply_chat_template(input_text, return_tensors="pt")
            input_ids = inputs.to(model.device)
            pred_ids = input_ids[:,-6]
        else:
            input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
            pred_ids = input_ids[:,-2]
        outputs = model(
            input_ids=input_ids,
        )
        if model_name.startswith('Mistral'):
            logits = outputs[0][:, -7, :].float()
        else:
            logits = outputs[0][:, -3, :].float()
        logits = F.softmax(logits, dim=-1)
        score = logits[:, pred_ids[0]].squeeze().cpu().numpy()
        del input_ids, outputs
        torch.cuda.empty_cache()
        return score
        

correct = 0
cnt = 0
cot_spliter = CoTLoader()
results = []

def model_generate(question):
    input = prompter.wrap_input(question, icl_cnt=5)
    config = get_config(model_name=model_name, strategy='greedy')
    if model_name.startswith('Baichuan'):
        model.generation_config = config
        return baichuan_generate(model, tokenizer, input, task='rt_answer')
    elif model_name.startswith('Chat'):
        return chat_generate(input, task='rt_answer')        
    elif model_name.startswith('Mistral'):
        return mistral_generate(model, config, tokenizer, input, task='rt_answer')
    else:
        return llama_generate(model, config, tokenizer, input, task='rt_answer')


cost = 0
for msg in tqdm(dataloader):
    start = time.time()
    idx = dataloader.idx - 1
    question = msg['question']
    label = msg['label']
    # if res:
    #     cot = cot_spliter.split_cot(res_cots[idx]['answer'])
    # else:
    cot = cot_spliter.split_cot(cot_data[idx]['answer'])
    cot = '.'.join(cot) + '.'
    prompter = get_prompter(model_name, dataset, 'rt_answer')
    input = f'Rationale: {cot}' + f'\nQuestion: {question}'
    
    
    pred1 = base_data[idx]['pred']
    pred2 = cot_data[idx]['pred']
    if pred1 == pred2 and res:
        pred = pred1
    else:
        result, pred = model_generate(input)
    #     prompter = get_prompter(model_name, dataset, 'cot_answer')
    #     rt_input = f'Rationale: {cot}' + f'\nQuestion: {question}'
    #     # print(rt_input)
    #     # prompt = rt_input
    #     prompt = prompter.wrap_input(rt_input, icl_cnt=5) 
    #     if model_name.startswith('Mistral'):
    #         input1 = prompt
    #         input1[-1]['content'] = prompt[-1]['content'] + f'Answer: ({pred1})'
    #         score1 = lm_logit(input1)
    #         input2 = prompt
    #         input2[-1]['content'] = prompt[-1]['content'] + f'Answer: ({pred2})'
    #         score2 = lm_logit(input2)
    #     else:
    #         score1 = lm_logit(prompt + f'Answer: ({pred1})')
    #         score2 = lm_logit(prompt + f'Answer: ({pred2})')
 
    #     if score1 > score2:
    #         pred = pred1 
    #     else:
    #         pred = pred2
    #     answer = res_cots[idx]['answer']
    #     # path = f'./result/{dataset}/fig-{idx}.png'
    #     # draw_plot(layers=[i+1 for i in range(40)], scores=[res_pred_scores, res_label_scores], labels=['pred','label'], path=path)
    # else:
    #     match = re.findall(r'[1-5]\)',pred)
    #     if match:
    #         pred = match[0][:-1]
    #     else:
    #         pred = 'None'
        print(pred)
        answer = pred.replace(',', '')  # remove middle ',' from numbers like '1,234'
        match = re.findall('\d+', answer)
        if match:
            answer = re.findall('\d+', answer)[-1]
        else:
            answer = 'None'
        pred = answer.strip()
        pred = pred.lstrip('0')
        if pred.isdigit():
            pred = eval(pred)
        else:
            pred = -114514
    print(pred)
    # end = time.time()
    # cost += end - start
    cor_flag = (pred == label)
    if cor_flag:
        correct += 1
    msg = {'question':question, 'pred':pred, 'label':label, 'cor_flag':cor_flag}
    results.append(msg)
    cnt += 1
    
# print(f'Acc:{correct / cnt}')
print(f'Time:{cost / cnt}')
results.append({'acc':correct / cnt})
with open(result_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=4)