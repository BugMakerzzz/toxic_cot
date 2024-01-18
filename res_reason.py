import os
import torch
import re
import argparse
import json
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModelForSeq2SeqLM
from prompts.wrap_prompt import LlamaPrompter
from load_data import DataLoader, CoTLoader
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='Llama-2-13b-chat-hf')
parser.add_argument('--datalength', type=int, default=2)
parser.add_argument('--dataset', type=str, default='csqa')
parser.add_argument('--task', type=str, default='cot_answer')
parser.add_argument('--strategy', type=str, default='sample')
parser.add_argument('--scale', type=int, default=None)
parser.add_argument('--weight', type=float, default=None)
parser.add_argument('--num_candidates', type=int, default=None)
parser.add_argument('--temperature', type=float, default=0.5)
parser.add_argument('--top_p', type=float, default=0.9)
parser.add_argument('--res', action='store_true')
parser.add_argument('--test', action='store_true')

args = parser.parse_args()


model_name = args.model
dataset = args.dataset
datalength = args.datalength
task = args.task
strategy = args.strategy
scale_factor = args.scale
penalty_weights = args.weight
num_attn_candidates = args.num_candidates
temperature = args.temperature
top_p = args.top_p
res = args.res
test = args.test

model_path = f'./model/{model_name}'
result_path = f'./result/{dataset}/res_result_{strategy}_{datalength}_s{scale_factor}_w{penalty_weights}_c{num_attn_candidates}_r{res}_t{test}.json'
cot_file_path  = f'./result/{dataset}/{model_name}_cot_answer_dev_2000_greedy.json'
base_file_path = f'./result/{dataset}/{model_name}_direct_answer_dev_2000.json'

model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)  
prompter = LlamaPrompter(dataset=dataset, task=task)
dataloader = DataLoader(dataset=dataset, data_length=datalength)
with open(cot_file_path, 'r') as f:
    cot_data = json.load(f)[:-1]
with open(base_file_path, 'r') as f:
    base_data = json.load(f)[:-1]


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords_ids:list):
        self.keywords = keywords_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids[0][-1] in self.keywords:
            return True
        return False

stop_words = ['</s>', '<s>', '</s><s>']
stop_ids = [tokenizer.encode(w)[0] for w in stop_words]
stop_criteria = KeywordsStoppingCriteria(stop_ids)

if dataset == 'csqa':
    index1 = [41, 49, 158, 161, 174, 244, 276, 283, 286, 297, 386, 394, 402, 413, 424, 431, 441, 443, 457, 523, 539, 652, 700, 709, 754, 869, 881, 898, 946]
    index2 = [2, 5, 7, 14, 26, 31, 34, 35, 48, 58, 66, 75, 92, 96, 103, 109, 122, 125, 126, 127, 175, 184, 185, 186, 191, 200, 209, 218, 245, 247]
    index1 = index1[:10]
    index2 = index2[:10]
    index = index1 + index2
elif dataset == 'wino':
    index1 = [1, 2, 20, 22, 23, 32, 34, 38, 42, 46, 54, 59, 62, 65, 77, 79, 81, 85, 89, 91, 96, 99, 101, 103, 104, 112, 122, 124, 127, 142, 144, 145, 170, 179, 182, 190, 191, 198, 205, 210, 212, 213, 215, 229, 239, 243, 247, 256, 265, 276, 279, 283, 291, 294, 297, 303, 311, 328, 335, 348, 349, 353, 356, 358, 360, 370, 371, 372, 379, 380, 384, 388, 401, 405, 414, 435, 437, 441, 442, 443, 452, 458, 462, 464, 470, 474, 484, 491, 505, 507, 509, 510, 513, 514, 517, 520, 523, 528, 532, 534, 544, 550, 554, 555, 564, 566, 576, 585, 591, 603, 606, 617, 621, 623, 629, 633, 639, 641, 644, 648, 655, 663, 665, 667, 672, 681, 685, 688, 691, 697, 702, 707, 719, 743, 744, 747, 748, 750, 763, 767, 778, 790, 803, 816, 820, 825, 826, 829, 830, 840, 841, 843, 853, 856, 871, 875, 876, 880, 889, 891, 897, 899, 901, 902, 904, 907, 909, 910, 914, 915, 921, 933, 941, 942, 944, 948, 956, 958, 965, 972, 974, 980, 993]
    index2 = [7, 15, 50, 53, 97, 108, 119, 121, 132, 201, 207, 209, 235, 253, 284, 285, 307, 316, 320, 338, 342, 347, 387, 390, 426]
    index1 = index1[:10]
    index2 = index2[:10]
    index = index1 + index2
else:
    indexloader = CoTLoader()
    _, index1 = indexloader.load_data(cot_file_path, base_file_path, mode='C2W', cnt=10)
    _, index2 = indexloader.load_data(cot_file_path, base_file_path, mode='W2C', cnt=10)
    index = index1 + index2

def res_inference(question):
    with torch.no_grad():
        input = prompter.wrap_input(question, icl_cnt=5)
        model.eval()
        inputs = tokenizer(input, return_tensors="pt")
        
        question_len = len(prompter.user_prompt.format(question))
        prompt = prompter.wrap_input(question, icl_cnt=5)[:-question_len]
        stem = '\n'.join(input.split('\n')[:-1])
        stem_end = len(tokenizer(stem, return_tensors="pt").input_ids[0])
        stem_start = len(tokenizer(prompt, return_tensors="pt").input_ids[0]) - 1
        question_end = len(tokenizer(input, return_tensors="pt").input_ids[0])
        key_position = {'start':stem_start, 'end':stem_end}
        
        input_ids = inputs["input_ids"].to(model.device)
        # if dataloader.idx - 1 in index:
        if strategy == 'greedy':
            output = model.generate(
                input_ids=input_ids,
                return_dict_in_generate=True,
                max_new_tokens=200,
                res_decoding=True,
                do_sample=False,
                key_position=key_position,
                scale_factor=scale_factor,
                num_attn_candidates=num_attn_candidates,
                penalty_weights=penalty_weights,
                stopping_criteria=[stop_criteria],
            )
            result = tokenizer.decode(output.sequences[0]).split(' [/INST] ')[-1]
            # result = result.split('\nNote')[0]
            split_result = result.split(':')
            pred = split_result[-1].strip()
        elif strategy == 'beam':
            output = model.generate(
                    input_ids=input_ids,
                    return_dict_in_generate=True,
                    max_new_tokens=200,
                    num_beams=5,
                    res_decoding=True,
                    do_sample=True,
                    temperature=temperature,
                    top_k=num_attn_candidates,
                    top_p=top_p,
                    num_return_sequences=3,
                    key_position=key_position,
                    scale_factor=scale_factor,
                    num_attn_candidates=num_attn_candidates, 
                    penalty_weights=penalty_weights,
                    stopping_criteria=[stop_criteria],
                )
            result_ls = []
            preds = []
            for i in range(3):
                result = tokenizer.decode(output.sequences[i]).split(' [/INST] ')[-1]
                # result = result.split('\nNote')[0]
                split_result = result.split(':')
                pred = split_result[-1].strip()
                preds.append(pred)
                result_ls.append(result)
            result = result_ls
            pred = max(preds,key=preds.count)
        else:
            output = model.generate(
                input_ids=input_ids,
                return_dict_in_generate=True,
                max_new_tokens=200,
                res_decoding=True,
                key_position=key_position,
                scale_factor=scale_factor,
                num_attn_candidates=num_attn_candidates,
                penalty_weights=penalty_weights,
                stopping_criteria=[stop_criteria],
            )
            result = tokenizer.decode(output.sequences[0]).split(' [/INST] ')[-1]
            # result = result.split('\nNote')[0]
            split_result = result.split(':')
            pred = split_result[-1].strip()
        match = re.findall(r'[1-5]\)',pred)
        if match:
            pred = match[-1][:-1]
        else:
            pred = 'None'
            
        del input_ids,output
    
    return result, pred

max_acc = 0
max_index = -1
idx = 0
weight_ls = []
max_results = []
# scale_ls = [30,40,50,60,70,80,90] if dataset == 'wino' else [100,90,80,70,60,50,40]

for scale_factor in [100,110]:
    for penalty_weights in [1.5, 2.0]:
        for num_attn_candidates in range(5, 11):
            config = {'s':scale_factor, 'p':penalty_weights, 'n':num_attn_candidates}
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
                if cot_msg['pred'] == base_msg['pred'] and res:
                    result = cot_msg['answer'] 
                    pred = cot_msg['pred']
                else:
                    result, pred = res_inference(question=question)

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
max_results.append({'acc':max_acc})
with open(result_path, 'w', encoding='utf-8') as f:
    json.dump(max_results, f, indent=4)
# print(f'Acc: {acc}')
# results.append({'acc':acc})
# with open(result_path, 'w', encoding='utf-8') as f:
#     json.dump(results, f, indent=4)