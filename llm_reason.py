# import os
# import torch
# import re
# import argparse
# import json
# import numpy as np
# from tqdm import tqdm
# from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModelForSeq2SeqLM
# from prompts.wrap_prompt import LlamaPrompter
# from load_data import DataLoader
# from accelerate import init_empty_weights, load_checkpoint_and_dispatch
# from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList


# parser = argparse.ArgumentParser()
# parser.add_argument('--model', type=str, default='Llama-2-13b-chat-hf')
# parser.add_argument('--datalength', type=int, default=2)
# parser.add_argument('--split', type=str, default='dev')
# parser.add_argument('--dataset', type=str, default='siqa')
# parser.add_argument('--task', type=str, default='direct_answer')
# parser.add_argument('--shuffle', type=str, default=False)
# args = parser.parse_args()


# model_name = args.model
# dataset = args.dataset
# datalength = args.datalength
# split = args.split
# task = args.task
# shuffle = args.shuffle
# model_path = f'./model/{model_name}'
# result_path = f'./result/{dataset}/{model_name}_{task}_{split}_{datalength}_{shuffle}.json'

# config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
# with init_empty_weights():
#     model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16, trust_remote_code=True)
#     # model = AutoModelForSeq2SeqLM.from_config(config, torch_dtype=torch.float16, trust_remote_code=True)
# no_split_modules = model._no_split_modules
# model = load_checkpoint_and_dispatch(
#     model, model_path, device_map="auto", no_split_module_classes=no_split_modules
# )
# tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)  
# prompter = LlamaPrompter(dataset=dataset, task=task)
# dataloader = DataLoader(dataset=dataset, data_length=datalength, split=split, shuffle=shuffle)

# class KeywordsStoppingCriteria(StoppingCriteria):
#     def __init__(self, keywords_ids:list):
#         self.keywords = keywords_ids

#     def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
#         if input_ids[0][-1] in self.keywords:
#             return True
#         return False

# stop_words = ['</s>', '<s>', '</s><s>']
# stop_ids = [tokenizer.encode(w)[0] for w in stop_words]
# stop_criteria = KeywordsStoppingCriteria(stop_ids)

# def score_generate(input, 
#                    layers=[1,5,10,15,20,25,30,35,40], 
#                    return_dict_in_generate=True,
#                    max_new_tokens=500,
#                    stopping_criteria=[stop_criteria]):
#     question_ids = tokenizer(input, return_tensors="pt").input_ids.to(model.device)
#     dict_outputs = []
#     output = model.generate(
#             input_ids=question_ids,
#             return_dict_in_generate=return_dict_in_generate,
#             max_new_tokens=max_new_tokens,
#             stopping_criteria=stopping_criteria,
#             # early_exit_layers=layers,
#             # split_token_ids=29889,
#         )
#     reg_logits = []
#     for dict_output in dict_outputs:
#         dict_logits = []
#         for layer in layers:
#             logits = dict_output[layer][0, -1, :]
#             dict_logits.append(logits.log_softmax(dim=-1).tolist())
#         dict_logits = np.array(dict_logits)
#         del dict_output[layer]
#         reg_logits.append(dict_logits)
#     options = input.split('\n')[-1].split('(')[1:]
#     option_scores = []
#     for option in options:
#         scores = []
#         full_text = input + ' So the answer is: (' + option.split(')')[0] + ')'
#         full_ids = tokenizer(full_text, return_tensors="pt").input_ids
#         answer_ids = full_ids[0, question_ids.shape[-1]:]
#         for dict_logits in reg_logits:
#             probs = dict_logits[:, answer_ids].sum(axis=-1).tolist()
#             scores.append(probs)
#         option_scores.append(scores)    
#     result = tokenizer.decode(output.sequences[0]).split(' [/INST] ')[-1]
#     pred = result.split(':')[-1]
#     match = re.findall(r'[1-5]\)',pred)
#     if match:
#         pred = match[-1][:-1]
#     else:
#         pred = 'None'
#     del question_ids
#     del output
#     torch.cuda.empty_cache()
#     return option_scores, result, pred


# correct = 0
# results = []
# for data in tqdm(dataloader):
#     question = data['question']
#     input = prompter.wrap_input(question, icl_cnt=5)
#     label = data['label']
#     torch.set_grad_enabled(False)
#     model.eval()
#     # inputs = tokenizer(input, return_tensors="pt")
#     scores, result, pred = score_generate(input)
#     cor_flag = (pred == label)
#     if cor_flag:
#         correct += 1
#     msg = {'question':question, 'answer':result, 'pred':pred, 'label':label, 'cor_flag':cor_flag, 'scores':scores}
#     results.append(msg)
    
# results.append({'acc':correct/datalength})
# print(f'Acc:{correct/datalength}')
# with open(result_path, 'w', encoding='utf-8') as f:
#     json.dump(results, f, indent=4)

### 以上为加了生成CoT过程中打分的版本

import os
import torch
import re
import argparse
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModelForSeq2SeqLM
from prompts.wrap_prompt import LlamaPrompter
from load_data import DataLoader
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='Llama-2-13b-chat-hf')
parser.add_argument('--datalength', type=int, default=2)
parser.add_argument('--split', type=str, default='dev')
parser.add_argument('--dataset', type=str, default='siqa')
parser.add_argument('--task', type=str, default='direct_answer')
parser.add_argument('--icl', type=int, default=5)
parser.add_argument('--search', type=str, default='greedy')
args = parser.parse_args()


model_name = args.model
dataset = args.dataset
datalength = args.datalength
split = args.split
task = args.task
icl = args.icl
search = args.search
model_path = f'./model/{model_name}'
result_path = f'./result/{dataset}/{model_name}_{task}_{split}_{datalength}.json'
if search == 'beam':
    result_path = f'./result/{dataset}/{model_name}_{task}_{split}_{datalength}_beam.json'

config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16, trust_remote_code=True)
    # model = AutoModelForSeq2SeqLM.from_config(config, torch_dtype=torch.float16, trust_remote_code=True)
no_split_modules = model._no_split_modules
model = load_checkpoint_and_dispatch(
    model, model_path, device_map="auto", no_split_module_classes=no_split_modules
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)  
prompter = LlamaPrompter(dataset=dataset, task=task)
dataloader = DataLoader(dataset=dataset, data_length=datalength, split=split)
sent_model = SentenceTransformer('./model/all-mpnet-base-v2')
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

correct = 0
results = []
for data in tqdm(dataloader):
    question = data['question']
    input = prompter.wrap_input(question, icl_cnt=icl)
    label = data['label']
    torch.set_grad_enabled(False)
    model.eval()
    inputs = tokenizer(input, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    if search == 'greedy':
        output = model.generate(
                input_ids=input_ids,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=500,
                stopping_criteria=[stop_criteria],
            )
        result = tokenizer.decode(output.sequences[0]).split(' [/INST] ')[-1]
        pred = result.split(':')[-1]
    elif search == 'beam':
        output = model.generate(
                input_ids=input_ids,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=500,
                num_beams=5,
                # diversity_penalty=1.0,
                do_sample=True,
                temperature=1.3,
                # num_beam_groups=3,
                top_k=7,
                top_p=0.9,
                num_return_sequences=3,
                stopping_criteria=[stop_criteria],
            )
        result_ls = []
        preds = []
        for i in range(3):
            result = tokenizer.decode(output.sequences[i]).split(' [/INST] ')[-1]
            pred = result.split(':')[-1].strip()
            preds.append(pred)
            result_ls.append(result)
        pred = max(preds,key=preds.count)
        result = result_ls
    if dataset == 'gsm8k':
        match = re.findall(r'[+-]?\d+',pred)
        if match:
            pred = int(match[0])
        else:
            pred = 0
    else:
        match = re.findall(r'[1-5]\)',pred)
        if match:
            pred = match[-1][:-1]
        else:
            pred = 'None'
    cor_flag = (pred == label)
    if cor_flag:
        correct += 1
    msg = {'question':question, 'answer':result, 'pred':pred, 'label':label, 'cor_flag':cor_flag}
    results.append(msg)
    del input_ids
    del output
    torch.cuda.empty_cache()
    
results.append({'acc':correct/datalength})
print(f'Acc:{correct/datalength}')
with open(result_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=4)