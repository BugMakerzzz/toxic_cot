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
parser.add_argument('--strategy', type=str, default=None)
args = parser.parse_args()


model_name = args.model
dataset = args.dataset
datalength = args.datalength
split = args.split
task = args.task
icl = args.icl
strategy = args.strategy
model_path = f'./model/{model_name}'
result_path = f'./result/{dataset}/{model_name}_{task}_{split}_{datalength}.json'
if strategy:
    result_path = f'./result/{dataset}/{model_name}_{task}_{split}_{datalength}_{strategy}.json'

# config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
# with init_empty_weights():
#     model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16, trust_remote_code=True)
#     # model = AutoModelForSeq2SeqLM.from_config(config, torch_dtype=torch.float16, trust_remote_code=True)
# no_split_modules = model._no_split_modules
# model = load_checkpoint_and_dispatch(
#     model, model_path, device_map="auto", no_split_module_classes=no_split_modules
# )
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)  
prompter = LlamaPrompter(dataset=dataset, task=task)
dataloader = DataLoader(dataset=dataset, data_length=datalength, split=split)
# sent_model = SentenceTransformer('./model/all-mpnet-base-v2')
model.eval()

def model_generate(question, strategy):
    with torch.no_grad():
        input = prompter.wrap_input(question, icl_cnt=icl)
        inputs = tokenizer(input, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        if strategy == 'greedy':
            output = model.generate(
                    input_ids=input_ids,
                    return_dict_in_generate=True,
                    max_new_tokens=500,
                    do_sample=False,
                    stopping_criteria=[stop_criteria],
                )
            result = tokenizer.decode(output.sequences[0]).split(' [/INST] ')[-1]
            pred = result.split(':')[-1]
        elif strategy == 'beam':
            output = model.generate(
                    input_ids=input_ids,
                    return_dict_in_generate=True,
                    max_new_tokens=500,
                    num_beams=5,
                    # diversity_penalty=1.0,
                    do_sample=True,
                    temperature=0.5,
                    # num_beam_groups=3,
                    top_k=10,
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
        else:
            output = model.generate(
                    input_ids=input_ids,
                    return_dict_in_generate=True,
                    max_new_tokens=500,
                    stopping_criteria=[stop_criteria],
                )
            result = tokenizer.decode(output.sequences[0]).split(' [/INST] ')[-1]
            pred = result.split(':')[-1]
        del input_ids
        del output
        
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
                
        return result, pred 


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
cnt = 0
results = []

for data in tqdm(dataloader):
    question = data['question']
    label = data['label']
    idx = dataloader.idx - 1
    result, pred = model_generate(question, strategy)
    cor_flag = (pred == label)
    cnt += 1
    if cor_flag:
        correct += 1
    msg = {'question':question, 'answer':result, 'pred':pred, 'label':label, 'cor_flag':cor_flag}
    results.append(msg)
    
    torch.cuda.empty_cache()
    
results.append({'acc':correct/cnt})
print(f'Acc:{correct/cnt}')
with open(result_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=4)