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
parser.add_argument('--dataset', type=str, default='csqa')
args = parser.parse_args()


model_name = args.model
dataset = args.dataset
datalength = args.datalength

model_path = f'./model/{model_name}'
result_path = f'./result/{dataset}/res_result_{datalength}.json'

model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)  
prompter = LlamaPrompter(dataset=dataset, task='cot_answer')
dataloader = DataLoader(dataset=dataset, data_length=datalength)

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
# index = [41,49,158,161,174,219,244,276,283,286,297,386,394,402,413,424,431,441,443,457]
index = [7,15,50,53,84,97,108,119,121,132,201,207,209,235,253]
for data in tqdm(dataloader):
    if dataloader.idx - 1 not in index:
        continue
    question = data['question']
    input = prompter.wrap_input(question, icl_cnt=5)
    label = data['label']
    torch.set_grad_enabled(False)
    model.eval()
    inputs = tokenizer(input, return_tensors="pt")
    
    question_len = len(prompter.user_prompt.format(question))
    prompt = prompter.wrap_input(question, icl_cnt=5)[:-question_len]
    stem = '\n'.join(input.split('\n')[:-1])
    stem_end = len(tokenizer(stem, return_tensors="pt").input_ids[0])
    stem_start = len(tokenizer(prompt, return_tensors="pt").input_ids[0]) - 1
    key_position = {'start':stem_start, 'end':stem_end}
    
    input_ids = inputs["input_ids"].to(model.device)
    
    output = model.generate(
            input_ids=input_ids,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=500,
            num_beams=5,
            res_decoding=True,
            do_sample=False,
            num_return_sequences=3,
            key_position=key_position,
            scale_factor=10,
            num_attn_candidates=5,
            penalty_weights=5,
            stopping_criteria=[stop_criteria],
        )
    result_ls = []
    preds = []
    for i in range(3):
        result = tokenizer.decode(output.sequences[i]).split(' [/INST] ')[-1]
        result = result.split('\nNote')[0]
        pred = result.split(':')[-1].strip()
        preds.append(pred)
        result_ls.append(result)
    pred = max(preds,key=preds.count)
    result = result_ls
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
    

results.append({'acc':correct/len(index)})
print(f'Acc:{correct/len(index)}')
with open(result_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=4)