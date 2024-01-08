import torch
import argparse
import json
import numpy as np
import torch.nn.functional as F
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModelForSeq2SeqLM
from prompts.wrap_prompt import LlamaPrompter
from load_data import DataLoader, CoTLoader
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
from metrics import draw_acc

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='Llama-2-13b-chat-hf')
parser.add_argument('--datalength', type=int, default=2)
parser.add_argument('--dataset', type=str, default='csqa')
parser.add_argument('--task', type=str, default='res_cot_answer')
args = parser.parse_args()

model_name = args.model
dataset = args.dataset
datalength = args.datalength
task = args.task

model_path = f'./model/{model_name}'
cot_file_path  = f'./result/{dataset}/{model_name}_cot_answer_dev_1000.json'
base_file_path = f'./result/{dataset}/{model_name}_direct_answer_dev_1000.json'
if dataset == 'csqa':  
    res_cot_file_path = f'./result/csqa/res_result_cot_answer_greedy_1000_s-90_w-1.5_c-10_r-True.json'
elif dataset == 'wino':
    res_cot_file_path = f'./result/wino/res_result_cot_answer_greedy_1000_s-40_w-2.0_c-4.json'
result_path = f'./result/{dataset}/test.json'

model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)  
prompter = LlamaPrompter(dataset=dataset, task=task)
dataloader = DataLoader(dataset=dataset, data_length=datalength)
with open(cot_file_path, 'r') as f:
    cot_data = json.load(f)
    f.close()
with open(base_file_path, 'r') as f:
    base_data = json.load(f)
    f.close()
with open(res_cot_file_path, 'r') as f:
    res_cots = json.load(f)
    f.close()

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

def lm_logit(input_text, layers):
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
    pred_ids = input_ids[:,-2]
    dict_outputs, _ = model(
        input_ids=input_ids,
        return_dict=True,
        output_attentions=False,
        output_hidden_states=False,
        early_exit_layers=layers,
    )
    scores = []
    for layer in layers:
        logits = dict_outputs[layer][:, -3, :].float()
        logits = F.softmax(logits, dim=-1)
        probs = logits[:, pred_ids[0]].squeeze().cpu().numpy()
        scores.append(probs)
    del input_ids, dict_outputs
    torch.cuda.empty_cache()
    return scores

def llm_score(question, pred, cot, res_cot, layers, diff=True):
    with torch.no_grad():
        cot_question = prompter.wrap_input(question, icl_cnt=5) 
        scores = lm_logit(cot_question + cot + f'. So the answer is: ({pred})', layers)
        res_scores = lm_logit(cot_question + res_cot + f'. So the answer is: ({pred})', layers)
        if diff:
            diff_scores = np.array(res_scores) - np.array(scores)
            return diff_scores
        else:
            return scores, res_scores

correct = 0
cnt = 0
cot_spliter = CoTLoader()
results = []
for msg in tqdm(dataloader):
    idx = dataloader.idx - 1
    question = msg['question']
    label = msg['label']
    cot = cot_spliter.split_cot(res_cots[idx]['answer'])
    cot = '.'.join(cot)
    input = f'Knowledge: {cot}. ' + f'\nQuestion: {question}\n'
    input_text = prompter.wrap_input(question, icl_cnt=5) + input
    inputs = tokenizer(input_text, return_tensors="pt")
    # question_len = len(prompter.user_prompt.format(question))
    # prompt = prompter.wrap_input(question, icl_cnt=5)[:-question_len]
    # stem = '\n'.join(input.split('\n')[:-1])
    # stem_end = len(tokenizer(stem, return_tensors="pt").input_ids[0])
    # stem_start = len(tokenizer(prompt, return_tensors="pt").input_ids[0]) - 1
    # key_position = {'start':stem_start, 'end':stem_end}
    input_ids = inputs["input_ids"].to(model.device)
    output = model.generate(
            input_ids=input_ids,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=200,
            stopping_criteria=[stop_criteria],
        )
    result = tokenizer.decode(output.sequences[0]).split(' [/INST] ')[-1]
    results.append(result)
    # result = result.split('\nNote')[0]
    split_result = result.split(':')
    pred = split_result[-1].strip()
    match = re.findall(r'[1-5]\)',pred)
    if match:
        pred = match[-1][:-1]
    else:
        pred = 'None'
        
    cnt += 1
    if pred == label:
        cor_flag = True
        correct += 1
    else:
        cor_flag = False
    msg = {'question':question, 'answer':result, 'pred':pred, 'label':label, 'cor_flag':cor_flag}
    results.append(msg)
    
print(f'Acc:{correct / cnt}')
with open(result_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=4)