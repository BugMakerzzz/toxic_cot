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
args = parser.parse_args()


model_name = args.model
dataset = args.dataset
datalength = args.datalength
split = args.split
task = args.task
model_path = f'./model/{model_name}'
result_path = f'./result/{dataset}/{model_name}_{task}_{split}_{datalength}.json'

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


def split_cot(text):
    if text[-1] != '.':
        text += '.'
    answer = text.split('.')[-1]
    text = text.replace(answer, '')
    cots = text.split('.')[:-1]
    return cots

def lm_score(input_text1, input_text2, mature_layer=None, premature_layer=None, post_softmax=True, **kwargs):
    with torch.no_grad():
        input_text = input_text1 + input_text2
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
        prefix_ids = tokenizer(input_text1, return_tensors="pt").input_ids.to(model.device)
        continue_ids = input_ids[0, prefix_ids.shape[-1]:]

        dict_outputs, outputs = model(
            input_ids=input_ids,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
            early_exit_layers=[premature_layer, mature_layer],
        )

        assert premature_layer is not None
        base_logits = dict_outputs[premature_layer][0, prefix_ids.shape[-1] - 1: -1, :]
        final_logits = dict_outputs[mature_layer][0, prefix_ids.shape[-1] - 1: -1, :]
        final_logits = final_logits.log_softmax(dim=-1)
        base_logits = base_logits.log_softmax(dim=-1)
        base_probs = base_logits[range(base_logits.shape[0]), continue_ids].sum().item()
        final_probs = final_logits[range(final_logits.shape[0]), continue_ids].sum().item()
        diff_logits = final_logits - base_logits
        if post_softmax:
            diff_logits = diff_logits.log_softmax(dim=-1)
            
        log_probs = diff_logits[range(diff_logits.shape[0]), continue_ids].sum().item()

    return  base_probs, final_probs, log_probs

with open('./result/wino/gpt-3.5-turbo-0613_2_i5_s1_l1000.json', 'r') as f:
    dataloader = json.load(f)

correct = 0
cnt = 0
result = []
corrects = [0] * 41
for data in tqdm(dataloader):
    cnt += 1
    if 'question' not in data.keys():
        break
    question = data['question']
    cot = data['answer']
    cots = split_cot(cot)
    suffix = '\nSo the answer is:'
    prefix = question
    label = data['label']
    wr_label = '1' if label == '2' else '2'
    options = question.split('\n')[-1].split(' ')
    cor_answer = options[eval(label)*2-2] + ' ' + options[eval(label)*2-1]
    wr_answer = options[eval(wr_label)*2-2] + ' ' + options[eval(wr_label)*2-1]
    torch.set_grad_enabled(False)
    model.eval()
    scores = []
    wr_scores = []
    for cot in cots:
        prefix += cot
    for i in range(1, 41):
        score, _, _  = lm_score(prefix+suffix, cor_answer, mature_layer=-1, premature_layer=i)
        wr_score, _, _ = lm_score(prefix+suffix, wr_answer, mature_layer=-1, premature_layer=i)
        if score > wr_score:
            corrects[i] += 1
    else:
        result.append({'question': question, 'answer':cot, 'label':label})
    # if cnt >= 100:
    #     break
for correct in corrects:        
    print(correct / cnt)