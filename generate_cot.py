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
# parser.add_argument('--icl', type=int, default=5)
args = parser.parse_args()


model_name = args.model
dataset = args.dataset
datalength = args.datalength
split = args.split
# icl = args.icl
model_path = f'./model/{model_name}'
result_path = f'./result/{dataset}/{model_name}_cot_{split}_{datalength}.json'

config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16, trust_remote_code=True)
    # model = AutoModelForSeq2SeqLM.from_config(config, torch_dtype=torch.float16, trust_remote_code=True)
no_split_modules = model._no_split_modules
model = load_checkpoint_and_dispatch(
    model, model_path, device_map="auto", no_split_module_classes=no_split_modules
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)  
prompter = LlamaPrompter(dataset=dataset, task='generate_cot')
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
for data in tqdm(dataloader):
    question = data['question']
    option = data['option']
    cots = []
    for i in range(len(option)):
        pred = '\nAnswer: ' + f'({i+1}) ' + option[i]
        msg = question + pred
        input = prompter.wrap_input(msg, icl_cnt=5)
        torch.set_grad_enabled(False)
        model.eval()
        inputs = tokenizer(input, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        output = model.generate(
                input_ids=input_ids,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=500,
                stopping_criteria=[stop_criteria]
            )
        result = tokenizer.decode(output.sequences[0])
        result = result.split(' [/INST] ')[-1]
        cots.append(result)
        torch.cuda.empty_cache()
        del input_ids
        del output
    msg = {'question':question, 'answer':cots}
    results.append(msg)

with open(result_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=4)
    
print('Done!!!')