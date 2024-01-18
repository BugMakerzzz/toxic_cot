import os
import torch
import re
import argparse
import json
import random
import numpy as np
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
parser.add_argument('--mode', type=str, default=None)
parser.add_argument('--cnt', type=int, default=5)
args = parser.parse_args()


model_name = args.model
dataset = args.dataset
datalength = args.datalength
split = args.split
task = args.task
icl = args.icl
mode = args.mode
cnt = args.cnt
model_path = f'./model/{model_name}'
result_path = f'./result/{dataset}/{model_name}_{task}_{split}_{datalength}.json'


model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True, device_map='auto')
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
if mode == 'C2W':
    if dataset == 'csqa':
        idx = [41, 49, 161, 174, 219, 244, 276, 283, 286, 297, 394, 402, 413, 424, 441, 443, 457, 523, 526, 539, 652, 700, 709, 754, 869, 881, 898, 946][cnt]
    else:
        idx = [7,15,50,53,84,97,108,119,121,132,201,207,209,235,253,284,285,307,320,338,342,347,387,390,426,453,467,475,478,482,490,498,512][:10]
elif mode == 'W2C':
    if dataset == 'csqa':
        idx = [2, 7, 14, 26, 31, 34, 35, 48, 58, 66, 75, 88, 92, 96, 103, 109, 125, 126, 127, 175, 185, 186, 191, 200, 209, 215, 218, 247, 248, 249, 250, 253, 260, 267, 274, 293, 295, 314, 322, 324, 352, 356, 363, 364, 370, 376, 380, 385, 387, 398, 407, 412, 429, 438, 446, 513, 516, 524, 532, 543, 550, 566, 567, 588, 590, 592, 593, 601, 602, 607, 616, 622, 624, 628, 633, 639, 640, 644, 646, 659, 673, 705, 713, 718, 721, 723, 744, 747, 755, 756, 758, 760, 768, 771, 776, 781, 791, 805, 818, 827][cnt]
    else:
        idx = [1, 2, 20, 22, 23, 32, 34, 38, 42, 46, 54, 59, 62, 65, 77, 79, 81, 85, 89, 91, 96, 99, 101, 103, 104, 112, 122, 124, 127, 142, 144, 145, 170, 179, 182, 190, 191, 198, 205, 210, 212, 213, 215, 229, 239, 243, 247, 256, 265, 276, 279, 283, 291, 294, 297, 303, 311, 328, 335, 348, 349, 353, 356, 358][:20]
else:
    pool = range(datalength)
    idx = random.sample(pool, 20)

model.eval()
with torch.no_grad():
    scores = []
    for data in tqdm(dataloader):
        question = data['question']
        input = prompter.wrap_input(question, icl_cnt=icl)
        if dataloader.idx - 1 != idx:
            continue
        # print(question)
        # print(f"Label: {data['label']}")
        question_len = len(prompter.user_prompt.format(question))
        prompt = input[:-question_len]
        prompt_end = len(tokenizer(prompt, return_tensors="pt").input_ids[0]) - 1

        inputs = tokenizer(input, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        output = model.generate(
                input_ids=input_ids,
                return_dict_in_generate=True,
                max_new_tokens=200,
                do_sample=False,
                stopping_criteria=[stop_criteria],
            )
        result = tokenizer.decode(output.sequences[0]).split(' [/INST] ')[-1]
        label = result.split(':')[-1]
        match = re.findall(r'[1-5]\)',label)
        if match:
            label = match[-1][:-1]
        else:
            label = 'None'
        print(f"Orignal Pred: {label}")
        print(f'Orignal CoT: {result}')
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0,prompt_end:])
        print(f'Tokens: {tokens}')
        if mode == 'W2C' and label != data['label'] or mode == 'C2W' and label == data['label']:
            continue
        cnt = 0
        change = 0
        mask_index = [[prompt_end+3,prompt_end+7], [prompt_end+7,prompt_end+11], [prompt_end+12,prompt_end+13]]
        for i,j in mask_index:
            token = tokenizer.convert_ids_to_tokens(input_ids[0,i:j])
            print(f'Token: {token}')
            idx_range = list(range(i,j))
            print(f'Idx: {idx_range}')
            output = model.generate(
                    input_ids=input_ids,
                    return_dict_in_generate=True,
                    max_new_tokens=200,
                    do_sample=False,
                    stopping_criteria=[stop_criteria],
                    attention_mask_idx=idx_range
                )
            result = tokenizer.decode(output.sequences[0]).split(' [/INST] ')[-1]
            pred = result.split(':')[-1]
            match = re.findall(r'[1-5]\)',pred)
            if match:
                pred = match[-1][:-1]
            else:
                pred = 'None'
            if pred != label:
                change += 1
            print(f'Pred:{pred}\tCoT:{result}')
            cnt += 1
            # msg = {'question':question, 'answer':result, 'pred':pred, 'label':label, 'cor_flag':cor_flag}
            # results.append(msg)
            del output
        del input_ids
        score = change / cnt
        print(score)
        scores.append(change)
        torch.cuda.empty_cache()
# mean_score = np.array(scores).mean(axis=0)
# print(f'{mode}: {mean_score}')
# results.append({'acc':correct/datalength})
# print(f'Acc:{correct/datalength}')
# with open(result_path, 'w', encoding='utf-8') as f:
#     json.dump(results, f, indent=4)