import os
import torch
import re
import argparse
import json
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
parser.add_argument('--dataset', type=str, default='csqa')
parser.add_argument('--task', type=str, default='cot_answer')
parser.add_argument('--sample', type=str, default='greedy')
parser.add_argument('--scale', type=int, default=90)
parser.add_argument('--weight', type=float, default=1.5)
parser.add_argument('--cand_num', type=int, default=10)
parser.add_argument('--res', action='store_true')

args = parser.parse_args()


model_name = args.model
dataset = args.dataset
datalength = args.datalength
task = args.task
sample = args.sample
scale_factor = args.scale
penalty_weights = args.weight
num_attn_candidates = args.cand_num
res = args.res

model_path = f'./model/{model_name}'
result_path = f'./result/{dataset}/res_result_{task}_{sample}_{datalength}_s-{scale_factor}_w-{penalty_weights}_c-{num_attn_candidates}_r-{res}.json'
cot_file_path  = f'./result/{dataset}/{model_name}_cot_answer_dev_1000.json'
base_file_path = f'./result/{dataset}/{model_name}_direct_answer_dev_1000.json'

model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True, device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)  
prompter = LlamaPrompter(dataset=dataset, task=task)
dataloader = DataLoader(dataset=dataset, data_length=datalength)
with open(cot_file_path, 'r') as f:
    cot_data = json.load(f)
with open(base_file_path, 'r') as f:
    base_data = json.load(f)


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

# if dataset == 'csqa':
#     index1 = [36, 41, 49, 158, 161, 174, 219, 244, 276, 283, 286, 287, 297, 331, 340, 341, 348, 355, 379, 386, 394, 395, 396, 402, 413, 416, 420, 424, 431, 441, 443, 450, 457]
#     index2 = [2, 5, 7, 14, 26, 31, 34, 35, 48, 58, 66, 75, 92, 96, 103, 109, 122, 125, 126, 127]
#     index1 = index1[:10]
#     index2 = index2[:10]
#     index = index1 + index2
# else:
#     w2c_index = [1, 2, 20, 22, 23, 32, 34, 38, 42, 46, 54, 59, 62, 65, 77, 79, 81, 85, 89, 91, 96, 99, 101, 103, 104, 112, 122, 124, 127, 142, 144, 145, 170, 179, 182, 190, 191, 198, 205, 210, 212, 213, 215, 229, 239, 243, 247, 256, 265, 276, 279, 283, 291, 294, 297, 303, 311, 328, 335, 348, 349, 353, 356, 358, 360, 370, 371, 372, 379, 380, 384, 388, 401, 405, 414, 435, 437, 441, 442, 443, 452, 458, 462, 464, 470, 474, 484, 491, 505, 507, 509, 510, 513, 514, 517, 520, 523, 528, 532, 534, 544, 550, 554, 555, 564, 566, 576, 585, 591, 603, 606, 617, 621, 623, 629, 633, 639, 641, 644, 648, 655, 663, 665, 667, 672, 681, 685, 688, 691, 697, 702, 707, 719, 743, 744, 747, 748, 750, 763, 767, 778, 790, 803, 816, 820, 825, 826, 829, 830, 840, 841, 843, 853, 856, 871, 875, 876, 880, 889, 891, 897, 899, 901, 902, 904, 907, 909, 910, 914, 915, 921, 933, 941, 942, 944, 948, 956, 958, 965, 972, 974, 980, 993]
#     c2w_index = [4, 7, 15, 27, 40, 41, 47, 50, 53, 60, 71, 73, 76, 80, 84, 97, 100, 108, 113, 114, 119, 121, 132, 151, 158, 160, 171, 175, 180, 183, 185, 189, 197, 199, 201, 206, 207, 209, 232, 235, 245, 253, 255, 266, 272, 274, 284, 285, 292, 306, 307, 316, 320, 323, 327, 333, 338, 342, 347, 381, 387, 390, 393, 407, 409, 418, 423, 426, 427, 433, 439, 444, 453, 454, 455, 459, 467, 473, 475, 478, 479, 481, 482, 490, 493, 498, 512, 518, 525, 529, 531, 535, 538, 543, 557, 560, 568, 573, 574, 580, 582, 595, 597, 600, 605, 610, 620, 627, 638, 640, 646, 654, 661, 666, 677, 678, 686, 689, 693, 695, 710, 711, 712, 714, 721, 733, 735, 739, 740, 745, 752, 753, 759, 760, 766, 768, 772, 774, 776, 780, 782, 798, 808, 819, 824, 831, 836, 842, 848, 849, 861, 868, 869, 872, 873, 882, 893, 903, 911, 916, 920, 927, 928, 930, 943, 960, 962, 967, 973, 976, 977, 979, 981, 984, 995, 997]
#     w2c_index = w2c_index[:10]
#     c2w_index = c2w_index[:10]
#     index = w2c_index + c2w_index

# max_acc = 0
# max_index = -1
# idx = 0
# weight_ls = []
# max_results = []
# for num_attn_candidates in tqdm(range(10, 1, -2)):
#     for scale_factor in tqdm(range(100, 0, -10)):
#         for penalty_weights in tqdm(np.arange(2.0, 0, -0.5)):
            # weight_ls.append({'num_attn_candidates':num_attn_candidates, 'scale_factor':scale_factor, 'penalty_weights':penalty_weights})
            # print({'num_attn_candidates':num_attn_candidates, 'scale_factor':scale_factor, 'penalty_weights':penalty_weights})
def res_inference(question):
    input = prompter.wrap_input(question, icl_cnt=5)
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
    # if dataloader.idx - 1 in index:
    if sample == 'greedy':
        output = model.generate(
            input_ids=input_ids,
            return_dict_in_generate=True,
            output_scores=True,
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
    elif sample == 'beam_search':
        output = model.generate(
                input_ids=input_ids,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=200,
                num_beams=5,
                res_decoding=True,
                do_sample=False,
                # temperature=1.3,
                # top_k=3,
                # top_p=0.75,
                num_return_sequences=3,
                key_position=key_position,
                scale_factor=50,
                num_attn_candidates=5,
                penalty_weights=1.5,
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
    match = re.findall(r'[1-5]\)',pred)
    if match:
        pred = match[-1][:-1]
    else:
        pred = 'None'
        
    del input_ids
    del output
    
    return result, pred

correct = 0
results = []
dataloader.idx = 0
cnt = 0
for data in tqdm(dataloader):
   
    # if dataloader.idx - 1 not in index:
    #     continue    
    question = data['question']
    label = data['label']
    msg = cot_data[dataloader.idx - 1]
    base_msg = base_data[dataloader.idx - 1]
    if msg['pred'] == base_msg['pred'] and res:
        pred = msg['pred']
        result = msg['answer']
    else:
        result, pred = res_inference(question)
    
    # else:
    # output = model.generate(
    #         input_ids=input_ids,
    #         return_dict_in_generate=True,
    #         output_scores=True,
    #         max_new_tokens=200,
    #         stopping_criteria=[stop_criteria],
    #     )
    # result = tokenizer.decode(output.sequences[0]).split(' [/INST] ')[-1]
    # split_result = result.split(':')
    # if len(split_result) >= 2:
    #     pred = split_result[-1].strip()
    # else:
    #     pred = 'None'
    cor_flag = (pred == label)
    if cor_flag:
        correct += 1
    cnt += 1  
    msg = {'question':question, 'answer':result, 'pred':pred, 'label':label, 'cor_flag':cor_flag}
    results.append(msg)
    
    torch.cuda.empty_cache()

acc = correct / cnt
            # # print(f'Acc: {acc}')
            # if acc > max_acc:
            #     max_acc = acc
            #     max_index = idx
            #     max_results = results
            #     print(f'Acc: {max_acc}')
            #     print(f'Config: {weight_ls[max_index]}')
            # idx += 1
            
# print(f'Acc: {max_acc}')
# print(f'Config: {weight_ls[max_index]}')
# max_results.append({'acc':max_acc})
# with open(result_path, 'w', encoding='utf-8') as f:
#     json.dump(max_results, f, indent=4)
print(f'Acc: {acc}')
results.append({'acc':acc})
with open(result_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=4)