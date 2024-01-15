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
from metrics import draw_acc, draw_plot

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='Llama-2-13b-chat-hf')
parser.add_argument('--datalength', type=int, default=2)
parser.add_argument('--dataset', type=str, default='csqa')
parser.add_argument('--task', type=str, default='cot_answer')
parser.add_argument('--res', action='store_true')
args = parser.parse_args()

model_name = args.model
dataset = args.dataset
datalength = args.datalength
task = args.task
res = args.res

model_path = f'./model/{model_name}'
cot_file_path  = f'./result/{dataset}/{model_name}_cot_answer_dev_1000.json'
base_file_path = f'./result/{dataset}/{model_name}_direct_answer_dev_1000.json'
if dataset == 'csqa':  
    res_cot_file_path = f'./result/csqa/res_result_cot_answer_greedy_1000_s-90_w-1.5_c-10_r-True.json'
elif dataset == 'wino':
    res_cot_file_path = f'./result/wino/res_result_cot_answer_greedy_1000_s-40_w-2.0_c-4.json'
result_path = f'./result/{dataset}/test_{datalength}_{res}.json'

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

def llm_score(question, pred, res_cot, layers, cot=None, diff=True):
    with torch.no_grad():
        prompt = prompter.wrap_input(question, icl_cnt=5) 
        if diff:
            base_input = f'Knowledge: {cot}. ' + f'\nQuestion: {question}\n'
            scores = lm_logit(prompt + base_input + f'. So the answer is: ({pred})', layers)
        res_input = f'Knowledge: {res_cot}. ' + f'\nQuestion: {question}\n'
        res_scores = lm_logit(prompt + res_input + f'. So the answer is: ({pred})', layers)
        if diff:
            diff_scores = np.array(res_scores) - np.array(scores)
            return diff_scores
        else:
            return res_scores
        
def llm_generate(question, cot):
    input = prompter.wrap_input(question, icl_cnt=5) 
    inputs = tokenizer(input + cot + '. So the answer is: (', return_tensors="pt")
    question_len = len(prompter.user_prompt.format(question))
    prompt = prompter.wrap_input(question, icl_cnt=5)[:-question_len]
    stem = '\n'.join(input.split('\n')[:-1])
    stem_end = len(tokenizer(stem, return_tensors="pt").input_ids[0])
    stem_start = len(tokenizer(prompt, return_tensors="pt").input_ids[0]) - 1
    question_end = len(tokenizer(input, return_tensors="pt").input_ids[0])
    key_position = {'start':stem_start, 'end':stem_end}
    input_ids = inputs["input_ids"].to(model.device)
    output = model.generate(
            input_ids=input_ids,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=2,
            stopping_criteria=[stop_criteria],
            res_decoding=False,
            do_sample=False,
            key_position=key_position,
            scale_factor=100,
            num_attn_candidates=10,
            penalty_weights=2,
        )
    result = tokenizer.decode(output.sequences[0]).split(' [/INST] ')[-1]
    result = result.split('\nNote')[0]
    split_result = result.split(':')
    pred = split_result[-1].strip()
    match = re.findall(r'[1-5]\)',pred)
    if match:
        pred = match[-1][:-1]
    else:
        pred = 'None'
    return pred

correct = 0
cnt = 0
cot_spliter = CoTLoader()
results = []

if dataset == 'csqa':
    index1 = [36, 41, 49, 158, 161, 174, 219, 244, 276, 283, 286, 287, 297, 331, 340, 341, 348, 355, 379, 386, 394, 395, 396, 402, 413, 416, 420, 424, 431, 441, 443, 450, 457]
    index2 = [2, 5, 7, 14, 26, 31, 34, 35, 48, 58, 66, 75, 92, 96, 103, 109, 122, 125, 126, 127]
    index1 = index1[:20]
    index2 = index2[:20]
    index = index1 + index2
else:
    w2c_index = [1, 2, 20, 22, 23, 32, 34, 38, 42, 46, 54, 59, 62, 65, 77, 79, 81, 85, 89, 91, 96, 99, 101, 103, 104, 112, 122, 124, 127, 142, 144, 145, 170, 179, 182, 190, 191, 198, 205, 210, 212, 213, 215, 229, 239, 243, 247, 256, 265, 276, 279, 283, 291, 294, 297, 303, 311, 328, 335, 348, 349, 353, 356, 358, 360, 370, 371, 372, 379, 380, 384, 388, 401, 405, 414, 435, 437, 441, 442, 443, 452, 458, 462, 464, 470, 474, 484, 491, 505, 507, 509, 510, 513, 514, 517, 520, 523, 528, 532, 534, 544, 550, 554, 555, 564, 566, 576, 585, 591, 603, 606, 617, 621, 623, 629, 633, 639, 641, 644, 648, 655, 663, 665, 667, 672, 681, 685, 688, 691, 697, 702, 707, 719, 743, 744, 747, 748, 750, 763, 767, 778, 790, 803, 816, 820, 825, 826, 829, 830, 840, 841, 843, 853, 856, 871, 875, 876, 880, 889, 891, 897, 899, 901, 902, 904, 907, 909, 910, 914, 915, 921, 933, 941, 942, 944, 948, 956, 958, 965, 972, 974, 980, 993]
    # c2w_index = [4, 7, 15, 27, 40, 41, 47, 50, 53, 60, 71, 73, 76, 80, 84, 97, 100, 108, 113, 114, 119, 121, 132, 151, 158, 160, 171, 175, 180, 183, 185, 189, 197, 199, 201, 206, 207, 209, 232, 235, 245, 253, 255, 266, 272, 274, 284, 285, 292, 306, 307, 316, 320, 323, 327, 333, 338, 342, 347, 381, 387, 390, 393, 407, 409, 418, 423, 426, 427, 433, 439, 444, 453, 454, 455, 459, 467, 473, 475, 478, 479, 481, 482, 490, 493, 498, 512, 518, 525, 529, 531, 535, 538, 543, 557, 560, 568, 573, 574, 580, 582, 595, 597, 600, 605, 610, 620, 627, 638, 640, 646, 654, 661, 666, 677, 678, 686, 689, 693, 695, 710, 711, 712, 714, 721, 733, 735, 739, 740, 745, 752, 753, 759, 760, 766, 768, 772, 774, 776, 780, 782, 798, 808, 819, 824, 831, 836, 842, 848, 849, 861, 868, 869, 872, 873, 882, 893, 903, 911, 916, 920, 927, 928, 930, 943, 960, 962, 967, 973, 976, 977, 979, 981, 984, 995, 997]
    # c2w_index = [40,47,73,175,180,185,197,232,255,266,274,306,316,327,333,409,423,427,433,444,454,481][:cnt]
    c2w_index = [7, 15, 50, 53, 84, 97, 108, 119, 121, 132, 201, 207, 209, 235, 253][:cnt]
    w2c_index = w2c_index[:10]
    c2w_index = c2w_index[:10]
    index = w2c_index + c2w_index


for msg in tqdm(dataloader):
    idx = dataloader.idx - 1
    if idx not in index:
        continue
    question = msg['question']
    label = msg['label']
    if res:
        cot = cot_spliter.split_cot(res_cots[idx]['answer'])
    else:
        cot = cot_spliter.split_cot(cot_data[idx]['answer'])
    cot = '.'.join(cot)
    pred1 = base_data[idx]['pred']
    pred2 = cot_data[idx]['pred']
    
    pred = llm_generate(question, cot)
    # if pred1 == pred2:
    #     pred = pred1
    #     answer = cot_data[idx]['answer']
    # else:
    #     if pred1 == label:
    #         pred = pred2
    #     else:
    #         pred = pred1
    #     res_pred_scores = llm_score(question, pred, cot, layers=range(40), diff=False)
    #     res_label_scores = llm_score(question, label, cot, layers=range(40), diff=False)
    #     if res_label_scores[-1] > res_pred_scores[-1]:
    #         pred = label
    #     answer = res_cots[idx]['answer']
    #     # path = f'./result/{dataset}/fig-{idx}.png'
    #     # draw_plot(layers=[i+1 for i in range(40)], scores=[res_pred_scores, res_label_scores], labels=['pred','label'], path=path)
    cnt += 1
    if pred == label:
        cor_flag = True
        correct += 1
    else:
        cor_flag = False
    msg = {'question':question, 'pred':pred, 'label':label, 'cor_flag':cor_flag}
    results.append(msg)
    
print(f'Acc:{correct / cnt}')
with open(result_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=4)