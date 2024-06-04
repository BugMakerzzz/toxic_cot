import argparse
import json
import os
from prompts.wrap_prompt import LlamaPrompter
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='Llama-2-13b-chat-hf')
parser.add_argument('--dataset', type=str, default='csqa')
args = parser.parse_args()

model_name = args.model
dataset = args.dataset


def get_cost(task):
    if task == 'direct':
        path = f'./result/{dataset}/{model_name}_direct_answer_dev_2000.json'
    elif task == 'cot':
        path = f'./result/{dataset}/{model_name}_cot_answer_dev_2000_greedy.json'
    elif task == 'sc':
        path = f'./result/{dataset}/{model_name}_sc_2000.json'
    elif task == 'l2m':
        path = f'./result/{dataset}/{model_name}_l2m_question_dev_2000.json'
    elif task == 'sr':
        path = f'./result/{dataset}/{model_name}_self_refine_dev_2000.json'
    elif task == 'cont':
        path = f'./result/{dataset}/{model_name}_cons_answer_dev_2000.json'
    else:
        path = f'./result/{dataset}/res.json'
        

    if not os.path.exists(path):
        return -1
    with open(path, 'r') as f:
        data = json.load(f)[:-1]
        f.close()
    cnt = 0
    sum_cost = 0
    for msg in data:
        question = msg['question']
        if task in ['sc', 'res', 'cot']:
            prompter = LlamaPrompter(dataset, 'cot_answer')
            question = prompter.wrap_input(question, icl_cnt=5)
        elif task == 'sr':
            prompter = LlamaPrompter(dataset, 'sr_feedback')
            question = prompter.wrap_input(question, icl_cnt=5)
            prompter = LlamaPrompter(dataset, 'sr_answer')
            question = prompter.wrap_input(question, icl_cnt=5)
        elif task == 'l2m':
            prompter = LlamaPrompter(dataset, 'l2m_question')
            question = prompter.wrap_input(question, icl_cnt=5)
            prompter = LlamaPrompter(dataset, 'l2m_mid_answer')
            question = prompter.wrap_input(question, icl_cnt=5)
            prompter = LlamaPrompter(dataset, 'l2m_final_answer')
            question = prompter.wrap_input(question, icl_cnt=5)
        elif task == 'cont':
            prompter = LlamaPrompter(dataset, 'cons_answer')
            question = prompter.wrap_input(question, icl_cnt=5)
        else:
            prompter = LlamaPrompter(dataset, 'cot_answer')
            question += prompter.wrap_input(question, icl_cnt=5)
        
        if task == 'sc':
            result = ""
            for text in msg['answer']:
                result += text
        elif task == 'our':
            result = msg['answer'] * 2
        else:
            result = msg['answer']
        
        cost = len(question + result)
        sum_cost += cost
        cnt += 1
    return sum_cost / cnt


test_data = []
task_ls = ['cot','sc', 'sr', 'l2m', 'cont', 'our']
for task in task_ls:
    cost = get_cost(task)
    print(f'{task}: Avg cost:{cost}')
# if dataset == 'csqa':
#     index1 = [41,49,158,161,174,244,276,283,286,297,386,394,402,413,424,431,441,443,457,523,539,652,700,709,754,869,881,898,939,946]
#     index2 = [36,331,379,395,521,525,527,599,654,826,893,913,998]
# elif dataset == 'wino':
#     index1 = [7,15,50,53,97,108,119,121,132,201,207,209,235,253,284,285,307,320,338,342,347,387,390,426,453,467,475,478,482,490,498]
#     index2 = [40,47,73,175,180,185,197,232,255,266,274,306,316,327,333,409,423,427,433,444,454,481,493]
# task_ls = ['cot', 'res', 'sps', 'riders']
# for task in task_ls:
#     acc1 = get_drift_acc(task, index1)
#     acc2 = get_drift_acc(task, index2)
#     print(f'{task}:    Type1:{acc1}    Type2:{acc2}')