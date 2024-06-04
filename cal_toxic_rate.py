import argparse
import json
import os
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='Llama-2-13b-chat-hf')
parser.add_argument('--dataset', type=str, default='csqa')
args = parser.parse_args()

model_name = args.model
dataset = args.dataset

base_file_path = f'./result/{dataset}/{model_name}_direct_answer_2000.json'
base_data = []
with open(base_file_path, 'r') as f:
    base_data = json.load(f)[:-1]
    f.close()

def get_drift_acc(task, index):
    if task == 'cot':
        path = f'./result/{dataset}/{model_name}_cot_answer_2000.json'
    elif task == 'sc':
        path = f'./result/{dataset}/{model_name}_sc_2000.json'
    elif task == 'l2m':
        path = f'./result/{dataset}/{model_name}_l2m_question_dev_2000.json'
    elif task == 'sr':
        path = f'./result/{dataset}/{model_name}_self_refine_dev_2000.json'
    elif task == 'cont':
        path = f'./result/{dataset}/{model_name}_cons_answer_dev_2000.json'
    elif task == 'res':
        path = f'./result/{dataset}/res.json'
    elif task == 'sps':
        path = f'./result/{dataset}/{model_name}_rt_result_2000_rFalse.json'
    else:
        path = f'./result/{dataset}/{model_name}_rt_result_2000_rTrue.json'
    with open(path, 'r') as f:
        test_data = json.load(f)[:-1]
        f.close()
    cnt = 0
    correct = 0
    for i in range(len(test_data)):
        if i not in index:
            continue
        if test_data[i]['cor_flag']:
            correct += 1
        cnt += 1
    return correct / cnt

def get_tr(task, base):
    if task == 'direct':
        path = f'./result/{dataset}/{model_name}_direct_answer_2000.json'
    elif task == 'cot':
        path = f'./result/{dataset}/{model_name}_cot_answer_2000.json'
    elif task == 'sc':
        path = f'./result/{dataset}/{model_name}_sc_2000.json'
    elif task == 'l2m':
        path = f'./result/{dataset}/{model_name}_l2m_2000.json'
    elif task == 'sr':
        path = f'./result/{dataset}/{model_name}_sr_2000.json'
    elif task == 'cont':
        path = f'./result/{dataset}/{model_name}_cons_answer_2000.json'
    elif task == 'res':
        path = f'./result/{dataset}/{model_name}_res.json'
    elif task == 'sps':
        path = f'./result/{dataset}/{model_name}_rt_result_2000_rFalse.json'
    else:
        path = f'./result/{dataset}/{model_name}_rt_result_2000_rTrue.json'

    if not os.path.exists(path):
        return -1
    with open(path, 'r') as f:
        test_data = json.load(f)[:-1]
        f.close()
    cnt = 0
    false = 0
    for i in range(len(test_data)):
        if test_data[i]['cor_flag']:
            continue
        if base[i]['cor_flag']:
            false += 1
        cnt += 1
    return false / cnt

def get_acc(task):
    if task == 'direct':
        path = f'./result/{dataset}/{model_name}_direct_answer_2000.json'
    elif task == 'cot':
        path = f'./result/{dataset}/{model_name}_cot_answer_2000.json'
    elif task == 'sc':
        path = f'./result/{dataset}/{model_name}_sc_2000.json'
    elif task == 'l2m':
        path = f'./result/{dataset}/{model_name}_l2m_2000.json'
    elif task == 'sr':
        path = f'./result/{dataset}/{model_name}_sr_2000.json'
    elif task == 'cont':
        path = f'./result/{dataset}/{model_name}_cons_answer_2000.json'
    elif task == 'res':
        path = f'./result/{dataset}/{model_name}_res.json'
    elif task == 'sps':
        path = f'./result/{dataset}/{model_name}_rt_result_2000_rFalse.json'
    else:
        path = f'./result/{dataset}/{model_name}_rt_result_2000_rTrue.json'
        
    if not os.path.exists(path):
        return -1
    with open(path, 'r') as f:
        acc = json.load(f)[-1]['acc']
    return acc

test_data = []
task_ls = ['direct', 'cot','sc', 'sr', 'l2m', 'cont', 'res', 'sps', 'riders']
for task in task_ls:
    tr = get_tr(task, base_data)
    acc = get_acc(task)
    print(f'{task}: Accuracy:{acc}  Toxic Rate:{tr}')
# if dataset == 'csqa':
#     index1 = [41,49,158,161,174,244,276,283,286,297,386,394,402,413,424,431,441,443,457,523,539,652,700,709,754,869,881,898,939,946]
#     index2 = [36,331,379,395,521,525,527,599,654,826,893,913,998]
# elif dataset == 'wino':
#     index1 = [7,15,50,53,97,108,119,121,132,201,207,209,235,253,284,285,307,320,338,342,347,387,390,426,453,467,475,478,482,490,498]
#     index2 = [40,47,73,175,180,185,197,232,255,266,274,306,316,327,333,409,423,427,433,444,454,481,493]
# task_ls = ['cot', 'sc','sr','l2m','con', 'res', 'sps', 'riders']
# for task in task_ls:
#     acc1 = get_drift_acc(task, index1)
#     acc2 = get_drift_acc(task, index2)
#     print(f'{task}:    Type1:{acc1}    Type2:{acc2}')