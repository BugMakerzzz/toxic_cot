from utils import get_phrases
from tqdm import tqdm
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='Llama-2-13b-chat-hf')
parser.add_argument('--dataset', type=str, default='wino')
parser.add_argument('--datalength', type=int, default=1000)
parser.add_argument('--mode', type=str, default='C2W')
parser.add_argument('--cnt', type=int, default=1000)
args = parser.parse_args()


model_name = args.model
dataset = args.dataset
datalength = args.datalength
mode = args.mode
cnt = args.cnt

def split_cot(cot):
    steps = cot.split('.')[:-1]
    cots = []
    tags = []
    for step in steps:
        units = get_phrases(step)
        if len(units) == 1:
            if 'answer is' in units[0] or '(' in units[0]:
                continue
            tags.append(['[S]'])
            cots.append(units)
        else:
            tag = []
            cot = []
            for i in range(len(units)):
                unit = units[i]
                if 'answer is' in unit or '(' in unit:
                    continue
                if i == len(units) - 1:
                    tag.append('[S]')
                else:
                    tag.append('[P]')
                cot.append(unit)
            tags.append(tag)
            cots.append(cot)
    return cots, tags

cot_file_path  = f'./result/{dataset}/{model_name}_cot_answer_dev_{datalength}.json'
proc_cot_file_path = f'./result/{dataset}/{model_name}_proc_cot_answer_dev_{datalength}.json'
with open(cot_file_path, 'r') as f:
    data = json.load(f)
    f.close()
results = []
data = data[:-1]
for msg in tqdm(data):
    cot = msg['answer']
    cots, tags = split_cot(cot)
    msg['cots'] = cots
    msg['tags'] = tags
    results.append(msg)
with open(proc_cot_file_path, 'w') as f:
    json.dump(results, f, indent=4)
    f.close()
print('Done!!!')