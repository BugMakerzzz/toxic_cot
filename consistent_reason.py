import torch
import argparse
import json
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModelForSeq2SeqLM
from prompts.wrap_prompt import LlamaPrompter
from load_data import DataLoader
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
from metrics import draw

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='Llama-2-13b-chat-hf')
parser.add_argument('--datalength', type=int, default=2)
parser.add_argument('--dataset', type=str, default='csqa')
args = parser.parse_args()


model_name = args.model
dataset = args.dataset
datalength = args.datalength

model_path = f'./model/{model_name}'
cot_file_path  = f'./result/{dataset}/{model_name}_cot_answer_dev_200.json'
base_file_path = f'./result/{dataset}/{model_name}_direct_answer_dev_200.json'
full_cot_path = f'./result/{dataset}/{model_name}_cot_dev_1000.json'
# result_path = f'./result/{dataset}/{model_name}_direct_answer_dev_{datalength}.json'

config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
with init_empty_weights():
    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16, trust_remote_code=True)
    # model = AutoModelForSeq2SeqLM.from_config(config, torch_dtype=torch.float16, trust_remote_code=True)
no_split_modules = model._no_split_modules
model = load_checkpoint_and_dispatch(
    model, model_path, device_map="auto", no_split_module_classes=no_split_modules
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)  


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


def lm_logit(question_text, pred_text, layers, diff_logits):
    input_text = question_text + pred_text
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
    prefix_ids = tokenizer(question_text, return_tensors="pt").input_ids.to(model.device)
    continue_ids = input_ids[0, prefix_ids.shape[-1]:].cpu().numpy()
    dict_outputs, outputs = model(
        input_ids=input_ids,
        return_dict=True,
        output_attentions=False,
        output_hidden_states=False,
        early_exit_layers=layers,
    )
    
    scores = []
    for layer in layers:
        logits = dict_outputs[layer][0, prefix_ids.shape[-1] - 1: -1, :]
        logits = logits.log_softmax(dim=-1)
        if diff_logits:
            if layer == 0:
                continue
            if diff_logits == 'base':
                base_logits = dict_outputs[0][0, prefix_ids.shape[-1] - 1: -1, :]
                base_logits = base_logits.log_softmax(dim=-1)
            else:
                base_layer = layers[layers.index(layer)-1]
                base_logits = dict_outputs[base_layer][0, prefix_ids.shape[-1] - 1: -1, :]
                base_logits = base_logits.log_softmax(dim=-1)
            logits = logits - base_logits
            logits = logits.log_softmax(dim=-1)
            # probs = diff_logits[range(diff_logits.shape[0]), continue_ids].sum().item()
        scores.append(logits.cpu().numpy())  
    torch.cuda.empty_cache()
    return scores, continue_ids


def cal_diff_score(logits_ls, ids_ls, diff_cot):
    scores = []
    for i in range(len(logits_ls)):
        logits = np.array(logits_ls[i])
        ids = ids_ls[i]
        if diff_cot and i == 0:
            continue
        if diff_cot == 'base':
            base_logits = np.array(logits_ls[0])
        elif diff_cot == 'adj':
            base_logits = np.array(logits_ls[i-1])
        
        if diff_cot:
            logits = logits - base_logits

        probs = logits[:,range(logits.shape[1]), ids].sum(axis=-1)
        probs = (probs - probs[0])[1:].tolist()
        scores.append(probs)

    return scores[-1]


def cot_score(question, pred, cot, layers=[0, 5, 10, 15, 20, 25, 30, 35, 40], diff_cot=None, diff_logits=None):
    with torch.no_grad():
        pred_ids = []
        pred_logits = []
        pred_scores = []
        # for i in range(len(cot)+1):
            # if i == 0:
        cot_question = prompter.wrap_input(question, icl_cnt=5) + cot
    
            # else:
            #     cot_question += cot[i-1] + '.'
        logits, ids = lm_logit(cot_question+' So the answer is: ', pred, layers, diff_logits)
        pred_logits.append(logits)
        pred_ids.append(ids)
        pred_scores = cal_diff_score(pred_logits, pred_ids, diff_cot)

    return pred_scores


prompter = LlamaPrompter(dataset=dataset, task='cot_answer')
correct = 0
cnt = 0
layers = []
results = []
with open(cot_file_path, 'r') as f:
    cot_data = json.load(f)
with open(base_file_path, 'r') as f:
    base_data = json.load(f)
with open(full_cot_path, 'r') as f:
    cots = json.load(f)

 
layers = [5, 10, 15, 20, 25, 30, 35, 40]
corrects = [0] * len(layers)
for i in tqdm(range(len(cot_data))):
    base_msg = base_data[i]
    msg = cot_data[i]
    question = msg['question']
    answers = msg['answer']
    label = msg['label']
    if msg['cor_flag'] == base_msg['cor_flag']:
        pred = [msg['pred']] * len(layers)
    else:
        options = question.split('\n')[-1].split('(')
        pred1 = base_msg['pred']
        pred2 = msg['pred']
        if pred2 == 'None' or eval(pred2) > len(options):
            pred = [pred1] * len(layers)
        elif pred1 == 'None' or eval(pred1) > len(options):
            pred = [pred2] * len(layers)
        else:
            pred1_option = f'({options[eval(pred1)]}' 
            pred2_option = f'({options[eval(pred2)]}'
            
            # steps = answers.split('.')[:-2]
            # if len(steps) == 0:
            #     pred = pred1
            # pred1_steps = pred2_steps = steps
            steps = cots[i]['answer']
            pred1_steps = steps[eval(pred1)-1]
            pred2_steps = steps[eval(pred2)-1]
            # pred1_cot = '.'.join(pred1_steps)
            # pred2_cot = '.'.join(pred2_steps)
            # else:        
            pred1_score = cot_score(question, pred1_option, pred1_steps)
            pred2_score = cot_score(question, pred2_option, pred2_steps)
            pred = []
            for i in range(len(pred1_score)):
                score1 = pred1_score[i]
                score2 = pred2_score[i]
                if score1 > score2:
                    pred.append(pred1)
                else:
                    pred.append(pred2)
    # print(pred)
    for i in range(len(pred)):
        if pred[i] == label:
            corrects[i] += 1
    cnt += 1
    if cnt >= datalength:
        break
            
results = [corrects[i] / datalength for i in range(len(corrects))]
print(results)
draw(layers, results, dataset, './test.png', 'acc')