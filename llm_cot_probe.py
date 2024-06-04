import os
import torch
import argparse
import json
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import random
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModelForSeq2SeqLM
from utils import get_prompter, build_chat_input
from load_data import DataLoader, CoTLoader
from metrics import draw_plot, draw_heat, draw_line_plot, draw_attr_bar
from sklearn.preprocessing import normalize

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='Llama-2-13b-chat-hf')
parser.add_argument('--dataset', type=str, default='wino')
parser.add_argument('--mode', type=str, default=None)
parser.add_argument('--score', type=str, default=None)
parser.add_argument('--avg', action='store_true')
parser.add_argument('--cnt', type=int, default=1000)
parser.add_argument('--reg', action='store_true')
parser.add_argument('--loss', type=str, default='cot')
parser.add_argument('--diff', action='store_true')
parser.add_argument('--res', action='store_true')
parser.add_argument('--swap', action='store_true')
# parser.add_argument('--opt', action='store_true')
args = parser.parse_args()


model_name = args.model
dataset = args.dataset
mode = args.mode
avg = args.avg
cnt = args.cnt
reg = args.reg
score = args.score
loss_type = args.loss
diff = args.diff
res = args.res
swap = args.swap

model_path = f'./model/{model_name}'
if model_name[:5] == 'Llama':
    cot_file_path  = f'./result/{dataset}/{model_name}_cot_answer_dev_1000.json'
    base_file_path = f'./result/{dataset}/{model_name}_direct_answer_dev_1000.json'
else:
    cot_file_path  = f'./result/{dataset}/{model_name}_cot_answer_2000.json'
    base_file_path = f'./result/{dataset}/{model_name}_direct_answer_2000.json'
full_cot_path = f'./result/{dataset}/{model_name}_cot_dev_1000.json'
result_path = f'./result/{dataset}/fig/{score}/{model_name}_m-{mode}_c-{cnt}_r-{reg}_d-{diff}_l-{loss_type}_re-{res}_s{swap}'
res_cot_path = f'./result/{dataset}/res.json'
swap_file_path = f'./result/{dataset}/{model_name}_rt_result_2000_rFalse.json'

model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True, device_map='auto')
device_map = model.hf_device_map
model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)  
cot_prompter = get_prompter(model_name=model_name, dataset=dataset, task='cot_answer')
base_prompter = get_prompter(model_name=model_name, dataset=dataset, task='direct_answer')

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(17)

class Probe():
    def __init__(self) -> None:
        pass

    def cal_attn_attr(self, question, label, cot, layers, swap=None):
        if swap:
            input = f'Rational: {cot}' + f'\nQuestion: {question}'
            question_len = len(cot_prompter.user_prompt.format(input))
            prompt = cot_prompter.wrap_input(input, icl_cnt=5)[:-question_len]
            wrap_question = cot_prompter.wrap_input(input, icl_cnt=5)
            input_text = wrap_question + f'\nAnswer: ({label})'
            input_ids = tokenizer(input_text, return_tensors="pt").input_ids
            prompt_len = len(tokenizer(prompt, return_tensors="pt").input_ids[0]) - 1
            split_ls = wrap_question.split('\nQuestion:')
            cot = '\nQuestion:'.join(split_ls[:-1])
            cot_len = len(tokenizer(cot, return_tensors="pt").input_ids[0])
            stem = '\nQuestion:' + question.split('\n')[0]
            stem_len = len(tokenizer(cot + stem, return_tensors="pt").input_ids[0])
            question_len = len(tokenizer(wrap_question, return_tensors="pt").input_ids[0])
        else:
            if model_name[:5] == 'Llama':
                question_len = len(cot_prompter.user_prompt.format(question))
                prompt = cot_prompter.wrap_input(question, icl_cnt=5)[:-question_len]
                wrap_question = cot_prompter.wrap_input(question, icl_cnt=5)
                if loss_type == 'cot':
                    input_text = wrap_question + cot 
                else:
                    input_text = wrap_question + cot + f' So the answer is: ({label})' 
                input_ids = tokenizer(input_text, return_tensors="pt").input_ids
                
                prompt_len = len(tokenizer(prompt, return_tensors="pt").input_ids[0]) - 1
                question_len = len(tokenizer(wrap_question, return_tensors="pt").input_ids[0])
                stem = '\n'.join(wrap_question.split('\n')[:-1])
                stem_len = len(tokenizer(stem, return_tensors="pt").input_ids[0])
                cot_len = len(tokenizer(wrap_question + cot, return_tensors="pt").input_ids[0])
            else:
                question_len = len(tokenizer(question, return_tensors="pt").input_ids[0])
                question_msg = cot_prompter.wrap_input(question, icl_cnt=5)
                question_ids = build_chat_input(model, tokenizer, question_msg)
                prompt_len = len(question_ids[0]) - question_len - 1
                question_len = len(question_ids[0])
                
                if loss_type == 'cot':
                    assistant_msg = [{'role':'assistant', 'content':f'{cot}'}]
                else:
                    assistant_msg = [{'role':'assistant', 'content':f'{cot} So the answer is: ({label})'}]
                input_msg = question_msg + assistant_msg
                input_ids = build_chat_input(model, tokenizer, input_msg)
                
                stem_len = len(tokenizer(question.split('\n')[0],return_tensors="pt").input_ids[0])
                stem_len = prompt_len + stem_len
                cot_msg = question_msg + [{'role':'assistant', 'content':f'{cot}'}]
                cot_len = len(build_chat_input(model, tokenizer, cot_msg)[0])

        
        labels = torch.full_like(input_ids, -100)
        if loss_type == 'cot':
            labels[:, question_len:] = input_ids[:, question_len:]
        else:
            labels[:, -2] = input_ids[:, -2]
        if model_name[:5] == 'Llama':
            margin = 6
        else:
            margin = 1

        
        stem_scores = []
        option_scores = []
        cot_scores = []
        
        loss = None 
        outputs = None 
        
        
        outputs = model(
            input_ids=input_ids.to(model.device),
            labels=labels.to(model.device),
            return_dict=True,
            output_attentions=True,
            output_hidden_states=False,
        )
        del input_ids, labels
        loss = outputs['loss']
        
        
        for layer in layers: 
            attn_values = outputs['attentions'][layer]
            attn_grad = torch.autograd.grad(loss, attn_values, create_graph=True, allow_unused=True)[0].detach().cpu()
            attn_values = torch.squeeze(attn_values).detach().cpu()
            attn_grad = torch.squeeze(attn_grad)
            attn_scores = torch.zeros_like(attn_values[0,:,:])
            for i in range(40):
                attn_scores += self.cal_attr_score(attn_values[i,:,:], attn_grad[i,:,:])
            attn_scores = attn_scores[prompt_len:,prompt_len:].cpu().numpy()
            
            if loss_type == 'cot':
                stem_attr = attn_scores[question_len-prompt_len:cot_len-prompt_len, :stem_len-prompt_len].sum() 
                option_attr = attn_scores[question_len-prompt_len:cot_len-prompt_len, stem_len-prompt_len:question_len-prompt_len-margin].sum() 
                cot_attr = attn_scores[question_len-prompt_len:cot_len-prompt_len, question_len-prompt_len:cot_len-prompt_len]
                for i in range(cot_len-question_len):
                    cot_attr[i,i] = 0
                cot_attr = cot_attr.sum() 
            else:
                if swap:
                    stem_attr = attn_scores[-3, cot_len-prompt_len:stem_len-prompt_len].sum() 
                    option_attr = attn_scores[-3, stem_len-prompt_len:question_len-prompt_len-margin].sum() 
                    cot_attr = attn_scores[-3, :cot_len-prompt_len].sum()
                else:
                    stem_attr = attn_scores[-3, :stem_len-prompt_len].sum() 
                    option_attr = attn_scores[-3, stem_len-prompt_len:question_len-prompt_len-margin].sum() 
                    cot_attr = attn_scores[-3, question_len-prompt_len:cot_len-prompt_len].sum()
            
            stem_scores.append(stem_attr)
            option_scores.append(option_attr)
            cot_scores.append(cot_attr)
            
            del attn_values, attn_grad, attn_scores
            model.zero_grad() 
            torch.cuda.empty_cache()
            
        del loss, outputs
        torch.cuda.empty_cache()
        return [stem_scores, option_scores, cot_scores]
    
    
    def cal_attr_score(self, value, grad, steps=20):
        grad_int = torch.zeros_like(value*grad)
        for i in range(steps):
            k = (i+1) / steps
            grad_int += k * grad
        scores = 1 / steps * value * grad_int
        return torch.abs(scores) 
    
    
    def cal_attn(self, question, label, cot, layers):  
        if model_name.startswith('Llama'):
            question_len = len(cot_prompter.user_prompt.format(question))
            prompt = cot_prompter.wrap_input(question, icl_cnt=5)[:-question_len]
            wrap_question = cot_prompter.wrap_input(question, icl_cnt=5)
            input_text = wrap_question + cot + f' So the answer is: ({label})'
            input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
            prompt_len = len(tokenizer(prompt, return_tensors="pt").input_ids[0]) - 1
            stem = '\n'.join(wrap_question.split('\n')[:-1])
            stem_len = len(tokenizer(stem, return_tensors="pt").input_ids[0])
            question_len = len(tokenizer(wrap_question, return_tensors="pt").input_ids[0])
            cot_len = len(tokenizer(wrap_question + cot, return_tensors="pt").input_ids[0])
        else:
            question_len = len(tokenizer(question, return_tensors="pt").input_ids[0])
            question_msg = cot_prompter.wrap_input(question, icl_cnt=5)
            question_ids = build_chat_input(model, tokenizer, question_msg)
            prompt_len = len(question_ids[0]) - question_len - 1
            question_len = len(question_ids[0])
            assistant_msg = [{'role':'assistant', 'content':f'{cot} So the answer is: ({label})'}]
            input_msg = question_msg + assistant_msg
            input_ids = build_chat_input(model, tokenizer, input_msg)  
            stem_len = len(tokenizer(question.split('\n')[0],return_tensors="pt").input_ids[0])
            stem_len = prompt_len + stem_len
            cot_msg = question_msg + [{'role':'assistant', 'content':f'{cot}'}]
            cot_len = len(build_chat_input(model, tokenizer, cot_msg)[0])

        model.eval()
        outputs = model(
            input_ids=input_ids.to(model.device),
            return_dict=True,
            output_attentions=True,
            output_hidden_states=False,
        )
        del input_ids
        scores = []
        for layer in layers:
            attn_values = outputs['attentions'][layer]
            attn_values = torch.squeeze(attn_values)
            attn_scores = attn_values[:, prompt_len:, prompt_len:].detach().cpu().numpy()  
            if loss_type == 'cot':
                attn_scores = attn_scores[:, question_len-prompt_len:cot_len-prompt_len, :stem_len-prompt_len].sum(axis=-1).sum(axis=-1)
            else:
                attn_scores = attn_scores[:, cot_len-prompt_len:, :stem_len-prompt_len].sum()
            # fig_path = os.path.join(fold_path, f'layer-{layer+1}.pdf')
            # draw_attr_heat(attn_scores, x_tokens, y_tokens, fig_path)
            scores.append(attn_scores)
            del attn_values, attn_scores
            
            torch.cuda.empty_cache()
        del outputs
        torch.cuda.empty_cache()
        return scores
    
    
    def cal_mlp_attr(self, question, label, cot, layers, loss='cot'):
        question_len = len(cot_prompter.user_prompt.format(question))
        wrap_question = cot_prompter.wrap_input(question, icl_cnt=5)
        if loss_type == 'cot':
            input_text = wrap_question + cot 
        else:
            input_text = wrap_question + cot + f' So the answer is: ({label})'
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids
        
        question_len = len(tokenizer(wrap_question, return_tensors="pt").input_ids[0])
        cot_len = len(tokenizer(wrap_question + cot, return_tensors="pt").input_ids[0])
        
        labels = torch.full_like(input_ids, -100)
        if loss_type =='cot':
            labels[:, question_len:cot_len] = input_ids[:, question_len:cot_len]
        else:
            labels[:, -3] = input_ids[:, -3]

        model.train()
        up_scores = []
        down_scores = []
        outputs = None 
        loss = None 
        idx = -1
        
        outputs = model(
            input_ids=input_ids.to(model.device),
            labels=labels.to(model.device),
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False,
        )
        loss = outputs['loss']
        
        for layer in layers:
            mlp_up_values = model.model.layers[layer].mlp.up_proj.weight 
            mlp_down_values = model.model.layers[layer].mlp.down_proj.weight 
            mlp_up_grad = torch.autograd.grad(loss, mlp_up_values, create_graph=True, allow_unused=True)[0]
            mlp_down_grad = torch.autograd.grad(loss, mlp_down_values, create_graph=True, allow_unused=True)[0]
            mlp_up_score = self.cal_attr_score(mlp_up_values, mlp_up_grad).sum().detach().cpu().numpy()
            mlp_down_score = self.cal_attr_score(mlp_down_values, mlp_down_grad).sum().detach().cpu().numpy()
            up_scores.append(mlp_up_score)
            down_scores.append(mlp_down_score)
            
            del mlp_up_values, mlp_up_grad, mlp_down_values, mlp_down_grad
            torch.cuda.empty_cache()
            
        del loss, outputs
        torch.cuda.empty_cache()
        
        return [up_scores, down_scores]
    
    def probe_data(self, data, index):
        if reg:
            with open(full_cot_path, 'r') as f:
                cots = json.load(f)
        if res:
            with open(res_cot_path, 'r') as f:
                cots = json.load(f)
        pred_scores_ls = []   
            
        for msg in tqdm(data):
            label = msg['label']
            pred = msg['pred']
            idx = index[data.index(msg)]
            question = msg['question']
            steps = msg['steps']
            pred_cot = '.'.join(steps) + '.'
            
            if score == 'attn':
                if reg or res:
                    reg_answer = cots[idx]['answer']
                    if reg:
                        reg_answer = reg_answer[eval(label)-1]
                    reg_steps = dataloader.split_cot(reg_answer)
                    reg_cot = '.'.join(reg_steps) + '.'
                    if reg:
                        reg_cot = reg_cot.replace(' Reason: ',"")
                layers = range(40)
                scores = self.cal_attn(question, pred, pred_cot, layers)
                if loss_type == 'label':
                    new_scores = self.cal_attn(question, label, pred_cot, layers)
                    scores = scores + new_scores
        
                if reg:
                    reg_scores = self.cal_attn(question, label, reg_cot, layers)
                    scores = np.array(reg_scores) / len(reg_steps) - np.array(scores) / len(steps)
                
                if avg:
                    pred_scores_ls.append(scores) 
                else:   
                    fig_path = result_path + f'_idx-{idx}.pdf'
                    layers = [i+1 for i in layers]
                    index = list(range(len(scores.shape[1])))
                    draw_heat(layers,  index, scores.tolist(), fig_path)
            
            elif score == 'attn_attr':
                if reg or res:
                    reg_answer = cots[idx]['answer']
                    if reg:
                        reg_answer = reg_answer[eval(label)-1]
                    reg_steps = dataloader.split_cot(reg_answer)
                    reg_cot = '.'.join(reg_steps) + '.'
                    if reg:
                        reg_cot = reg_cot.replace(' Reason: ',"")

                layers = range(40)
                
                scores = self.cal_attn_attr(question, pred, pred_cot, layers)
                # if loss_type == 'label':
                #     if swap:
                #         new_scores = self.cal_attn_attr(question, label, pred_cot, layers, swap=True)
                #     else: 
                #         new_scores = self.cal_attn_attr(question, label, pred_cot, layers)
                #     scores = scores + new_scores
                    
        
                if reg or res or loss_type == 'label':
                    if reg or res:
                        reg_scores = self.cal_attn_attr(question, label, reg_cot, layers)
                    else:
                        if swap: 
                            reg_scores = self.cal_attn_attr(question, label, pred_cot, layers, swap=True)
                        else:
                            reg_scores = self.cal_attn_attr(question, label, pred_cot, layers)
                        reg_steps = steps
                    
                    if diff:
                        scores = np.array(reg_scores[0]) / len(reg_steps) - np.array(scores[0]) / len(steps)
                    else:
                        scores = np.concatenate([np.array(scores[0]) / len(steps), np.array(reg_scores[0])/ len(reg_steps)], axis=-1)
                        
                if avg:
                    pred_scores_ls.append(scores) 
                else:   
                    fig_path = result_path + f'_idx-{idx}.pdf'
                    layers = [i+1 for i in layers]
                    if reg:
                        if not diff:
                            draw_plot(layers, list(scores), ['Drift', 'Correct'], fig_path)
                    else:
                        draw_plot(layers, list(scores), ['stem', 'option', 'cot'], fig_path)
            
            elif score == 'mlp_attr':
                if reg:
                    reg_answers = cots[idx]['answer']
                    reg_answer = reg_answers[eval(label)-1]
                    reg_steps = dataloader.split_cot(reg_answer)
                    reg_cot = '.'.join(reg_steps) + '.'
                    reg_cot = reg_cot.replace(' Reason: ',"")
                layers = range(40)

                
                scores = self.cal_mlp_attr(question, pred, pred_cot, layers)
                
                if reg or res or loss_type == 'label':
                    if reg:
                        reg_scores = self.cal_mlp_attr(question, label, reg_cot, layers)
                    else:
                        if swap: 
                            reg_scores = self.cal_mlp_attr(question, label, pred_cot, layers, swap=True)
                        else:
                            reg_scores = self.cal_mlp_attr(question, label, pred_cot, layers)
                        reg_steps = steps
                    
                    if diff:
                        scores = np.array(reg_scores[cnt]) - np.array(scores[cnt])
                    else:
                        scores = np.concatenate([scores[cnt], reg_scores[cnt]], axis=-1)
                
                if avg:
                    pred_scores_ls.append(scores)
                else:   
                    fig_path = result_path + f'_idx-{idx}.pdf'
                    layers = [i+1 for i in layers]
                    draw_plot(layers, list(scores), ['up_proj', 'down_proj'], fig_path)
            
            
        if score in ['attn_attr', 'mlp_attr'] and avg:
            if diff:
                return np.array(pred_scores_ls).mean(axis=0)
            else:
                return np.array(pred_scores_ls)
        
        elif score in ['attn'] and avg:
            if loss_type == 'label':
                return np.array(pred_scores_ls)
            else:
                return np.array(pred_scores_ls).mean(axis=0)


def get_index(mode, dataset):
    index = None
    if model_name[:5] == 'Llama':
        if mode == 'C2W':
            if dataset == 'csqa':
                # index = [41,49,158,161,174,244,276,283,286,297,386,394,402,413,424,431,441,443,457,523,539,652,700,709,754,869,881,898,939,946]
                if loss_type == 'label':
                    index = [36,331,379,395,521,525,527,599,654,826,893,913,998]
                else:
                    index =[41,49,158,161,174,244,276,283,286,297,386,394,402,413,424,431,441,443,457,523,539,652,700,709,754,869,881,898,939,946]
            elif dataset == 'wino':
                if loss_type == 'label':
                    index = [40,47,73,175,180,185,197,232,255,266,274,306,316,327,333,409,423,427,433,444,454,481,493]
                else:
                    index = [7,15,50,53,97,108,119,121,132,201,207,209,235,253,284,285,307,320,338,342,347,387,390,426,453,467,475,478,482,490,498]
    else:
        if mode == 'C2W':
            if dataset == 'csqa':
                if loss_type == 'label':
                    index = [86,221,263,279,280,342,352,395,399,408,471,545,599,761,857,877,913]
                else:
                    index =[3,8,56,158,167,175,189,225,257,283,298,318,323,439,457,492,499,540,568,578,580,582,596,623,679,700,792,838,839,860,865,899,915,954,988]
            elif dataset == 'wino':
                if loss_type == 'label':
                    index = [28,53,90,93,97,102,145,148,158,183,185,201,261,316,327,348,366,393,429,437,453,465,506,584,642,658,661,678,696,710,732,734,755,756,771,805,843,882]
                else:
                    index = [56,69,132,142,146,147,157,163,167,203,218,221,307,320,321,342,378,383,440,512,524,591,594,610,620,624,627,645,685,686,745,751,753,760,767,832,876,930,955,966,986]
    if mode == 'RAND':
        index = random.sample(range(1000), cnt)
        sorted(index)
    return index 

if __name__ == '__main__':
    probe = Probe()
    dataloader = CoTLoader()
    layers = [i+1 for i in range(40)]
    
    # if reg: 
    #     w_index = get_index(mode='C2W', dataset=dataset)
    #     w_data, index = dataloader.load_data(cot_file=cot_file_path, base_file=base_file_path, mode='C2W', cnt=cnt, index=w_index)
    #     w_results = probe.probe_data(w_data, index)
    #     if diff:
    #         c_index = get_index(mode='W2C', dataset=dataset)
    #         c_data, index = dataloader.load_data(cot_file=cot_file_path, base_file=base_file_path, mode='W2C', cnt=cnt, index=c_index)
    #         c_results = probe.probe_data(c_data, index)
        
    # else:
    index = get_index(mode='C2W', dataset=dataset)
    data, index = dataloader.load_data(cot_file=cot_file_path, base_file=base_file_path, cnt=cnt, index=index)
    if res:
        res_data = []
        res_index = []
        with open(res_cot_path, 'r') as f:
            res_cots = json.load(f)
        for i in range(len(data)):
            if res_cots[index[i]]['cor_flag']:
                res_data.append(data[i])
                res_index.append(index[i])
        index = res_index
        data = res_data
    if swap:
        swap_data = []
        swap_index = []
        with open(swap_file_path, 'r') as f:
            flags = json.load(f)
        for i in range(len(data)):
            if flags[index[i]]['cor_flag']:
                swap_data.append(data[i])
                swap_index.append(index[i])
        index = swap_index
        data = swap_data
    
    results = probe.probe_data(data, index)
   
    fig_path = result_path + f'.pdf'
    
    if score == 'attn':
        if loss_type == 'cot':    
            index = layers
            draw_heat(index, results.tolist(), fig_path) 
        else:  
            labels = ['Drift', 'Correct'] 
            draw_line_plot(layers, results, labels, fig_path, 'Attn')
        
    if score in ['attn_attr', 'mlp_attr']:
        if reg or res:
            if diff:
                # c_data, index = dataloader.load_data(cot_file=cot_file_path, base_file=base_file_path, mode='W2C', cnt=len(index), index=c_index)
                c_data, index = dataloader.load_data(cot_file=cot_file_path, base_file=base_file_path, mode=mode, cnt=len(index))
                c_results = probe.probe_data(c_data, index)
                results = np.concatenate([results, c_results], axis=-1)     
                labels = ['Drift', 'Normal'] 
                draw_line_plot(layers, [results], labels, fig_path, 'Attr Div')
            else:
                if reg:
                    labels = ['Drift', 'Correct'] 
                elif res:
                    labels = ['CoT', 'RD'] 
                draw_line_plot(layers, results, labels, fig_path, 'Attr')
        else:
            if swap:
                labels = ['CoT', 'SPS'] 
                draw_line_plot(layers, results, labels, fig_path, 'Attr')
            elif loss_type == 'label':   
                labels = ['Drift', 'Correct'] 
                draw_line_plot(layers, results, labels, fig_path, 'Attr')
            else:
                
                scores = []
                for result in results:
                    scores.append(np.concatenate(result.tolist(), axis=-1))
                if score == 'attn_attr':
                    labels = ['context', 'option', 'cot']
                else:
                    labels = ['up_proj', 'down_proj']
                fig_path = result_path + f'.pdf'
                draw_line_plot(layers, scores, labels, fig_path, 'Attr') 
                

                        
    # if reg:
    #     if diff:
    #         for rel in label_dic.keys():
    #             key = label_dic[rel]
    #             fig_path = result_path + f'_{rel}.png'
    #             w_scores = np.array(w_results)[:, key, :]
    #             c_scores = np.array(c_results)[:, key, :]
    #             results = np.concatenate([w_scores, c_scores], axis=-1)     
    #             labels = ['Drift', 'Correct'] 
    #             draw_line_plot(layers, results, labels, fig_path) 
    #     else:
    #         fig_path = result_path + f'.png'
    #         w_scores = np.array(w_results)
    #         # c_scores = np.array(c_results)
    #         results = np.concatenate(w_results, axis=-1)     
    #         labels = ['Drift', 'Correct'] 
    #         draw_line_plot(layers, results, labels, fig_path) 
    # else:
    #     scores = []
    #     for result in results:
    #         scores.append(np.concatenate(result))
    #     labels = list(label_dic.keys())
    #     fig_path = result_path + f'.png'
    #     draw_line_plot(layers, scores, labels, fig_path) 
    
    