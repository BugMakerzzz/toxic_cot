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
from prompts.wrap_prompt import LlamaPrompter
from load_data import DataLoader, CoTLoader
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, Accelerator
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
from metrics import draw_plot, draw_heat, draw_line_plot, draw_attr_heat
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
random.seed(17)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='Llama-2-13b-chat-hf')
parser.add_argument('--dataset', type=str, default='wino')
parser.add_argument('--datalength', type=int, default=1000)
parser.add_argument('--mode', type=str, default=None)
parser.add_argument('--score', type=str, default=None)
parser.add_argument('--diff_cot', type=str, default=None)
parser.add_argument('--diff_logits', type=str, default=None)
parser.add_argument('--avg', type=str, default=None)
parser.add_argument('--cnt', type=int, default=1000)
parser.add_argument('--reg', action='store_true')
parser.add_argument('--split', action='store_true')
parser.add_argument('--direct', action='store_true')
parser.add_argument('--loss', type=str, default='cot')
args = parser.parse_args()


model_name = args.model
dataset = args.dataset
datalength = args.datalength
mode = args.mode
diff_logits = args.diff_logits
diff_cot = args.diff_cot
avg = args.avg
cnt = args.cnt
reg = args.reg
direct = args.direct
split = args.split
score = args.score
loss_type = args.loss
model_path = f'./model/{model_name}'
cot_file_path  = f'./result/{dataset}/{model_name}_cot_answer_dev_{datalength}.json'
base_file_path = f'./result/{dataset}/{model_name}_direct_answer_dev_{datalength}.json'
result_path = f'./result/{dataset}/fig/{model_name}_s-{score}_m-{mode}_c-{cnt}_r-{reg}_l-{loss_type}'
full_cot_path = f'./result/{dataset}/{model_name}_cot_dev_1000.json'


# device_map = {'model.embed_tokens': 0, 'model.layers.0': 0, 'model.layers.1': 3, 'model.layers.2': 3, 'model.layers.3': 3, 'model.layers.4': 2, 'model.layers.5': 1, 'model.layers.6': 0, 'model.layers.7': 0, 'model.layers.8': 0, 'model.layers.9': 0, 'model.layers.10': 1, 'model.layers.11': 1, 'model.layers.12': 1, 'model.layers.13': 1, 'model.layers.14': 1, 'model.layers.15': 1, 'model.layers.16': 1, 'model.layers.17': 1, 'model.layers.18': 1, 'model.layers.19': 0, 'model.layers.20': 2, 'model.layers.21': 2, 'model.layers.22': 2, 'model.layers.23': 2, 'model.layers.24': 2, 'model.layers.25': 2, 'model.layers.26': 2, 'model.layers.27': 2, 'model.layers.28': 2, 'model.layers.29': 0, 'model.layers.30': 2, 'model.layers.31': 3, 'model.layers.32': 3, 'model.layers.33': 3, 'model.layers.34': 0, 'model.layers.35': 3, 'model.layers.36': 3, 'model.layers.37': 3, 'model.layers.38': 0, 'model.layers.39': 0, 'model.norm': 3, 'lm_head': 3}
device_map = {'model.embed_tokens': 0, 'model.layers.0': 0, 'model.layers.1': 0, 'model.layers.2': 0, 'model.layers.3': 0, 'model.layers.4': 0, 'model.layers.5': 0, 'model.layers.6': 0, 'model.layers.7': 0, 'model.layers.8': 0, 'model.layers.9': 0, 'model.layers.10': 1, 'model.layers.11': 1, 'model.layers.12': 1, 'model.layers.13': 1, 'model.layers.14': 1, 'model.layers.15': 1, 'model.layers.16': 1, 'model.layers.17': 1, 'model.layers.18': 1, 'model.layers.19': 1, 'model.layers.20': 2, 'model.layers.21': 2, 'model.layers.22': 2, 'model.layers.23': 2, 'model.layers.24': 2, 'model.layers.25': 2, 'model.layers.26': 2, 'model.layers.27': 2, 'model.layers.28': 2, 'model.layers.29': 2, 'model.layers.30': 2, 'model.layers.31': 3, 'model.layers.32': 3, 'model.layers.33': 3, 'model.layers.34': 3, 'model.layers.35': 3, 'model.layers.36': 3, 'model.layers.37': 3, 'model.layers.38': 3, 'model.layers.39': 3, 'model.norm': 3, 'lm_head': 3}
device_num = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))

model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True, device_map=device_map)
# model = accelerator.prepare_model(model)
# config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
# with init_empty_weights():
#     model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16, trust_remote_code=True)
#     # model = AutoModelForSeq2SeqLM.from_config(config, torch_dtype=torch.float16, trust_remote_code=True)
# no_split_modules = model._no_split_modules
# model = load_checkpoint_and_dispatch(
#     model, model_path, device_map="auto", no_split_module_classes=no_split_modules
# )

model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)  
cot_prompter = LlamaPrompter(dataset=dataset, task='cot_answer')
base_prompter = LlamaPrompter(dataset=dataset, task='direct_answer')

# accelerator = Accelerator()
# model = accelerator.prepare_model(model)

class Probe():
    def __init__(self) -> None:
        pass
    
    
    def cal_probs(self, question_text, pred_text, layers, diff_logits=None):
        input_text = question_text + pred_text
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
        # prefix_ids = tokenizer(question_text, return_tensors="pt").input_ids.to(model.device)
        # continue_ids = input_ids[0, prefix_ids.shape[-1]:].cpu().numpy()
        continue_ids = input_ids[0, -2].cpu().numpy()
        outputs = model(
            input_ids=input_ids,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=True,
        )
     
        hidden_states = outputs['hidden_states']
        # attn_values = outputs['attentions']
        # assert len(hidden_states) == 41 
        scores = []
        for layer in layers:
            logits = model.lm_head(hidden_states[layer])
            logits = logits[:, -3, :].float()
            logits = F.softmax(logits, dim=-1)
            probs = probs[:, continue_ids].squeeze()
            # probs = logits[:,range(logits.shape[1]), res_tok].sum(axis=-1).tolist()
            

            # logits = model.lm_head(hidden_states[layer])[0, prefix_ids.shape[-1] - 1: -1, :]
            
            # logits = logits.log_softmax(dim=-1)
            # if diff_logits and diff_logits != 'contrast':
            #     if layer == 0:
            #         continue
            #     if diff_logits == 'base':
            #         base_logits = model.lm_head(hidden_states[0])[0, prefix_ids.shape[-1] - 1: -1, :]
            #         base_logits = base_logits.log_softmax(dim=-1)
            #     elif diff_logits == 'adj':
            #         base_layer = layers[layers.index(layer)-1]
            #         base_logits = model.lm_head(hidden_states[base_layer])[0, prefix_ids.shape[-1] - 1: -1, :]
            #         base_logits = base_logits.log_softmax(dim=-1)
            #     logits = logits - base_logits
            #     logits = logits.log_softmax(dim=-1)
            #     # print(logits)
            #     # probs = diff_logits[range(diff_logits.shape[0]), continue_ids].sum().item()
            # scores.append(logits.cpu().numpy())  
            scores.append(probs.cpu().numpy())
            del logits  
        del input_ids, outputs
        torch.cuda.empty_cache()
        return scores
        # return scores, continue_ids


    def cal_attn_attr(self, question, label, cot, layers):
        question_len = len(cot_prompter.user_prompt.format(question))
        prompt = cot_prompter.wrap_input(question, icl_cnt=5)[:-question_len]
        wrap_question = cot_prompter.wrap_input(question, icl_cnt=5)
        if loss_type == 'cot':
            input_text = wrap_question + cot 
        else:
            input_text = wrap_question + cot + f' So the answer is: {(label)}' 
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids
        
        prompt_len = len(tokenizer(prompt, return_tensors="pt").input_ids[0])
        question_len = len(tokenizer(wrap_question, return_tensors="pt").input_ids[0])
        stem = '\n'.join(wrap_question.split('\n')[:-1])
        stem_len = len(tokenizer(stem, return_tensors="pt").input_ids[0])
        cot_len = len(tokenizer(wrap_question + cot, return_tensors="pt").input_ids[0])
        
        labels = torch.full_like(input_ids, -100)
        if loss_type == 'cot':
            labels[:, question_len:] = input_ids[:, question_len:]
        else:
            labels[:, -2] = input_ids[:, -2]

        stem_scores = []
        option_scores = []
        cot_scores = []
        
        loss = None 
        outputs = None 
        idx = -1
        
        model.train()
        
        for layer in layers: 
            key = f'model.layers.{layer}'
            cuda_idx = device_map[key]
            if cuda_idx != idx:
                idx = cuda_idx
                device = f'cuda:{cuda_idx}'
                outputs = model(
                    input_ids=input_ids.to(device),
                    labels=labels.to(device),
                    return_dict=True,
                    output_attentions=True,
                    output_hidden_states=False,
                )
                loss = outputs['loss']
            attn_values = outputs['attentions'][layer]
            attn_grad = torch.autograd.grad(loss, attn_values, create_graph=True, allow_unused=True)[0]
            attn_values = torch.squeeze(attn_values)
            attn_grad = torch.squeeze(attn_grad)
            attn_scores = torch.zeros_like(attn_values[0,:,:])
            for i in range(40):
                attn_scores += self.cal_attr_score(attn_values[i,:,:], attn_grad[i,:,:])
            attn_scores = attn_scores[prompt_len:,prompt_len:].detach().cpu().numpy()
            
            if loss_type == 'cot':
                stem_attr = attn_scores[question_len-prompt_len:cot_len-prompt_len, :stem_len-prompt_len].sum() 
                option_attr = attn_scores[question_len-prompt_len:cot_len-prompt_len, stem_len-prompt_len:question_len-prompt_len-6].sum() 
                cot_attr = attn_scores[question_len-prompt_len:cot_len-prompt_len, question_len-prompt_len:cot_len-prompt_len]
                for i in range(cot_len-question_len):
                    cot_attr[i,i] = 0
                cot_attr = cot_attr.sum() 
            else:
                stem_attr = attn_scores[-3, :stem_len-prompt_len].sum() 
                option_attr = attn_scores[-3, stem_len-prompt_len:question_len-prompt_len-6].sum() 
                cot_attr = attn_scores[-3, question_len-prompt_len:cot_len-prompt_len].sum()
            
            stem_scores.append(stem_attr)
            option_scores.append(option_attr)
            cot_scores.append(cot_attr)
            
            del attn_values, attn_grad, attn_scores  
            torch.cuda.empty_cache()
            
        del input_ids, loss, outputs
        torch.cuda.empty_cache()
        
        return [stem_scores, option_scores, cot_scores]
    
    
    def cal_attr_score(self, value, grad, steps=20):
        grad_int = torch.zeros_like(value*grad)
        for i in range(steps):
            k = (i+1) / steps
            grad_int += k * grad
        scores = 1 / steps * value * grad_int
        return torch.abs(scores) 
    
    
    def cal_attn(self, question, label, cot, layers, fold_path):  
 
        question_len = len(cot_prompter.user_prompt.format(question))
        prompt = cot_prompter.wrap_input(question, icl_cnt=5)[:-question_len]
        wrap_question = cot_prompter.wrap_input(question, icl_cnt=5)
        input_text = wrap_question + cot + f' So the answer is: ({label})'
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
        prompt_len = len(tokenizer(prompt, return_tensors="pt").input_ids[0]) - 1

        question_len = len(tokenizer(wrap_question, return_tensors="pt").input_ids[0])
        cot_len = len(tokenizer(wrap_question + cot, return_tensors="pt").input_ids[0])
        
            
        fig_tokens = tokenizer.convert_ids_to_tokens(input_ids[0, prompt_len:])
        y_tokens = fig_tokens[question_len-prompt_len:]
        x_tokens = fig_tokens[:cot_len-prompt_len]
        # print(fig_tokens)

        outputs = None 
        
        model.eval()
        outputs = model(
            input_ids=input_ids.to(model.device),

            return_dict=True,
            output_attentions=True,
            output_hidden_states=False,
        )
            
        for layer in layers:
            attn_values = outputs['attentions'][layer]
            attn_scores = torch.squeeze(attn_values)
            attn_scores = attn_scores[:, prompt_len:, prompt_len:].sum(axis=0).detach().cpu().numpy()
                
            attn_scores = attn_scores[question_len-prompt_len:, :cot_len-prompt_len]
            fig_path = os.path.join(fold_path, f'layer-{layer+1}.pdf')
            draw_attr_heat(attn_scores, x_tokens, y_tokens, fig_path)
            del attn_values
            del attn_scores
            torch.cuda.empty_cache()

        del outputs
        torch.cuda.empty_cache()
        return 
    
    
    def cal_mlp_attr(self, question, label, cot, layers, loss='cot'):
        question_len = len(cot_prompter.user_prompt.format(question))
        wrap_question = cot_prompter.wrap_input(question, icl_cnt=5)
        if loss_type == 'cot':
            input_text = wrap_question + cot 
        else:
            input_text = wrap_question + cot + f' So the answer is: {(label)}'
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
        
        for layer in layers:
            model.train()
            key = f'model.layers.{layer}'
            cuda_idx = device_map[key]
            if cuda_idx != idx:
                idx = cuda_idx
                device = f'cuda:{cuda_idx}'
                outputs = model(
                    input_ids=input_ids.to(device),
                    labels=labels.to(device),
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                )
                loss = outputs['loss']
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
    
    def cal_logits_score(self, logits_ls, ids_ls, diff_cot):
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
            elif diff_cot == 'add':
                base_logits = - np.array(logits_ls[0])

            if diff_cot:
                logits = logits - base_logits

            # if diff_logits == 'score':
            #     probs = logits[:,range(logits.shape[1]), ids].sum(axis=-1)
            #     probs = (probs - probs[0])[1:].tolist()
            # else:
            probs = logits[:,range(logits.shape[1]), ids].sum(axis=-1).tolist()
            
            scores.append(probs)
        return scores

    def llm_score(self, question, pred, cot, layers, direct=False, split=False, diff_cot=None, diff_logits=None):
        with torch.no_grad():
            pred_ids = []
            pred_logits = []
            pred_scores = []
            if split:
                cot = cot.split('.')
            else:
                cot = [cot]
            for i in range(len(cot)+1):
                if i == 0:
                    cot_question = cot_prompter.wrap_input(question, icl_cnt=5)   
                    if not direct:
                        continue
                else:
                    cot_question += cot[i-1] 

                if i == len(cot) or i == 0:
                    probs = self.cal_probs(cot_question + ' So the answer is: ', f'({pred})', layers, diff_logits)
                else:
                    probs = self.cal_probs(cot_question + '...So the answer is: ', f'({pred})', layers, diff_logits)
                pred_scores.append(probs)
                # pred_ids.append(ids)
            # pred_scores = self.cal_logits_score(pred_logits, pred_ids, diff_cot)
            
        return pred_scores

    def get_avg(self, scores, avg):
        if avg == 'norm':
            max_len = 0
            for score in scores:
                layer_len = len(score[0])
                if len(score) > max_len:
                    max_len = len(score)
            avg_scores = np.ma.empty((max_len, layer_len, len(scores))) 
            avg_scores.mask = True
            for i in range(len(scores)):
                score = np.array(scores[i])
                avg_scores[:score.shape[0], :score.shape[1], i] = score            
            scores = avg_scores.mean(axis=-1).tolist()[:5]
        elif avg == 'steps':
            score_dic = {}
            for score in scores:
                if len(score) not in score_dic.keys():
                    score_dic[len(score)] = [score]
                else:
                    score_dic[len(score)].append(score)
            for i, score in score_dic.items():
                score = np.array(score).mean(axis=0).tolist()
                score_dic[i] = score
            scores = score_dic
        return scores

    def llm_avg_score(self, pred_scores_ls, label_scores_ls, layers, avg, x_range, result_path):
        if avg == 'heat':
            pred_fig_path = result_path + '_heat-pred.png'
            label_fig_path = result_path + '_heat-label.png'
            pred_first_scores = np.zeros(len(layers))
            pred_mid_scores = np.zeros(len(layers))
            pred_last_scores = np.zeros(len(layers))
            label_first_scores = np.zeros(len(layers))
            label_mid_scores = np.zeros(len(layers))
            label_last_scores = np.zeros(len(layers))
            for scores in pred_scores_ls:
                steps = len(scores)
                pred_first_scores += np.array(scores[0])
                pred_last_scores += np.array(scores[-1])
                if steps <= 2:
                    pred_mid_scores += np.array([scores[0], scores[-1]]).mean(axis=0)
                else:
                    pred_mid_scores += np.array(scores[1:-1]).mean(axis=0)
            for scores in label_scores_ls:
                steps = len(scores)
                label_first_scores += np.array(scores[0])
                label_last_scores += np.array(scores[-1])
                if steps <= 2:
                    label_mid_scores += np.array([scores[0], scores[-1]]).mean(axis=0)
                else:
                    label_mid_scores += np.array(scores[1:-1]).mean(axis=0) 
            # pred_cnt = len(pred_scores_ls)
            # label_cnt = len(label_scores_ls)
            pred_scores_ls = [(pred_first_scores).tolist(), (pred_mid_scores).tolist(), (pred_last_scores).tolist()]
            label_scores_ls = [(label_first_scores).tolist(), (label_mid_scores).tolist(), (label_last_scores).tolist()]
            index = ['First', 'Mid', 'Last']
            draw_heat(x_range, index, pred_scores_ls, pred_fig_path)
            draw_heat(x_range, index, label_scores_ls, label_fig_path)
        else:
            pred_scores = self.get_avg(pred_scores_ls, avg)
            label_scores = self.get_avg(label_scores_ls, avg)
            if avg == 'norm':
                legends = []
                for i in range(len(pred_scores)):
                    legends.append(f'pred_step_{i+1}')
                for i in range(len(label_scores)):
                    legends.append(f'label_step_{i+1}')
                fig_path = result_path + '_norm-avg.png'
                draw_plot(x_range, pred_scores+label_scores, legends, fig_path)
            elif avg == 'steps':
                fold_path = result_path + '_steps-avg/'
                if not os.path.exists(fold_path):
                    os.mkdir(fold_path)
                for i in sorted(pred_scores.keys()):
                    pred_score = pred_scores[i]
                    label_score = label_scores[i]
                    pred_legend = [f'l{i}_pred_step_{x}' for x in range(1,i+1)]
                    label_legend = [f'l{i}_label_step_{x}' for x in range(1,i+1)]
                    fig_path = os.path.join(fold_path, f'l{i}.png')
                    draw_plot(x_range, pred_score+label_score, pred_legend+label_legend, fig_path)
        return 
    
    
    def probe_data(self, data, index):
        if reg:
            with open(full_cot_path, 'r') as f:
                cots = json.load(f)
        if score == 'llm':
            pred_scores_ls = []  
            label_scores_ls = []
            layers = range(0,41)
            x_range = layers
        elif score in ['sim', 'attn', 'mlp']:
            scores_ls = []
            x_range = []
        elif score in ['attn_attr', 'mlp_attr']:
            pred_scores_ls = []   
            
        for msg in tqdm(data):
            label = msg['label']
            pred = msg['pred']
            idx = index[data.index(msg)]
            question = msg['question']
            answers = msg['answer']
            steps = msg['steps']
            pred_cot = '.'.join(steps) + '.'
            label_cot = '.'.join(steps) + '.'
            
            if score == 'llm':
                if reg:
                    steps = cots[idx]['answer']
                    if eval(pred)-1 >= len(steps):
                        continue
                    pred_cot = steps[eval(pred)-1]
                    pred_steps = dataloader.split_cot(pred_cot)
                    pred_cot = '.'.join(pred_steps) + '.' 
                    label_cot = steps[eval(label)-1]
                    label_steps = dataloader.split_cot(label_cot)
                    label_cot = '.'.join(label_steps) + '.'
                    
                if not steps:
                    print(f'Num {idx} question\'s cot is too short!!!')
                    continue
                
                if dataset == 'gsm8k':
                    label_option = str(label)
                    pred_option = str(pred)
                else:
                    label_option = f'({label})'
                    pred_option = f'({pred})'
                pred_scores = self.llm_score(question, pred_option, pred_cot, layers, direct, split, diff_cot, diff_logits)
                label_scores = self.llm_score(question, label_option, label_cot, layers, direct, split, diff_cot, diff_logits)
                
                if diff_logits == 'contrast':
                    pred_sub_scores = self.llm_score(question, label_option, pred_cot, layers, direct, split, diff_cot, diff_logits)
                    label_sub_scores = self.llm_score(question, pred_option, label_cot, layers, direct, split, diff_cot, diff_logits)
                    pred_scores = (np.array(pred_scores) - np.array(pred_sub_scores)).tolist()
                    label_scores = (np.array(label_scores) - np.array(label_sub_scores)).tolist()   
                pred_legends = [f'pred_step_{i+1}' for i in range(len(pred_scores))]
                label_legends = [f'label_step_{i+1}' for i in range(len(label_scores))]
                if diff_logits and diff_logits != 'contrast':
                    x_range = layers[1:]
                else:
                    x_range = layers
                if avg:
                    pred_scores_ls.append(pred_scores)
                    label_scores_ls.append(label_scores)
                else:
                    fig_path = result_path + f'_{idx}.png'
                    draw_plot(x_range, pred_scores+label_scores, pred_legends+label_legends, fig_path)
        
            elif score == 'attn':
                layers = range(40)
                fold_path = result_path + f'_attn_heat_{idx}/'
                if not os.path.exists(fold_path):
                    os.mkdir(fold_path)
                self.cal_attn(question, pred, pred_cot, layers, fold_path)

            
            elif score == 'attn_attr':
                if reg:
                    reg_answers = cots[idx]['answer']
                    reg_answer = reg_answers[eval(label)-1]
                    reg_steps = dataloader.split_cot(reg_answer)
                    reg_cot = '.'.join(reg_steps) + '.'
                    reg_cot = reg_cot.replace(' Reason: ',"")
                layers = range(40)


                scores = self.cal_attn_attr(question, label, pred_cot, layers)
        
                if reg:
                    reg_scores = self.cal_attn_attr(question, label, reg_cot, layers)
                    scores = (np.array(scores) / len(steps) - np.array(reg_scores) / len(reg_steps)) / (np.array(reg_scores) / len(reg_steps))
                
                if avg:
                    pred_scores_ls.append(scores) 
                else:   
                    fig_path = result_path + f'_idx-{idx}.pdf'
                    layers = [i+1 for i in layers]
                    draw_plot(layers, list(scores), ['stem', 'option', 'cot', 'sum'], fig_path)
            
            elif score == 'mlp_attr':
                if reg:
                    reg_answers = cots[idx]['answer']
                    reg_answer = reg_answers[eval(label)-1]
                    reg_steps = dataloader.split_cot(reg_answer)
                    reg_cot = '.'.join(reg_steps) + '.'
                    reg_cot = reg_cot.replace(' Reason: ',"")
                layers = range(40)

                
                scores = self.cal_mlp_attr(question, label, pred_cot, layers)
                
                if reg:
                    reg_scores = self.cal_mlp_attr(question, label, reg_cot, layers)
                    scores = (np.array(scores) / len(steps) - np.array(reg_scores) / len(reg_steps)) / (np.array(reg_scores) / len(reg_steps))
                
                if avg:
                    pred_scores_ls.append(scores)
                else:   
                    fig_path = result_path + f'_idx-{idx}.pdf'
                    layers = [i+1 for i in layers]
                    draw_plot(layers, list(scores), ['up_proj', 'down_proj'], fig_path)
            
            
        if score == 'llm' and avg:
            self.llm_avg_score(pred_scores_ls, label_scores_ls, layers, avg, x_range, result_path)
        
        elif score == 'sim':
            fig_path = result_path + f'_sim.png'
            draw_line_plot(x_range, scores_ls, fig_path)    
            
        elif score in ['attn_attr', 'mlp_attr'] and avg:
            return pred_scores_ls


def get_index(mode, dataset):
    index = None
    if mode == 'C2W':
        if dataset == 'csqa':
            index = [41,49,158,161,174,244,276,283,286,297,386,394,402,413,424,431,441,443,457,523,539,652,700,709,754,869,881,898,946][:cnt]
            # index = [108,119,121,132,201]
        elif dataset == 'wino':
            # index = [40,47,73,175,180,185,197,232,255,266,274,306,316,327,333,409,423,427,433,444,454,481,493][:cnt] 
            index = [7,15,50,53,84,97,108,119,121,132,201,207,209,235,253,284,285,307,320,338,342,347,387,390,426,453,467,475,478,482,490,498,512][:cnt]
    if mode == 'RAND':
        index = random.sample(range(1000), cnt)
        sorted(index)
    return index 

if __name__ == '__main__':
    probe = Probe()
    dataloader = CoTLoader()
    layers = [i+1 for i in range(40)]
    
    if reg: 
        w_index = get_index(mode='C2W', dataset=dataset)
        w_data, index = dataloader.load_data(cot_file=cot_file_path, base_file=base_file_path, mode='C2W', cnt=cnt, index=w_index)
        w_results = probe.probe_data(w_data, index)
        
        c_index = get_index(mode='W2C', dataset=dataset)
        c_data, index = dataloader.load_data(cot_file=cot_file_path, base_file=base_file_path, mode='W2C', cnt=cnt, index=c_index)
        c_results = probe.probe_data(c_data, index)
        
    else:
        index = get_index(mode=mode, dataset=dataset)
        data, index = dataloader.load_data(cot_file=cot_file_path, base_file=base_file_path, mode=mode, cnt=cnt, index=index)
        results = probe.probe_data(data, index)

    if score == 'attn_attr':    
        label_dic = {'stem':0, 'option':1, 'cot':2}
    elif score == 'mlp_attr':
        label_dic = {'up_proj':0, 'down_proj':1}

    if reg:
        for rel in label_dic.keys():
            key = label_dic[rel]
            fig_path = result_path + f'_{rel}.png'
            w_scores = np.array(w_results)[:, key, :]
            c_scores = np.array(c_results)[:, key, :]
            results = np.concatenate([w_scores, c_scores], axis=-1)     
            labels = ['Drift', 'Correct'] 
            draw_line_plot(layers, results, labels, fig_path) 
    else:
        scores = []
        for result in results:
            scores.append(np.concatenate(result))
        labels = list(label_dic.keys())
        fig_path = result_path + f'.png'
        draw_line_plot(layers, scores, labels, fig_path) 
    
    