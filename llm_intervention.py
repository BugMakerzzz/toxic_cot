import os
import torch
import argparse
import json
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModelForSeq2SeqLM
from prompts.wrap_prompt import LlamaPrompter
from load_data import CoTLoader, InterventionData
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
from metrics import draw_plot, draw_heat, draw_line_plot, draw_attr_heat
from intervention_model import Model
random.seed(17)

## argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='Llama-2-13b-chat-hf')
parser.add_argument('--dataset', type=str, default='wino')
parser.add_argument('--mode', type=str, default='W2C')
parser.add_argument('--cnt', type=int, default=10)
parser.add_argument('--exp', type=str, default='mlp')
args = parser.parse_args()
model_name = args.model
dataset = args.dataset
mode = args.mode
cnt = args.cnt
exp = args.exp

## Path 
model_path = f'./model/{model_name}'
cot_file_path  = f'./result/{dataset}/{model_name}_cot_answer_dev_1000.json'
base_file_path = f'./result/{dataset}/{model_name}_direct_answer_dev_1000.json'
full_cot_path = f'./result/{dataset}/{model_name}_cot_dev_1000.json'
result_path = f'./result/{dataset}/fig/{model_name}_{mode}_'

## Load Model
model = Model(model_name=model_name)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)  
cot_prompter = LlamaPrompter(dataset=dataset, task='cot_answer')

## Load Data
index = None 
if mode == 'C2W':
    if dataset == 'csqa':
        index = [41,49,158,161,174,219,244,276,283,286,297,386,394,402,413,424,431,441,443,457][:cnt]
        # index = [108,119,121,132,201]
    elif dataset == 'wino':
        index = [40,47,73,175,180,185,197,232,255,266,274][:cnt]
        #  index = [7, 15, 50, 53, 84, 97, 108,119,121,132,201, 207,209, 235,253][:cnt]
        
dataloader = CoTLoader()
data, index = dataloader.load_data(cot_file=cot_file_path, base_file=base_file_path, mode=mode, cnt=cnt, index=index)
with open(full_cot_path, 'r') as f:
    cots = json.load(f)
inter_data_list = []
for msg in data:
    idx = data.index(msg)
    steps = cots[idx]['answer']
    reg_cot = steps[eval(msg['label'])-1]
    reg_cot = reg_cot.replace(' Reason: ',"")
    reg_steps = dataloader.split_cot(reg_cot)
    msg['reg'] = reg_steps
    inter_data_list.append(InterventionData(msg, tokenizer, cot_prompter))

if exp == 'mlp':
    results = model.intervention_experiment(inter_data_list)
    x_range = range(0, 41)
else:
    results = model.attention_experiment(inter_data_list)
    x_range = range(1, 41)
    
fold_path = result_path + f'_{exp}-inter/'
if not os.path.exists(fold_path):
    os.mkdir(fold_path)
for i, result in results.items():
    _, _, base_result, alt_result = result
    
    labels = []
    values = []
    for idx, scores in alt_result.items():
        if idx == 0:
            label = 'stem'
        elif idx == 1:
            label = 'option'
        else:
            label = f'cot_step-{idx-1}'
        values.append(scores.squeeze().numpy())
        labels.append(label)
    path = os.path.join(fold_path, f'label-idx-{index[i]}.png')
    draw_plot(x_range,values, labels, path)
    
    labels = []
    values = []
    for idx, scores in base_result.items():
        if idx == 0:
            label = 'stem'
        elif idx == 1:
            label = 'option'
        else:
            label = f'cot_step-{idx-1}'
        values.append(scores.squeeze().numpy())
        labels.append(label)
    path = os.path.join(fold_path, f'pred-idx-{index[i]}.png')

    draw_plot(x_range,values, labels, path)
    

# import os
# import hydra
# import json
# import random
# import torch
# import wandb
# from omegaconf import DictConfig, OmegaConf
# from transformers import GPT2Tokenizer, BertTokenizer, AutoTokenizer, BloomTokenizerFast, GPTNeoXTokenizerFast, LlamaTokenizer
# from intervention_models.intervention_model import load_model
# from interventions.intervention import get_data


# @hydra.main(config_path='conf', config_name='config')
# def run_experiment(args: DictConfig):
#     print(OmegaConf.to_yaml(args))
#     print("Model:", args.model)

#     print('args.intervention_type', args.intervention_type)

#     if 'llama_models_hf/7B' in args.model:
#         model_str = 'llama7B'
#     elif 'llama_models_hf/13B' in args.model:
#         model_str = 'llama13B'
#     elif 'llama_models_hf/30B' in args.model:
#         model_str = 'llama30B'
#     elif 'alpaca' in args.model:
#         model_str = 'alpaca'
#     else:
#         model_str = args.model

#     # initialize logging
#     log_directory = args.output_dir
#     log_directory += f'/{model_str}'
#     if args.model_ckpt:
#         ckpt_name = '_'.join(args.model_ckpt.split('/')[5:9])
#         log_directory += f'_from_ckpt_{ckpt_name}'
#     log_directory += f'/n_operands{args.n_operands}'
#     log_directory += f'/template_type{args.template_type}'
#     log_directory += f'/max_n{args.max_n}'
#     log_directory += f'/n_shots{args.n_shots}'
#     log_directory += f'/examples_n{args.examples_per_template}'
#     log_directory += f'/seed{args.seed}'
#     print(f'log_directory: {log_directory}')
#     os.makedirs(log_directory, exist_ok=True)
#     wandb_name = ('random-' if args.random_weights else '')
#     wandb_name += f'{model_str}'
#     wandb_name += f' -p {args.template_type}'
#     wandb.init(project='mathCMA', name=wandb_name, notes='', dir=log_directory,
#                settings=wandb.Settings(start_method='fork'), mode=args.wandb_mode)
#     args_to_log = dict(args)
#     args_to_log['out_dir'] = log_directory
#     print("\n" + json.dumps(str(args_to_log), indent=4) + "\n")
#     wandb.config.update(args_to_log)
#     del args_to_log

#     random.seed(args.seed)
#     torch.manual_seed(args.seed)

#     # Initialize Model and Tokenizer
#     model = load_model(args)
#     tokenizer_class = (GPT2Tokenizer if model.is_gpt2 or model.is_gptneo or model.is_opt else
#                        BertTokenizer if model.is_bert else
#                        AutoTokenizer if model.is_gptj or model.is_flan or model.is_pythia else
#                        BloomTokenizerFast if model.is_bloom else
#                        GPTNeoXTokenizerFast if model.is_neox else
#                        LlamaTokenizer if model.is_llama else
#                        None)
#     if not tokenizer_class:
#         raise Exception(f'Tokenizer for model {args.model} not found')

#     if 'goat' in args.model:
#         tokenizer_id = 'decapoda-research/llama-7b-hf'
#     else:
#         tokenizer_id = args.model

#     tokenizer = tokenizer_class.from_pretrained(tokenizer_id, cache_dir=args.transformers_cache_dir)
#     model.create_vocab_subset(tokenizer, args)

#     intervention_list = get_data(tokenizer, args)

#     if args.debug_run:
#         intervention_list = intervention_list[:2]

#     print('================== INTERVENTIONS ==================')
#     for intervention in intervention_list[:5]:
#         print(f'BASE: {intervention.base_string} {intervention.res_base_string}')
#         print(f'ALT: {intervention.alt_string} {intervention.res_alt_string}')

#     if args.intervention_loc.startswith('attention_'):
#         attention_int_loc = '_'.join(args.intervention_loc.split('_')[1:])
#         results = model.attention_experiment(interventions=intervention_list,
#                                              effect_type=args.effect_type,
#                                              intervention_loc=attention_int_loc,
#                                              get_full_distribution=args.get_full_distribution,
#                                              all_tokens=args.all_tokens)
#     else:
#         results = model.intervention_experiment(interventions=intervention_list,
#                                                 effect_type=args.effect_type,
#                                                 intervention_loc=args.intervention_loc,
#                                                 get_full_distribution=args.get_full_distribution,
#                                                 all_tokens=args.all_tokens)

#     df_results = model.process_intervention_results(intervention_list, model.word_subset, results, args)

#     random_w = 'random_' if args.random_weights else ''
#     f_name: str = f'{random_w}intervention_{args.intervention_type}'
#     f_name += f'_{args.representation}'
#     f_name += f'_{args.effect_type}'
#     f_name += f'_{args.intervention_loc}'
#     f_name += '_all_tokens' if args.all_tokens else ''
#     f_name += '_int8' if args.int8 else ''
#     out_path = os.path.join(log_directory, f_name + ".feather")
#     print('out_path: ', out_path)
#     df_results.to_feather(out_path)


# if __name__ == "__main__":
#     run_experiment()


