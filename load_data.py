import json
import random
import re

from config import *

random.seed(17)

class DataLoader():
    def __init__(self, dataset,  data_length, split='dev', shuffle=True) -> None:
        self.__question_stem_ls = []
        self.__label_ls = []
        self.__answer_ls = []
        self.__option_ls = []
        self.idx = 0
        self.__len = data_length
        self.__load_data(dataset, split=split)
        if self.__len > len(self.__label_ls):
            self.__len = len(self.__label_ls)
        if shuffle and dataset != 'hella':
            self.__shuffle_data()
    
    def __load_labels(self, label_path):
        labels = []
        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f:
                labels.append(line[:-1])
        return labels
    
    
    def __load_data(self, dataset, split):
        if dataset == 'csqa':
            if split == 'train':
                datapath = csqa_train_data_path
            else:
                datapath = csqa_dev_data_path
        elif dataset == 'wino':
            if split == 'train':
                datapath = wino_train_data_path
            else:
                datapath = wino_dev_data_path
        elif dataset == 'hella':
            if split == 'train':
                datapath = hella_train_data_path
            else:
                datapath = hella_dev_data_path
        elif dataset == 'siqa':
            if split == 'train':
                datapath = siqa_train_data_path
                labelpath = siqa_train_label_path
            else:
                datapath = siqa_dev_data_path
                labelpath = siqa_dev_label_path
        elif dataset == 'gsm8k':
            if split == 'train':
                datapath = gsm8k_train_data_path
            else:
                datapath = gsm8k_dev_data_path
        elif dataset == 'strategy':
            datapath = strategy_data_path
        
        with open(datapath, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                label = ""
                answer = ""
                options = []
                if dataset == 'csqa':
                    question = data['question']['stem']
                    label = str(ord(data['answerKey']) - ord("A") + 1)
                    options = []
                    for i in range(len(data['question']['choices'])):
                        tup = data['question']['choices'][i]
                        options.append(tup['text'])
                elif dataset == 'strategy':
                    question = data['question']
                    label = data['answer']
                    if label == "true":
                        label = '1'
                    else:
                        label = '2' 
                    options = ['yes', 'no']
                elif dataset == 'wino':
                    question = data['sentence']
                    label = data['answer']
                    options = [data['option1'], data['option2']]
                elif dataset == 'hella':
                    question = data['ctx']
                    label = str(data['label'] + 1)
                    options = []
                    for i in range(len(data['endings'])):
                        options.append(data['endings'][i])
                elif dataset == 'siqa':
                    context = data['context']
                    question = data['question']
                    question = context + " " + question
                    options = [data['answerA'], data['answerB'], data['answerC']]
                elif dataset == 'gsm8k':
                    question = data['question']
                    answer = data['answer']
                    answer = re.sub(r'<<(.*?)>>', '', answer)
                    answer, label = answer.split('\n#### ')  
                    label = eval(label)             
                self.__question_stem_ls.append(question)
                self.__option_ls.append(options)
                self.__answer_ls.append(answer)
                if label:
                    self.__label_ls.append(label)
            if dataset == 'siqa':
                self.__label_ls = self.__load_labels(labelpath)


    def __shuffle_data(self):
        tuples = list(zip(self.__question_stem_ls, self.__option_ls, self.__label_ls))
        random.shuffle(tuples)
        self.__question_stem_ls, self.__option_ls, self.__label_ls = zip(*tuples)
        return 
    
    
    def __get_next_question_stem(self):
        return self.__question_stem_ls[self.idx]
      
        
    def __get_next_options(self):
        return self.__option_ls[self.idx]
    
    def __get_next_answer(self):
        return self.__answer_ls[self.idx]
    
    
    def __get_next_option_string(self):
        options = self.__get_next_options()
        option_string = ""
        for i in range(len(options)):
            option = options[i]
            option_string += f"({i+1}) " + option + " "
        return option_string
    
    
    def __get_next_question(self):
        question_stem = self.__get_next_question_stem()
        question = question_stem + '\n' + self.__get_next_option_string()
        return question 
        
    
    def __get_next_label(self):
        return self.__label_ls[self.idx]
    
    def __iter__(self):
        return self

    def __next__(self):
        if self.idx < self.__len:
            stem = self.__get_next_question_stem()
            question = self.__get_next_question()
            option = self.__get_next_options()
            label = self.__get_next_label()
            answer = self.__get_next_answer()
            self.idx += 1
            return {'stem':stem, 'question':question, 'option':option, 'label':label, 'answer':answer}
        else:
            raise StopIteration()
    
    def __len__(self):
        return self.__len


class CoTLoader():
    def __init__(self) -> None:
        return 
    
    def __collect_tf_data(self, dataloader):
        cor_data = []
        wr_data = []
        for data in dataloader:
            if data['pred'] == 'None':
                continue
            if data['cor_flag']:
                cor_data.append(data)
            else:
                wr_data.append(data)
        return cor_data, wr_data
    
    
    def split_cot(self, cot):
        steps = cot.split('.')[:-1]
        cots = []
        for step in steps:
            if 'answer is' in step:
                continue
            cots.append(step)
        return cots
    
    def load_data(self, cot_file, base_file, mode=None, cnt=None, index=None):
        with open(cot_file, 'r') as f:
            full_cot_data = json.load(f)[:-1]
        cot_cor_data, cot_wr_data = self.__collect_tf_data(full_cot_data)
        with open(base_file, 'r') as f:
            full_base_data = json.load(f)[:-1]
        base_cor_data, base_wr_data = self.__collect_tf_data(full_base_data)
        if not index:
            if mode == 'W2C':
                base_data = base_wr_data
                cot_data = cot_cor_data
            elif mode  == 'C2W':
                base_data = base_cor_data
                cot_data = cot_wr_data  
            elif mode  == 'W2W':
                base_data = base_wr_data
                cot_data = cot_wr_data  
            elif mode  == 'C2C':
                base_data = base_cor_data
                cot_data = cot_cor_data
            else:
                return  
            index = []
            question_set = []
            for data in base_data:
                question_set.append(data['question'])
            for data in cot_data:
                if data['question'] in question_set:
                    index.append(full_cot_data.index(data))
            index = index[:cnt]
        data = []
        for idx in index:
            if idx >= len(full_cot_data):
                break
            question = full_cot_data[idx]['question']
            cot = full_cot_data[idx]['answer']
            cots = self.split_cot(cot)
            pred = full_cot_data[idx]['pred']
            label = full_cot_data[idx]['label']
            msg = {'question':question, 'answer':cot, 'steps':cots, 'pred':pred, 'label':label}
            data.append(msg)       
        return data, index
    
    
class InterventionData():
    def __init__(self, msg, tokenizer, prompter) -> None:
        self.question = None 
        self.cot = None 
        self.pred = None 
        self.load_data(msg)
        
        self.prompt_end = None 
        self.question_end = None
        self.cot_end = None  
        self.cot_input_ids = None 
        self.pred_ids = None
        self.tokenize_data(tokenizer, prompter)
        
        self.cot_intervention_idx = {}
        self.get_intervention_idx(tokenizer)

        return 
    
    def load_data(self, msg):
        self.question = msg['question']
        self.cot = '.'.join(msg['steps']) + '.'
        self.pred = msg['pred']

        return 
    
    
    def tokenize_data(self, tokenizer, prompter):
        cot_question = prompter.wrap_input(self.question, icl_cnt=5)
        cot_input = cot_question + self.cot + f' So the answer is: ({self.pred})'
        
        question_len = len(prompter.user_prompt.format(cot_question))
        prompt = prompter.wrap_input(cot_question, icl_cnt=5)[:-question_len]
        self.cot_input_ids = tokenizer(cot_input, return_tensors="pt").input_ids
        self.pred_ids = self.cot_input_ids[:,-2]
        self.prompt_end = len(tokenizer(prompt, return_tensors="pt").input_ids[0]) - 1
        self.question_end = len(tokenizer(cot_question, return_tensors="pt").input_ids[0])
        self.cot_end = len(tokenizer(cot_question + self.cot, return_tensors="pt").input_ids[0])
        # print(tokenizer.convert_ids_to_tokens(self.pred_ids))
        
        return 
        
        
    def get_intervention_idx(self, tokenizer):
        tokens = tokenizer.convert_ids_to_tokens(self.cot_input_ids[0, self.prompt_end:self.question_end])
        # reg_tokens = tokenizer.convert_ids_to_tokens(self.reg_input_ids[0, self.prompt_end:self.reg_end+1])
        # self.cot_intervention_idx[0] = list(range(self.prompt_end))
        i = 0
        self.cot_intervention_idx[1] = []
        while tokens[i] != '<0x0A>':
            self.cot_intervention_idx[1].append(i+self.prompt_end)
            i += 1
        i += 1
        self.cot_intervention_idx[2] = []
        while tokens[i] != '▁[':
            self.cot_intervention_idx[2].append(i+self.prompt_end)
            i += 1
        self.cot_intervention_idx[3] = list(range(self.question_end, self.cot_end))
        self.cot_intervention_idx[4] = [len(self.cot_input_ids[0])-3]
        # print(tokenizer.convert_ids_to_tokens(self.cot_input_ids[:,self.cot_intervention_idx[4]]))
        # self.reg_intervention_idx.append(len(self.reg_input_ids[0])-1)
    
        return 

        