import json
import random
import re
random.seed(17)


wino_train_data_path = './data/winogrande_1.1/train_l.jsonl'
wino_dev_data_path = './data/winogrande_1.1/dev.jsonl'

hella_train_data_path = './data/hellaswag/hellaswag_train.jsonl'
hella_dev_data_path = './data/hellaswag/hellaswag_val.jsonl'

siqa_train_data_path = './data/SocialIQA/train.jsonl'
siqa_train_label_path = './data/SocialIQA/train-labels.lst'
siqa_dev_data_path = './data/SocialIQA/dev.jsonl'
siqa_dev_label_path = './data/SocialIQA/dev-labels.lst'

gsm8k_train_data_path = './data/grade-school-math/grade_school_math/data/train.jsonl'
gsm8k_dev_data_path = './data/grade-school-math/grade_school_math/data/test.jsonl'


class DataLoader():
    def __init__(self, dataset,  data_length, split='dev', shuffle=True) -> None:
        self.__question_stem_ls = []
        self.__label_ls = []
        self.__answer_ls = []
        self.__option_ls = []
        self.__idx = 0
        self.__len = data_length
        self.__load_data(dataset, split=split)
        if shuffle:
            self.__shuffle_data()
    
    def __load_labels(self, label_path):
        labels = []
        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f:
                labels.append(line[:-1])
        return labels
    
    
    def __load_data(self, dataset, split):
        if dataset == 'wino':
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
        
        with open(datapath, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                label = ""
                answer = ""
                options = []
                if dataset == 'wino':
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
        return self.__question_stem_ls[self.__idx]
      
        
    def __get_next_options(self):
        return self.__option_ls[self.__idx]
    
    def __get_next_answer(self):
        return self.__answer_ls[self.__idx]
    
    
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
        return self.__label_ls[self.__idx]
    
    def __iter__(self):
        return self

    def __next__(self):
        if self.__idx < self.__len:
            stem = self.__get_next_question_stem()
            question = self.__get_next_question()
            option = self.__get_next_options()
            label = self.__get_next_label()
            answer = self.__get_next_answer()
            self.__idx += 1
            return {'stem':stem, 'question':question, 'option':option, 'label':label, 'answer':answer}
        else:
            raise StopIteration()
    
    def __len__(self):
        return self.__len
    