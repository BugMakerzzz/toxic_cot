import json

class Prompter():
    def __init__(self, dataset, task) -> None:
        self.dataset = dataset
        self.task = task
        self.name = None 
        pass
    
    def load_sys_instruction(self):
        sys_msg_path = f'./prompts/{self.task}/sys_instruction.json'
        with open(sys_msg_path, 'r') as f:
            sys_msg = json.load(f)
            sys_msg = sys_msg['instruction']
        sys_msg = self.wrap_msg(sys_msg, 'sys')
        return sys_msg
        
    def wrap_msg(self, msg, role):
        pass
    
    
    def load_examples(self, icl_cnt):
        if icl_cnt == 0:
            return ""
        else:
            example_path = f'./prompts/{self.task}/{self.dataset}.json'
            with open(example_path, 'r') as f:
                examples = json.load(f)
            if self.name in ['Baichuan','Mistral']:
                example_seq = []
            else:
                example_seq = ""
            cnt = 0
            for example in examples:
                user_msg = example['question']
                model_msg = example['answer']
                user_msg = self.wrap_msg(user_msg, 'user')
                model_msg = self.wrap_msg(model_msg, 'model')
                example_seq += user_msg + model_msg
                cnt += 1
                if cnt >= icl_cnt:
                    break
        return example_seq
                
        
    def wrap_input(self, msg, icl_cnt):
        sys_msg = self.load_sys_instruction()
        example_msg = self.load_examples(icl_cnt=icl_cnt)
        user_msg = self.wrap_msg(msg, 'user')
        if self.name == 'Mistral':
            example_msg[0]['content'] = sys_msg[0]['content'] + example_msg[0]['content']
            return example_msg + user_msg
        else:
            return sys_msg + example_msg + user_msg
                
                
class LlamaPrompter(Prompter):
    def __init__(self, dataset, task) -> None:
        super().__init__(dataset=dataset, task=task)
        self.sys_prompt = "<s>[INST] <<SYS>>\n{}\n<</SYS>>\n\n"
        self.user_prompt = "{} [/INST] "
        self.model_prompt = "{} </s><s>[INST] "
        self.name = 'Llama'
    
    def wrap_msg(self, msg, role):
        if role == 'sys':
            msg = self.sys_prompt.format(msg) 
        elif role == 'user':
            msg = self.user_prompt.format(msg)
        else:
            msg = self.model_prompt.format(msg)
        return msg

class VicunaPrompter(Prompter):
    def __init__(self, dataset, task) -> None:
        super().__init__(dataset=dataset, task=task)
        self.sys_prompt = "{}\n"
        self.user_prompt = "{} "
        self.model_prompt = "{}\n"
        self.name = ""
    
    def wrap_msg(self, msg, role):
        if role == 'sys':
            msg = self.sys_prompt.format(msg) 
        elif role == 'user':
            msg = self.user_prompt.format(msg)
        else:
            msg = self.model_prompt.format(msg)
        return msg
    

class GPTPrompter(Prompter):
    def __init__(self, dataset, task) -> None:
        super().__init__(dataset, task)
        self.name = 'Baichuan'
    
    def wrap_msg(self, msg, role):
        if role == 'sys':
            msg = [{"role":"system", "content": msg}]
        elif role == 'user':
            msg = [{"role":"user", "content": msg}]
        else:
            msg = [{"role":"assistant", "content": msg}]
        return msg

class MistralPrompter(Prompter):
    def __init__(self, dataset, task) -> None:
        super().__init__(dataset, task)
        self.name = 'Mistral'
    
    def wrap_msg(self, msg, role):
        if role == 'sys':
            msg = [{"role":"user", "content": msg}]
        elif role == 'user':
            msg = [{"role":"user", "content": msg}]
        else:
            msg = [{"role":"assistant", "content": msg}]
        return msg