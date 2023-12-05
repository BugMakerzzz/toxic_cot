import json

class Prompter():
    def __init__(self, dataset, task) -> None:
        self.dataset = dataset
        self.task = task
        self.sys_prompt = "{}"
        self.user_prompt = "{}"
        self.model_prompt = "{}"
        pass
    
    def load_sys_instruction(self):
        sys_msg_path = f'./prompts/{self.task}/sys_instruction.json'
        with open(sys_msg_path, 'r') as f:
            sys_msg = json.load(f)
            sys_msg = sys_msg['instruction']
        sys_msg = self.sys_prompt.format(sys_msg) 
        return sys_msg
        
    
    def load_examples(self, icl_cnt):
        if icl_cnt == 0:
            return ""
        else:
            example_path = f'./prompts/{self.task}/{self.dataset}.json'
            with open(example_path, 'r') as f:
                examples = json.load(f)
            example_str = ""
            cnt = 0
            for example in examples:
                user_msg = example['question']
                model_msg = example['answer']
                user_msg = self.user_prompt.format(user_msg)
                model_msg = self.model_prompt.format(model_msg)
                example_str += user_msg + model_msg
                cnt += 1
                if cnt >= icl_cnt:
                    break
        return example_str
                
        
    def wrap_input(self, msg, icl_cnt):
        sys_msg = self.load_sys_instruction()
        example_msg = self.load_examples(icl_cnt=icl_cnt)
        if not example_msg and self.task == 'cot_answer':
            msg += " Let's think step by step: "
        user_msg = self.user_prompt.format(msg)
        
        return sys_msg + example_msg + user_msg
                
                
class LlamaPrompter(Prompter):
    def __init__(self, dataset, task) -> None:
        super().__init__(dataset=dataset, task=task)
        self.sys_prompt = "<s>[INST] <<SYS>>\n{}\n<</SYS>>\n\n"
        self.user_prompt = "{} [/INST] "
        self.model_prompt = "{} </s><s>[INST] "
    

class GPTPrompter(Prompter):
    def __init__(self, dataset, task) -> None:
        super().__init__(dataset, task)