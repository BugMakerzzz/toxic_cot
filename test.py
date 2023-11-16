import torch
import transformers
data = [
    {
        "question": "Needing a boost Emily asked Rachel to put them on their shoulders in order to reach the roof, _  is tall in height.\n(1) Emily (2) Rachel ",
        "answer": "  A person who is tall in height can reach the roof easily. Since Emily needed a boost to reach the roof, Emily is not tall in height. \nSo the answer is: (1) Emily.</s>",
        "label": "2"
    },
    {
        "question": "The computer of Victoria ran slower than that of Carrie because _ downloaded less files.\n(1) Victoria (2) Carrie ",
        "answer": "  A computer runs slowly when it has too many files. A person who downloads less files may have a faster computer. Since Victoria's computer ran slower than Carrie's, Victoria downloaded less files. \nSo the answer is: (1) Victoria.</s>",
        "label": "2"
    },
    {
        "question": "The TV that Samantha bought costs more than that of Carrie, because _ was poor.\n(1) Samantha (2) Carrie ",
        "answer": "  A person who is poor may not afford to buy an expensive thing. A person who can afford to buy an expensive thing may be rich. Since Samantha's TV costs more than Carrie's, Samantha may be rich. \nSo the answer is: (1) Samantha.</s>",
        "label": "2"
    },
    
    
]


def llm_score(self, input_text1, input_text2, pmi=False, max_new_tokens=256, top_p=0.95, top_k=0, temperature=0.8, mature_layer=None, premature_layer=None, candidate_premature_layers=[], mode='baseline', verbose=True, remove_stop_words=False, relative_top=0.1, relative_top_value=-1000.0, post_softmax=True, **kwargs):
    with torch.no_grad():
        input_text = input_text1 + input_text2
        input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids.to(self.device)
        prefix_ids = self.tokenizer(input_text1, return_tensors="pt").input_ids.to(self.device)
        continue_ids = input_ids[0, prefix_ids.shape[-1]:]
        if mode == 'baseline':
            outputs = self.model(input_ids)[0].squeeze(0)
            outputs = outputs.log_softmax(-1)  # logits to log probs

            # skip tokens in the prompt -- we only care about the answer
            outputs = outputs[prefix_ids.shape[-1] - 1: -1, :]

            # get logprobs for each token in the answer
            log_probs = outputs[range(outputs.shape[0]), continue_ids].sum().item()
            
        elif mode == 'dola-static':
            dict_outputs, outputs = self.model(
                input_ids=input_ids,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
                early_exit_layers=[premature_layer, mature_layer],
            )

            assert premature_layer is not None
            base_logits = dict_outputs[premature_layer][0, prefix_ids.shape[-1] - 1: -1, :]
            final_logits = dict_outputs[mature_layer][0, prefix_ids.shape[-1] - 1: -1, :]
            final_logits = final_logits.log_softmax(dim=-1)
            base_logits = base_logits.log_softmax(dim=-1)
            diff_logits = final_logits - base_logits
            if post_softmax:
                diff_logits = diff_logits.log_softmax(dim=-1)
            if relative_top > 0.0:
                relative_top_mask = self.get_relative_top_filter(final_logits, relative_top)
                diff_logits = torch.where(relative_top_mask, relative_top_value, diff_logits)
                
            log_probs = diff_logits[range(diff_logits.shape[0]), continue_ids].sum().item()