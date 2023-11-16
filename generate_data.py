import json

file1 = './result/wino/Llama-2-13b-chat-hf_cot_answer_dev_1000.json'
file2 = './result/wino/Llama-2-13b-chat-hf_direct_answer_dev_1000.json'

with open(file1, 'r') as f:
    data1 = json.load(f)
with open(file2, 'r') as f:
    data2 = json.load(f)

wrong_questions = set()
for item in data1:
    if 'question' not in item.keys():
        break
    question = item['question']
    cor_flag = item['cor_flag']
    if not cor_flag:
        wrong_questions.add(question)
for item in data2:
    if 'question' not in item.keys():
        break
    question = item['question']
    cor_flag = item['cor_flag']
    if not cor_flag and question in wrong_questions:
        wrong_questions.remove(question)
result = []
for item in data1:
    if 'question' not in item.keys():
        break
    question = item['question']
    answer = item['answer']
    label = item['label']
    if question in wrong_questions:
        result.append({'question':question, 'answer':answer, 'label':label})
with open('probe_data.json', 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=4)
    