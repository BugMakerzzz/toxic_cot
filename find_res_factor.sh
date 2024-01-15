CUDA_VISIBLE_DEVICES=4,5,6,7 python llm_cot_probe.py --dataset csqa --mode C2W --score attn_attr --cnt 25 --avg norm --reg
wait 
CUDA_VISIBLE_DEVICES=4,5,6,7 python llm_cot_probe.py --dataset csqa --mode C2W --score attn_attr --cnt 25 --avg norm --reg
wait
CUDA_VISIBLE_DEVICES=4,5,6,7 python llm_cot_probe.py --dataset csqa --mode C2W --score attn_attr --cnt 25 --avg norm --reg