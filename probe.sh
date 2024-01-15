CUDA_VISIBLE_DEVICES=4,5,6,7 python llm_cot_probe.py --dataset csqa --mode C2W --score attn_attr --cnt 30 --avg norm --reg
wait
CUDA_VISIBLE_DEVICES=4,5,6,7 python llm_cot_probe.py --dataset csqa --mode C2W --score mlp_attr --cnt 30 --avg norm --reg
wait 
CUDA_VISIBLE_DEVICES=4,5,6,7 python llm_cot_probe.py --dataset csqa --mode W2C --score attn_attr --cnt 100 --avg norm 
wait 
CUDA_VISIBLE_DEVICES=4,5,6,7 python llm_cot_probe.py --dataset csqa --mode W2C --score mlp_attr --cnt 100 --avg norm 
wait
CUDA_VISIBLE_DEVICES=4,5,6,7 python llm_cot_probe.py --dataset wino --mode C2W --score attn_attr --cnt 30 --avg norm --reg
wait
CUDA_VISIBLE_DEVICES=4,5,6,7 python llm_cot_probe.py --dataset wino --mode C2W --score mlp_attr --cnt 30 --avg norm --reg
wait 
CUDA_VISIBLE_DEVICES=4,5,6,7 python llm_cot_probe.py --dataset wino --mode W2C --score attn_attr --cnt 100 --avg norm 
wait 
CUDA_VISIBLE_DEVICES=4,5,6,7 python llm_cot_probe.py --dataset wino --mode W2C --score mlp_attr --cnt 100 --avg norm 