python3 validate_multi.py --gold /home/yangliu26/data/train/train.json \
--baseline /home/yangliu26/data/candidates_vllm/sim_result.json \
--preds /home/yangliu26/data/candidates_vllm/cot_result.json \
 /home/yangliu26/data/candidates_vllm/os_results.json \
 /home/yangliu26/data/candidates_vllm/qp_result.json \
--names cot os qp \
--out_dir /home/yangliu26/CHASE/pairwise/result_vllm \
--single_timeout 300 \
--workers 32