python3 validate_multi.py --gold /home/yangliu26/data/train/train.json \
--baseline /home/yangliu26/data/candidates/sim_result_merged.json \
--preds /home/yangliu26/data/candidates/COT_results.json \
 /home/yangliu26/data/candidates/os_results.json \
 /home/yangliu26/data/candidates/qp_result_merged.json \
--names cot os qp \
--out_dir /home/yangliu26/CHASE/pairwise/result \
--single_timeout 300 \
--workers 32