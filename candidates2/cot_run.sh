python /home/yangliu26/CHASE/candidates2/run_nl2sql.py \
  --template_path /home/yangliu26/CHASE/template/template_generate_candidate_one.txt \
  --fixer_template_path /home/yangliu26/CHASE/template/template_query_fixer.txt \
  --input_json /home/yangliu26/data/schema_linking/dev_schema_linking_result.json \
  --output_dir /home/yangliu26/CHASE/candidates2/2025_5_22/result/cot_result \
  --model_name /data/Qwen2.5-32B-Instruct \
  --mschema_path /home/yangliu26/data/mschema/dev_mschemas.json \
  --num_generations 3 \
  --batch_size 32