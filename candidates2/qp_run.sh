python run_nl2sql.py \
  --template_path /home/yangliu26/CHASE/template/template_generate_candidate_two.txt \
  --input_json /home/yangliu26/data/schema_linking/dev_schema_linking_result.json \
  --output_dir /home/yangliu26/CHASE/candidates2/result/qp_result \
  --model_name /data/XiYanSQL-QwenCoder-32B-2412 \
  --mschema_path /home/yangliu26/data/mschema/dev_mschemas.json \
  --batch_size 32