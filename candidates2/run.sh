python -m vllm.entrypoints.openai.api_server \
    --model /data/XiYanSQL-QwenCoder-32B-2412 \
    --tensor-parallel-size 4 \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype float16