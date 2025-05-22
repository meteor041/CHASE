python -m vllm.entrypoints.openai.api_server \
    --model /data/Qwen2.5-32B-Instruct \
    --tensor-parallel-size 4 \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype float16