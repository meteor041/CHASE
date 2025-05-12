# multi_gpu_os_runner.py
# 多 GPU 并行运行 Online Synthetic SQL 示例生成（适配 4 张 A100）

import os
import json
from pathlib import Path
from multiprocessing import Process
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from dataclasses import dataclass
from typing import List, Dict, Any
import torch
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

# ========== 配置 ==========
@dataclass
class Config:
    model_name: str = "/home/yangliu26/qwen3-8b"
    input_json: str = "/home/yangliu26/data/train/schema_linking_result.json"
    output_dir: str = "/home/yangliu26/CHASE/candidates/os_results"
    tables_json: str = "/home/yangliu26/CHASE/utils/converted_schema.json"
    use_fp16: bool = True
    num_general_examples: int = 3
    num_schema_aware_examples: int = 3
    max_new_tokens: int = 256
    batch_size: int = 16
    num_gpus: int = 3
    device_map = {"" : 0}
    max_memory={0: "30GiB"} 

CFG = Config()

# ========== 工具函数 ==========
def split_data(data: List[dict], num_chunks: int) -> List[List[dict]]:
    n = len(data)
    base = n // num_chunks          # 每块至少拥有的数据量
    remainder = n % num_chunks      # 需要再平均分配的余量

    chunks: List[List[dict]] = []
    start = 0
    for i in range(num_chunks):
        end = start + base + (1 if i < remainder else 0)
        chunks.append(data[start:end])
        start = end                 # 下一块从 end 开始
    return chunks

def batched(iterable: List[Any], n: int):
    """将列表分批切片 (Python 3.12 里有 itertools.batched)"""
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]
        
def load_model_and_tokenizer(cfg: Config):
    quant_cfg = None
    if not cfg.use_fp16:
        quant_cfg = BitsAndBytesConfig(load_in_4bit=True)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        trust_remote_code=True,
        padding_side="left"
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if cfg.use_fp16 else torch.float32,
        quantization_config=quant_cfg,
        device_map=cfg.device_map,
        max_memory=cfg.max_memory
    )
    model.generation_config.enable_thinking = False
    return tokenizer, model

def merge_results(output_dir: str, num_gpus: int, merged_file: str):
    results = []
    for i in range(num_gpus):
        path = Path(output_dir) / f"os_result_gpu{i}.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                results.extend(json.load(f))
        else:
            logging.critical(f"[GPU-{i}] {path} don't exist!")
    with open(Path(output_dir) / merged_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
def run_worker(gpu_id: int, data: List[dict], tables_info: Dict[str, str]):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    logging.info(f"[GPU {gpu_id}] 正在加载模型...")
    tokenizer, model = load_model_and_tokenizer(CFG)
    generator = [model, tokenizer]

    from online_synthetic import process_multiple_item  # 重用现有逻辑

    results = []
    data_chunk = list(batched(data, CFG.batch_size))
    for items in tqdm(data_chunk, desc=f"[GPU {gpu_id}] Running"):
        for item in items:
            item["db_whole_schema"] = tables_info.get(item.get("db_id"))
        result = process_multiple_item(items, generator)
        if result:
            results.extend(result)

    output_path = os.path.join(CFG.output_dir, f"os_result_gpu{gpu_id}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logging.info(f"[GPU {gpu_id}] 完成，结果保存至 {output_path}")

# ========== 主函数 ==========
def run_multi_gpu():
    os.makedirs(CFG.output_dir, exist_ok=True)
    with open(CFG.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    # data = data[:100]
    with open(CFG.tables_json, "r", encoding="utf-8") as f:
        tables_info = json.load(f)

    num_gpus = CFG.num_gpus
    chunks = split_data(data, num_gpus)
    processes = []

    for gpu_id in range(num_gpus):
        p = Process(target=run_worker, args=(gpu_id, chunks[gpu_id], tables_info))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    logging.info("所有 GPU 子进程执行完毕。")
    merge_results(CFG.output_dir, CFG.num_gpus, "os_result_merged.json")
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
    run_multi_gpu()
