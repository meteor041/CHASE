#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Query Plan方法实现NL2SQL
"""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List
import re
import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
import os
import math
import logging
from multiprocessing import Process
import time
import openai
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",  # 对于某些本地服务，API密钥可以是任意非空字符串
    base_url="http://localhost:8000/v1" # 指向你的 vLLM 服务
)

# ---------- 可调参数 ----------
@dataclass
class Config:
    model_name: str = r"/home/yangliu26/qwen3-8b"
    prompt_file: str = r"/home/yangliu26/CHASE/template/QP_prompt.txt"
    input_json: str = r"/home/yangliu26/data/schema_linking/schema_linking_result.json"
    output_dir: str = "qp_results"
    # 文本生成超参
    max_new_tokens: int = 512
    do_sample: bool = False
    temperature: float = None
    top_p: float = None
    top_k: int = None
    # 性能设置
    batch_size: int = 4      # 根据显存灵活调整
    use_fp16: bool = True        # 或用 8bit/4bit 量化
    load_in_8bit: bool = False
    device_map= {"" : 0}
    max_memory={0: "30GiB"}     # V100-32G 留 2G 缓冲


CFG = Config()
# --------------------------------


def load_prompt_template(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()

def format_prompt(template: str, db_id: str, question: str,
                  evidence: str, DDL: str) -> str:
    """
    使用 str.format_map 直接填充占位符，保持模板可读性。
    模板里写 {db_id}、{question}、{evidence}、{schema_linking}
    """
    return template.format_map({
        "db_id": db_id,
        "question": question,
        "evidence": evidence,
        "schema_linking": DDL,
    })

def call_single_prompt(prompt: str, max_tokens: int = 128) -> Tuple[str, str]:
    try:
        response = client.chat.completions.create(
            model=CFG.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.7,
        )
        text = response.choices[0].message.content.strip()
    except Exception as e:
        text = f"[ERROR]: {e}"

    if "</think>" in text:
        think, content = text.split("</think>", 1)
        think = think.replace("<think>", "").strip()
        content = content.strip()
    else:
        think = ""
        content = text.strip()

    return think, content

def llm_batch_call_vllm(
    prompts: List[str],
    max_new_tokens_num=8192,
    do_sample: bool = True, 
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20
) -> Tuple[List[str], List[str]]:
    """并发提交 prompt 到 vLLM 的 OpenAI 接口"""
    thinking_results, content_results = [None] * len(prompts), [None] * len(prompts)

    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_index = {
            executor.submit(call_single_prompt, prompt, max_new_tokens_num): i
            for i, prompt in enumerate(prompts)
        }
        for future in as_completed(future_to_index):
            i = future_to_index[future]
            think, content = future.result()
            thinking_results[i] = think
            content_results[i] = content

    return thinking_results, content_results

# ========== 工具 ==========
def split_data(data: List[dict], num_chunks: int) -> List[List[dict]]:
    n = len(data)
    base = n // num_chunks
    remainder = n % num_chunks
    chunks = []
    start = 0
    for i in range(num_chunks):
        end = start + base + (1 if i < remainder else 0)
        chunks.append(data[start:end])
        start = end
    return chunks
    
def batched(iterable: List[Any], n: int):
    """将列表分批切片 (Python 3.12 里有 itertools.batched)"""
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]

def extract_sql_block(generated_text: str) -> str:
    """从模型输出中提取 ```sql ... ``` 中间内容"""
    pattern = r"```sql\s+(.*?)\s*```"
    match = re.search(pattern, generated_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return generated_text.strip()  # fallback

def merge_results(output_dir: str, num_gpus: int, merged_file: str):
    results = []
    for i in range(num_gpus):
        path = Path(output_dir) / f"qp_result_gpu{i}.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    results.append(json.loads(line))
    with open(Path(output_dir) / merged_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
# ========== 实际工作 ==========
def run_worker(data_chunk: List[dict], prompt_tpl: str):
    out_path = Path(CFG.output_dir) / f"qp_result.json"
    out_path.write_text("") # 清空
    results = []
    batched_data = list(batched(data_chunk, CFG.batch_size))
    for i, batch in enumerate(tqdm(batched_data, desc=f"Running")):
        prompts = [
            format_prompt(
                prompt_tpl,
                item.get("db_id", ""),
                item.get("question", ""),
                item.get("evidence", ""),
                item.get("DDL", ""),
            )
            for item in batch
        ]
        _, outputs  = llm_batch_call_vllm(
                prompts, 
                8192, CFG.do_sample, CFG.temperature, CFG.top_p, CFG.top_k
            )
        for item, gen_text in zip(batch, outputs):
            sql = extract_sql_block(gen_text)
            results.append({
                "db_id": item["db_id"],
                "question": item["question"],
                "sql": sql,
                "text": gen_text
            })
        if (i + 1) % 10 == 0 or (i + 1) == len(batched_data):
            with open(out_path, "a", encoding="utf-8") as f:
                for r in results:
                    logging.info(r)
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            results = []  # 清空已写入部分缓存
    logging.info(f"已完成")

# ========== 主逻辑 ==========
def run():
    os.makedirs(CFG.output_dir, exist_ok=True)
    with open(CFG.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    data = data[:1]
    prompt_tpl = load_prompt_template(CFG.prompt_file)
    run_worker(data, prompt_tpl)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
    run()
