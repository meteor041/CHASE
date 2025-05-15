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
# ---------- 可调参数 ----------
@dataclass
class Config:
    model_name: str = r"/home/yangliu26/qwen3-8b"
    prompt_file: str = r"/home/yangliu26/CHASE/template/QP_prompt.txt"
    input_json: str = r"/home/yangliu26/data/train/schema_linking_result.json"
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
    num_gpus: int = 4


CFG = Config()
# --------------------------------


def load_prompt_template(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def format_prompt(template: str, db_id: str, question: str,
                  evidence: str, schema_linking: Dict[str, Any]) -> str:
    """
    使用 str.format_map 直接填充占位符，保持模板可读性。
    模板里写 {db_id}、{question}、{evidence}、{schema_linking}
    """
    return template.format_map({
        "db_id": db_id,
        "question": question,
        "evidence": evidence,
        "schema_linking": json.dumps(schema_linking, ensure_ascii=False),
    })


def load_model_and_tokenizer(cfg: Config):
    if cfg.load_in_8bit:
        quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
        torch_dtype = torch.float16
    else:
        quant_cfg = None
        torch_dtype = torch.float16 if cfg.use_fp16 else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        trust_remote_code=True,
        padding_side="left",  # 对于 text-generation 更稳妥
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        quantization_config=quant_cfg,
        device_map=cfg.device_map,
        max_memory=cfg.max_memory,
        local_files_only=True,
    )
    # if hasattr(model.generation_config, "enable_thinking"):
    model.generation_config.enable_thinking = True
    model.config.attn_implementation = "flash_attention_2"
    return tokenizer, model

def llm_call(model, tokenizer, prompts: List[str],
             max_new_tokens_num=128,
             do_sample: bool = True, 
             temperature: float = 0.6,
             top_p: float = 0.95,
             top_k: int = 20)-> str:
    messages = [
        {
            "role": "user", "content": prompt
        }
        for prompt in prompts
    ]
    kwargs = dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
    # if "enable_thinking" in tokenizer.apply_chat_template.__code__.co_varnames:
    #     kwargs["enable_thinking"] = False
    text_batch = [tokenizer.apply_chat_template([m], **kwargs) for m in messages]
    model_inputs = tokenizer(
        text_batch, 
        padding=True, 
        truncation=True,         # ✅ 若超过模型 max length 则截断
        return_tensors="pt"
    ).to(model.device)
    
    # conduct text completion
    outputs = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens_num,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k
    )
    # 解码每个生成结果（去掉prompt部分）
    # decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    # prompts_decoded = tokenizer.batch_decode(model_inputs["input_ids"], skip_special_tokens=True)
    # completions = [full[len(prompt):].strip() for full, prompt in zip(decoded, prompts_decoded)]

    # return completions
    output_ids = [
        output[len(input_ids):].tolist()
        for output, input_ids in zip(outputs, model_inputs.input_ids)
    ]
    
    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        indexes = [
            len(output_id) - output_ids[::-1].index(151668) 
            for output_id in output_ids
        ]
    except ValueError:
        indexes = [0 for _ in output_ids]
    
    contents = [
        tokenizer.decode(output_id[index:], skip_special_tokens=True).strip("\n") 
        for index, output_id in zip(indexes, output_ids)
    ]

    return contents 

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
def run_worker(gpu_id: int, data_chunk: List[dict], prompt_tpl: str):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    logging.info(f"[GPU {gpu_id}] 加载模型...")
    tokenizer, model = load_model_and_tokenizer(CFG)
    out_path = Path(CFG.output_dir) / f"qp_result_gpu{gpu_id}.json"
    out_path.write_text("")
    results = []
    batched_data = list(batched(data_chunk, CFG.batch_size))
    for i, batch in enumerate(tqdm(batched_data, desc=f"[GPU {gpu_id}] Running")):
        prompts = [
            format_prompt(
                prompt_tpl,
                item.get("db_id", ""),
                item.get("question", ""),
                item.get("evidence", ""),
                item.get("schema_linking", {}),
            )
            for item in batch
        ]
        outputs  = llm_call(
                model, tokenizer, prompts, 
                512, CFG.do_sample, CFG.temperature, CFG.top_p, CFG.top_k
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

    # out_path = Path(CFG.output_dir) / f"qp_result_gpu{gpu_id}.json"
    # with open(out_path, "w", encoding="utf-8") as f:
    #     json.dump(results, f, ensure_ascii=False, indent=2)
    logging.info(f"[GPU {gpu_id}] 结果已保存至 {out_path}")

# ========== 主逻辑 ==========
def run_multi_gpu():
    os.makedirs(CFG.output_dir, exist_ok=True)
    with open(CFG.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    data = data[:1]
    prompt_tpl = load_prompt_template(CFG.prompt_file)
    chunks = split_data(data, CFG.num_gpus)

    for i, chunk in enumerate(chunks):
        logging.info(f"分配给 GPU-{i}: {len(chunk)} 条样本")
    import multiprocessing as mp
    # ctx = mp.get_context("spawn")
    processes = []
    for gpu_id in range(CFG.num_gpus):
        p = mp.Process(target=run_worker, args=(gpu_id, chunks[gpu_id], prompt_tpl))
        p.start()
        processes.append(p)
        if gpu_id != CFG.num_gpus - 1:
            time.sleep(10)  # ✅ 每个进程启动之间隔 10 秒，避免瞬间爆显存

    for p in processes:
        p.join()
    merge_results(CFG.output_dir, CFG.num_gpus, "qp_result_merged.json")
    logging.info("[Main] 所有 GPU 子进程执行完毕")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
    run_multi_gpu()
