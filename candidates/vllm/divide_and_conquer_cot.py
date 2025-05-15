#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Divide-and-Conquer CoT 方法实现
- 将复杂自然语言问题分解为多个子问题
- 对每个子问题分别生成SQL片段
- 最后合并这些片段，构造完整的SQL
"""
import concurrent.futures
import json
import os
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import time
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
import openai
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",  # 对于某些本地服务，API密钥可以是任意非空字符串
    base_url="http://localhost:8000/v1" # 指向你的 vLLM 服务
)

# ---------- 可调参数 ----------
@dataclass
class Config:
    model_name: str = r"/home/yangliu26/qwen3-8b"  # 请根据实际模型路径调整
    input_json: str = r"/home/yangliu26/data/train/schema_linking_result.json"
    output_dir: str = r"/home/yangliu26/CHASE/candidates/vllm/cot_result"
    # 文本生成超参
    max_new_tokens: int = 1024
    do_sample: bool = True
    temperature: float = 0.7
    enable_thinking: bool = True
    # 性能设置
    num_gpus: int = 1
    batch_size: int = 8
    use_fp16: bool = True
    device_map = {"" : 0}

# 配置实例
CFG = Config()

def batched(iterable: List[Any], n: int):
    """将列表分批切片"""
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]

def load_prompt_template(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()

PARTIAL_SQL_TEMPLATE = load_prompt_template(r"/home/yangliu26/CHASE/template/partial_sql_template.txt")
ASSEMBLE_TEMPLATE = load_prompt_template(r"/home/yangliu26/CHASE/template/assemble_template.txt")
DECOMPOSE_TEMPLATE = load_prompt_template(r"/home/yangliu26/CHASE/template/decompose_template.txt")

def extract_sql_block(generated_text: str) -> str:
    """从模型输出中提取 ```sql ... ``` 中间内容"""
    pattern = r"```sql\s+(.*?)\s*```"
    match = re.search(pattern, generated_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return generated_text.strip()  # fallback
    
def call_single_prompt(prompt: str, max_tokens: int = 128) -> Tuple[str, str]:
    try:
        response = client.chat.completions.create(
            model=CFG.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.7,
        )
        text = response["choices"][0]["message"]["content"].strip()
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
    max_new_tokens_num=128,
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
    
def decompose_question(batch, prompt_list, init_pos: int):
    # 分解步骤的提示词模板
    """将问题分解为多个子问题"""
    prompts = [DECOMPOSE_TEMPLATE.format(
        question=item.get("question"),
        db_schema=item.get("schema_linking")
        ) 
               for item in batch]
    
    thinking_texts, texts = llm_batch_call_vllm(prompts, None)
    # 解析子问题
    sub_questions = []
    for text in texts:
        sub_questions.append([])
        for line in text.strip().split('\n'):
            if line.strip() and any(line.strip().startswith(str(i)) for i in range(1, 10)):
                # 移除序号前缀
                sub_q = line.strip()
                for i in range(1, 10):
                    prefix = f"{i}. "
                    if sub_q.startswith(prefix):
                        sub_q = sub_q[len(prefix):]
                        break
                sub_questions[-1].append(sub_q)
    for i, idx in enumerate(range(init_pos, init_pos + len(batch))):
        prompt_list[idx].append(prompts[i])
    return sub_questions

def _format_history(history: List[Tuple[str, str]]) -> str:
    if not history or len(history) == 0:
        return ""
    chunks = []
    for i, (q, sql) in enumerate(history, 1):
        chunks.append(f"{i}. prev‑subQ: {q}\n   SQL: {sql}")
    return "Former Questions and SQL:\n" + "\n".join(chunks)
    
def generate_partial_sql(batch, sub_questions_list, prompt_list, init_pos) -> str:
    """为子问题生成SQL片段"""
    max_len = max(len(sub_questions) for sub_questions in sub_questions_list)
    partial_sqls = [[] for _ in range(len(batch))]
    for i in range(max_len):
        histories = [_format_history(list(zip(sub_questions[:i], partial_sql))) 
                     for sub_questions, partial_sql in zip(sub_questions_list, partial_sqls)]
        current_sub_questions = [sub_questions[i] if i < len(sub_questions) else "" for sub_questions in sub_questions_list]
        prompts = [
            PARTIAL_SQL_TEMPLATE.format(
                sub_question=sub_question,
                evidence=item.get("evidence"),
                history=history,
                db_schema=item.get("schema_linking")
            ) for sub_question, item, history in zip(current_sub_questions, batch, histories)]
        _, ret = llm_batch_call_vllm(prompts, None)
        for j, sql in enumerate(ret):
            if i < len(sub_questions_list[j]):
                partial_sqls[j].append(extract_sql_block(sql))
                prompt_list[init_pos+j].append(prompts[j])
    return partial_sqls

def assemble_sql(batch, sub_questions_list: List[List[str]], 
                partial_sqls_list: List[List[str]], 
                prompt_list, init_pos: int) -> str:
    """组合SQL片段为完整SQL"""
    # 格式化子问题和SQL片段
    sub_qs_and_sqls = []
    for j, (sub_questions, partial_sqls) in enumerate(zip(sub_questions_list, partial_sqls_list)):
        sub_qs_and_sqls.append("")
        for i, (q, sql) in enumerate(zip(sub_questions, partial_sqls), 1):
            sub_qs_and_sqls[j] += f"{i}. sub question: {q}\n   SQL: {sql}\n\n"
    
    prompts = [ASSEMBLE_TEMPLATE.format(
            question=item.get("question"),
            db_schema=item.get("schema_linking"),
            sub_questions_and_sqls=sub_qs_and_sql
        ) for item, sub_qs_and_sql in zip(batch, sub_qs_and_sqls)]
    
    for i, k in enumerate(range(init_pos, init_pos + len(batch))):
        prompt_list[k].append(prompts[i])
    _, rets = llm_batch_call_vllm(prompts, None)
    return [extract_sql_block(ret) for ret in rets]
    
def divide_and_conquer_sql(batch, prompt_list, init_pos: int):
    """主函数：使用分而治之方法生成SQL"""    
    # 步骤1: 分解问题
    sub_questions_list = decompose_question(batch, prompt_list, init_pos)
    # 步骤2: 生成每个子问题的SQL片段
    partial_sqls = generate_partial_sql(batch, sub_questions_list, prompt_list, init_pos)
    # 步骤3: 汇总构造最终SQL
    final_sqls = assemble_sql(batch, sub_questions_list, partial_sqls, prompt_list, init_pos)

    return sub_questions_list, partial_sqls, final_sqls

def process_item(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    batched_data = list(batched(data, CFG.batch_size))
    prompt_list = [[] for _ in range(len(data))]
    results = []
    out_path = Path(CFG.output_dir) / f"cot_result.jsonl"
    out_path.write_text("")
    for i, batch in enumerate(tqdm(batched_data, desc=f" Running")):
        idx_offset = i * CFG.batch_size
        sub_questions_list, partial_sqls_list, final_sqls = divide_and_conquer_sql(
            batch, prompt_list, idx_offset
        )
        for j in range(len(batch)):
            item = batch[j]
            results.append({
                "db_id": item["db_id"],
                "question": item["question"],
                "db_schema": item["schema_linking"],
                "sql": final_sqls[j],
                "subquestion": sub_questions_list[j],
                "partial_sqls": partial_sqls_list[j],
                "prompts": prompt_list[idx_offset + j]
            })
        if (i + 1) % 10 == 0 or (i + 1) == len(batched_data):
            with open(out_path, "a", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            results = []  # 清空已写入部分缓存

def process_data_parallel():
    os.makedirs(CFG.output_dir, exist_ok=True)
    
    with open(CFG.input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    data = data[:10]
    process_item(data)
        
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, 
        format="[%(asctime)s] %(levelname)s - %(message)s"
    )
    process_data_parallel()