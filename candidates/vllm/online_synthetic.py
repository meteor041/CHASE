#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Online Synthetic (OS) 方法实现
- 根据当前问题动态生成多个输入-输出样例
- 作为Prompt输入来指导SQL生成
"""
import multiprocessing
import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import torch
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig,
)
import re
import logging
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
import openai
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",  # 对于某些本地服务，API密钥可以是任意非空字符串
    base_url="http://localhost:8000/v1" # 指向你的 vLLM 服务
)

logging.basicConfig(level=logging.INFO)
# ---------- 可调参数 ----------
@dataclass
class Config:
    model_name: str = r"/home/yangliu26/qwen3-8b"  # 请根据实际模型路径调整
    input_json: str = r"/home/yangliu26/data/schema_linking/schema_linking_result.json"
    output_dir: str =  r"/home/yangliu26/CHASE/candidates/os_results"
    tables_json: str = r"/home/yangliu26/CHASE/utils/converted_schema.json"
    # 文本生成超参
    max_new_tokens: int = 256
    do_sample: bool = True
    temperature: float = 0.7
    # 性能设置
    batch_size: int = 4
    use_fp16: bool = True
    device_map: str = "auto"
    # OS特定参数
    num_general_examples: int = 3
    num_schema_aware_examples: int = 3

# 配置实例
CFG = Config()

with open(CFG.tables_json, "r") as f:
    data = f.read()
    tables_info = json.loads(data)
    
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

def batched(iterable: List[Any], n: int):
    """将列表分批切片"""
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]

def load_prompt_template(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()

# GENERAL_EXAMPLES_TEMPLATE是生成通用示例的提示模板
GENERAL_EXAMPLES_TEMPLATE = load_prompt_template("/home/yangliu26/CHASE/template/general_examples_template.txt")
SCHEMA_AWARE_TEMPLATE = load_prompt_template("/home/yangliu26/CHASE/template/schema_aware_template.txt")
FEW_SHOT_TEMPLATE = load_prompt_template("/home/yangliu26/CHASE/template/os_few_shot_template.txt")

def generate_examples_by_sql_features(items, num_examples: int) -> List[Tuple[str, str]]:
    """生成涵盖不同SQL特性的示例"""
    prompts = [
        GENERAL_EXAMPLES_TEMPLATE.format(
            db_schema=tables_info.get(item.get("db_id")),
            num_examples=num_examples
        ) 
        for item in items
    ]
    _, texts = llm_batch_call_vllm(prompts, 8192)
    # 解析示例
    examples_list = [[] for _ in range(len(items))]
    
    for j, text in enumerate(texts):
        lines = text.split('\n')
        i = 0
        while i < len(lines):
            if lines[i].startswith("Question:"):
                question = lines[i][9:].strip()
                i += 1
                # 寻找SQL行
                while i < len(lines) and not lines[i].startswith("SQL:"):
                    i += 1
                if i < len(lines):
                    sql = lines[i][4:].strip()
                    examples_list[j].append((question, sql))
            i += 1
    
    return prompts, texts, [examples[:num_examples] for examples in examples_list]  # 确保不超过请求的数量

def generate_examples_by_schema(items, num_examples: int) -> List[Tuple[str, str]]:
    """生成基于特定schema的示例"""
    prompts = [
        SCHEMA_AWARE_TEMPLATE.format(
            db_schema=item.get("schema_linking"),
            num_examples=num_examples
        )
        for item in items
    ]
    _, texts = llm_batch_call_vllm(prompts, 8192)
    # 解析示例
    examples_list = [[] for _ in range(len(items))]
    for j, text in enumerate(texts):
        lines = text.split('\n')
        i = 0
        while i < len(lines):
            if lines[i].startswith("Question:"):
                question = lines[i][9:].strip()
                i += 1
                # 寻找SQL行
                while i < len(lines) and not lines[i].startswith("SQL:"):
                    i += 1
                if i < len(lines):
                    sql = lines[i][4:].strip()
                    examples_list[j].append((question, sql))
            i += 1
    
    return prompts, texts, [examples[:num_examples] for examples in examples_list]  # 确保不超过请求的数量

def format_few_shot_prompt(examples_list: List[List[Tuple[str, str]]], items) -> str:
    """格式化few-shot提示"""
    examples_text_list = ["" for _ in range(len(items))]
    for j, examples in enumerate(examples_list):
        for i, (q, sql) in enumerate(examples, 1):
            examples_text_list[j] += f"Example {i}:\nQuestion: {q}\nSQL: {sql}\n"
    
    return [
        FEW_SHOT_TEMPLATE.format(
            db_schema=item.get("schema_linking"),
            examples=examples_text,
            question=item.get("question")
        )
        for item, examples_text in zip(items, examples_text_list)
    ]

def extract_sql(text: str) -> str:
    logging.info("text: %s", text)
    """从生成的文本中提取SQL"""
    pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    
    if not match:
        print("no match")
        return text                 # 没找到代码块

    json_str = match.group(1).strip()
    print("json_str: ", json_str)
    try:
        obj = json.loads(json_str)  # ❷ 解析 JSON
    except json.JSONDecodeError as e:
        print(f"[JSON 解析失败] {e}")
        print(f"原语句 {json_str}")
        return json_str
    except Exception as e:
        print(f"[其他失败] {e}")
        print(f"原语句 {json_str}")
    print("obj: ", obj)
    # 允许大小写差异
    try:
        sql = obj.get("sql") or obj.get("SQL")
    except Exception as e:
        print("JSON 中未找到 'sql'或者'SQL` 字段", obj)
    if not sql:
        print("JSON 中未找到 'sql'或者'SQL` 字段")
        return obj
    return sql.strip()

def online_synthetic_icl(items,  
                         num_general: int = 3, 
                         num_schema_aware: int = 3):
    """主函数：使用在线合成示例方法生成SQL"""
    logging.info("步骤1: 使用常见SQL特征生成通用示例")
    prompt1, raw_output1, general_examples = generate_examples_by_sql_features(items, num_general)
    logging.info("完成步骤1")
    logging.info("步骤2: 生成schema-aware示例")
    prompt2, raw_output2, schema_examples = generate_examples_by_schema(items, num_schema_aware)
    logging.info("完成步骤2")
    logging.info("步骤3: 组合所有示例 + 当前问题进入Prompt")
    example_combined = [
        general_example + schema_example 
        for general_example, schema_example in zip(general_examples, schema_examples)
    ]
    prompts = format_few_shot_prompt(example_combined, items)
    logging.info("完成步骤3")
    logging.info("步骤4: 生成SQL")
    _, responses = llm_batch_call_vllm(prompts, 8192)
    logging.info("完成SQL生成")
    return prompt1, raw_output1, general_examples, prompt2, raw_output2, schema_examples, prompts, [extract_sql(response) for response in responses]

def process_multiple_item(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    # 使用OS方法生成SQL
    try:
        prompt1_list, raw_output1_list, general_examples_list, prompt2_list, raw_output2_list, schema_examples_list, prompt_list, generated_sql_list = online_synthetic_icl(
            items,
            num_general=CFG.num_general_examples,
            num_schema_aware=CFG.num_schema_aware_examples
        )
        
        # 保存结果
        return [
            {
            "id": item.get("id", ""),
            "question": item.get("question"),
            "db_schema": item.get("schema_linking"),
            "sql": generated_sql,
            "prompt1": prompt1,
            "raw_output1": raw_output1,
            "general_examples": general_examples,
            "prompt2": prompt2,
            "raw_output2": raw_output2,
            "schema_examples": schema_examples,
            "prompt": prompt
            }
            for item, generated_sql, prompt1, raw_output1, general_examples, prompt2, raw_output2, schema_examples, prompt in 
                zip(items, generated_sql_list, prompt1_list, raw_output1_list, general_examples_list, prompt2_list, raw_output2_list, schema_examples_list, prompt_list)
        ]
    except Exception as e:
        logging.info(f"处理样本 {items[0].get('id', '')} 时出错: {e}")
        return None
        
def process_data():
    """处理数据并生成SQL"""
    # 创建输出目录
    os.makedirs(CFG.output_dir, exist_ok=True)
    logging.info("---开始加载数据---")
    # 加载数据
    with open(CFG.input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    data = data[:2]
    logging.info(f"加载数据完成,读取并选择{len(data)}个数据")
    data_chunks = list(batched(data, CFG.batch_size))
    results = []
    for item in tqdm(data_chunks):
        res = process_multiple_item(item)
        if res:
            results.extend(res)
    # 保存所有结果
    output_path = os.path.join(CFG.output_dir, "os_results.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logging.info(f"处理完成，结果已保存到 {output_path}")

if __name__ == "__main__":
    process_data()