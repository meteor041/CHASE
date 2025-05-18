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
from typing import Dict, Any, List, Tuple, Optional
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    model_name: str = r"/data/XiYanSQL-QwenCoder-32B-2412"  # 请根据实际模型路径调整
    input_json: str = r"/home/yangliu26/data/schema_linking/dev_schema_linking_result.json"
    output_dir: str =  r"/home/yangliu26/CHASE/candidates2/result/os_result"
    tables_json: str = r"/home/yangliu26/data/mschema/dev_mschemas.json"
    # 文本生成超参
    max_new_tokens: int = 256
    do_sample: bool = True
    temperature: float = 0.7
    # 性能设置
    batch_size: int = 48
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

    with ThreadPoolExecutor(max_workers=CFG.batch_size) as executor:
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

GENERAL_EXAMPLES_TEMPLATE = load_prompt_template("/home/yangliu26/CHASE/template/online_synthetic1.txt")
SCHEMA_AWARE_TEMPLATE = load_prompt_template("/home/yangliu26/CHASE/template/online_synthetic2.txt")
FEW_SHOT_TEMPLATE = load_prompt_template("/home/yangliu26/CHASE/template/online_synthetic3.txt")

def split_llm_sql_examples(text: str):
    # 以 "Example N)" 为起始标志，按组切分
    pattern = r"Example\s+\d+\)[\s\S]*?(?=(?:Example\s+\d+\)|\Z))"
    return [match.strip() for match in re.findall(pattern, text)]
    
def generate_examples_by_sql_features(items, num_examples: int) -> List[Tuple[str, str]]:
    """生成涵盖不同SQL特性的示例"""
    prompts = [
        GENERAL_EXAMPLES_TEMPLATE.format(
            TARGET_DATABASE_SCHEMA=item.get('DDL'),
            num_examples=num_examples
        ) 
        for item in items
    ]
    thoughts, texts = llm_batch_call_vllm(prompts, 8192)
    return prompts, thoughts, texts

def generate_examples_by_schema(items, num_examples: int) -> List[Tuple[str, str]]:
    """生成基于特定schema的示例"""
    prompts = [
        SCHEMA_AWARE_TEMPLATE.format(
            TARGET_DATABASE_SCHEMA=item.get("DDL")
        )
        for item in items
    ]
    thoughts, texts = llm_batch_call_vllm(prompts, 8192)
    return prompts, thoughts, texts

def format_few_shot_prompt(examples_text_list, items) -> str:
    """格式化few-shot提示"""
    return [
        FEW_SHOT_TEMPLATE.format(
            DATABASE_SCHEMA=tables_info.get(item.get('db_id')),
            ONLINE_EXAMPLES=examples_text,
            QUESTION=item.get("question"),
            HINT=item.get('evidence'),
        )
        for item, examples_text in zip(items, examples_text_list)
    ]

def extract_sql_from_llm(output: str) -> Optional[str]:
        # 1) Primary extraction: between <FINAL_ANSWER> tags
        tag_regex = re.compile(
            r"<FINAL_ANSWER>\s*(.*?)\s*</FINAL_ANSWER>",
            flags=re.IGNORECASE | re.DOTALL,
        )
        m = tag_regex.search(output)
        if m:
            sql = m.group(1).strip()
    
            # Optional: strip any code-fence left inside the tag
            if sql.startswith("```"):
                sql = re.sub(r"^```[^`\n]*\n?", "", sql)       # drop opening fence
                sql = re.sub(r"\n?```$", "", sql)              # drop closing fence
                sql = sql.strip()
    
            return sql or None
    
        # 2) Fallback: grab the last fenced SQL block (```) in the answer
        fence_regex = re.compile(
            r"```(?:sql)?\s*(SELECT[\s\S]*?)```",
            flags=re.IGNORECASE,
        )
        blocks = fence_regex.findall(output)
        if blocks:
            return blocks[-1].strip()
    
        # 3) Nothing found
        return output

def online_synthetic_icl(items,  
                         num_general: int = 3, 
                         num_schema_aware: int = 3):
    """主函数：使用在线合成示例方法生成SQL"""
    logging.info("步骤1: 使用常见SQL特征生成通用示例")
    prompt1, thoughts1, general_examples = generate_examples_by_sql_features(items, num_general)
    logging.info("完成步骤1")
    logging.info("步骤2: 生成schema-aware示例")
    prompt2, thoughts2, schema_examples = generate_examples_by_schema(items, num_schema_aware)
    logging.info("完成步骤2")
    logging.info("步骤3: 组合所有示例 + 当前问题进入Prompt")
    example_combined = [
        general_example + schema_example 
        for general_example, schema_example in zip(general_examples, schema_examples)
    ]
    prompts = format_few_shot_prompt(example_combined, items)
    logging.info("完成步骤3")
    logging.info("步骤4: 生成SQL")
    thoughts, responses = llm_batch_call_vllm(prompts, 8192)
    logging.info("完成SQL生成")
    return prompt1, thoughts1, general_examples, prompt2, thoughts2, schema_examples, prompts, thoughts, responses, [extract_sql_from_llm(response) for response in responses]

def process_multiple_item(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    # 使用OS方法生成SQL
    try:
        prompt1_list, thoughts1, general_examples_list, prompt2_list, thoughts2, schema_examples_list, prompt_list, thoughts, responses, generated_sql_list = online_synthetic_icl(
            items,
            num_general=CFG.num_general_examples,
            num_schema_aware=CFG.num_schema_aware_examples
        )
        
        # 保存结果
        return [
            {
                "id": item.get("id", ""),
                "question": item.get("question"),
                "sql": generated_sql,
                "prompt1": prompt1,
                "thought1": thought1,
                "general_examples": general_examples,
                "prompt2": prompt2,
                "thought2": thought2,
                "schema_examples": schema_examples,
                "prompt": prompt,
                "thought": thought,
                "response": response
            }
            for item, generated_sql, prompt1, thought1, general_examples, prompt2, thought2, schema_examples, prompt, thought, response in 
                zip(items, generated_sql_list, prompt1_list, thoughts1, general_examples_list, prompt2_list, thoughts2, schema_examples_list, prompt_list, thoughts, responses)
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
    data = data
    logging.info(f"加载数据完成,读取并选择{len(data)}个数据")
    data_chunks = list(batched(data, CFG.batch_size))
    results = []
    for item in tqdm(data_chunks):
        res = process_multiple_item(item)
        if res:
            results.extend(res)
    # 保存所有结果
    output_path = os.path.join(CFG.output_dir, "dev_os_results.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logging.info(f"处理完成，结果已保存到 {output_path}")

if __name__ == "__main__":
    process_data()