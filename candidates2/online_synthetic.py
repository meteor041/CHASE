#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Online Synthetic (OS) 方法实现
- 根据当前问题动态生成多个输入-输出样例
- 作为Prompt输入来指导SQL生成
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pairwise')))
from db_utils import subprocess_sql_executor, build_db_path
import multiprocessing
import json
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
import argparse
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
import openai
from openai import OpenAI
import sqlglot
client = OpenAI(
    api_key="EMPTY",  # 对于某些本地服务，API密钥可以是任意非空字符串
    base_url="http://localhost:8000/v1" # 指向你的 vLLM 服务
)

wrong_cnt = 0
logging.basicConfig(level=logging.INFO)
# ---------- 可调参数 ----------
@dataclass
class CFG:
    model_name: str
    input_json: str
    output_dir: str
    mschema_path: str
    # 性能设置
    num_generations: int
    batch_size: int
    # 文本生成超参
    max_new_tokens: int = 256
    do_sample: bool = True
    temperature: float = 0.2
    gen_temperature: float = 0.6
    use_fp16: bool = True
    device_map: str = "auto"
    # OS特定参数
    num_general_examples: int = 3
    num_schema_aware_examples: int = 3

def parse_cfg() -> CFG:
    parser = argparse.ArgumentParser(description="NL2SQL baseline runner")
    parser.add_argument("--input_json",  required=True,
                        help="Path to schema_linking JSON (e.g. dev_schema_linking_result.json)")
    parser.add_argument("--output_dir",  required=True,
                        help="Directory to save generated SQL results")
    parser.add_argument("--model_name",  required=True,
                        help="LLM name or checkpoint path for vLLM OpenAI API")
    parser.add_argument("--mschema_path", required=True,
                        help="Path to merged schema JSON (mschema)")
    parser.add_argument("--num_generations", type=int, default=1,
                        help="Number of SQL generations per question (default: 1)")
    parser.add_argument("--batch_size",  type=int, default=32,
                        help="Concurrent request batch size (default: 32)")
    args = parser.parse_args()
    return CFG(**vars(args))

CFG = parse_cfg()

def load_prompt_template(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()

GENERAL_EXAMPLES_TEMPLATE = load_prompt_template(r"/home/yangliu26/CHASE/template/online_synthetic1.txt")
SCHEMA_AWARE_TEMPLATE = load_prompt_template(r"/home/yangliu26/CHASE/template/online_synthetic2.txt")
FEW_SHOT_TEMPLATE = load_prompt_template(r"/home/yangliu26/CHASE/template/online_synthetic3.txt")
FIXER_TEMPLATE = load_prompt_template(r"/home/yangliu26/CHASE/template/template_query_fixer.txt")

with open(CFG.mschema_path, "r") as f:
    TABLES_INFO = json.loads(f.read())

def _is_valid_sql(sql: str, db_path) -> Tuple[bool, Any]:
    """返回 SQL 语法是否合法（SQLite 方言）"""
    if not sql:
        return False, ""
    try:
        # 语法错误
        sqlglot.parse_one(sql, read="sqlite")
    except Exception as e:
        return False, f"[PARSE_ERROR] {e}"
    try:
        ok, raw = subprocess_sql_executor(db_path, sql)
        return ok, str(raw)[:100]
    except TimeoutError:
        return False, "[TIMEOUT]"
    except Exception as e:
        return False, e

def call_single_prompt(prompt: str, 
                       temperature
                      ) -> Tuple[str, str]:
    try:
        response = client.chat.completions.create(
            model=CFG.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
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

def call_single_prompt_with_fixer(prompt:str, 
                                  item, 
                                  max_retry:int = 1)-> Tuple[List[str], List[str]]:
    db_path = build_db_path(item.get('db_id'))
    thoughts = []
    contents = []
    while max_retry > 0:
        think, content = call_single_prompt(prompt, CFG.temperature)
        thoughts.append(think)
        contents.append(content)
        sql = extract_sql_from_llm(content)
        max_retry -= 1
        ok, ret = _is_valid_sql(sql, db_path)
        if ok:
            return thoughts, contents
        else:
            prompt = FIXER_TEMPLATE.format_map({
                        "DATABASE_SCHEMA": item.get("DDL"),
                        "QUESTION": item.get('question'),
                        "HINT": item.get('evidence'),
                        "QUERY": sql,
                        "RESULT": ret
                    })
    logging.critical(f"未能通过有效性测试, sql:{sql}")
    global wrong_cnt
    wrong_cnt += 1
    return thoughts, contents

def llm_batch_call_vllm(prompts, temperature) -> Tuple[List[str], List[str]]:
    """并发提交 prompt 到 vLLM 的 OpenAI 接口"""
    thinking_results, content_results = [None] * len(prompts), [None] * len(prompts)
    with ThreadPoolExecutor(max_workers=CFG.batch_size) as executor:
        future_to_index = {
                    executor.submit(call_single_prompt, prompt, temperature): i
            for i, prompt in enumerate(prompts)
        }
        for future in as_completed(future_to_index):
            i = future_to_index[future]
            think, content = future.result()
            thinking_results[i] = think
            content_results[i] = content

    return thinking_results, content_results
    
def llm_batch_call_vllm_sql(prompts,
                            items) -> Tuple[List[str], List[str]]:
    """并发提交 prompt 到 vLLM 的 OpenAI 接口"""
    thinking_results, content_results = [None] * len(items), [None] * len(items)
    with ThreadPoolExecutor(max_workers=CFG.batch_size) as executor:
        future_to_index = {
                    executor.submit(call_single_prompt_with_fixer, prompt, item): i
            for i, (prompt, item) in enumerate(zip(prompts, items))
        }
        for future in as_completed(future_to_index):
            i = future_to_index[future]
            thoughts, contents = future.result()
            think = thoughts[-1]
            content = contents[-1]
            thinking_results[i] = thoughts
            content_results[i] = contents

    return thinking_results, content_results

def batched(iterable: List[Any], n: int):
    """将列表分批切片"""
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]

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
    thoughts, texts = llm_batch_call_vllm(prompts, CFG.gen_temperature)
    return prompts, thoughts, texts

def generate_examples_by_schema(items, num_examples: int = 3) -> List[Tuple[str, str]]:
    """生成基于特定schema的示例"""
    prompts = [
        SCHEMA_AWARE_TEMPLATE.format(
            TARGET_DATABASE_SCHEMA=item.get("DDL"),
            num_examples=num_examples
        )
        for item in items
    ]
    thoughts, texts = llm_batch_call_vllm(prompts, CFG.gen_temperature)
    return prompts, thoughts, texts

def format_few_shot_prompt(examples_text_list, items) -> str:
    """格式化few-shot提示"""
    return [
        FEW_SHOT_TEMPLATE.format(
            DATABASE_SCHEMA=TABLES_INFO.get(item.get('db_id')),
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
    thoughts, responses = llm_batch_call_vllm_sql(prompts, items)
    logging.info("完成SQL生成")
    return prompt1, thoughts1, general_examples, prompt2, thoughts2, schema_examples, prompts, thoughts, responses, [extract_sql_from_llm(response[-1]) for response in responses]

def process_multiple_item(items: List[Dict[str, Any]], num_generations) -> Dict[str, Any]:
    # 使用OS方法生成SQL
    prompt1_list_all = [[] for _ in range(len(items))]
    thoughts1_all = [[] for _ in range(len(items))]
    general_examples_list_all = [[] for _ in range(len(items))]
    prompt2_list_all = [[] for _ in range(len(items))]
    thoughts2_all = [[] for _ in range(len(items))]
    schema_examples_list_all = [[] for _ in range(len(items))]
    prompt_list_all = [[] for _ in range(len(items))]
    thoughts_all = [[] for _ in range(len(items))]
    responses_all = [[] for _ in range(len(items))]
    generated_sql_list_all = [[] for _ in range(len(items))]
    try:
        for _ in range(num_generations):
            (
                prompt1_list, 
                thoughts1, 
                general_examples_list, 
                prompt2_list, 
                thoughts2, 
                schema_examples_list, 
                prompt_list, 
                thoughts, 
                responses, 
                generated_sql_list
            ) = online_synthetic_icl(
                items,
                num_general=CFG.num_general_examples,
                num_schema_aware=CFG.num_schema_aware_examples
            )
            for i, generated_sql in enumerate(generated_sql_list):
                generated_sql_list_all[i].append(generated_sql)
        # 保存结果
        return [
            {
                "id": item.get("id", ""),
                "question": item.get("question"),
                "sql": generated_sql_list if len(generated_sql_list) > 1 else generated_sql_list[0],
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
            for item, generated_sql_list, prompt1, thought1, general_examples, prompt2, thought2, schema_examples, prompt, thought, response in 
                zip(items, generated_sql_list_all, prompt1_list, thoughts1, general_examples_list, prompt2_list, thoughts2, schema_examples_list, prompt_list, thoughts, responses)
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
    # data = data[:2]
    logging.info(f"加载数据完成,读取并选择{len(data)}个数据")
    data_chunks = list(batched(data, CFG.batch_size))
    results = []
    for item in tqdm(data_chunks):
        res = process_multiple_item(item, CFG.num_generations)
        if res:
            results.extend(res)
    # 保存所有结果
    output_path = os.path.join(CFG.output_dir, "dev_os_results.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logging.info(f"处理完成，结果已保存到 {output_path}")
    logging.info(f"错误SQL个数:{wrong_cnt}")

if __name__ == "__main__":
    process_data()