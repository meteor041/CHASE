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

logging.basicConfig(level=logging.INFO)
# ---------- 可调参数 ----------
@dataclass
class Config:
    model_name: str = r"/home/yangliu26/qwen3-8b"  # 请根据实际模型路径调整
    input_json: str = r"/home/yangliu26/data/train/schema_linking_result.json"
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
    
def load_model_and_tokenizer(cfg: Config):
    """加载模型和分词器"""
    # 量化配置
    quant_cfg = None
    if not cfg.use_fp16:
        quant_cfg = BitsAndBytesConfig(load_in_4bit=True)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        trust_remote_code=True,
        padding_side="left",
        # enable_thinking=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if cfg.use_fp16 else torch.float32,
        quantization_config=quant_cfg,
        device_map=cfg.device_map,
    )
    model.generation_config.enable_thinking = False
    return tokenizer, model
    
def llm_call(model, tokenizer, prompt: str, max_new_tokens_num=128)-> str:
    messages = [
        {
            "role": "user", "content": prompt
        }
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False # 一定要关闭深度思考
    )
    model_inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens_num
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
    
    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0
    
    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    
    # print("thinking content:", thinking_content)
    # logging.info("content:", content)

    return content 

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

def generate_examples_by_sql_features(db_schema: str, num_examples: int, generator) -> List[Tuple[str, str]]:
    """生成涵盖不同SQL特性的示例"""
    prompt = GENERAL_EXAMPLES_TEMPLATE.format(
        db_schema=db_schema,
        num_examples=num_examples
    )
    text = llm_call(generator[0], generator[1], prompt, 256).strip()
    # response = generator(prompt, max_new_tokens=256, do_sample=True, temperature=0.5, enable_thinking=False)
    # text = response[0]["generated_text"].strip()
    logging.info("步骤1: %s", text)
    # 解析示例
    examples = []
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
                examples.append((question, sql))
        i += 1
    
    return prompt, text, examples[:num_examples]  # 确保不超过请求的数量

def generate_examples_by_schema(db_schema: str, num_examples: int, generator) -> List[Tuple[str, str]]:
    """生成基于特定schema的示例"""
    prompt = SCHEMA_AWARE_TEMPLATE.format(
        db_schema=db_schema,
        num_examples=num_examples
    )
    text = llm_call(generator[0], generator[1], prompt, 256).strip()
    # response = generator(prompt, max_new_tokens=256, do_sample=True, temperature=0.5, enable_thinking=False)
    # text = response[0]["generated_text"].strip()
    logging.info("步骤2: %s",text)
    # 解析示例
    examples = []
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
                examples.append((question, sql))
        i += 1
    
    return prompt, text, examples[:num_examples]  # 确保不超过请求的数量

def format_few_shot_prompt(examples: List[Tuple[str, str]], question: str, db_schema: str) -> str:
    """格式化few-shot提示"""
    examples_text = ""
    for i, (q, sql) in enumerate(examples, 1):
        examples_text += f"Example {i}:\nQuestion: {q}\nSQL: {sql}\n"
    
    return FEW_SHOT_TEMPLATE.format(
        db_schema=db_schema,
        examples=examples_text,
        question=question
    )

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

def online_synthetic_icl(question: str, db_whole_schema: str, db_schema: str, generator, 
                         num_general: int = 3, num_schema_aware: int = 3):
    """主函数：使用在线合成示例方法生成SQL"""
    logging.info("步骤1: 使用常见SQL特征生成通用示例")
    prompt1, raw_output1, general_examples = generate_examples_by_sql_features(db_whole_schema, num_general, generator)
    logging.info("完成步骤1")
    logging.info("步骤2: 生成schema-aware示例")
    prompt2, raw_output2, schema_examples = generate_examples_by_schema(db_schema, num_schema_aware, generator)
    logging.info("完成步骤2")
    logging.info("步骤3: 组合所有示例 + 当前问题进入Prompt")
    prompt = format_few_shot_prompt(general_examples + schema_examples, question, db_schema)
    logging.info("完成步骤3")
    logging.info("步骤4: 生成SQL")
    response = llm_call(generator[0], generator[1], prompt, 128)
    # response = generator(prompt, max_new_tokens=128, do_sample=True, enable_thinking=False)
    logging.info("完成SQL生成")
    return prompt1, raw_output1, general_examples, prompt2, raw_output2, schema_examples, prompt, extract_sql(response)

# def safe_process_single_item(item: Dict[str, Any]) -> Dict[str, Any]:
#     # 每个进程内加载模型
#     tokenizer, model = load_model_and_tokenizer(CFG)
#     generator = [model, tokenizer]
#     return process_single_item(item, generator)
    
def process_single_item(item: Dict[str, Any], generator) -> Dict[str, Any]:
    question = item.get("question", "")
    db_schema = item.get("schema_linking", "")
    db_whole_schema = tables_info.get(item.get("db_id"))
    # 使用OS方法生成SQL
    try:
        prompt1, raw_output1, general_examples, prompt2, raw_output2, schema_examples, prompt, generated_sql = online_synthetic_icl(
            question, 
            db_whole_schema,
            db_schema, 
            generator,
            num_general=CFG.num_general_examples,
            num_schema_aware=CFG.num_schema_aware_examples
        )
        logging.info(f"生成 prompt 成功，正在生成 SQL: id={item.get('id', '')}")
        
        # 保存结果
        return {
            "id": item.get("id", ""),
            "question": question,
            "db_schema": db_schema,
            "sql": generated_sql,
            "prompt1": prompt1,
            "raw_output1": raw_output1,
            "general_examples": general_examples,
            "prompt2": prompt2,
            "raw_output2": raw_output2,
            "schema_examples": schema_examples,
            "prompt": prompt
        }
    except Exception as e:
        logging.info(f"处理样本 {item.get('id', '')} 时出错: {e}")
        return None

from queue import Queue
from threading import Thread

def model_worker(model, tokenizer, task_queue: Queue, result_list: list):
    while True:
        item = task_queue.get()
        if item is None:
            break
        result = process_single_item(item, [model, tokenizer])
        if result is not None:
            result_list.put(result)
        task_queue.task_done()
        
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
    # results = []
    # logging.info("加载模型")
    # tokenizer, model = load_model_and_tokenizer(CFG)
    # generator = [model, tokenizer]
    # with ProcessPoolExecutor(max_workers=4) as executor:
    #     # 提交所有任务
    #     futures = [executor.submit(safe_process_single_item, item, generator) for item in data]
    #     # 使用tqdm显示进度
    #     for future in tqdm(futures, total=len(data), desc="处理样本"):
    #         result = future.result()
    #         if result is not None:
    #             results.append(result) 
    # 主函数
    task_queue = Queue()
    result_queue = Queue()
    tokenizer, model = load_model_and_tokenizer(CFG)
    
    # 启动模型线程（串行模型调用）
    model_thread = Thread(target=model_worker, args=(model, tokenizer, task_queue, result_queue))
    model_thread.start()
    
    # 将任务提交到队列（可以并行准备数据）
    for item in data:
        task_queue.put(item)
    
    # 等待任务完成
    task_queue.join()
    task_queue.put(None)  # 终止信号
    model_thread.join()

    results = []
    while not result_queue.empty():
        results.append(result_queue.get())
    
    # 保存所有结果
    output_path = os.path.join(CFG.output_dir, "os_results.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logging.info(f"处理完成，结果已保存到 {output_path}")

if __name__ == "__main__":
    process_data()