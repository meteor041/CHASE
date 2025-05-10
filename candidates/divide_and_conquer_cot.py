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
from concurrent.futures import ThreadPoolExecutor
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig,
)

# ---------- 可调参数 ----------
@dataclass
class Config:
    model_name: str = r"/home/yangliu26/qwen3-8b"  # 请根据实际模型路径调整
    input_json: str = r"/home/yangliu26/data/train/schema_linking_result.json"
    output_dir: str = r"/home/yangliu26/CHASE/candidates/cot_result"
    # 文本生成超参
    max_new_tokens: int = 1024
    do_sample: bool = True
    temperature: float = 0.7
    # 性能设置
    batch_size: int = 4
    use_fp16: bool = True
    device_map: str = "auto"

# 配置实例
CFG = Config()

def load_model_and_tokenizer(cfg: Config):
    """加载模型和分词器"""
    # 量化配置
    quant_cfg = None
    if not cfg.use_fp16:
        quant_cfg = BitsAndBytesConfig(load_in_8bit=True)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model_name,
        trust_remote_code=True,
        padding_side="left",
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if cfg.use_fp16 else torch.float32,
        quantization_config=quant_cfg,
        device_map=cfg.device_map,
    )
    return tokenizer, model

def batched(iterable: List[Any], n: int):
    """将列表分批切片"""
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]

def load_prompt_template(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()

def extract_sql_block(generated_text: str) -> str:
    """从模型输出中提取 ```sql ... ``` 中间内容"""
    pattern = r"```sql\s+(.*?)\s*```"
    match = re.search(pattern, generated_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return generated_text.strip()  # fallback
    
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
    # print("content:", content)

    return content 
    
def decompose_question(question: str, db_schema: str, generator) -> List[str]:
    # 分解步骤的提示词模板
    DECOMPOSE_TEMPLATE = load_prompt_template(r"/home/yangliu26/CHASE/template/decompose_template.txt")

    """将问题分解为多个子问题"""
    prompt = DECOMPOSE_TEMPLATE.format(
        question=question,
        db_schema=db_schema
    )
    
    # response = generator(prompt, max_new_tokens=512, do_sample=False)
    # text = response[0]["generated_text"]
    text = llm_call(generator[0], generator[1], prompt, 512)
    # 解析子问题
    sub_questions = []
    for line in text.strip().split('\n'):
        if line.strip() and any(line.strip().startswith(str(i)) for i in range(1, 10)):
            # 移除序号前缀
            sub_q = line.strip()
            for i in range(1, 10):
                prefix = f"{i}. "
                if sub_q.startswith(prefix):
                    sub_q = sub_q[len(prefix):]
                    break
            sub_questions.append(sub_q)
    
    return prompt, sub_questions


def _format_history(history: List[Tuple[str, str]]) -> str:
    if not history or len(history) == 0:
        return ""
    chunks = []
    for i, (q, sql) in enumerate(history, 1):
        chunks.append(f"{i}. prev‑subQ: {q}\n   SQL: {sql}")
    return "Former Questions and SQL:\n" + "\n".join(chunks)
    
def generate_partial_sql(sub_question: str, evidence: str, db_schema: str, generator, history) -> str:
     # 生成SQL片段的提示词模板
    PARTIAL_SQL_TEMPLATE = load_prompt_template(r"/home/yangliu26/CHASE/template/partial_sql_template.txt")
    
    """为子问题生成SQL片段"""
    prompt = PARTIAL_SQL_TEMPLATE.format(
        sub_question=sub_question,
        evidence=evidence,
        history=_format_history(history),
        db_schema=db_schema
    )
    
    # response = generator(prompt, max_new_tokens=512, do_sample=False)
    # return response[0]["generated_text"].strip()
    return prompt, extract_sql_block(llm_call(generator[0], generator[1], prompt, 512).strip())

def assemble_sql(question: str, db_schema: str, sub_questions: List[str], 
                partial_sqls: List[str], generator) -> str:
    # 组合SQL的提示词模板
    ASSEMBLE_TEMPLATE = load_prompt_template(r"/home/yangliu26/CHASE/template/assemble_template.txt")
    
    """组合SQL片段为完整SQL"""
    # 格式化子问题和SQL片段
    sub_qs_and_sqls = ""
    for i, (q, sql) in enumerate(zip(sub_questions, partial_sqls), 1):
        sub_qs_and_sqls += f"{i}. sub question: {q}\n   SQL: {sql}\n\n"
    
    prompt = ASSEMBLE_TEMPLATE.format(
        question=question,
        db_schema=db_schema,
        sub_questions_and_sqls=sub_qs_and_sqls
    )
    
    # response = generator(prompt, max_new_tokens=1024, do_sample=False)
    # return response[0]["generated_text"].strip()
    return prompt, llm_call(generator[0], generator[1], prompt, 128)

def optimize_sql(sql: str, generator) -> str:
    """优化SQL查询（可选）"""
    # 这里可以添加SQL优化逻辑，如去除冗余等
    # 简单实现，直接返回
    return sql
    
def divide_and_conquer_sql(question: str, db_schema: str, evidence: str, generator):
    """主函数：使用分而治之方法生成SQL"""
    prompt_list = []
    # 步骤1: 分解问题
    decompose_prompt, sub_questions = decompose_question(question, db_schema, generator)
    prompt_list.append(decompose_prompt)
    # 步骤2: 生成每个子问题的SQL片段
    partial_sqls = []
    for idx, q in enumerate(sub_questions):
        history = list(zip(sub_questions[: idx], partial_sqls))  # 前 idx 个已解
        partial_prompt, partial_sql = generate_partial_sql(q, evidence, db_schema, generator, history)
        partial_sqls.append(partial_sql)
        prompt_list.append(partial_prompt)
    
    # 步骤3: 汇总构造最终SQL
    assemeble_prompt, final_sql = assemble_sql(question, db_schema, sub_questions, partial_sqls, generator)
    prompt_list.append(assemeble_prompt)
    return sub_questions, partial_sqls, final_sql, prompt_list

def process_item(item: Dict[str, Any], idx: int, generator) -> Dict[str, Any]:
    """多进程处理函数，每个进程独立加载模型和处理一条数据"""
    # tokenizer, model = load_model_and_tokenizer(cfg)
    # generator = pipeline(
    #     "text-generation",
    #     model=model,
    #     tokenizer=tokenizer,
    #     return_full_text=False,
    # )
    # generator = [model, tokenizer]
    question = item.get("question", "")
    db_schema = item.get("schema_linking", "")
    evidence = item.get("evidence", "")
    db_id = item.get("db_id", f"db_{idx}")

    sub_questions, partial_sqls, sql, prompts = divide_and_conquer_sql(question, db_schema, evidence, generator)

    return {
        "db_id": db_id,
        "question": question,
        "db_schema": db_schema, 
        "sql": sql,
        "subquestion": sub_questions,
        "partial_sqls": partial_sqls,
        "prompts": prompts
    }

def process_data_parallel():
    os.makedirs(CFG.output_dir, exist_ok=True)
    
    with open(CFG.input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # data = data[:1]
    tokenizer, model = load_model_and_tokenizer(CFG)
    generator = [model, tokenizer]
    results = []
    with ThreadPoolExecutor(max_workers=CFG.batch_size) as executor:
        futures = [
            executor.submit(process_item, item, i, generator)
            for i, item in enumerate(data)
        ]
        for i, future in enumerate(tqdm(futures, total=len(futures), desc="Processing")):
            try:
                result = future.result()
                results.append(result)
                if i % 10 == 9:
                    flush_path = os.path.join(CFG.output_dir, "COT_result_temp.json")
                    with open(flush_path, "w", encoding="utf-8") as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)
                    print(f"已处理 {i}/{len(data)}，中间结果写入 {flush_path}")
            except Exception as e:
                print(f"Error in item {i}: {e}")
                
    out_path = os.path.join(CFG.output_dir, "COT_results.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"任务完成,结果存储在{out_path}")
        
if __name__ == "__main__":
    process_data_parallel()