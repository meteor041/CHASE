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
import logging
import time
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
    num_gpus: int = 4
    batch_size: int = 1
    use_fp16: bool = True
    device_map: str = {"": "cuda:0"}

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
    """将列表分批切片"""
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]

def load_prompt_template(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()

PARTIAL_SQL_TEMPLATE = load_prompt_template(r"/home/yangliu26/CHASE/template/partial_sql_template.txt")
ASSEMBLE_TEMPLATE = load_prompt_template(r"/home/yangliu26/CHASE/template/assemble_template.txt")
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

def merge_results(output_dir: str, num_gpus: int, merged_file: str, gpu_id: int):
    results = []
    for i in range(num_gpus):
        path = Path(output_dir) / f"cot_result_gpu{i}.jsonl"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    results.append(json.loads(line))
        else:
            logging.critical(f"[GPU-{gpu_id}] {path} don't exist!")
    with open(Path(output_dir) / merged_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
def llm_batch_call(model, tokenizer, prompts: List[str],
             max_new_tokens_num=128,
             do_sample: bool = True, 
             temperature: float = 0.6,
             top_p: float = 0.95,
             top_k: int = 20)-> List[str]:
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
 
def decompose_question(batch, generator, prompt_list, init_pos: int):
    # 分解步骤的提示词模板
    DECOMPOSE_TEMPLATE = load_prompt_template(r"/home/yangliu26/CHASE/template/decompose_template.txt")
    """将问题分解为多个子问题"""
    prompts = [DECOMPOSE_TEMPLATE.format(
        question=item.get("question"),
        db_schema=item.get("schema_linking")
        ) 
               for item in batch]
    
    texts = llm_batch_call(generator[0], generator[1], prompts, 512)
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
    i = 0
    for idx in range(init_pos, len(sub_questions)):
        prompt_list[idx].append(prompts[i])
        i += 1
    return sub_questions


def _format_history(history: List[Tuple[str, str]]) -> str:
    if not history or len(history) == 0:
        return ""
    chunks = []
    for i, (q, sql) in enumerate(history, 1):
        chunks.append(f"{i}. prev‑subQ: {q}\n   SQL: {sql}")
    return "Former Questions and SQL:\n" + "\n".join(chunks)
    
def generate_partial_sql(batch, sub_questions_list, generator, prompt_list, init_pos) -> str:
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
        ret = llm_batch_call(generator[0], generator[1], prompts, 512)
        for j, sql in enumerate(ret):
            if i < len(sub_questions_list[j]):
                partial_sqls[j].append(sql)
                prompt_list[init_pos+j].append(prompts[j])
    return partial_sqls

def assemble_sql(batch, sub_questions_list: List[List[str]], 
                partial_sqls_list: List[List[str]], generator, 
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
    
    for i, k in enumerate(range(init_pos, len(prompts))):
        prompt_list[k].append(prompts[i])
    return llm_batch_call(generator[0], generator[1], prompts, 128)

def optimize_sql(sql: str, generator) -> str:
    """优化SQL查询（可选）"""
    # 这里可以添加SQL优化逻辑，如去除冗余等
    # 简单实现，直接返回
    return sql
    
def divide_and_conquer_sql(batch, generator, prompt_list, init_pos: int):
    """主函数：使用分而治之方法生成SQL"""    
    # 步骤1: 分解问题
    sub_questions_list = decompose_question(batch, generator, prompt_list, init_pos)
    # 步骤2: 生成每个子问题的SQL片段
    partial_sqls = generate_partial_sql(batch, sub_questions_list, generator, prompt_list, init_pos)
    # 步骤3: 汇总构造最终SQL
    final_sqls = assemble_sql(batch, sub_questions_list, partial_sqls, generator, prompt_list, init_pos)

    return sub_questions_list, partial_sqls, final_sqls

def process_item(gpu_id: int, data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """多进程处理函数，每个进程独立加载模型和处理一条数据"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    tokenizer, model = load_model_and_tokenizer(CFG)
    generator = [model, tokenizer]
    batched_data = list(batched(data, CFG.batch_size))
    
    prompt_list = [[] for _ in range(len(data))]
    results = []
    out_path = Path(CFG.output_dir) / f"cot_result_gpu{gpu_id}.jsonl"
    out_path.write_text("")
    for i, batch in enumerate(batched_data):
        idx_offset = i * CFG.batch_size
        sub_questions_list, partial_sqls_list, final_sqls = divide_and_conquer_sql(
            batch, generator, prompt_list, idx_offset
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
                    logging.info(f"[GPU-{gpu_id}] {r}")
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            results = []  # 清空已写入部分缓存

def process_data_parallel():
    os.makedirs(CFG.output_dir, exist_ok=True)
    
    with open(CFG.input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    chunks = split_data(data, CFG.num_gpus)

    for i, chunk in enumerate(chunks):
        logging.info(f"分配给 GPU-{i}: {len(chunk)} 条样本")
    import multiprocessing as mp
    # ctx = mp.get_context("spawn")
    processes = []
    for gpu_id in range(CFG.num_gpus):
        p = mp.Process(target=process_item, args=(gpu_id, chunks[gpu_id]))
        p.start()
        processes.append(p)
        if gpu_id != CFG.num_gpus - 1:
            time.sleep(10)  # ✅ 每个进程启动之间隔 10 秒，避免瞬间爆显存

    for p in processes:
        p.join()
                
    merge_results(CFG.output_dir, CFG.num_gpus, "cot_result_merged.json", gpu_id)
        
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, 
        format="[%(asctime)s] %(levelname)s - %(message)s"
    )
    process_data_parallel()