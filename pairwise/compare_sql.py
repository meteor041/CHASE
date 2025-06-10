import json
import os
from typing import List, Dict, Tuple
from pathlib import Path
from collections import defaultdict
from db_utils import subprocess_sql_executor, build_db_path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import logging
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig,
)
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",  # 对于某些本地服务，API密钥可以是任意非空字符串
    base_url="http://localhost:8000/v1" # 指向你的 vLLM 服务
)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")

# ---------- 可调参数 ----------
@dataclass
class Config:
    # model_name: str = r"/home/yangliu26/CHASE/pairwise/pairwise_selector_model/qwen3-8b-lora"
    # model_name: str = r"/data/XiYanSQL-QwenCoder-32B-2412"
    model_name: str = r"/home/yangliu26/Qwen2.5-32B"
    output_file: str = r'/home/yangliu26/CHASE/pairwise/result/2025_5_20_qwen2.5-32B_lora/cmp_results.json'
    # 文本生成超参
    max_new_tokens: int = 1024
    do_sample: bool = True
    temperature: float = 0.2
    # 性能设置
    batch_size: int = 4
    use_fp16: bool = True
    device_map: str = "auto"

# 配置实例
CFG = Config()
    
def load_prompt_template() -> str:
    """加载SQL比较的提示模板"""
    template_path = r"/home/yangliu26/CHASE/template/selection_agent_train_prompt.txt"
    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read()

def prepare_comparison_prompt(question:str, evidence: str, schema: str, sql1: str, sql2: str, res1: str, res2: str, template: str) -> str:
    """准备用于比较的提示词"""
    return template.format(
        **{
            "DATABASE SCHEMA": schema,
            "QUESTION": question,
            "HINT": evidence,
            "CANDIDATE A QUERY":sql1, 
            "CANDIDATE B QUERY":sql2,
            "CANDIDATE A RESULT":res1,
            "CANDIDATE B RESULT":res2
        }
    )

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
    
def compare_sql_pairs(template, question, evidence, schema, kwargs) -> int:
    """比较两个SQL语句，返回较好的SQL的索引（0或1）"""
    prompts = [prepare_comparison_prompt(question, evidence, schema, kwarg[0], kwarg[1], kwarg[2], kwarg[3], template) for kwarg in kwargs]
    
    # 使用模型进行预测
    _, responses = llm_batch_call_vllm(prompts, 128)
    # 确保响应是0或1
    # results = [response.strip() for response in responses]
    results = [response.split('\n', 1)[0].strip() for response in responses]
    for result in results:
        assert result in ['A', 'B'], f"模型输出必须是A或B，但得到了：{result}"
    
    return prompts, results
    
def tournament_comparison(sql_list) -> Tuple[str, List[int], List[str]]:
    """使用锦标赛方式比较所有SQL语句"""
    sqls = sql_list.get('sql_list')
    db_id = sql_list.get('db_id')
    question = sql_list.get('question', '')
    evidence = sql_list.get('evidence', '')
    schema = sql_list.get('schema_linking', '')
    template = load_prompt_template()
    n = len(sqls)
    scores = [0] * n  # 每个SQL的得分
    results = [None] * n
    # results_exec = [None] * n
    logging.info(f"Starting tournament for db_id={db_id}, {n} candidates")
    db_path = build_db_path(db_id)
    for i, sql in enumerate(sqls):
        try:
            res, raw_result = subprocess_sql_executor(db_path, sql)
        except TimeoutError:
            logging.warning(f"SQL {i} 执行超时，内容如下：\n{sql}")
            raw_result = "[TIMEOUT]"
        except Exception as e:
            logging.warning(f"SQL {i} 执行失败: {e}，内容如下：\n{sql}")
            raw_result = "[ERROR]"
        # results_exec[i] = raw_result
        # 对结果进行截断
        if isinstance(raw_result, str) and len(raw_result) > 100:
            raw_result = raw_result[:100] + '...'
        elif isinstance(raw_result, list) or isinstance(raw_result, dict):
            raw_result = json.dumps(raw_result, ensure_ascii=False)
            if len(raw_result) > 100:
                raw_result = raw_result[:100] + '...'
    
        results[i] = raw_result

    # 用于存储所有理论上会生成的提示，以便返回值与原函数行为一致
    all_prompts_that_would_be_generated = [] 
    
    # 两两比较
    kwargs_for_llm_processing = []
    indices_for_llm_processing = [] # 存储 (original_i_index, original_j_index)
    for i in range(n):
        for j in range(i + 1, n):
            # 1. 准备当前SQL对的提示信息 (无论是否送入LLM，都为all_prompts_that_would_be_generated列表准备)
            prompt_for_this_pair = prepare_comparison_prompt(
                question, evidence, schema,
                sqls[i], sqls[j],
                results[i], results[j], # 使用已执行的SQL结果
                template
            )
            all_prompts_that_would_be_generated.append(prompt_for_this_pair)

            # 2. 【核心修改点】检查执行结果是否一致
            if results[i] == results[j]:
                scores[i] += 1  # SQL i (候选 A) 直接得分
                logging.info(f"Pair (SQL {i}, SQL {j}): Results are identical. SQL {i} gets +1 point directly.")
            else:
                # 结果不同，准备交由LLM处理
                kwargs_for_llm_processing.append([sqls[i], sqls[j], results[i], results[j]])
                indices_for_llm_processing.append((i, j)) # 记录原始索引对

    # 3. 如果存在需要LLM比较的SQL对，则调用LLM
    if kwargs_for_llm_processing:
        # compare_sql_pairs 返回 (实际发送给LLM的prompts, LLM的响应)
        # 这里我们只需要LLM的响应
        _, llm_responses_for_these_pairs = compare_sql_pairs(
            template, question, evidence, schema,
            kwargs_for_llm_processing # 只传递需要LLM处理的子集
        )

        # 4. 根据LLM的响应更新分数
        for k, winner_char in enumerate(llm_responses_for_these_pairs):
            original_i, original_j = indices_for_llm_processing[k] # 获取原始的i, j索引
            if winner_char == 'A':
                scores[original_i] += 1
            elif winner_char == 'B':
                scores[original_j] += 1
            else:
                logging.error(f"LLM output for pair (SQL {original_i}, SQL {original_j}) was '{winner_char}', expected 'A' or 'B'.")
    
    # 找出得分最高的SQL
    best_idx = scores.index(max(scores))
    return sqls[best_idx], scores, all_prompts_that_would_be_generated

def process_json_files(input_files: List[str], output_file: str):
    """处理多个JSON文件中的SQL语句"""
    all_results = []
    sql_lists = defaultdict(dict)
    with open(input_files[0], 'r', encoding='utf-8') as f:
        sql_lists = json.load(f) 
        # sql_lists = sql_lists[:2]
    # 读取所有JSON文件
    for i, file_path in enumerate(tqdm(input_files[1:],desc="Reading JSON files")):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # data = data[:2]
        logging.info(f"Loaded {len(data)} items from {file_path}")
        for k, item in enumerate(data):
            if i == 0:
                sql_lists[k]['sql_list'] = []
            sql_lists[k]['sql_list'].append(item['sql'])

    for i, sql_list in enumerate(tqdm(sql_lists, desc="Comparing SQL sets")):
        logging.info(f"[{i+1}/{len(sql_lists)}] Comparing {len(sql_list['sql_list'])} SQLs for question: {sql_list.get('question')!r}")
        best_sql, scores, prompts = tournament_comparison(sql_list)
        # 记录结果
        result = {
            'question': sql_list.get('question', ''),
            'best_sql': best_sql,
            'all_sqls': sql_list.get('sql_list'),
            'scores': scores,
            'prompts': prompts
        }
        all_results.append(result)
    # 保存结果
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    return all_results

def main():
    logging.info(f"Starting comparison script with config: {CFG}")
    input_files = [
        '/home/yangliu26/data/schema_linking/dev_schema_linking_result.json',
        '/home/yangliu26/CHASE/candidates2/result/2025_5_20/cot_result/dev_result.json',
        '/home/yangliu26/CHASE/candidates2/result/2025_5_20/qp_result/dev_result.json',
        '/home/yangliu26/CHASE/candidates2/result/2025_5_20/os_result/dev_os_results.json'
    ]
    results = process_json_files(input_files, CFG.output_file)
    print(f'处理完成，结果已保存到：{CFG.output_file}')

if __name__ == '__main__':
    main()