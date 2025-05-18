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
    model_name: str = r"/data/XiYanSQL-QwenCoder-32B-2412"
    output_file: str = r'/home/yangliu26/CHASE/pairwise/result/cmp_results.json'
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

# def load_model_and_tokenizer(cfg: Config):
#     """加载模型和分词器"""
#     # 量化配置
#     quant_cfg = None
#     if not cfg.use_fp16:
#         quant_cfg = BitsAndBytesConfig(load_in_8bit=True)

#     tokenizer = AutoTokenizer.from_pretrained(
#         cfg.model_name,
#         trust_remote_code=True,
#         padding_side="left",
#         local_files_only=True,
#     )
#     model = AutoModelForCausalLM.from_pretrained(
#         cfg.model_name,
#         trust_remote_code=True,
#         torch_dtype=torch.float16 if cfg.use_fp16 else torch.float32,
#         quantization_config=quant_cfg,
#         device_map=cfg.device_map,
#     )
#     logging.info(f"Loaded tokenizer & model from {cfg.model_name}, fp16={cfg.use_fp16}")
#     return tokenizer, model

# def llm_call(model, tokenizer, prompt: str, max_new_tokens_num=128)-> str:
#     print('-------------------\n' + prompt + '\n--------------------\n')
#     messages = [
#         {
#             "role": "user", "content": prompt
#         }
#     ]
#     text = tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True,
#         enable_thinking=False # 一定要关闭深度思考
#     )
#     model_inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
#     # conduct text completion
#     generated_ids = model.generate(
#         **model_inputs,
#         max_new_tokens=max_new_tokens_num
#     )
#     output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
    
#     # parsing thinking content
#     try:
#         # rindex finding 151668 (</think>)
#         index = len(output_ids) - output_ids[::-1].index(151668)
#     except ValueError:
#         index = 0
    
#     thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
#     content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    
#     # print("thinking content:", thinking_content)
#     # print("content:", content)

#     return content
    
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
    
# def llm_batch_call(model, tokenizer, prompts: List[str],
#              max_new_tokens_num=128,
#              do_sample: bool = True, 
#              temperature: float = 0.6,
#              top_p: float = 0.95,
#              top_k: int = 20)-> str:
#     messages = [
#         {
#             "role": "user", "content": prompt
#         }
#         for prompt in prompts
#     ]
#     kwargs = dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
#     # if "enable_thinking" in tokenizer.apply_chat_template.__code__.co_varnames:
#     #     kwargs["enable_thinking"] = False
#     text_batch = [tokenizer.apply_chat_template([m], **kwargs) for m in messages]
#     model_inputs = tokenizer(
#         text_batch, 
#         padding=True, 
#         truncation=True,         # ✅ 若超过模型 max length 则截断
#         return_tensors="pt"
#     ).to(model.device)
    
#     # conduct text completion
#     outputs = model.generate(
#         **model_inputs,
#         max_new_tokens=max_new_tokens_num,
#         do_sample=do_sample,
#         temperature=temperature,
#         top_p=top_p,
#         top_k=top_k
#     )
#     # 解码每个生成结果（去掉prompt部分）
#     # decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
#     # prompts_decoded = tokenizer.batch_decode(model_inputs["input_ids"], skip_special_tokens=True)
#     # completions = [full[len(prompt):].strip() for full, prompt in zip(decoded, prompts_decoded)]

#     # return completions
#     output_ids = [
#         output[len(input_ids):].tolist()
#         for output, input_ids in zip(outputs, model_inputs.input_ids)
#     ]
    
#     # parsing thinking content
#     try:
#         # rindex finding 151668 (</think>)
#         indexes = [
#             len(output_id) - output_ids[::-1].index(151668) 
#             for output_id in output_ids
#         ]
#     except ValueError:
#         indexes = [0 for _ in output_ids]
    
#     contents = [
#         tokenizer.decode(output_id[index:], skip_special_tokens=True).strip("\n") 
#         for index, output_id in zip(indexes, output_ids)
#     ]

#     return contents 
    
def compare_sql_pairs(template, question, evidence, schema, kwargs) -> int:
    """比较两个SQL语句，返回较好的SQL的索引（0或1）"""
    prompts = [prepare_comparison_prompt(question, evidence, schema, kwarg[0], kwarg[1], kwarg[2], kwarg[3], template) for kwarg in kwargs]
    
    # 使用模型进行预测
    _, responses = llm_batch_call_vllm(prompts, 128)
    # 确保响应是0或1
    results = [response.strip() for response in responses]
    for result in results:
        assert result in ['A', 'B'], f"模型输出必须是A或B，但得到了：{result}"
    
    return prompts, results

# def build_db_path(db_id: str)-> str:
#     """
#     构建数据库文件的完整路径
    
#     参数:
#         db_id: 数据库标识符字符串，将作为目录名和文件名的一部分
        
#     返回:
#         数据库文件的完整绝对路径
        
#     异常:
#         ValueError: 如果db_id包含不安全的路径字符或为空
#     """
#     # 输入验证
#     if not db_id or not isinstance(db_id, str):
#         raise ValueError("db_id must be a non-empty string")
    
#     # 安全检查 - 防止目录遍历攻击
#     if os.path.sep in db_id or (os.path.altsep and os.path.altsep in db_id):
#         raise ValueError("db_id cannot contain path separators")
    
#     # 使用Path对象进行安全的路径拼接
#     root_dir = Path('/home/yangliu26/data/train/train_databases')
#     db_dir = root_dir / db_id
#     db_file = db_dir / f"{db_id}.sqlite"
    
#     # 转换为字符串返回
#     return str(db_file.absolute())
    
def tournament_comparison(sql_list) -> Tuple[str, List[int]]:
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
        # db_path = build_path
        # res, raw_result = subprocess_sql_executor(db_path, sql)
        # 对结果进行截断（仅限字符串）
        if isinstance(raw_result, str) and len(raw_result) > 100:
            raw_result = raw_result[:100] + '...'
        elif isinstance(raw_result, list) or isinstance(raw_result, dict):
            raw_result = json.dumps(raw_result, ensure_ascii=False)
            if len(raw_result) > 100:
                raw_result = raw_result[:100] + '...'
    
        results[i] = raw_result
    # 两两比较
    kwargs = []
    for i in range(n):
        for j in range(i + 1, n):
            kwargs.append([sqls[i], sqls[j], results[i], results[j]])
            
    prompts, winners = compare_sql_pairs(template, question, evidence, schema, kwargs)
    cnt = 0
    for i in range(n):
        for j in range(i+1, n):
            winner = winners[cnt]
            cnt += 1
            if winner == 'A':
                scores[i] += 1
            elif winner == 'B':
                scores[j] += 1
            else:
                print(f"输出winner不符合格式要求:{winner}")
    
    # 找出得分最高的SQL
    best_idx = scores.index(max(scores))
    return sqls[best_idx], scores, prompts

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
        '/home/yangliu26/CHASE/candidates2/result/cot_result/dev_result.json',
        '/home/yangliu26/CHASE/candidates2/result/qp_result/dev_result.json',
        '/home/yangliu26/CHASE/candidates2/result/os_result/dev_os_results.json'
    ]
    
    
    results = process_json_files(input_files, CFG.output_file)
    print(f'处理完成，结果已保存到：{CFG.output_file}')

if __name__ == '__main__':
    main()