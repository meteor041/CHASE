import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'pairwise')))
import json
from func_timeout import func_timeout, FunctionTimedOut
from concurrent.futures import ThreadPoolExecutor, as_completed
from db_utils import subprocess_sql_executor, build_db_path
import sqlglot
from typing import List, Tuple, Dict, Any, List, Optional
from tqdm import tqdm
import re
import time
import logging
from pathlib import Path
from openai import OpenAI
import argparse
from dataclasses import dataclass
import random 

client = OpenAI(
    api_key="EMPTY",  # 对于某些本地服务，API密钥可以是任意非空字符串
    base_url="http://localhost:8000/v1" # 指向你的 vLLM 服务
)

# 配置参数
@dataclass
class CFG:
    template_path: str
    fixer_template_path: str
    input_json: str
    output_dir: str
    model_name: str
    mschema_path: str
    num_generations: int # 每个问题的生成次数,默认为1
    batch_size: int = 48
    temperature: float = 0.6
    top_p: float = 0.9

def parse_cfg() -> CFG:
    parser = argparse.ArgumentParser(description="NL2SQL baseline runner")
    parser.add_argument("--template_path", required=True)
    parser.add_argument("--fixer_template_path", required=True)
    parser.add_argument("--input_json",  required=True,
                        help="Path to schema_linking JSON (e.g. dev_schema_linking_result.json)")
    parser.add_argument("--output_dir",  required=True,
                        help="Directory to save generated SQL results")
    parser.add_argument("--model_name",  required=True,
                        help="LLM name or checkpoint path for vLLM OpenAI API")
    parser.add_argument("--mschema_path", required=True,
                        help="Path to merged schema JSON (mschema)")
    parser.add_argument("--batch_size",  type=int, default=48,
                        help="Concurrent request batch size (default: 48)")
    parser.add_argument("--num_generations", type=int, default=1,
                        help="Number of SQL generations per question (default: 1)")
    args = parser.parse_args()
    return CFG(**vars(args))

CFG = parse_cfg()

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
    
        
class SimpleNL2SQL:
    def __init__(self, model_path):
        self.model_path = model_path
        with open(CFG.mschema_path, "r") as f:
            self.mschema = json.load(f)
        self.PROMPT_TEMPLATE = self.load_prompt_template(CFG.template_path)
        self.FIXER_TEMPLATE = self.load_prompt_template(CFG.fixer_template_path)

    def load_prompt_template(self, path: str) -> str:
        with open(path, encoding="utf-8") as f:
            return f.read()
    
    def extract_sql_from_llm(self, output: str) -> Optional[str]:
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
        return output.strip()
        
    def format_prompt(self, 
                      template: str, 
                      db_id: str, 
                      question: str,
                      evidence: str,
                      DDL: str) -> str:
        return template.format_map({
            "DATABASE_SCHEMA": self.mschema.get(db_id),
            "QUESTION": question,
            "HINT": evidence,
            "DDL": DDL,
        })
        
    def call_single_prompt(self, 
                           prompt: str, 
                          ) -> Tuple[str, str]:
        try:
            response = client.chat.completions.create(
                model=CFG.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=CFG.temperature,
                top_p=CFG.top_p,
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

    def call_single_prompt_with_fixer(self, 
                                      prompt:str, 
                                      item, 
                                      max_retry:int = 1)-> Tuple[List[str], List[str]]:
        db_path = build_db_path(item.get('db_id'))
        for _ in range(max_retry):
            think, content = self.call_single_prompt(prompt)
            sql = self.extract_sql_from_llm(content)
            ok, ret = _is_valid_sql(sql, db_path)
            if ok:
                return think, content
            else:
                prompt = self.FIXER_TEMPLATE.format_map({
                            "DATABASE_SCHEMA": self.mschema.get(item.get('db_id')),
                            "QUESTION": item.get('question'),
                            "HINT": item.get('evidence'),
                            "QUERY": sql,
                            "RESULT": ret
                        })
        logging.critical(f"循环{max_retry}次仍然未能通过有效性测试, sql:{sql}")
        return think, content
                
    def llm_batch_call_vllm(
        self,
        items
    ) -> Tuple[List[str], List[str]]:
        """并发提交 prompt 到 vLLM 的 OpenAI 接口"""
        thinking_results, content_results = [None] * len(items), [None] * len(items)
        prompts = [
            self.format_prompt(
                self.PROMPT_TEMPLATE,
                item.get('db_id'),
                item.get("question", ""),
                item.get("evidence", ""),
                item.get("DDL", ""),
            )
            for item in items
        ] 
        with ThreadPoolExecutor(max_workers=CFG.batch_size) as executor:
            future_to_index = {
                        executor.submit(self.call_single_prompt_with_fixer, prompt, item): i
                for i, (prompt, item) in enumerate(zip(prompts, items))
            }
            for future in as_completed(future_to_index):
                i = future_to_index[future]
                think, content = future.result()
                thinking_results[i] = think
                content_results[i] = content
    
        return thinking_results, content_results
        
    def generate_sql(self, items_in_batch: List[Dict], num_generations_per_item: int) -> List[Dict]:
        expanded_items_for_llm = []  # 用于存储扩展后的 item 列表，每个原始 item 重复 N 次
        expanded_prompts = []        # 用于存储对应的 prompt 列表
        original_item_metadata = []  # 存储每个扩展 item 对应的原始 item 信息和生成尝试次数 
        prompts = [
            self.format_prompt(
                self.PROMPT_TEMPLATE,
                item.get('db_id'),
                item.get("question", ""),
                item.get("evidence", ""),
                item.get("DDL", ""),
            )
            for item in items_in_batch
        ] 
        thoughts_list = [[] for _ in range(len(items_in_batch))]
        contents_list = [[] for _ in range(len(items_in_batch))]
        for _ in range(num_generations_per_item):
            thoughts_temp, contents_temp = self.llm_batch_call_vllm(items_in_batch)
            for i, (thought, content) in enumerate(zip(thoughts_temp, contents_temp)):
                thoughts_list[i].append(thought)
                contents_list[i].append(content)
        return [
            {
                "question": item.get("question"),
                "sql": [self.extract_sql_from_llm(content) for content in contents] if len(contents) > 1 else self.extract_sql_from_llm(contents[0]),
                "prompt": prompt,
                "think": thoughts,
                "content": contents
            }
            for item, prompt, thoughts, contents in zip(items_in_batch, prompts, thoughts_list, contents_list)
        ]
         

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
        
def worker_run(data):
    nl2sql = SimpleNL2SQL(CFG.model_name)
    output_path = os.path.join(CFG.output_dir, f"dev_sim_result.jsonl")
    with open(output_path, "w", encoding="utf-8") as fout:
        logging.info(f"len(data)={len(data)}, batch_size={CFG.batch_size}")
        batched_data = [data[i:i + CFG.batch_size] for i in range(0, len(data), CFG.batch_size)]
        for idx, item in enumerate(tqdm(batched_data, desc="generating"), 1):
            logging.info(f"流程{idx}, 处理{len(item)}个数据")
            sqls = nl2sql.generate_sql(item, CFG.num_generations)
            for sql in sqls:
                fout.write(json.dumps(sql, ensure_ascii=False) + "\n")

            # ---- 每 10 条立即落盘 ----
            if idx % 10 == 0:
                fout.flush()          # 把缓冲区刷到磁盘
                os.fsync(fout.fileno())
                tqdm.write(f"已写入 {idx}/{len(data)} 条 → {output_path}")
        fout.flush()          # 把缓冲区刷到磁盘
        os.fsync(fout.fileno())
    results = []
    with open(output_path, "r") as fin:
        for line in fin:
            if line.strip():
                results.append(json.loads(line))
    final_output_path = os.path.join(CFG.output_dir, f"dev_result.json")
    with open(final_output_path, "w") as fout:
        json.dump(results, fout, ensure_ascii=False, indent=2)
        
    print(f"全部完成，结果已保存到 {output_path}")
    
# ========== 主逻辑 ==========
def run_multi_gpu():
    os.makedirs(CFG.output_dir, exist_ok=True)
    with open(CFG.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    # data = random.sample(data, CFG.batch_size)  # 随机取 48 个样本
    # data = data
    worker_run(data)
    
    logging.info("程序执行完毕")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
    run_multi_gpu()
