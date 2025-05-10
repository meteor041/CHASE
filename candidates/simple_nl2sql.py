"""
NL2SQL直接实现
检测模型baseline
"""
import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import re
import time
import logging
from pathlib import Path
from typing import Dict, Any, List

class SimpleNL2SQL:
    def __init__(self, model_path: str = r"/home/yangliu26/qwen3-8b"):
        logging.info("正在加载模型...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left",
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16,
            max_memory={0: "30GiB"} 
        )
        self.model.generation_config.enable_thinking = False  
        logging.info("模型加载完成")

    def extract_sql(self, text: str) -> str:
        logging.info(f"text: {text}")
        """从生成的文本中提取SQL"""
        pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
        match = re.search(pattern, text, flags=re.IGNORECASE)
        
        if not match:
            logging.info("no match")
            return text                 # 没找到代码块
    
        json_str = match.group(1).strip()
        # logging.info(f"json_str: {json_str}")
        try:
            obj = json.loads(json_str)  # ❷ 解析 JSON
        except json.JSONDecodeError as e:
            logging.info(f"[JSON 解析失败] {e}")
            logging.info(f"原语句 {json_str}")
            return json_str
        except Exception as e:
            logging.info(f"[其他失败] {e}")
            logging.info(f"原语句 {json_str}")
        # logging.info(f"obj: {obj}")
        # 允许大小写差异
        try:
            sql = obj.get("sql") or obj.get("SQL")
        except Exception as e:
            logging.info("JSON 中未找到 'sql'或者'SQL` 字段 {obj}", )
        if not sql:
            logging.info("JSON 中未找到 'sql'或者'SQL` 字段")
            return obj
        return sql.strip()
    
    def generate_sql(self, gpu_id: int, items) -> List[str]:
        """批量生成多条SQL"""     
        messages = [
            {
                "role": "user",
                "content": (
                    "You are a senior SQL generator.\n"
                    "You MUST respond only with a JSON code block like this:\n"
                    "```json\n{\"sql\": \"...\"}\n```\n"
                    "No explanations or text outside."
                    f"Schema:\n{item.get('schema_linking')}\n\n"
                    f"Hint:\n{item.get('evidence')}\n\n"
                    f"Question:\n{item.get('question')}"
                )
            }
            for item in items
        ] 
        kwargs = dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
        text_batch = [self.tokenizer.apply_chat_template([m], **kwargs) for m in messages]
        model_inputs = self.tokenizer(
            text_batch, 
            padding=True, 
            truncation=True,         # ✅ 若超过模型 max length 则截断
            return_tensors="pt"
        ).to(self.model.device)
        
        # conduct text completion
        outputs = self.model.generate(
            **model_inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None
        )
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
            self.tokenizer.decode(output_id[index:], skip_special_tokens=True).strip("\n") 
            for index, output_id in zip(indexes, output_ids)
        ]

        return [
            {
                "question": item.get("question"),
                "sql": self.extract_sql(content),
                "prompt": message["content"]
            }
            for item, content, message in zip(items, contents, messages)
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
        
def merge_results(output_dir: str, num_gpus: int, merged_file: str):
    results = []
    for i in range(num_gpus):
        path = Path(output_dir) / f"result_gpu_{i}.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    results.append(json.loads(line))
    logging.info(f"{results}")
    with open(Path(output_dir) / merged_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logging.info(f"合并成功,保存至{Path(output_dir) / merged_file}")
        
def worker_run(gpu_id:int, data, CFG):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    nl2sql = SimpleNL2SQL()
    output_path = os.path.join(CFG.output_dir, f"result_gpu_{gpu_id}.json")
    with open(output_path, "w", encoding="utf-8") as fout:
        logging.info(f"len(data)={len(data)}, batch_size={CFG.batch_size}")
        batched_data = [data[i:i + CFG.batch_size] for i in range(0, len(data), CFG.batch_size)]
        for idx, item in enumerate(tqdm(batched_data, desc="generating"), 1):
            logging.info(f"流程{idx}, 处理{len(item)}个数据")
            sqls = nl2sql.generate_sql(
                gpu_id, item
            )
            for sql in sqls:
                fout.write(json.dumps(sql, ensure_ascii=False) + "\n")

            # ---- 每 10 条立即落盘 ----
            if idx % 10 == 0:
                fout.flush()          # 把缓冲区刷到磁盘
                os.fsync(fout.fileno())
                tqdm.write(f"已写入 {idx}/{len(data)} 条 → {output_path}")
        fout.flush()          # 把缓冲区刷到磁盘
        os.fsync(fout.fileno())
    print(f"全部完成，结果已保存到 {output_path}")
    
# ========== 主逻辑 ==========
def run_multi_gpu():
   # 配置参数
    class CFG:
        input_json = "/home/yangliu26/data/train/schema_linking_result.json"
        output_dir = "/home/yangliu26/CHASE/candidates/sim_results"
        num_gpus = 4
        batch_size = 48
    logging.info(f"启动了{CFG.num_gpus}个GPU")
    os.makedirs(CFG.output_dir, exist_ok=True)
    with open(CFG.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    # data = data[:100]
    chunks = split_data(data, CFG.num_gpus)
    
    for i, chunk in enumerate(chunks):
        logging.info(f"分配给 GPU-{i}: {len(chunk)} 条样本")
    import multiprocessing as mp
    # ctx = mp.get_context("spawn")
    # 初始化模型
    processes = []
    for gpu_id in range(CFG.num_gpus):
        p = mp.Process(
            target=worker_run, args=(gpu_id, chunks[gpu_id], CFG))
        p.start()
        processes.append(p)
        if gpu_id != CFG.num_gpus - 1:
            time.sleep(10)  # ✅ 每个进程启动之间隔 10 秒，避免瞬间爆显存

    for p in processes:
        p.join()
    merge_results(CFG.output_dir, CFG.num_gpus, "qp_result_merged.json")
    logging.info("[Main] 所有 GPU 子进程执行完毕")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
    run_multi_gpu()
