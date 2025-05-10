from typing import List, Union
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import asyncio
import json
import re
from tqdm.asyncio import tqdm

class KeywordExtractor:
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
            enable_thinking=False,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True
        )
        # 保险起见，再显式关一次 generation_config
        self.model.generation_config.enable_thinking = False  
        
        self.model.eval()
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def build_prompt(self, question: str, prompt_template: str = None) -> str:
        if prompt_template is None:
            with open(r"/home/yangliu26/CHASE/template/schema_linking_template.txt", "r") as f:
                prompt_template = f.read()
        return prompt_template.format(question=question)

    def extract_json_block(self, text: str) -> dict | None:
        """
        从文本中抓取第一个合法的 JSON 对象并反序列化。
        返回 dict；若没找到返回 None。
        """
        # 非贪婪匹配最外层 {...}
        match = re.search(r"\{[\s\S]*?\}", text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                return None
        return None

    
    def clean_output(self, result: str) -> str:
        # 去除包裹的 ```json ``` 和其他markdown
        cleaned = re.sub(r"^```(?:json)?\s*([\s\S]*?)\s*```$", r"\1", result.strip(), flags=re.DOTALL)
        # 尝试提取第一个出现的大括号 {} 或数组 []
        match = re.search(r"(\{[\s\S]*?\}|\[[\s\S]*?\])", cleaned)
        if match:
            print(f"匹配成功:{match.group(0)}")
            return match.group(0)
        print(f"匹配失败:{cleaned}")
        return cleaned  # fallback

    def parse_keywords(self, cleaned: str) -> List[str]:
        try:
            parsed = self.extract_json_block(cleaned)
            if isinstance(parsed, list):
                parsed = parsed[0]
            return parsed.get("keywords", [])
        except json.JSONDecodeError as e:
            print(f"[Parsing Failed] Content: {cleaned}")
            print(f"[Error Info] {e}")
            return []

    async def extract_keywords(self, question: str, prompt_template: str = None, timeout_sec: int = 20) -> List[str]:
        prompt = self.build_prompt(question, prompt_template)
        # inputs = self.tokenizer(
        #     prompt,
        #     max_length=2048,
        #     truncation=True,
        #     return_tensors="pt"
        # ).to(self.model.device)

        try:
            output = await asyncio.wait_for(self._generate(prompt), timeout=timeout_sec)
            # print(output)
            cleaned = self.clean_output(output)
            keywords = self.parse_keywords(cleaned)
            return keywords
        except asyncio.TimeoutError:
            print("[Timeout Warning] Response generation timeout, skipping this question.")
            return []

    async def _generate(self, prompt) -> str:
        messages = [
            {
                "role": "user", "content": prompt
            }
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False # 一定要关闭深度思考
        )
        model_inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        # conduct text completion
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=128
        )
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
        
        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0
        
        # thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        
        # print("thinking content:", thinking_content)
        # print("content:", content)
    
        return content 

    # async def batch_extract(self, questions: List[str], prompt_template: str = None) -> List[List[str]]:
    #     tasks = [self.extract_keywords(q, prompt_template) for q in questions]
    #     results = await asyncio.gather(*tasks)
    #     return results
    

    async def batch_extract(self, questions: List[str], 
                            prompt_template: str = None, 
                            max_concurrency: int = 12) -> List[List[str]]:
        """
        并发抽取关键词，保持返回顺序，并用 tqdm 显示进度
        """
        sem = asyncio.Semaphore(max_concurrency)
        async def _wrap(idx: int, coro: asyncio.Future) -> tuple[int, list[str]]:
            async with sem:
                """包装每个 coroutine，把索引和结果一起返回"""
                try:
                    res = await coro
                except Exception as e:
                    print(f"[Error] {e}")
                    res = []
                return idx, res
    
        # 先把每个问题的提取任务封装好
        tasks = [
            asyncio.create_task(_wrap(i, self.extract_keywords(q, prompt_template)))
            for i, q in enumerate(questions)
        ]
    
        results: list[list[str] | None] = [None] * len(questions)
    
        # 用 as_completed 驱动进度条
        for fut in tqdm(asyncio.as_completed(tasks),
                        total=len(tasks),
                        desc="Extracting keywords"):
            idx, res = await fut         # 拿到索引和结果
            results[idx] = res           # 放回正确位置
    
        # 类型守恒：全为 list[str]
        return results                  # type: ignore[return-value]

if __name__ == "__main__":
    model_path = "/home/yangliu26/qwen3-8b"
    extractor = KeywordExtractor(model_path)

    async def main():
        questions = [
            "Which stores have sales exceeding 50,000?",
            "What are the popular restaurants in Beijing in 2023?",
            "List the training institutions that offer courses in Python and deep learning."
        ]
        results = await extractor.batch_extract(questions)
        for q, keywords in zip(questions, results):
            print(f"Question: {q}\nKeywords: {keywords}\n")

    asyncio.run(main())
