from typing import List
import asyncio
import json
import re
from tqdm.asyncio import tqdm
from openai import OpenAI

with open(r"/home/yangliu26/CHASE/template/schema_linking_template.txt", "r") as f:
    PROMPT_TEMPLATE = f.read()
    
class KeywordExtractor:
    def __init__(self, model_path: str):
        self.client = OpenAI(
            api_key="EMPTY",
            base_url="http://localhost:8000/v1"
        )
        self.model_name = model_path

    def build_prompt(self, question: str) -> str:
        return PROMPT_TEMPLATE.format(question=question)

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
            if not parsed:
                return []
            if isinstance(parsed, list):
                parsed = parsed[0]
            return parsed.get("keywords", [])
        except json.JSONDecodeError as e:
            print(f"[Parsing Failed] Content: {cleaned}")
            print(f"[Error Info] {e}")
            return []

    async def extract_keywords(self, question: str, timeout_sec: int = 20) -> List[str]:
        prompt = self.build_prompt(question)

        try:
            output = await asyncio.wait_for(self._generate(prompt), timeout=timeout_sec)
            cleaned = self.clean_output(output)
            keywords = self.parse_keywords(cleaned)
            return keywords
        except asyncio.TimeoutError:
            print("[Timeout Warning] Response generation timeout, skipping this question.")
            return []

    async def _generate(self, prompt: str) -> str:
        response = await asyncio.to_thread(          # 非阻塞包装
            self.client.chat.completions.create,
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=128
        )
        text =  response.choices[0].message.content.strip()
        if "</think>" in text:
            think, content = text.split("</think>", 1)
            think = think.replace("<think>", "").strip()
            content = content.strip()
        else:
            think = ""
            content = text.strip()
    
        return content
    

    async def batch_extract(self, questions: List[str], 
                            max_concurrency: int = 64) -> List[List[str]]:
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
            asyncio.create_task(_wrap(i, self.extract_keywords(q)))
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
