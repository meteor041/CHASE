# schema_linker.py

import faiss
import numpy as np
import difflib
import torch
from typing import List, Tuple, Dict
from sentence_transformers import SentenceTransformer

class SchemaLinker:
    def __init__(self, embedding_model_name: str = "/home/yangliu26/all-MiniLM-L6-v2", use_gpu = False):
        self.model = SentenceTransformer(embedding_model_name)
        self.index = None
        self.index_mapping = []  # 保留原始字符串
        self.device = "cuda" if (torch.cuda.is_available() and use_gpu) else "cpu"

    def build_index(self, schema_info: Dict):
        """
        用数据库schema里的表名和列名建立LSH索引。
        """
        table_names = schema_info.get("table_names_original", [])
        column_infos = schema_info.get("column_names_original", [])

        self.index_mapping.clear()
        # for table in table_names:
        #     self.index_mapping.append((table, table))
        for table_idx, column in column_infos:
            if 0 <= table_idx < len(table_names):
                self.index_mapping.append((column,
                                           table_names[table_idx]))
        all_names = [item[0] for item in self.index_mapping]
        embeddings = self.model.encode(all_names, 
                                       convert_to_tensor=True,
                                       # normalize_embeddings=True,
                                       device=self.device)
        # print(embeddings.shape)
        dim = embeddings.shape[1]
        if self.device == "cuda":
            res = faiss.StandardGpuResources()
            cpu_i = faiss.IndexFlatIP(dim)
            self.index = faiss.index_cpu_to_gpu(res, 0, cpu_i)
            self.index.add(embeddings.detach())
        else:
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(embeddings.cpu().numpy())
        

    def search(self, 
               keywords: List[str], 
               top_k: int = 5
              ) -> List[List[Tuple[str, str, float]]]:
        """
        针对每个keyword，检索top_k个最相近的schema元素，返回列表（带得分）
        """
        results = []
        for kw in keywords:
            kw_emb = self.model.encode(kw, convert_to_tensor=True, device=self.device)
            if self.device == 'cuda':
                D, I = self.index.search(kw_emb.detach()[None, :], top_k)
            else:
                D, I = self.index.search(kw_emb.cpu().numpy()[None, :], top_k)  # 查询
            matches = []
            for idx, dist in zip(I[0], D[0]):
                if idx < len(self.index_mapping):
                    schema_item, table_name = self.index_mapping[idx]
                    edit_sim = self._edit_similarity(kw, schema_item)
                    combined_score = self._combine_score(dist, edit_sim)
                    matches.append((kw, schema_item, table_name, combined_score))
            results.append(sorted(matches, key=lambda x: -x[3]))  # 按最终得分降序
        return results

    def _edit_similarity(self, s1: str, s2: str) -> float:
        """
        计算编辑相似度（归一化Levenshtein）
        """
        return difflib.SequenceMatcher(None, s1.lower(), s2.lower()).ratio()

    def _combine_score(self, cos_sim: float, edit_sim: float) -> float:
        """
        将faiss距离和编辑相似度合并成最终得分。
        注意：faiss距离小是好事，相似度高是好事，所以负向处理
        """
        return cos_sim * 0.7 + edit_sim * 0.3  # 可以调整权重

if __name__ == "__main__":
    # 示例代码
    dummy_schema = {
        "table_names": ["Store", "Employee", "Product"],
        "column_names": [(0, "store_id"), (0, "store_name"), (1, "employee_id"), (1, "employee_name"), (2, "product_id"), (2, "product_name")]
    }

    linker = SchemaLinker()
    print("SchemaLinker 初始化完成")
    linker.build_index(dummy_schema)
    print("build index 完成")
    test_keywords = ["store", "employee", "product", "store name", "product id"]

    results = linker.search(test_keywords)
    print("search 完成")
    for keyword_results in results:
        print("\nKeyword Linking:")
        for kw, schema_item, table_name, score in keyword_results:
            print(f"Keyword: {kw} -> Schema: {schema_item}(Table: {table_name}), Score: {score:.4f}")

