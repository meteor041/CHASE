import json
from schema_parser import load_schema
from schema_linker import SchemaLinker
from async_keyword_extractor import KeywordExtractor
from typing import Dict, Any
import os
import asyncio

# 训练集
TRAIN_JSON_PATH = r"/home/yangliu26/data/train/train.json"
SCHEMA_JSON_PATH = r"/home/yangliu26/data/train/train_tables.json"
# 验证集
# TRAIN_JSON_PATH = r"/home/yangliu26/data/dev/dev.json"
# SCHEMA_JSON_PATH = r"/home/yangliu26/data/dev/dev_tables.json"
MODEL_PATH = r"/home/yangliu26/qwen3-8b"

# 加载schema信息
def get_schema_map(schema_json_path: str) -> Dict[str, Any]:
    schema = load_schema(schema_json_path)
    db_schema_map = {}
    for db in schema:
        db_id = db["db_id"] if isinstance(db, dict) and "db_id" in db else db.get("db_id", "")
        db_schema_map[db_id] = db
    return db_schema_map

async def async_main():
    with open(TRAIN_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 取前10个数据作为测试
    # data = data[:40]
    schema_map = get_schema_map(SCHEMA_JSON_PATH)
    print("---schema_map提取完成---")
    extractor = KeywordExtractor(MODEL_PATH)
    print("---extractor建立完成---")
    linker = SchemaLinker()
    print("---linker建立完成---")
    # 提取出所有问题
    questions = [sample["question"] for sample in data]
    print("---问题提取完成---")
    # 提取出每个问题的关键词
    print("---开始提取关键词---")
    all_keywords = await extractor.batch_extract(questions)
    print("---提取关键词完成---")
    results = []
    for idx, (sample, keywords) in enumerate(zip(data, all_keywords), 1):
        db_id = sample["db_id"]
        question = sample["question"]
        evidence = sample["evidence"]
        schema_info = schema_map.get(db_id, {})
        # schema-linking
        linker.build_index(schema_info)
        linking_results = linker.search(keywords)
        # 格式化
        formatted_linking = {}
        for matches in linking_results:
            for kw, schema_item, table_name, col_type, is_pk, fk_pair, score, this_idx in matches:
                formatted_linking.setdefault(table_name, [])
                # fk_target = None
                # if fk_pair is not None:
                #     child_idx, parent_idx = fk_pair
                #     ref_idx = parent_idx if linker.index_mapping[child_idx][0] == schema_item else child_idx
                #     ref_col, ref_tbl, *_ = linker.index_mapping[ref_idx]
                #     fk_target = {"table":ref_tbl, "column": ref_col}
                
                formatted_linking[table_name].append(
                    {
                        "keyword": kw,
                        "column": schema_item,
                        "type": col_type or "TEXT",
                        "is_primary": is_pk,
                        "foreign_key": None,
                        "score": float(score)
                    }
                )
                # -- 若存在外键，再补充引用表/列 -------------------------------
                if fk_pair is not None:
                    child_idx, parent_idx = fk_pair
                    ref_idx = parent_idx if this_idx == child_idx else child_idx
                    ref_col, ref_tbl, ref_type, ref_is_pk, _ = linker.index_mapping[ref_idx]

                    # 更新当前列的外键信息
                    formatted_linking[table_name][-1]["foreign_key"] = {
                        "table":  ref_tbl,
                        "column": ref_col
                    }

                    # 把被引用列也放进 formatted_linking，以确保生成 DDL 时存在
                    formatted_linking.setdefault(ref_tbl, [])
                    if all(col["column"] != ref_col for col in formatted_linking[ref_tbl]):
                        formatted_linking[ref_tbl].append({
                            "keyword": "<ref>",      # 占位
                            "column":  ref_col,
                            "type":    ref_type or "TEXT",
                            "is_primary": ref_is_pk, # 可能是主键
                            "foreign_key": None,
                            "score": 0.0
                        })

        # ---------- 基于 formatted_linking 生成 DDL_schema ----------
        ddl_parts = []
        for tbl, cols in formatted_linking.items():
            col_defs, pk_cols, fk_defs = [], [], []
            used = set()
            for c in cols:
                if c['column'] in used:
                    continue
                used.add(c['column'])
                col_defs.append(f"  {c['column']} {c['type']}")
                if c["is_primary"]:
                    pk_cols.append(c["column"])
                if c["foreign_key"]:
                    fk_defs.append(
                        f"  FOREIGN KEY ({c['column']}) "
                        f"REFERENCES {c['foreign_key']['table']}({c['foreign_key']['column']})"
                    )
            if pk_cols:
                col_defs.append(f"  PRIMARY KEY ({', '.join(pk_cols)})")
            col_defs.extend(fk_defs)
            ddl_parts.append(f"CREATE TABLE {tbl} (\n" + ",\n".join(col_defs) + "\n);")
        DDL_schema = "\n\n".join(ddl_parts)

        results.append({
            "db_id": db_id,
            "question": question,
            "evidence": evidence,
            "keywords": keywords,
            "schema_linking": formatted_linking,
            "DDL": DDL_schema
        })
        
        # --- 每 10 条立即保存 ---
        if idx % 10 == 0:
            flush_path = os.path.join(os.path.dirname(__file__), "results/schema_linking_result.json")
            with open(flush_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"已处理 {idx}/{len(data)}，中间结果写入 {flush_path}")
        
    # 输出结果到当前文件目录下的schema_linking_result.json
    out_path = os.path.join(os.path.dirname(__file__), "results/schema_linking_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Schema linking结果已保存到: {out_path}")

if __name__ == "__main__":
    asyncio.run(async_main())