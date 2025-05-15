import json
from typing import Dict, Any

def load_schema(schema_json_path: str) -> Dict[str, Any]:
    """
    加载数据库schema信息，返回结构化的表、列等信息。
    """
    with open(schema_json_path, 'r', encoding='utf-8') as f:
        schema = json.load(f)
    return schema

if __name__ == "__main__":
    schema_path = r"E:\code\CHASE_NL2SQL\data\train\train_tables.json"
    schema = load_schema(schema_path)
    print(f"Loaded schema for {len(schema)} databases.")
    print(schema[:3])