#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
将JSONL文件转换为格式化的JSON文件

功能：
1. 读取JSONL文件（每行一个JSON对象）
2. 将所有JSON对象合并为一个列表
3. 输出格式化的JSON文件，便于阅读
"""

import json
from pathlib import Path
from typing import List, Dict, Any

def convert_jsonl_to_json(input_file: str, output_file: str) -> None:
    """
    将JSONL文件转换为格式化的JSON文件
    
    Args:
        input_file: JSONL文件路径
        output_file: 输出JSON文件路径
    """
    # 读取JSONL文件
    results: List[Dict[str, Any]] = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                # 解析每一行的JSON对象
                json_obj = json.loads(line.strip())
                results.append(json_obj)
            except json.JSONDecodeError as e:
                print(f"警告：无法解析的行: {line.strip()}")
                print(f"错误信息: {e}")
                continue
    
    # 将结果写入格式化的JSON文件
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 转换完成！")
    print(f"📄 输入文件: {input_file}")
    print(f"📄 输出文件: {output_file}")
    print(f"📊 共处理 {len(results)} 条记录")

def main():
    # 设置输入输出路径
    input_file = r"e:\code\CHASE_work\CHASE_work\final_result2_merged.jsonl"
    output_file = r"e:\code\CHASE_work\CHASE_work\utils\final_result2_merged.json"
    
    # 执行转换
    convert_jsonl_to_json(input_file, output_file)

if __name__ == '__main__':
    main()