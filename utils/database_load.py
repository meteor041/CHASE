import json
import os

def convert_schema(json_path, output_path=None):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    db_dict = {}
    for db in data:
        db_id = db["db_id"]
        table_names = db["table_names"]
        column_names_original = db["column_names_original"]
        # 构建 table_name 到 column_names_original 的映射
        table_columns = {table: [] for table in table_names}
        for col in column_names_original:
            table_idx, col_name = col
            if table_idx >= 0:
                table = table_names[table_idx]
                table_columns[table].append(col_name)
        db_dict[db_id] = table_columns
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(db_dict, f, ensure_ascii=False, indent=2)
    return db_dict

if __name__ == "__main__":
    input_path = os.path.join(r'/home/yangliu26/data/train/train_tables.json')
    output_path = os.path.join(os.path.dirname(__file__), "converted_schema.json")
    result = convert_schema(input_path, output_path)
    print(f"转换完成，结果已保存到: {output_path}")