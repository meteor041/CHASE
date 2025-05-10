from pathlib import Path
import os
import json

def merge_results(output_dir: str, num_gpus: int, merged_file: str):
    results = []
    for i in range(num_gpus):
        path = Path(output_dir) / f"os_results_gpu{i}.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    results.append(json.loads(line))
    # logging.info(f"{results}")
    with open(Path(output_dir) / merged_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    # logging.info(f"合并成功,保存至{Path(output_dir) / merged_file}")

def merge_results2(output_dir: str, num_gpus: int, merged_file: str):
    results = []
    for i in range(num_gpus):
        path = Path(output_dir) / f"os_results_gpu{i}.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                results.extend(json.loads(f.read()))
    # logging.info(f"{results}")
    with open(Path(output_dir) / merged_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    # logging.info(f"合并成功,保存至{Path(output_dir) / merged_file}")
if __name__ == '__main__':
    merge_results2("/home/yangliu26/data/candidates", 4, "os_results.json")
    