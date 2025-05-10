import json


def original_to_pairwise_data(data):
    """
    将原始数据转换为pairwise数据
    """
    pairwise_datas = []
    with open(r"selection_agent_train_prompt.txt", "r", encoding='utf-8') as f:
        prompt_tpl = f.read()
    for i in range(len(data)):
        question = data[i]["question"]
        schema_str = data[i]["schema_str"]
        evidence = data[i]["evidence"]
        pos = data[i]["pos"]
        neg = data[i]["neg"]
        pos_res = data[i]["neg_result"]
        neg_res = data[i]["neg_result"]
        correct_answer = data[i]["correct_answer"]
        template_vars = {
            "DATABASE SCHEMA": schema_str,
            "QUESTION": question,
            "HINT": evidence,
            "CANDIDATE A QUERY": pos if correct_answer == "A" else neg,
            "CANDIDATE B QUERY": neg if correct_answer == "A" else pos,
            "CANDIDATE A RESULT": pos_res if correct_answer == "A" else neg_res,
            "CANDIDATE B RESULT": neg_res if correct_answer == "A" else pos_res
        }
        prompt = prompt_tpl.format(**template_vars)
        pairwise_datas.append({
            "prompt": prompt,
            "label": correct_answer,
        })
            
    return pairwise_datas

if __name__ == "__main__":
    with open(r"/home/yangliu26/data/pairwise/final_result2_merged.json", "r", encoding='utf-8') as f:
        data = json.load(f)
    # data = data[:10]
    pairwise_datas = original_to_pairwise_data(data)
    with open(r"pairwise_datas.json", "w") as f:
        json.dump(pairwise_datas, f, ensure_ascii=False, indent=2)