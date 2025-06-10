# CHASE-SQL: 多路径推理和偏好优化候选选择的Text-to-SQL实现

本项目是对论文《CHASE-SQL: MULTI-PATH REASONING AND PREFERENCE OPTIMIZED CANDIDATE SELECTION IN TEXT-TO-SQL》的复现工作，主要实现了基于多种提示词工程方法生成SQL候选，并通过LLM选择最优SQL的方法。

## 项目结构

```
├── candidates2/        # 使用不同方法执行Text-to-SQL生成
├── pairwise/           # SQL比较和模型训练
├── schema_linking/     # 关键词提取和数据库链接
├── template/           # 提示词工程的提示词模板
├── utils/              # 工具函数
```

## 模块功能

### candidates2

该模块实现了多种Text-to-SQL生成方法，包括：

1. **DC-COT (Divide-and-Conquer Chain-of-Thought)**：将复杂问题分解为子问题，为每个子问题生成伪SQL查询，然后组合成最终SQL查询。
2. **QP-COT (Query Plan Chain-of-Thought)**：利用查询执行计划概念，模拟数据库引擎执行查询的步骤来生成SQL。
3. **OS (Online Synthetic)**：动态生成合成的问题-SQL对，增强提供给LLM的少样本示例。
4. **Simple**：基础的Text-to-SQL生成方法。

#### 执行命令

```bash
# DC-COT方法
python candidates2/run_nl2sql.py \
  --template_path template/template_generate_candidate_one.txt \
  --fixer_template_path template/template_query_fixer.txt \
  --input_json data/schema_linking/dev_schema_linking_result.json \
  --output_dir candidates2/result/cot_result \
  --model_name <model_path> \
  --mschema_path data/mschema/dev_mschemas.json \
  --num_generations 1 \
  --batch_size 32

# QP-COT方法
python candidates2/run_nl2sql.py \
  --template_path template/template_generate_candidate_two.txt \
  --fixer_template_path template/template_query_fixer.txt \
  --input_json data/schema_linking/dev_schema_linking_result.json \
  --output_dir candidates2/result/qp_result \
  --model_name <model_path> \
  --mschema_path data/mschema/dev_mschemas.json \
  --num_generations 1 \
  --batch_size 32

# OS方法
python candidates2/online_synthetic.py \
  --input_json data/schema_linking/dev_schema_linking_result.json \
  --output_dir candidates2/result/os_result \
  --model_name <model_path> \
  --mschema_path data/mschema/dev_mschemas.json \
  --num_generations 1 \
  --batch_size 32

# Simple方法
python candidates2/run_nl2sql.py \
  --template_path template/template_generate_candidate_simple.txt \
  --fixer_template_path template/template_query_fixer.txt \
  --input_json data/schema_linking/dev_schema_linking_result.json \
  --output_dir candidates2/result/sim_result \
  --model_name <model_path> \
  --mschema_path data/mschema/dev_mschemas.json \
  --num_generations 1 \
  --batch_size 32
```

### pairwise

该模块实现了SQL比较和模型训练功能：

1. **compare_sql.py**：使用LLM从多个候选SQL中筛选最优SQL，实现了锦标赛式比较方法。
2. **db_utils.py**：实现与数据库的连接和访问，包括SQL执行和结果比较。
3. **pairwise_train_lora.py**：实现LoRA方法微调大模型，用于SQL选择任务。
4. **validate_multi.py**：实现多线程验证Text-to-SQL生成的SQL与gold SQL比较执行结果是否正确。

#### 执行命令

```bash
# 验证生成的SQL与gold SQL的执行结果
python pairwise/validate_multi.py \
  --gold data/train/train.json \
  --baseline data/candidates_vllm/sim_result.json \
  --preds data/candidates_vllm/cot_result.json \
  data/candidates_vllm/os_results.json \
  data/candidates_vllm/qp_result.json \
  --names cot os qp \
  --out_dir pairwise/result_vllm \
  --single_timeout 300 \
  --workers 32
```

### schema_linking

该模块实现了Text-to-SQL中的关键词提取和数据库链接功能：

1. **async_keyword_extractor.py**：使用LLM从自然语言问题中提取关键词，支持异步批处理。
2. **schema_linker.py**：使用局部敏感哈希（LSH）从数据库中检索与提取的关键词在句法上相似的值，并基于嵌入相似度和编辑距离进行重新排序。
3. **schema_linking_main.py**：整合关键词提取和schema链接的主流程。

#### 执行命令

```bash
# 运行schema_linking处理
python schema_linking/schema_linking_main.py \
  --input_json data/train/train.json \
  --output_json data/schema_linking/train_schema_linking_result.json \
  --schema_json data/train/train_schema.json \
  --model_path <model_path>
```

### template

该目录包含各种提示词工程的模板：

1. **template_generate_candidate_one.txt**：DC-COT方法的提示词模板。
2. **template_generate_candidate_two.txt**：QP-COT方法的提示词模板。
3. **template_generate_candidate_three.txt**：另一种Text-to-SQL生成方法的模板。
4. **template_generate_candidate_simple.txt**：简单方法的提示词模板。
5. **online_synthetic1.txt/2.txt/3.txt**：OS方法的提示词模板。
6. **selection_agent_train_prompt.txt**：SQL选择代理的训练提示词。
7. **sql_comparison_template.txt**：SQL比较的提示词模板。

### utils

该目录包含各种工具函数：

1. **database_load.py**：加载数据库相关功能。
2. **jsonl_to_json.py**：将JSONL文件转换为JSON文件。
3. **merge.py**：合并多个结果文件。
4. **original_to_pairwise_data.py**：将原始数据转换为pairwise训练数据。

#### 执行命令

```bash
# 将JSONL文件转换为JSON文件
python utils/jsonl_to_json.py \
  --input_file results.jsonl \
  --output_file results.json

# 合并多个结果文件
python utils/merge.py \
  --output_dir results/ \
  --num_gpus 4 \
  --merged_file merged_results.json
```

## 完整流程

1. 使用schema_linking提取关键词并链接到数据库schema
2. 使用candidates2中的不同方法生成SQL候选
3. 使用pairwise中的compare_sql从候选中选择最优SQL
4. 使用pairwise中的validate_multi验证生成SQL的正确性
5. 使用pairwise_train_lora微调模型以提高SQL选择性能

## 环境要求

- Python 3.8+
- PyTorch
- SQLite
- FAISS
- Sentence-Transformers
- OpenAI API或本地LLM服务