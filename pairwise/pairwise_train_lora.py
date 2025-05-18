from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import torch
import json
import os
# import evaluate
import numpy as np

# 格式化pairwise示例
def format_pairwise_example(example):
    return {
        "text": example['prompt'],
        "label": 0 if example["label"] == "A" else 1
    }
    
# 加载 JSON 数据集
def load_json_dataset(path):
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    return Dataset.from_list([format_pairwise_example(ex) for ex in raw])

# # 计算评估指标
# def compute_metrics(eval_pred):
#     accuracy = evaluate.load("accuracy")
#     predictions, labels = eval_pred
#     predictions = predictions.argmax(axis=1)
#     return accuracy.compute(predictions=predictions, references=labels)

# 计算评估指标（纯本地实现，不依赖 evaluate）
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = (preds == labels).mean().item()
    return {"accuracy": acc}

# 主函数
def main():
    # 设置路径和参数
    data_path = "/home/yangliu26/CHASE/utils/pairwise_datas.json"
    model_name = "/data/Qwen2.5-32B-Instruct"
    output_dir = "./pairwise_selector_model/Qwen2.5-32B-Instruct-lora/"
    
    # LoRA配置
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,                     # LoRA矩阵的秩
        lora_alpha=32,            # LoRA的缩放参数
        lora_dropout=0.1,         # LoRA层的dropout率
        bias="none",              # 是否训练偏置项
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # 需要应用LoRA的模块
    )
    
    print("正在加载数据集...")
    # 加载和分割数据集
    dataset = load_json_dataset(data_path)
    dataset = dataset.train_test_split(test_size=0.1)
    
    print("正在加载模型和分词器...")
    # 加载分词器和模型
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )
    
    # 应用LoRA配置
    print("正在应用LoRA配置...")
    model = get_peft_model(base_model, lora_config)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.print_trainable_parameters()  # 打印可训练参数比例
    
    # 编码函数
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)
    
    print("正在处理数据集...")
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    
    print("训练集大小:", len(tokenized_dataset["train"]))
    print("测试集大小:", len(tokenized_dataset["test"]))
    
    # 初始化分布式训练（如果需要）
    if not torch.distributed.is_initialized():
        try:
            import deepspeed
            deepspeed.init_distributed()
            print(f"✅ 手动初始化完成: Rank {torch.distributed.get_rank()}")
        except Exception as e:
            print(f"❌ DeepSpeed 手动初始化失败: {e}")
    
    print("正在初始化训练参数...")
    # 训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=4,  # 使用LoRA可以增加批量大小
        per_device_eval_batch_size=4,
        num_train_epochs=5,
        bf16=True,
        logging_dir="./logs",
        logging_steps=10,
        learning_rate=5e-5,  # LoRA通常可以使用更高的学习率
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        ddp_find_unused_parameters=False,
        deepspeed="/home/yangliu26/CHASE/pairwise/ds_config_lora.json",  # 该文件存储LoRA优化的DeepSpeed配置
        gradient_accumulation_steps=2,  # 梯度累积
        warmup_ratio=0.1,  # 预热比例
    )
    
    print("⚙️ 正在初始化 Trainer...")
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    print("🔥 开始训练...")
    trainer.train()
    print("✅ 训练完成")
    
    # 保存最终模型
    print("💾 保存模型...")
    trainer.save_model(output_dir)
    
    # 评估模型
    print("📊 评估模型...")
    eval_results = trainer.evaluate()
    print(f"评估结果: {eval_results}")
    
    print(f"模型已保存到 {output_dir}")

if __name__ == "__main__":
    main()