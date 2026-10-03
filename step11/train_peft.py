"""Latih adapter LoRA Step 11 dari base model, tanpa merge."""

from pathlib import Path
import json

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


STEP_DIR = Path(__file__).resolve().parent
BASE_MODEL_PATH = Path("/Users/fajri/Projects/mytestmodel")
DATASET_PATH = STEP_DIR / "dataset" / "dataset.jsonl"
RESPONSE_DATASET_PATH = STEP_DIR / "dataset" / "response.jsonl"
STYLE_DATASET_PATH = STEP_DIR / "dataset" / "response_style_long.example.jsonl"
MULTI_VALUE_DATASET_PATH = STEP_DIR / "dataset" / "multi_value.example.jsonl"
OUTPUT_PATH = STEP_DIR / "output"

if not (BASE_MODEL_PATH / "config.json").is_file():
    raise FileNotFoundError(f"Base model tidak ditemukan: {BASE_MODEL_PATH}")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH, torch_dtype=torch.float32, local_files_only=True
)
model.config.use_cache = False
model.enable_input_require_grads()
records = [
    json.loads(line)
    for line in DATASET_PATH.read_text().splitlines()
    if line.strip()
]
if RESPONSE_DATASET_PATH.is_file():
    records.extend(
        {
            "messages": [
                {"role": "user", "content": item["user"]},
                {"role": "assistant", "content": item["assistant"]},
            ]
        }
        for item in (
            json.loads(line)
            for line in RESPONSE_DATASET_PATH.read_text().splitlines()
            if line.strip()
        )
    )
if STYLE_DATASET_PATH.is_file():
    for line in STYLE_DATASET_PATH.read_text().splitlines():
        if not line.strip():
            continue
        style_record = json.loads(line)
        records.append(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": f"Gunakan gaya {style_record['style']}.",
                    },
                    {"role": "user", "content": style_record["user"]},
                    {"role": "assistant", "content": style_record["assistant"]},
                ]
            }
        )
if MULTI_VALUE_DATASET_PATH.is_file():
    records.extend(
        json.loads(line)
        for line in MULTI_VALUE_DATASET_PATH.read_text().splitlines()
        if line.strip()
    )
dataset = Dataset.from_list(records)
print(f"Total contoh training Step 11: {len(dataset)}")

trainer = SFTTrainer(
    model=model,
    args=SFTConfig(
        output_dir=str(OUTPUT_PATH),
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        max_length=256,
        save_strategy="epoch",
        save_total_limit=3,
        gradient_checkpointing=True,
        report_to="none",
    ),
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ),
)
trainer.train()
trainer.save_model(str(OUTPUT_PATH))
tokenizer.save_pretrained(str(OUTPUT_PATH))
print(f"Adapter Step 11 disimpan di: {OUTPUT_PATH}")
