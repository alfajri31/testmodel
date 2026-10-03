"""Latih adapter LoRA Step 9 dari base model, tanpa merge."""

from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


STEP_DIR = Path(__file__).resolve().parent
BASE_MODEL_PATH = Path("/Users/fajri/Projects/mytestmodel")
DATASET_PATH = STEP_DIR / "dataset" / "dataset.jsonl"
OUTPUT_PATH = STEP_DIR / "output"

if not (BASE_MODEL_PATH / "config.json").is_file():
    raise FileNotFoundError(f"Base model tidak ditemukan: {BASE_MODEL_PATH}")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH, torch_dtype=torch.float32, local_files_only=True
)
model.config.use_cache = False
model.enable_input_require_grads()
dataset = load_dataset("json", data_files=str(DATASET_PATH), split="train")

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
print(f"Adapter Step 9 disimpan di: {OUTPUT_PATH}")
