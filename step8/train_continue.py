"""Latih adapter LoRA tahap kedua dari model yang sudah di-merge.

Model sumber adalah step5/merged_model, sehingga bobot hasil merge tahap
sebelumnya tetap menjadi titik awal training ini.
"""

from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


STEP_DIR = Path(__file__).resolve().parent
MERGED_MODEL_PATH = Path(
    "/Users/fajri/Projects/testmodel/step5/merged_model"
)
DATASET_PATH = STEP_DIR / "dataset" / "dataset.jsonl"
OUTPUT_PATH = STEP_DIR / "output"


if not (MERGED_MODEL_PATH / "config.json").is_file():
    raise FileNotFoundError(
        "Merged model tahap pertama tidak ditemukan di: "
        f"{MERGED_MODEL_PATH}\n"
        "Jalankan merge model pada step5 terlebih dahulu."
    )

if not DATASET_PATH.is_file():
    raise FileNotFoundError(f"Dataset tidak ditemukan di: {DATASET_PATH}")


print(f"Memuat merged model tahap pertama: {MERGED_MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(
    MERGED_MODEL_PATH,
    local_files_only=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MERGED_MODEL_PATH,
    torch_dtype=torch.float32,
    local_files_only=True,
)

model.config.use_cache = False
model.enable_input_require_grads()

dataset = load_dataset(
    "json",
    data_files=str(DATASET_PATH),
    split="train",
)

# Ini adalah adapter LoRA tahap kedua. Bobot model yang dimuat di atas sudah
# berisi hasil merge tahap pertama dan tetap dibekukan selama training ini.
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

training_config = SFTConfig(
    output_dir=str(OUTPUT_PATH),
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    max_length=256,
    logging_steps=1,
    save_strategy="epoch",
    gradient_checkpointing=True,
    report_to="none",
)

trainer = SFTTrainer(
    model=model,
    args=training_config,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=lora_config,
)

trainer.model.print_trainable_parameters()
trainer.train()

trainer.save_model(str(OUTPUT_PATH))
tokenizer.save_pretrained(str(OUTPUT_PATH))

print(f"Adapter LoRA tahap kedua disimpan di: {OUTPUT_PATH}")
