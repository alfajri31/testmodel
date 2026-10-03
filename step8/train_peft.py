"""Latih adapter LoRA standalone dari base model."""

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


def choose_resume_checkpoint():
    """Tawarkan resume interaktif jika checkpoint training sudah tersedia."""
    checkpoints = sorted(
        OUTPUT_PATH.glob("checkpoint-*"),
        key=lambda path: int(path.name.removeprefix("checkpoint-"))
        if path.name.removeprefix("checkpoint-").isdigit()
        else -1,
    )
    if not checkpoints:
        return None

    print("Checkpoint yang tersedia:")
    for checkpoint in checkpoints:
        print(f"  {checkpoint.name}")
    print("Masukkan nomor checkpoint untuk melanjutkan, atau 0 untuk mulai dari awal.")

    while True:
        selected = input("Checkpoint [0]: ").strip() or "0"
        if selected == "0":
            return None
        if selected.isdigit():
            selected_path = OUTPUT_PATH / f"checkpoint-{selected}"
            if selected_path in checkpoints:
                return selected_path
        print("Pilihan tidak valid. Masukkan nomor checkpoint yang tersedia atau 0.")


if not (BASE_MODEL_PATH / "config.json").is_file():
    raise FileNotFoundError(
        "Base model tidak ditemukan di: "
        f"{BASE_MODEL_PATH}"
    )

if not DATASET_PATH.is_file():
    raise FileNotFoundError(f"Dataset tidak ditemukan di: {DATASET_PATH}")

RESUME_CHECKPOINT = choose_resume_checkpoint()


print(f"Memuat base model standalone: {BASE_MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    local_files_only=True,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
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

# Adapter LoRA dilatih di atas base model. Bobot base model tetap dibekukan.
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
    save_total_limit=3,
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
if RESUME_CHECKPOINT is None:
    trainer.train()
else:
    print(f"Melanjutkan training dari: {RESUME_CHECKPOINT}")
    trainer.train(resume_from_checkpoint=str(RESUME_CHECKPOINT))

trainer.save_model(str(OUTPUT_PATH))
tokenizer.save_pretrained(str(OUTPUT_PATH))

print(f"Adapter LoRA Step 8 disimpan di: {OUTPUT_PATH}")
