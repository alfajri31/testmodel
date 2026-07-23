import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

BASE_MODEL_PATH = "/Users/fajri/Projects/mytestmodel"
DATASET_PATH = "/Users/fajri/Projects/testmodel/step5/dataset/dataset.jsonl"
OUTPUT_PATH = "/Users/fajri/Projects/testmodel/step5/output"

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    local_files_only=True,
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.float32,
    local_files_only=True,
)

base_model.config.use_cache = False
base_model.enable_input_require_grads()

dataset = load_dataset(
    "json",
    data_files=DATASET_PATH,
    split="train",
)

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
    output_dir=OUTPUT_PATH,
    num_train_epochs=5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    max_length=256,
    logging_steps=1,
    save_strategy="epoch",
    gradient_checkpointing=True,
    report_to="tensorboard",
       logging_dir=(
        "/Users/fajri/Projects/testmodel/"
        "step5/tensorboard_logs"
    ),
)

trainer = SFTTrainer(
    model=base_model,
    args=training_config,
    train_dataset=dataset,
    processing_class=tokenizer,
    peft_config=lora_config,
)

trainer.model.print_trainable_parameters()
trainer.train()

trainer.save_model(OUTPUT_PATH)
tokenizer.save_pretrained(OUTPUT_PATH)

print(f"Adapter disimpan di: {OUTPUT_PATH}")