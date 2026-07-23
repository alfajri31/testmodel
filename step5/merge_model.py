import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_MODEL_PATH = "/Users/fajri/Projects/mytestmodel"
ADAPTER_PATH = "/Users/fajri/Projects/testmodel/step5/output"
MERGED_MODEL_PATH = "/Users/fajri/Projects/testmodel/step5/merged_model"


print("Memuat tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    local_files_only=True,
)

print("Memuat base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.float32,
    local_files_only=True,
)

print("Memasang adapter LoRA...")
peft_model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH,
    local_files_only=True,
)

print("Menggabungkan base model dengan adapter LoRA...")
merged_model = peft_model.merge_and_unload()

print("Menyimpan model gabungan...")
merged_model.save_pretrained(
    MERGED_MODEL_PATH,
    safe_serialization=True,
)
tokenizer.save_pretrained(MERGED_MODEL_PATH)

print(f"Model gabungan tersimpan di: {MERGED_MODEL_PATH}")
