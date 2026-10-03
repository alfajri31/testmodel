"""Gabungkan adapter LoRA tahap kedua ke model hasil merge tahap pertama."""

from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


STEP_DIR = Path(__file__).resolve().parent
MERGED_MODEL_INPUT_PATH = Path(
    "/Users/fajri/Projects/testmodel/step5/merged_model"
)
ADAPTER_PATH = STEP_DIR / "output"
MERGED_MODEL_OUTPUT_PATH = STEP_DIR / "merged_model"


if not (MERGED_MODEL_INPUT_PATH / "config.json").is_file():
    raise FileNotFoundError(
        "Merged model tahap pertama tidak ditemukan di: "
        f"{MERGED_MODEL_INPUT_PATH}"
    )

if not (ADAPTER_PATH / "adapter_config.json").is_file():
    raise FileNotFoundError(
        "Adapter LoRA tahap kedua tidak ditemukan di: "
        f"{ADAPTER_PATH}\n"
        "Jalankan train_continue.py terlebih dahulu."
    )

if MERGED_MODEL_OUTPUT_PATH.exists() and any(
    MERGED_MODEL_OUTPUT_PATH.iterdir()
):
    raise FileExistsError(
        "Folder target merge sudah berisi file: "
        f"{MERGED_MODEL_OUTPUT_PATH}\n"
        "Gunakan folder baru atau pindahkan hasil merge lama terlebih dahulu."
    )


print(f"Memuat merged model tahap pertama: {MERGED_MODEL_INPUT_PATH}")
tokenizer = AutoTokenizer.from_pretrained(
    MERGED_MODEL_INPUT_PATH,
    local_files_only=True,
)
base_model = AutoModelForCausalLM.from_pretrained(
    MERGED_MODEL_INPUT_PATH,
    torch_dtype=torch.float32,
    local_files_only=True,
)

print(f"Memasang adapter LoRA tahap kedua: {ADAPTER_PATH}")
model_with_adapter = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH,
    local_files_only=True,
)

print("Menggabungkan adapter tahap kedua ke bobot model...")
merged_model = model_with_adapter.merge_and_unload()

print(f"Menyimpan hasil merge lanjutan: {MERGED_MODEL_OUTPUT_PATH}")
merged_model.save_pretrained(
    MERGED_MODEL_OUTPUT_PATH,
    safe_serialization=True,
)
tokenizer.save_pretrained(MERGED_MODEL_OUTPUT_PATH)

print("Merge lanjutan selesai.")
