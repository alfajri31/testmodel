"""Uji base model dengan adapter LoRA hasil training Step 8."""

from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_MODEL_PATH = Path("/Users/fajri/Projects/mytestmodel")
ADAPTER_PATH = Path(__file__).resolve().parent / "output"


if not (BASE_MODEL_PATH / "config.json").is_file():
    raise FileNotFoundError(
        f"Base model tidak ditemukan di: {BASE_MODEL_PATH}"
    )
if not (ADAPTER_PATH / "adapter_config.json").is_file():
    raise FileNotFoundError(
        f"Adapter LoRA tidak ditemukan di: {ADAPTER_PATH}\n"
        "Jalankan train_peft.py terlebih dahulu."
    )


tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    local_files_only=True,
)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.float32,
    local_files_only=True,
)
model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH,
    local_files_only=True,
)
model.eval()

messages = [
    {
        "role": "system",
        "content": "Jawab pertanyaan dengan singkat dan jelas.",
    }
]

print("Model siap. Ketik /exit untuk keluar atau /reset untuk menghapus riwayat.")

while True:
    question = input("\nKamu: ").strip()

    if not question:
        continue

    if question.lower() == "/exit":
        break

    if question.lower() == "/reset":
        messages = messages[:1]
        print("Riwayat percakapan sudah dihapus.")
        continue

    messages.append({"role": "user", "content": question})

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    answer = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    ).strip()

    print(f"Model: {answer}")
    messages.append({"role": "assistant", "content": answer})
