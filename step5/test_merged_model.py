import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_PATH = "/Users/fajri/Projects/testmodel/step5/merged_model"


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float32,
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
