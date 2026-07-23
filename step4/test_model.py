import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_PATH = "/Users/fajri/Projects/mytestmodel"

ADAPTER_PATH = "/Users/fajri/Projects/testmodel/step3/output"

print(f"Memuat model dari bucket lokal: {MODEL_PATH}")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto",
    device_map="auto",
    local_files_only=True
)

model = PeftModel.from_pretrained(
    model,
    ADAPTER_PATH,
    local_files_only=True,
)

model.eval()

# Riwayat percakapan message = []
messages = []

print("\nModel siap digunakan...")
print("Perintah:")
print(" /reset Menghapus riwayat percakapan")
print(" /exit Menutup program")
print(" /help Menampilkan bantuan")

while True:
    
    user_input = input("Kamu: ").strip()
        
    command = user_input.lower()

    if command in {"/exit","exit","quit","/quit"}:
        print("Program Selesai.")
        break

    if command in {"/reset","reset"}:
        messages = []
        print("Riwayat percakapan dihapus.")
        continue
    
    if command in {"/help","help"}:
        print("Perintah:")
        print(" /reset Menghapus riwayat percakapan")
        print(" /exit Menutup program")
        print(" /help Menampilkan bantuan")
        continue
    
    if not user_input:
        continue

    messages.append({"role":"user","content":user_input})
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
        )

    input_length = inputs["input_ids"].shape[1]

    generated_tokens = outputs[0][input_length:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    print(f"Model: {response}")

    messages.append({"role":"assistant","content":response})