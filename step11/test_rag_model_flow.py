"""Menampilkan hubungan tahap RAG dan inference LLM.

Jalankan dari root project:
    python3 step11/test_rag_model_flow.py

Script ini memakai helper retrieval dan konfigurasi database dari Step 9.
RAG hanya membaca dokumen MongoDB; model hanya membaca natural text hasil RAG.
"""

from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from step9.test_model_rag import (
    COLLECTION_NAME,
    DATABASE_NAME,
    EMBEDDING_MODEL_NAME,
    MIN_SIMILARITY,
    MONGODB_URI,
    load_rag_documents,
    retrieve,
)


BASE_MODEL_PATH = Path("/Users/fajri/Projects/mytestmodel")
STEP11_ADAPTER_PATH = ROOT_DIR / "step11" / "output"
FALLBACK_ADAPTER_PATH = ROOT_DIR / "step8" / "output"
ADAPTER_PATH = (
    STEP11_ADAPTER_PATH
    if (STEP11_ADAPTER_PATH / "adapter_config.json").is_file()
    else FALLBACK_ADAPTER_PATH
)


RESET = "\033[0m"
RAG_COLOR = "\033[36m"      # cyan: hasil penemuan/retrieval RAG
AI_COLOR = "\033[32m"       # green: jawaban yang dirangkai AI
INFO_COLOR = "\033[90m"     # gray: status proses
CONTEXT_COLOR = "\033[33m"  # yellow: data yang masuk context window


def colored(color, text):
    """Warnai output jika terminal mendukung ANSI escape code."""
    return f"{color}{text}{RESET}"


STYLE_INSTRUCTIONS = {
    "common": "Gunakan gaya common.",
    "tegas": "Gunakan gaya common.",
    "moody": "Gunakan gaya moody.",
    "profesional": "Gunakan gaya profesional.",
    "comedian": "Gunakan gaya comedian.",
}
DEFAULT_STYLE = "profesional"
STYLE_GENERATION = {
    "common": {"do_sample": False},
    "tegas": {"do_sample": False},
    "moody": {"do_sample": False},
    "profesional": {"do_sample": False},
    "comedian": {"do_sample": False},
}

GUIDE_QUESTIONS = [
    "Sebutkan para pemain Inter Miami CF.",
    "Sebutkan salah satu pemain Inter Miami CF.",
    "Bagaimana RAG menemukan data pemain?",
    "Jelaskan daftar pemain tersebut dengan gaya comedian.",
    "Apa perbedaan RAG dan model bahasa?",
]


def generate_answer_with_style(question, rag_context, style, tokenizer, model):
    """Buat jawaban dengan fakta RAG dan gaya bahasa yang dipilih."""
    active_style = style if style in STYLE_INSTRUCTIONS else DEFAULT_STYLE
    messages = [
        {
            "role": "system",
            "content": (
                "Jawab berdasarkan referensi RAG yang diberikan. Jika fakta "
                "tidak tersedia di referensi, katakan bahwa informasinya tidak "
                "ditemukan. Jika referensi berisi daftar, pertahankan semua "
                "item dan jangan meringkasnya. Gunakan gaya berikut: "
                f"{STYLE_INSTRUCTIONS[active_style]}"
                "\nJika gaya tidak tersedia, gunakan gaya profesional."
                "\n\n"
                f"<REFERENSI_RAG>\n{rag_context}\n</REFERENSI_RAG>"
            ),
        },
        {"role": "user", "content": question},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(prompt, return_tensors="pt")

    generation_options = {
        "max_new_tokens": 260,
        "pad_token_id": tokenizer.eos_token_id,
        **STYLE_GENERATION.get(
            active_style,
            STYLE_GENERATION[DEFAULT_STYLE],
        ),
    }

    with torch.inference_mode():
        outputs = model.generate(**inputs, **generation_options)

    new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    if not (BASE_MODEL_PATH / "config.json").is_file():
        raise FileNotFoundError(
            f"Base model tidak ditemukan di {BASE_MODEL_PATH}."
        )
    if not (ADAPTER_PATH / "adapter_config.json").is_file():
        raise FileNotFoundError(
            f"Adapter tidak ditemukan di {ADAPTER_PATH}. "
            "Jalankan training Step 8 atau Step 11 terlebih dahulu."
        )

    print("[1/5] Memuat embedding model untuk RAG...")
    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL_NAME,
        local_files_only=True,
    )

    print("[2/5] Memuat model jawaban...")
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

    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=10_000)
    try:
        collection = client[DATABASE_NAME][COLLECTION_NAME]
        descriptions, vectors = load_rag_documents(collection)
        print(colored(INFO_COLOR, f"[3/5] RAG siap: {len(descriptions):,} dokumen tersedia."))

        style = DEFAULT_STYLE
        print("\nContoh pertanyaan yang sesuai dengan dataset:")
        for guide_question in GUIDE_QUESTIONS:
            print(f"- {guide_question}")
        print("\nKetik pertanyaan. Ketik /style untuk mengganti gaya, /exit untuk keluar.")
        while True:
            question = input("\nUser: ").strip()
            if not question:
                continue
            if question.lower() == "/exit":
                break
            if question.lower() == "/style":
                requested_style = input(
                    "Pilih gaya (common/moody/profesional/comedian): "
                ).strip().lower()
                if requested_style in STYLE_INSTRUCTIONS:
                    style = requested_style
                    print(f"Gaya aktif: {style}")
                else:
                    style = DEFAULT_STYLE
                    print(f"Gaya tidak tersedia. Kembali ke gaya {DEFAULT_STYLE}.")
                continue

            print(colored(INFO_COLOR, "[4/5] RAG membuat query vector dan mencari dokumen..."))
            sources = retrieve(
                question,
                embedding_model,
                descriptions,
                vectors,
            )
            best_score, rag_context = sources[0]
            print(colored(RAG_COLOR, f"RAG / similarity: {best_score:.3f}"))

            if best_score < MIN_SIMILARITY:
                print(colored(AI_COLOR, "AI / Informasi RAG belum cukup untuk menjawab."))
                continue

            print(colored(
                CONTEXT_COLOR,
                f"[5/5] RAG masuk context window LLM dengan gaya '{style}':\n"
                f"{rag_context}",
            ))
            answer = generate_answer_with_style(
                question,
                rag_context,
                style,
                tokenizer,
                model,
            )
            print(colored(AI_COLOR, f"AI / jawaban:\n{answer}"))
    finally:
        client.close()


if __name__ == "__main__":
    main()
