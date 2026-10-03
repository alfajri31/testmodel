"""Uji model Step 8 dengan RAG pemain sepak bola dari MongoDB.

MongoDB memberi natural text yang relevan. Model Step 8 hanya membaca text
tersebut di context window lalu merangkai jawaban.

Jalankan:
    python3 step9/test_model_rag.py
"""

from pathlib import Path

import numpy as np
import torch
from pymongo import MongoClient
from peft import PeftModel
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


STEP_DIR = Path(__file__).resolve().parent
BASE_MODEL_PATH = Path("/Users/fajri/Projects/mytestmodel")
ADAPTER_PATH = STEP_DIR.parent / "step8" / "output"

# Database yang sama dengan footballmodel/step.
MONGODB_URI = (
    "mongodb://admin:admin123@fajris-mac-mini.tail2fab46.ts.net:27017/"
    "gosoccer?authSource=admin"
)
DATABASE_NAME = "gosoccer_vector"
COLLECTION_NAME = "football_collection_id"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
TOP_K = 1
MIN_SIMILARITY = 0.65


def load_rag_documents(collection):
    """Muat natural text dan embedding yang sudah tersedia di MongoDB."""
    documents = list(
        collection.find(
            {},
            {"_id": 0, "description": 1, "embedding": 1},
        )
    )

    valid_documents = [
        document
        for document in documents
        if document.get("description") and document.get("embedding")
    ]
    if not valid_documents:
        raise RuntimeError(
            f"Collection '{collection.name}' tidak memiliki description dan embedding."
        )

    descriptions = [document["description"] for document in valid_documents]
    vectors = np.asarray(
        [document["embedding"] for document in valid_documents],
        dtype=np.float32,
    )
    return descriptions, vectors


def retrieve(question, embedding_model, descriptions, vectors):
    """Cari natural text dengan embedding yang paling mirip dengan pertanyaan."""
    question_vector = embedding_model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )[0]

    # Embedding dokumen dan query sudah dinormalisasi. Dot product sama dengan
    # cosine similarity pada kondisi tersebut.
    scores = vectors @ question_vector
    best_indices = np.argsort(scores)[-TOP_K:][::-1]
    return [
        (float(scores[index]), descriptions[index])
        for index in best_indices
    ]


def generate_answer(question, rag_context, tokenizer, model):
    """Masukkan text RAG ke context window, lalu minta LLM merangkai jawaban."""
    messages = [
        {
            "role": "system",
            "content": (
                "Kamu adalah asisten data sepak bola. Gunakan REFERENSI RAG "
                "sebagai sumber fakta utama. Jangan membuat nama pemain, klub, "
                "atau statistik baru. Jika referensi tidak cukup, katakan bahwa "
                "datanya tidak tersedia. Jawab dalam bahasa Indonesia.\n\n"
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

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    if not (BASE_MODEL_PATH / "config.json").is_file():
        raise FileNotFoundError(
            f"Base model tidak ditemukan di: {BASE_MODEL_PATH}"
        )
    if not (ADAPTER_PATH / "adapter_config.json").is_file():
        raise FileNotFoundError(
            f"Adapter Step 8 tidak ditemukan di: {ADAPTER_PATH}\n"
            "Jalankan step8/train_peft.py terlebih dahulu."
        )

    print(f"Memuat embedding model RAG: {EMBEDDING_MODEL_NAME}")
    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL_NAME,
        local_files_only=True,
    )

    print(f"Memuat base model: {BASE_MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, local_files_only=True)
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
        print(f"RAG siap: {len(descriptions):,} dokumen dari {COLLECTION_NAME}.")
        print("Ketik /rag untuk melihat sumber terakhir, /exit untuk keluar.")

        last_sources = []
        while True:
            question = input("\nKamu: ").strip()

            if not question:
                continue
            if question.lower() == "/exit":
                break
            if question.lower() == "/rag":
                if not last_sources:
                    print("Belum ada hasil RAG.")
                else:
                    for rank, (score, description) in enumerate(
                        last_sources,
                        start=1,
                    ):
                        print(f"\nSumber #{rank} | similarity={score:.3f}\n{description}")
                continue

            last_sources = retrieve(
                question,
                embedding_model,
                descriptions,
                vectors,
            )
            best_score, rag_context = last_sources[0]

            print(f"RAG similarity tertinggi: {best_score:.3f}")
            if best_score < MIN_SIMILARITY:
                print("Model: Maaf, belum ada informasi yang cukup di database.")
                continue

            answer = generate_answer(
                question,
                rag_context,
                tokenizer,
                model,
            )
            print(f"Model: {answer}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
