# Belajar Fine-Tuning LLM

Project ini berisi proses belajar dari mengambil base model sampai menjalankan hasil fine-tuning di Ollama.

## Daftar Materi

### Step 0 — Dasar Token, Embedding, dan Contextual Vector

Memahami bagaimana teks diubah menjadi token, diproses oleh attention, lalu digunakan untuk memprediksi token berikutnya.

[Baca Step 0](step0-concepts/README.md)

### Step 1 — Persiapan Python

Memahami versi Python, `python3`, `python3.11`, `pip`, dan virtual environment.

[Baca Step 1](step0/README.md)

### Step 2 — Menjalankan Base Model

Mengambil model dari Hugging Face Bucket, menyimpannya di lokal, dan menjalankannya menggunakan Python.

[Baca Step 2](step1/README-STEP-1.md)

### Step 3 — Fine-Tuning dengan LoRA

Membuat dataset JSONL, melatih adapter PEFT/LoRA, dan menguji hasil fine-tuning.

[Baca Step 3](step2/README-STEP-2.md)

### Step 4 — Dataset dan Jawaban Model

Memahami pengaruh dataset dan konfigurasi generation terhadap jawaban model.

[Baca Step 4](step3/README-STEP-3.md)

### Step 5 — Melihat Hasil Training

Memantau loss, accuracy, entropy, learning rate, dan gradient menggunakan TensorBoard.

[Baca Step 5](step4/README.md)

### Step 6 — Menggabungkan Model

Menggabungkan adapter LoRA dengan base model dan memahami token, layer, tensor, serta parameter.

[Baca Step 6](step5/README.md)

### Step 7 — Menjalankan Model di Ollama

Menjalankan Safetensors secara langsung atau mengonversinya menjadi GGUF untuk Ollama dan `llama.cpp`.

[Baca Step 7](step6/README.md)

## Alur Singkat

```text
Base Model
    ↓
Dataset
    ↓
Fine-Tuning LoRA
    ↓
Evaluasi Training
    ↓
Merge Model
    ↓
Konversi GGUF
    ↓
Deploy ke Ollama
```

## Ringkasan

```text
Fine-tuning → membuat model lebih condong mengikuti dataset
Merge       → menggabungkan LoRA dengan base model
Konversi    → mengubah format model untuk runtime
Deploy      → menjalankan model melalui Ollama
```
