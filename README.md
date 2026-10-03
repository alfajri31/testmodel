# Belajar Fine-Tuning LLM

Project ini berisi proses belajar dari mengambil base model sampai menjalankan hasil fine-tuning di Ollama.

## Daftar Materi

### Fondasi LLM — Token, Vector, Attention, dan Bobot

Memahami bagaimana teks diubah menjadi token dan vector, diproses oleh attention serta layer Transformer, lalu menggunakan bobot model untuk memprediksi token berikutnya.

[Baca Fondasi LLM](step0-concepts/README.md)

### Step 0 — Persiapan Python

Memahami versi Python, `python3`, `python3.11`, `pip`, dan virtual environment.

[Baca Step 0](step0/README.md)

### Step 1 — Menjalankan Base Model

Mengambil model dari Hugging Face Bucket, menyimpannya di lokal, dan menjalankannya menggunakan Python.

[Baca Step 1](step1/README-STEP-1.md)

### Step 2 — Fine-Tuning dengan LoRA

Membuat dataset JSONL, melatih adapter PEFT/LoRA, dan menguji hasil fine-tuning.

[Baca Step 2](step2/README-STEP-2.md)

### Step 3 — Dataset dan Jawaban Model

Memahami pengaruh dataset dan konfigurasi generation terhadap jawaban model.

[Baca Step 3](step3/README-STEP-3.md)

### Step 4 — Melihat Hasil Training

Memantau loss, accuracy, entropy, learning rate, dan gradient menggunakan TensorBoard.

[Baca Step 4](step4/README.md)

### Step 5 — Menggabungkan Model

Menggabungkan adapter LoRA dengan base model dan memahami token, layer, tensor, serta parameter.

[Baca Step 5](step5/README.md)

### Step 6 — Menjalankan Model di Ollama

Menjalankan Safetensors secara langsung atau mengonversinya menjadi GGUF untuk Ollama dan `llama.cpp`.

[Baca Step 6](step6/README.md)

### Step 7 — Visualisasi Struktur Model dan Uji Inference

Merekam struktur model sebagai teks dan graph di TensorBoard, lalu menguji adapter LoRA secara interaktif.

[Baca Step 7](step7/README.md)

### Step 8 — Continue Training Setelah Merge

Melatih adapter LoRA baru dari model hasil merge Step 5, lalu menggabungkannya kembali agar training tambahan tidak dimulai dari base model awal.

[Baca Step 8](step8/README.md)

### Step 9 — RAG dari MongoDB

Mencari natural text pemain sepak bola yang relevan dari MongoDB memakai embedding, lalu memberi hasilnya ke model sebagai konteks jawaban.

[Baca Step 9](step9/README.md)

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
    ↓
Visualisasi Struktur dan Uji Inference
    ↓
Continue Training dari Merged Model
    ↓
RAG dari MongoDB
```

## Ringkasan

```text
Fine-tuning → membuat model lebih condong mengikuti dataset
Merge       → menggabungkan LoRA dengan base model
Uji model   → menjalankan inference dan melihat struktur model
Konversi    → mengubah format model untuk runtime
Deploy      → menjalankan model melalui Ollama
Continue    → melatih LoRA baru dari bobot merged model sebelumnya
Merge lanjut → menggabungkan adapter training tambahan ke model tahap sebelumnya
RAG         → mengambil fakta MongoDB saat inference tanpa melatih ulang model
```
