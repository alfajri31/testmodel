# Belajar Fine-Tuning dan Deployment LLM

Project ini berisi proses belajar mulai dari mengambil base model, melakukan fine-tuning LoRA, mengevaluasi training, menggabungkan bobot, sampai menjalankan model di Ollama.

```text
Step 1 → Mengambil dan menjalankan base model
Step 2 → Fine-tuning menggunakan PEFT/LoRA
Step 3 → Memahami pengaruh dataset dan generation
Step 4 → Memantau training dengan TensorBoard
Step 5 → Menggabungkan base model dan LoRA
Step 6 → Mengonversi dan menjalankan model di Ollama
```

## Step 1 — Menjalankan Model Hugging Face dari Bucket

Materi:

- Struktur folder.
- Storage Bucket dan model lokal.
- Instalasi Hugging Face CLI.
- Login ke Hugging Face.
- Sinkronisasi bucket ke folder lokal.
- Pemeriksaan model hasil sinkronisasi.
- Persiapan Python environment.
- Menjalankan model lokal.
- Perbedaan Model ID dan local path.
- Pemeriksaan cache global Hugging Face.
- Sinkronisasi folder lokal kembali ke bucket.

Baca materi lengkap: [README Step 1](step1/README-STEP-1.md)

## Step 2 — Fine-Tuning Qwen dengan PEFT/LoRA

Materi:

- Konsep PEFT dan LoRA.
- Fungsi rank pada LoRA.
- Cara LoRA mempelajari dataset.
- Prioritas dan kualitas dataset.
- Format dataset JSON/JSONL.
- Fungsi `train_peft.py`.
- Fungsi `adapter_config.json`.
- Fungsi `adapter_model.safetensors`.
- Pengujian menggunakan `PeftModel`.
- Konfigurasi LoRA dan training.
- Menjalankan training.
- Menguji hasil fine-tuning.
- Checkpoint dan output hasil PEFT.
- Pemeriksaan model aktif.
- Git dan file output.
- Troubleshooting Python, package, dan adapter.

Baca materi lengkap: [README Step 2](step2/README-STEP-2.md)

## Step 3 — Pengaruh Dataset dan Konfigurasi Generation

Materi:

- Alur training dan pengujian.
- Dataset meningkatkan probabilitas, bukan menyimpan jawaban.
- Pengaruh jumlah dan variasi data latihan.
- Hubungan dataset dengan akurasi.
- Pengaruh konfigurasi `test_model.py`.
- Fungsi `do_sample`.
- Fungsi `temperature`.
- Fungsi `top_p`.
- Fungsi `max_new_tokens`.
- Konfigurasi evaluasi yang konsisten.
- Pengaruh riwayat percakapan.
- Training ulang setelah dataset berubah.
- Chat model secara interaktif.
- Evaluasi sederhana tanpa TensorBoard.

Baca materi lengkap: [README Step 3](step3/README-STEP-3.md)

## Step 4 — Training LoRA dan TensorBoard

Materi:

- Alur training LoRA.
- Hubungan dataset dengan token.
- Menjalankan training.
- Menyimpan log training.
- Menjalankan TensorBoard.
- Membaca accuracy, loss, dan entropy.
- Analogi loss dan entropy.
- Membaca gradient norm dan learning rate.
- Hubungan hasil training dengan dataset dan base model.
- Token API dan context window.
- Perbedaan context window dan kuota pemakaian.
- Pengaruh menghapus history atau memory.

Baca materi lengkap: [README Step 4](step4/README.md)

## Step 5 — Merge Base Model dan LoRA

Materi:

- Menggabungkan base model dengan adapter LoRA.
- Menjalankan proses merge.
- Memeriksa spesifikasi model.
- Perbedaan ukuran base model, adapter, dan merged model.
- Gambaran layer, module, tensor, matrix, dan parameter.
- Perbedaan token, aktivasi, dan parameter.
- Ukuran arsitektur Qwen3.
- Fungsi modul `q_proj`, `k_proj`, `v_proj`, dan `o_proj`.
- Fungsi `gate_proj`, `up_proj`, dan `down_proj`.
- Cara membaca input size dan output size.
- Perhitungan parameter base model.
- Embedding token dan parameter seluruh layer.
- Rank 8 dan perhitungan parameter LoRA.
- Perbedaan sebelum dan sesudah merge.
- Alasan setiap model mempunyai karakteristik berbeda.
- Fine-tuning mengubah kecenderungan jawaban melalui bobot.

Baca materi lengkap: [README Step 5](step5/README.md)

## Step 6 — Deploy Model ke Ollama

Materi:

- Deployment merged model ke Ollama.
- Perbandingan Safetensors dan GGUF.
- Import Safetensors secara langsung dengan fitur eksperimental Ollama.
- Fungsi `llama.cpp`.
- Fungsi `convert_hf_to_gguf.py`.
- Python `.venv` sebagai mesin converter.
- Library `llama.cpp` sebagai komponen converter.
- Konversi Safetensors menjadi GGUF F32.
- Deployment GGUF ke Ollama.
- Perbedaan `<think>` pada Safetensors dan GGUF.
- Pemeriksaan template, renderer, dan parser Ollama.
- Benchmark sederhana dengan `--verbose`.
- Fungsi `Modelfile` sebagai pengaturan deployment.
- Perbedaan `Modelfile` dan `Modelfile.llama_cpp`.
- Tujuan konversi format model.
- Pengaruh presisi F32, F16, BF16, Q8, dan Q4.
- Hal yang berubah dan tidak berubah ketika model dikonversi.

Baca materi lengkap: [README Step 6](step6/README.md)

## Alur Keseluruhan

```text
Hugging Face Bucket
        ↓
Base Model Lokal
        ↓
Dataset JSONL
        ↓
Fine-Tuning PEFT/LoRA
        ↓
Adapter LoRA
        ↓
Evaluasi dan TensorBoard
        ↓
Merge dengan Base Model
        ↓
Merged Model Safetensors
        ├── import langsung ke Ollama
        └── konversi ke GGUF → Ollama
```

## Kesimpulan

```text
Dataset     → contoh yang ingin dipelajari model
Fine-tuning → membuat model lebih condong mengikuti dataset
LoRA        → melatih parameter tambahan berukuran kecil
TensorBoard → memantau proses dan hasil training
Merge       → menggabungkan adapter dengan base model
Konversi    → mengubah format model untuk runtime
Modelfile   → menyimpan pengaturan deployment Ollama
Deploy      → mendaftarkan dan menjalankan model
```

Konversi bukan proses training dan tidak menambahkan dataset. Kemampuan yang dipelajari berasal dari proses fine-tuning, sedangkan konversi menyiapkan model agar dapat dijalankan pada runtime seperti `llama.cpp` dan Ollama.
