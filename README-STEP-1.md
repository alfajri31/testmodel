# README Step 1 — Menjalankan Model Hugging Face dari Bucket

Pada langkah pertama ini, kita mempelajari cara menyinkronkan model dari Hugging Face Storage Bucket ke komputer dan menjalankannya sebagai model lokal.

## Struktur folder

```text
/Users/fajri/Projects/
├── mytestmodel/          # Model Qwen hasil sinkronisasi bucket
└── testmodel/
    ├── test_model.py     # Script untuk menjalankan model
    └── README-STEP-1.md
```

## 1. Storage Bucket dan model lokal

Hugging Face Storage Bucket merupakan penyimpanan file seperti S3. Bucket bukan Git repository dan tidak memiliki commit, branch, pull request, atau riwayat versi.

Alur yang digunakan:

```text
Hugging Face Bucket
        ↓ hf buckets sync
/Users/fajri/Projects/mytestmodel
        ↓ from_pretrained(MODEL_PATH)
Model berjalan di komputer
```

## 2. Install Hugging Face CLI

```bash
python3 -m pip install --user --upgrade huggingface_hub
```

Periksa versi CLI:

```bash
hf version
```

## 3. Login ke Hugging Face

```bash
hf auth login
```

Periksa akun yang sedang aktif:

```bash
hf auth whoami
```

## 4. Sinkronkan bucket ke folder lokal

```bash
cd /Users/fajri/Projects
mkdir -p mytestmodel

hf buckets sync \
  hf://buckets/Fajri31/NAMA-BUCKET \
  ./mytestmodel \
  --verbose
```

Format dasar perintah:

```text
hf buckets sync SUMBER TUJUAN
```

Jika download terputus, jalankan kembali perintah yang sama. CLI akan membandingkan sumber dan tujuan, kemudian melanjutkan file yang masih diperlukan.

## 5. Periksa model hasil sinkronisasi

```bash
du -sh /Users/fajri/Projects/mytestmodel
ls -lah /Users/fajri/Projects/mytestmodel
```

Model Qwen lokal ini memiliki file penting berikut:

```text
config.json
generation_config.json
model.safetensors
tokenizer.json
tokenizer_config.json
```

Jenis model tidak ditentukan oleh nama folder. Arsitektur model dibaca dari `config.json`.

## 6. Siapkan Python environment

macOS biasanya menggunakan perintah `python3`, bukan `python`.

```bash
cd /Users/fajri/Projects
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install torch transformers accelerate safetensors
```

Setelah virtual environment aktif, perintah `python` dapat digunakan.

## 7. Jalankan model lokal

File `test_model.py` menggunakan path lokal berikut:

```python
MODEL_PATH = "/Users/fajri/Projects/mytestmodel"
```

Model dan tokenizer dimuat dengan:

```python
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto",
    device_map="auto",
    local_files_only=True,
)
```

`local_files_only=True` memastikan Transformers menggunakan model di komputer dan tidak mengunduh model dari internet.

Jalankan script:

```bash
cd /Users/fajri/Projects
source .venv/bin/activate
python testmodel/test_model.py
```

## 8. Model ID dan local path

Model ID seperti berikut akan menggunakan cache Hugging Face dan dapat mengunduh model dari internet:

```python
AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
```

Path berikut menggunakan folder hasil sinkronisasi bucket:

```python
AutoModelForCausalLM.from_pretrained(
    "/Users/fajri/Projects/mytestmodel",
    local_files_only=True,
)
```

Model yang aktif ditentukan oleh nilai yang diberikan kepada `from_pretrained()` pada program yang sedang berjalan.

## 9. Periksa cache global Hugging Face

Lokasi cache default:

```text
~/.cache/huggingface/hub
```

Periksa isi dan ukurannya:

```bash
ls -lah ~/.cache/huggingface/hub
du -sh ~/.cache/huggingface
du -sh ~/.cache/huggingface/hub/models--* 2>/dev/null
```

Folder hasil sinkronisasi dan cache global merupakan dua lokasi berbeda:

```text
Bucket sync -> /Users/fajri/Projects/mytestmodel
Model ID    -> ~/.cache/huggingface/hub/models--ORG--MODEL
```

## 10. Sinkronkan folder lokal kembali ke bucket

Lihat rencana perubahan tanpa mengunggah:

```bash
hf buckets sync \
  /Users/fajri/Projects/mytestmodel \
  hf://buckets/Fajri31/NAMA-BUCKET \
  --dry-run
```

Jika sudah benar, jalankan sinkronisasi:

```bash
hf buckets sync \
  /Users/fajri/Projects/mytestmodel \
  hf://buckets/Fajri31/NAMA-BUCKET
```

Hindari opsi `--delete` sebelum memahami dampaknya. Bucket tidak mempunyai riwayat versi sehingga file yang dihapus tidak dapat dipulihkan seperti commit Git.
