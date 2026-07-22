# README Step 2 — Fine-Tuning Qwen dengan PEFT/LoRA

Pada langkah kedua ini, model lokal Qwen3-0.6B dilatih menggunakan dataset percakapan sederhana. Training memakai PEFT dengan metode LoRA sehingga bobot utama Qwen tetap utuh dan hasil pembelajaran disimpan sebagai adapter yang jauh lebih kecil.

## Hasil langkah ini

Training pada proyek ini sudah berhasil menghasilkan folder `output`. Folder `output` merupakan **hasil dari menjalankan `python train_peft.py`**. Seluruh file dan subfolder di dalamnya adalah hasil generate dari PEFT/LoRA; file-file tersebut bukan dibuat atau diubah secara manual.

```text
python train_peft.py
        ↓
PEFT/LoRA melakukan training
        ↓
output/
```

```text
output/
├── adapter_config.json
├── adapter_model.safetensors
├── tokenizer.json
├── tokenizer_config.json
└── checkpoint-*/
```

Folder `output` tidak perlu dibuat atau diisi sendiri sebelum training. `SFTTrainer` akan membuatnya berdasarkan nilai `output_dir`, kemudian `trainer.save_model()` menyimpan adapter akhir ke folder tersebut.

```python
OUTPUT_PATH = "/Users/fajri/Projects/testmodel/step2/output"

training_config = SFTConfig(
    output_dir=OUTPUT_PATH,
    # konfigurasi lainnya
)

trainer.train()
trainer.save_model(OUTPUT_PATH)
tokenizer.save_pretrained(OUTPUT_PATH)
```

Pembagian hasil generate dari PEFT/LoRA:

```text
output/adapter_model.safetensors  # Bobot adapter LoRA final
output/adapter_config.json        # Konfigurasi adapter final
output/tokenizer*                 # Tokenizer yang disimpan otomatis
output/checkpoint-*/              # Snapshot sementara setiap epoch
output/training_args.bin          # Konfigurasi training tersimpan
```

Jangan mengedit file dalam `output` secara manual. Jika konfigurasi, dataset, atau proses training berubah, jalankan kembali training dan gunakan output baru.

Ringkasan training:

```text
Base model       : Qwen3-0.6B lokal
Dataset          : 25 contoh percakapan
Metode           : PEFT LoRA
Epoch            : 5
Training steps   : 35
LoRA rank        : 8
LoRA alpha       : 16
Adapter akhir    : sekitar 19 MB
```

## Struktur folder

```text
/Users/fajri/Projects/
├── mytestmodel/                  # Base model Qwen hasil sync bucket
└── testmodel/
    ├── step1/
    └── step2/
        ├── dataset/
        │   └── dataset.jsonl     # Bahan latihan
        ├── output/               # Adapter dan checkpoint hasil training
        ├── README-STEP-2.md
        ├── requirements.txt
        ├── test_model.py         # Menguji base model + adapter
        └── train_peft.py         # Menjalankan training LoRA
```

## Konsep PEFT dan LoRA

PEFT adalah cara menyesuaikan model dengan melatih sejumlah kecil parameter tambahan. Proyek ini menggunakan salah satu metode PEFT, yaitu LoRA.

```text
Base Qwen + dataset + LoraConfig
                  ↓ training
        adapter_model.safetensors
```

Bobot utama berikut tidak diubah:

```text
/Users/fajri/Projects/mytestmodel/model.safetensors
```

Perubahan yang dipelajari disimpan di:

```text
/Users/fajri/Projects/testmodel/step2/output/adapter_model.safetensors
```

Adapter tidak dapat berjalan sendiri. Saat inferensi, adapter harus dipasang ke base model yang kompatibel:

```text
Base Qwen + adapter LoRA → model hasil fine-tuning
```

## Fungsi file utama

### `dataset/dataset.jsonl`

Berisi contoh pertanyaan dan jawaban. Setiap baris adalah satu objek JSON:

```json
{"messages":[{"role":"user","content":"Siapa Raden Muhammad Alfajri?"},{"role":"assistant","content":"Raden Muhammad Alfajri adalah seorang programmer Java."}]}
```

Dataset bukan bobot model. Dataset baru memengaruhi perilaku model setelah diproses oleh training.

### `train_peft.py`

Script ini:

1. Memuat tokenizer dan base model lokal.
2. Membaca `dataset.jsonl`.
3. Membuat konfigurasi LoRA.
4. Menjalankan supervised fine-tuning dengan `SFTTrainer`.
5. Menyimpan adapter dan tokenizer ke folder `output`.

### `output/adapter_config.json`

File hasil generate dari PEFT yang berisi petunjuk pemasangan adapter, antara lain rank, alpha, dropout, tipe task, base model, dan target layer.

### `output/adapter_model.safetensors`

File hasil generate dari PEFT yang berisi bobot LoRA yang dipelajari saat training. File ini bukan salinan lengkap Qwen.

### `test_model.py`

Script ini memuat Qwen dengan `AutoModelForCausalLM`, memasang adapter dengan `PeftModel`, lalu menghasilkan jawaban.

## Persiapan environment

Buka terminal dan masuk ke proyek:

```bash
cd /Users/fajri/Projects/testmodel
```

Buat virtual environment jika belum tersedia:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Jika virtual environment proyek berada di `/Users/fajri/Projects/.venv`, gunakan:

```bash
source /Users/fajri/Projects/.venv/bin/activate
```

Install package:

```bash
python -m pip install --upgrade pip
python -m pip install -r step2/requirements.txt
```

Package utama yang digunakan:

```text
datasets
peft
torch
transformers
trl
```

Periksa instalasinya:

```bash
python -c "import datasets, peft, torch, transformers, trl; print('Dependencies OK')"
```

## Konfigurasi LoRA

Konfigurasi aktual pada `train_peft.py`:

```python
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)
```

Penjelasan singkat:

- `r=8` menentukan kapasitas matriks tambahan LoRA.
- `lora_alpha=16` mengatur skala pengaruh LoRA.
- `lora_dropout=0.05` membantu mengurangi overfitting.
- `target_modules` menentukan layer Qwen yang diberi adapter.
- `CAUSAL_LM` menunjukkan bahwa model digunakan untuk menghasilkan teks.

## Konfigurasi training

```python
training_config = SFTConfig(
    output_dir=OUTPUT_PATH,
    num_train_epochs=5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    max_length=256,
    logging_steps=1,
    save_strategy="epoch",
    gradient_checkpointing=True,
    report_to="none",
)
```

Dengan `save_strategy="epoch"`, trainer membuat checkpoint pada setiap epoch. Karena training menggunakan 5 epoch, folder output memiliki lima checkpoint.

## Menjalankan training

Dari root repository:

```bash
cd /Users/fajri/Projects/testmodel
source /Users/fajri/Projects/.venv/bin/activate
python step2/train_peft.py
```

Atau jika terminal sudah berada di folder `step2`:

```bash
cd /Users/fajri/Projects/testmodel/step2
source /Users/fajri/Projects/.venv/bin/activate
python train_peft.py
```

Perintah `python train_peft.py` menjalankan training PEFT/LoRA dan menghasilkan seluruh isi folder berikut:

```text
/Users/fajri/Projects/testmodel/step2/output
```

Ketika training dimulai, script menampilkan jumlah parameter yang benar-benar dilatih. Parameter utama Qwen dibekukan; hanya parameter LoRA yang diperbarui.

Setelah selesai, periksa adapter:

```bash
ls -lh step2/output/adapter_model.safetensors
cat step2/output/adapter_config.json
```

## Menguji hasil fine-tuning

Jalankan:

```bash
cd /Users/fajri/Projects/testmodel
source /Users/fajri/Projects/.venv/bin/activate
python step2/test_model.py
```

Bagian penting pada script pengujian:

```python
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto",
    device_map="auto",
    local_files_only=True,
)

model = PeftModel.from_pretrained(
    model,
    ADAPTER_PATH,
    local_files_only=True,
)
```

Urutannya harus seperti berikut:

```text
AutoModelForCausalLM → memuat base Qwen
PeftModel            → memasang adapter LoRA
model.generate()     → menghasilkan jawaban dengan adapter aktif
```

`LoraConfig` digunakan ketika membuat dan melatih adapter. `PeftModel` digunakan ketika memuat dan menguji adapter yang sudah selesai dilatih.

## Checkpoint

Folder seperti `checkpoint-7`, `checkpoint-14`, sampai `checkpoint-35` dibuat otomatis sebagai snapshot selama training. Checkpoint menyimpan adapter serta state optimizer, scheduler, dan trainer sehingga training dapat dilanjutkan.

Contoh melanjutkan training dari checkpoint terakhir:

```python
trainer.train(
    resume_from_checkpoint=(
        "/Users/fajri/Projects/testmodel/step2/output/checkpoint-35"
    )
)
```

Gunakan ini hanya jika memang ingin melanjutkan training, bukan memulai eksperimen baru dari awal.

## Memeriksa model yang aktif

Pada proyek ini:

```text
Base model : /Users/fajri/Projects/mytestmodel
Adapter    : /Users/fajri/Projects/testmodel/step2/output
```

Karena keduanya memakai path lokal dan `local_files_only=True`, proses inferensi tidak mengunduh model dari Hugging Face Hub.

## Git dan file output

Folder hasil generate berikut sudah diabaikan oleh `.gitignore`:

```text
step2/output
```

Ini berguna karena output training dan checkpoint dapat berukuran besar. File source, README, requirements, dan dataset tetap dapat dicatat oleh Git, sedangkan artefak model sebaiknya disimpan di Hugging Face.

## Troubleshooting

### `python: command not found`

Gunakan:

```bash
python3 --version
```

Di dalam virtual environment, perintah `python` akan tersedia setelah environment diaktifkan.

### Import berwarna kuning di Cursor

Pilih interpreter yang benar:

```text
Command + Shift + P
Python: Select Interpreter
/Users/fajri/Projects/.venv/bin/python
```

Lalu jalankan `Developer: Reload Window`.

### Adapter tidak ditemukan

Pastikan training sudah selesai dan dua file berikut tersedia:

```text
step2/output/adapter_config.json
step2/output/adapter_model.safetensors
```

### Jawaban masih tidak konsisten

Dataset ini hanya memiliki 25 contoh dan berfokus pada satu informasi sederhana. Hasilnya cocok untuk demonstrasi PEFT, tetapi belum cukup untuk generalisasi yang luas. Tingkatkan kualitas, variasi, dan jumlah data sebelum menaikkan epoch secara berlebihan.
