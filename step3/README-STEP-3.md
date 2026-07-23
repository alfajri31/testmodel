# README Step 3 — Pengaruh Dataset dan Konfigurasi Generation

Step 3 mempelajari cara menjalankan model secara interaktif serta memahami mengapa jawaban hasil fine-tuning tidak selalu sama dengan isi dataset.

Dokumentasi ini tidak menggunakan TensorBoard. Evaluasi dilakukan langsung dengan membandingkan pertanyaan dan jawaban model melalui `test_model.py`.

## Struktur folder

```text
step3/
├── dataset/
│   └── dataset.jsonl       # Data latihan
├── output/                 # Hasil generate PEFT/LoRA
├── README-STEP-3.md
├── test_model.py           # Chat interaktif dan konfigurasi generation
└── train_peft.py           # Training adapter PEFT/LoRA
```

## Alur Step 3

```text
dataset/dataset.jsonl
          ↓
python train_peft.py
          ↓
output/adapter_model.safetensors
          ↓
Base Qwen + PeftModel
          ↓
python test_model.py
          ↓
Percakapan interaktif
```

Seluruh isi folder `output` merupakan hasil generate dari training PEFT/LoRA. Mengubah `dataset.jsonl` saja tidak mengubah adapter yang sudah ada. Setelah dataset diubah, `train_peft.py` harus dijalankan kembali untuk menghasilkan adapter baru.

## Dataset meningkatkan probabilitas, bukan menyimpan jawaban

Model tidak membuka atau mencari teks secara langsung di `dataset.jsonl` ketika `test_model.py` berjalan. Dataset hanya digunakan pada saat training untuk menghitung loss dan memperbarui parameter LoRA.

```text
Pertanyaan dan jawaban target
          ↓
Model membuat prediksi
          ↓
Loss dihitung
          ↓
Gradient memperbarui parameter LoRA
          ↓
Probabilitas jawaban target meningkat
```

Contoh:

```text
Input  : Siapa Raden Muhammad Alfajri?
Target : Raden Muhammad Alfajri adalah seorang programmer Java.
```

Setelah training, tujuan yang diharapkan adalah:

```text
P("programmer Java" | pertanyaan tentang Alfajri) meningkat
```

Adapter tidak berubah menjadi tabel seperti:

```text
Alfajri → programmer Java
Reyga   → programmer Java di Askrindo
```

Informasi dari dataset direpresentasikan sebagai perubahan angka yang tersebar pada parameter LoRA. Karena itu, hasil fine-tuning bersifat probabilistik dan bukan lookup database.

## Pengaruh jumlah data latihan

Semakin banyak contoh yang relevan, konsisten, dan berkualitas, semakin sering model menerima arah pembelajaran yang sama. Hal ini dapat meningkatkan probabilitas munculnya jawaban yang diinginkan.

```text
Sedikit contoh relevan
→ sinyal pembelajaran lemah
→ jawaban target belum konsisten

Banyak contoh relevan dan bervariasi
→ sinyal pembelajaran lebih kuat
→ probabilitas jawaban target meningkat
```

Namun, **lebih banyak data tidak otomatis berarti lebih akurat**. Kualitas dan keseimbangan data juga penting.

Dataset yang baik memiliki karakteristik berikut:

- Jawaban faktual dan konsisten.
- Variasi cara bertanya yang natural.
- Jumlah contoh cukup untuk setiap topik atau entitas.
- Tidak memiliki jawaban yang saling bertentangan.
- Memiliki contoh penolakan untuk informasi yang tidak tersedia.
- Tidak hanya menyalin kalimat identik berulang kali.

Contoh variasi yang berguna:

```jsonl
{"messages":[{"role":"user","content":"Siapa Reyga?"},{"role":"assistant","content":"Reyga adalah teman kerja Raden Muhammad Alfajri di Askrindo dan juga seorang programmer Java."}]}
{"messages":[{"role":"user","content":"Apa pekerjaan Reyga?"},{"role":"assistant","content":"Reyga bekerja sebagai programmer Java."}]}
{"messages":[{"role":"user","content":"Di mana Reyga bekerja?"},{"role":"assistant","content":"Reyga bekerja di Askrindo."}]}
```

Jika satu entitas memiliki jauh lebih banyak contoh daripada entitas lain, model cenderung lebih kuat mengikuti pola entitas yang dominan. Dataset perlu diseimbangkan berdasarkan kemampuan yang ingin diuji.

## Akurasi tidak hanya dipengaruhi jumlah data

Hasil model dipengaruhi oleh beberapa komponen sekaligus:

```text
Kualitas dataset
      +
Keseimbangan dataset
      +
Jumlah epoch
      +
Learning rate
      +
Kapasitas/rank LoRA
      +
Prompt dan riwayat percakapan
      +
Konfigurasi generation
      ↓
Jawaban akhir model
```

Dataset kecil dapat membuat model menghafal pertanyaan training tetapi gagal menjawab variasi pertanyaan baru. Training loss yang rendah juga belum membuktikan bahwa model akurat pada data yang belum pernah dilihat.

## Konfigurasi `test_model.py` memengaruhi jawaban

`test_model.py` tidak melakukan pencarian langsung ke dataset. Konfigurasi pada `model.generate()` menentukan bagaimana token jawaban dipilih dari probabilitas yang dihasilkan model.

Konfigurasi saat ini:

```python
outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    do_sample=True,
    temperature=0.7,
    top_p=0.8,
)
```

### `do_sample`

```python
do_sample=True
```

Model memilih token dengan sampling sehingga jawaban dapat berbeda setiap kali dijalankan.

```python
do_sample=False
```

Model memilih token dengan probabilitas tertinggi. Hasil lebih deterministik dan lebih cocok untuk mengevaluasi fakta hasil fine-tuning.

### `temperature`

Temperature hanya digunakan ketika `do_sample=True`.

```text
Temperature rendah → lebih konservatif
Temperature tinggi → lebih bervariasi dan lebih berisiko melenceng
```

### `top_p`

`top_p` membatasi kumpulan kandidat token berdasarkan probabilitas kumulatif. Nilai lebih kecil membatasi pilihan, tetapi selama sampling aktif hasilnya masih dapat berubah.

### `max_new_tokens`

Untuk jawaban faktual yang singkat, nilai terlalu besar memberi kesempatan model menambahkan informasi yang tidak diminta.

```text
Jawaban fakta sederhana → 30–100 token biasanya cukup
```

## Konfigurasi untuk evaluasi yang lebih konsisten

Gunakan deterministic generation:

```python
model.generation_config.temperature = None
model.generation_config.top_p = None
model.generation_config.top_k = None

outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    do_sample=False,
    repetition_penalty=1.1,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
```

Ketika `do_sample=False`, jangan menggunakan `temperature`, `top_p`, atau `top_k`. Parameter tersebut hanya berlaku untuk sampling dan dapat menghasilkan peringatan jika tetap aktif.

Konfigurasi deterministic membuat hasil lebih konsisten, tetapi tidak menjamin model selalu benar. Jika sinyal tentang suatu fakta belum cukup kuat di dalam adapter, greedy decoding tetap dapat memilih jawaban yang salah atau mengatakan tidak tahu.

## Pengaruh riwayat percakapan

`test_model.py` menyimpan percakapan dalam:

```python
messages = []
```

Setiap pertanyaan dan jawaban ditambahkan kembali ke `messages`. Artinya, pertanyaan sebelumnya ikut memengaruhi jawaban berikutnya.

```text
Pertanyaan pertama
        ↓ disimpan
Jawaban pertama
        ↓ disimpan
Pertanyaan berikutnya
        ↓
Model menerima seluruh konteks sebelumnya
```

Gunakan `/reset` sebelum menguji topik berbeda agar evaluasi dimulai dari konteks kosong:

```text
Kamu: /reset
Riwayat percakapan dihapus.
```

## Menjalankan training ulang

Setelah mengubah dataset:

```bash
cd /Users/fajri/Projects/testmodel
source /Users/fajri/Projects/.venv/bin/activate
python step3/train_peft.py
```

Perintah tersebut menghasilkan atau memperbarui:

```text
step3/output/adapter_config.json
step3/output/adapter_model.safetensors
```

Untuk menjaga hasil training lama, gunakan output baru seperti `output-v2` dengan mengubah `OUTPUT_PATH` sebelum training.

## Menjalankan chat interaktif

```bash
cd /Users/fajri/Projects/testmodel
source /Users/fajri/Projects/.venv/bin/activate
python step3/test_model.py
```

Perintah interaktif:

```text
/reset → menghapus riwayat percakapan
/help  → menampilkan bantuan
/exit  → menutup program
```

Model akan terus menerima pertanyaan sampai `/exit` digunakan.

## Cara mengevaluasi tanpa TensorBoard

Siapkan kumpulan pertanyaan yang tidak semuanya sama persis dengan dataset:

```text
Pertanyaan training    : Siapa Reyga?
Variasi evaluasi       : Ceritakan tentang Reyga.
Pertanyaan fakta       : Di mana Reyga bekerja?
Pertanyaan tidak ada   : Berapa usia Reyga?
```

Catat hasilnya dalam tabel:

| Pertanyaan | Jawaban yang diharapkan | Jawaban model | Benar? |
|---|---|---|---|
| Siapa Reyga? | Profil Reyga | ... | Ya/Tidak |
| Di mana Reyga bekerja? | Askrindo | ... | Ya/Tidak |
| Berapa usia Reyga? | Informasi tidak tersedia | ... | Ya/Tidak |

Persentase sederhana dapat dihitung dengan:

```text
Akurasi = jumlah jawaban benar / jumlah seluruh pertanyaan × 100%
```

Bandingkan hasil menggunakan konfigurasi dan kumpulan pertanyaan yang sama agar eksperimen adil.

## Kesimpulan

```text
Lebih banyak data relevan dan berkualitas
→ memperkuat sinyal pembelajaran
→ meningkatkan probabilitas jawaban target

Konfigurasi test_model.py
→ memengaruhi cara token dipilih
→ memengaruhi konsistensi dan variasi jawaban
```

LoRA tidak menjamin jawaban persis seperti dataset dan `test_model.py` tidak mencari langsung ke dataset. Hasil terbaik diperoleh dengan dataset yang berkualitas dan seimbang, training yang sesuai, serta konfigurasi generation yang cocok dengan tujuan evaluasi.
