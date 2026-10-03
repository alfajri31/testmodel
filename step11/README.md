# 🟢 Step 11 — RAG, Style Jawaban, dan Inference

Step ini menjelaskan bagaimana RAG menyediakan fakta, lalu model menyusun jawaban
dengan style tertentu.

```text
Pertanyaan user
→ embedding query
→ pencarian dokumen MongoDB
→ natural text RAG
→ context window
→ model + adapter LoRA
→ jawaban
```

Step 11 tidak melakukan merge. Inference memakai base model dan adapter LoRA.

## 🟢 Peran komponen

RAG mencari fakta yang relevan dari MongoDB. Base model memahami bahasa dan
menyusun kalimat. Adapter LoRA mempelajari pola tambahan seperti gaya jawaban.
Context window membawa referensi RAG ke model hanya selama inference.

```text
RAG/database → fakta aktual
Base model   → kemampuan bahasa
Adapter      → pola tambahan dan style
```

## 🟢 Alur inference

1. Pertanyaan user diubah menjadi query vector.
2. Query dibandingkan dengan embedding dokumen MongoDB.
3. Dokumen dengan similarity tertinggi dipilih.
4. Natural text dimasukkan ke system prompt dan context window.
5. Model memproses prompt melalui layer Transformer.
6. `lm_head` memilih token jawaban satu per satu.
7. Jawaban ditampilkan dengan style aktif.

RAG menyediakan referensi, tetapi model masih dapat meringkas atau salah
menyalin daftar panjang.

## 🟢 Warna output

```text
Cyan    → similarity dan retrieval RAG
Kuning  → isi yang masuk context window
Hijau   → jawaban AI
Abu-abu → status proses
```

## 🟢 Style jawaban

```text
common       → jawaban dasar
moody        → dramatis dan reflektif
profesional  → formal dan terstruktur
comedian     → kocak dengan punchline
```

Style default adalah `profesional`. Ganti saat program berjalan:

```text
/style
Pilih gaya (common/moody/profesional/comedian): comedian
```

Semua style menggunakan `do_sample=False` supaya output lebih konsisten.

## 🟢 Dataset

```text
dataset/dataset.jsonl
→ contoh training dasar

dataset/response.jsonl
→ percakapan dasar sepak bola tanpa field style

dataset/response_style_long.example.jsonl
→ 200 contoh percakapan umum:
   common: 50
   moody: 50
   profesional: 50
   comedian: 50

dataset/multi_value.example.jsonl
→ contoh mempertahankan banyak item dalam satu jawaban
```

Dataset style sengaja tidak memakai nama klub atau pemain. Dataset tersebut hanya
mengajarkan cara berbicara.

## 🟢 Overfitting

Overfitting terjadi ketika model menghafal contoh training, bukan memahami pola
umumnya.

Jika dataset berulang kali berisi:

```text
User: Sebutkan pemain Inter Miami.
Jawaban: Lionel Messi dan Luis Suárez.
```

model dapat mengulang dua nama itu ketika pertanyaan berubah, walaupun RAG
memberikan daftar yang lebih panjang.

Karena itu, nama Inter Miami dan pemainnya tidak boleh mendominasi dataset style.
Fakta tersebut lebih tepat disimpan di RAG atau dataset fakta, sedangkan dataset
style memakai percakapan umum.

Cara mengurangi overfitting:

```text
1. Gunakan banyak parafrase pertanyaan.
2. Variasikan panjang dan konteks jawaban.
3. Pisahkan dataset style dari dataset fakta.
4. Uji dengan pertanyaan yang tidak sama persis dengan training.
```

## 🟢 Struktur folder

```text
step11/
├── dataset/
│   ├── dataset.jsonl
│   ├── response.jsonl
│   ├── response_style_long.example.jsonl
│   └── multi_value.example.jsonl
├── train_peft.py
├── test_rag_model_flow.py
├── requirements.txt
└── README.md
```

## 🟢 Menjalankan

Pastikan base model tersedia:

```bash
ls /Users/fajri/Projects/mytestmodel/config.json
```

Install dependency dan latih adapter:

```bash
python3 -m pip install -r step11/requirements.txt
python3 step11/train_peft.py
```

Jalankan RAG dan model:

```bash
python3 step11/test_rag_model_flow.py
```

Perintah interaktif:

```text
/style → mengganti style
/exit  → keluar
```

Contoh pertanyaan:

```text
Sebutkan para pemain Inter Miami CF.
Sebutkan salah satu pemain Inter Miami CF.
Bagaimana RAG menemukan data pemain?
Apa perbedaan RAG dan model bahasa?
```

## 🟢 Kesimpulan

```text
Training → mengajarkan pola dan style
RAG      → menyediakan fakta aktual
Context  → membawa fakta ke model
Inference→ merangkai jawaban
```

