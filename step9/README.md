# Step 9 — RAG dari MongoDB untuk Menjawab Data yang Tidak Ada di Model

Step ini menunjukkan cara model Step 8 menjawab daftar pemain Inter Miami CF ketika fakta pemain tidak diberikan melalui dataset fine-tuning Step 8. Sumber fakta saat inference adalah MongoDB.

Fakta pemain berasal dari MongoDB. Model hanya membaca fakta itu di context window dan merangkai jawabannya.

```text
MongoDB RAG memberi fakta
→ model Step 8 membaca fakta
→ model membuat jawaban natural
```

## Model dan database yang dipakai

```text
Model jawaban
→ /Users/fajri/Projects/testmodel/step8/merged_model

Database RAG
→ gosoccer_vector

Collection Indonesia
→ football_collection_id

Embedding model RAG
→ intfloat/multilingual-e5-small
```

Collection tersebut adalah database yang sama dengan `/Users/fajri/Projects/footballmodel/step`. Step 9 hanya membacanya. Script tidak mengubah koleksi atau menambahkan data baru.

URI MongoDB sudah disimpan statik pada bagian konfigurasi `test_model_rag.py`, mengikuti koneksi private yang dipakai project ini.

## Bentuk data di MongoDB

Sebelum RAG dipakai, data sepak bola telah diubah menjadi natural text dan embedding. Bentuk sederhananya:

```json
{
  "id": "inter-miami-roster-2025",
  "description": "Siapa saja pemain Inter Miami CF? Pemain Inter Miami CF yang tercatat antara lain Lionel Messi, Luis Suárez, Rodrigo De Paul, dan seterusnya.",
  "embedding": [0.021, -0.184, 0.552, "..."]
}
```

Arti setiap field:

```text
description
→ natural text yang nanti diberikan kepada LLM

embedding
→ vector untuk mencari description yang paling relevan
```

Vector di atas dibuat oleh embedding model, bukan oleh `lm_head` dan bukan contextual vector saat LLM menjawab.

## Alur ketika user bertanya

Misalnya user bertanya:

```text
Sebutkan nama pemain Inter Miami CF.
```

Step 9 menjalankan alur ini:

```text
Pertanyaan user
        ↓
Embedding model membuat query vector
        ↓
Bandingkan query vector dengan embedding semua description di MongoDB
        ↓
Ambil description dengan similarity tertinggi
        ↓
Masukkan description ke system prompt sebagai REFERENSI_RAG
        ↓
Model Step 8 memproses referensi + pertanyaan
        ↓
lm_head memilih token jawaban satu per satu
```

Perhitungan similarity pada contoh ini:

```text
query vector · document vector
→ similarity score
→ score tertinggi dipilih
```

Karena embedding query dan dokumen dinormalisasi, dot product pada script setara dengan cosine similarity.

## Mengapa model dapat menyebut nama yang tidak ada di bobotnya?

Contoh nama seperti Lionel Messi, Luis Suárez, atau Rodrigo De Paul tidak perlu ada dalam dataset Step 8 agar dapat disebut dalam jawaban ini. Nama tersebut dapat saja pernah muncul di data pretraining umum, tetapi Step 9 tidak mengandalkannya sebagai sumber fakta.

Natural text dari MongoDB sudah memasukkan nama tersebut ke context window:

```text
<REFERENSI_RAG>
Pemain Inter Miami CF yang tercatat antara lain Lionel Messi,
Luis Suárez, Rodrigo De Paul, dan seterusnya.
</REFERENSI_RAG>
```

Model kemudian memproses token referensi tersebut bersama token pertanyaan. Attention dapat memberi pengaruh lebih besar kepada token yang relevan, lalu `lm_head` menghasilkan token jawaban.

```text
RAG memberi: Lionel Messi, Luis Suárez, Rodrigo De Paul
        ↓
context window membawa token nama pemain
        ↓
model merangkai: "Pemain Inter Miami CF yang tercatat antara lain ..."
```

Model tidak membandingkan vector RAG langsung dengan vector `lm_head`. Ada dua perhitungan yang berbeda:

```text
RAG retrieval
→ query embedding dibandingkan dengan document embedding
→ memilih natural text dari MongoDB

LLM inference
→ natural text masuk context window
→ attention membentuk contextual vector
→ lm_head memilih next token
```

## Menjalankan contoh

Pastikan hasil merge Step 8 sudah tersedia:

```bash
ls /Users/fajri/Projects/testmodel/step8/merged_model/config.json
```

Pastikan package tersedia:

```bash
python3 -m pip install -r step9/requirements.txt
```

```bash
python3 step9/test_model_rag.py
```

Uji dengan pertanyaan:

```text
Sebutkan nama pemain Inter Miami CF.
```

Perintah interaktif:

```text
/rag  → menampilkan natural text dan similarity hasil retrieval terakhir
/exit → menutup program
```

## Jika similarity terlalu rendah

Script hanya memakai dokumen jika similarity setidaknya `0.65`:

```python
MIN_SIMILARITY = 0.65
```

Jika skor lebih rendah, script tidak mengirim dokumen tersebut kepada LLM dan memberi jawaban bahwa informasi tidak cukup. Nilai ini adalah titik awal; lihat hasil `/rag`, lalu sesuaikan berdasarkan data sebenarnya.

## Batas contoh ini

Step ini mengambil satu dokumen terbaik (`TOP_K = 1`) dan meminta LLM merangkai jawaban. Untuk daftar pemain yang harus lengkap 100%, aplikasi lebih baik mengembalikan daftar langsung dari database atau memakai query MongoDB terstruktur. LLM dapat menyingkat atau tidak menampilkan sebagian nama ketika menulis ulang text panjang.
