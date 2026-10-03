# 🟢 Step 11 — Hubungan RAG dan Model Saat Menjawab

Step ini menjelaskan bagaimana data dari RAG dapat dipakai model untuk menjawab pertanyaan, meskipun fakta tersebut tidak diberikan melalui dataset fine-tuning model.

Contoh utama:

```text
User bertanya:
“Sebutkan pemain Inter Miami.”

MongoDB/RAG menyediakan fakta pemain.
LLM membaca fakta tersebut di context window.
LLM merangkai jawaban secara natural.
```

RAG bukan pengganti model bahasa. RAG menyediakan referensi; model menyediakan kemampuan memahami instruksi dan menyusun jawaban.

## 🟢 Arsitektur singkat

```text
Pertanyaan user
        ↓
Embedding model RAG
        ↓
Query vector
        ↓
Pencarian similarity di database
        ↓
Natural text yang relevan
        ↓
System prompt + referensi + pertanyaan user
        ↓
LLM inference
        ↓
Transformer layer 1 sampai 28
        ↓
Contextual vector akhir
        ↓
lm_head memilih next token
        ↓
Jawaban user
```

## 🟢 Data di database

Dokumen RAG biasanya menyimpan natural text dan embedding-nya:

```json
{
  "id": "inter-miami-roster",
  "text": "Pemain Inter Miami yang tercatat antara lain Lionel Messi, Luis Suárez, dan Rodrigo De Paul.",
  "embedding": [0.021, -0.184, 0.552]
}
```

Field tersebut memiliki fungsi berbeda:

```text
text
→ fakta yang nanti dibaca oleh LLM

embedding
→ angka untuk mencari dokumen yang mirip dengan pertanyaan
```

Embedding RAG bukan embedding input LLM, bukan contextual vector, dan bukan output `lm_head`. Embedding RAG dipakai lebih dulu pada tahap pencarian dokumen.

Step 11 tidak memakai dataset RAG lokal. Sumber fakta diambil langsung dari MongoDB yang dikonfigurasi pada helper Step 9. Field `text` atau `description` dan `embedding` harus sudah tersedia di collection tersebut.

Gaya jawaban adalah data yang berbeda dari fakta RAG. Contoh format gaya tersedia di:

```text
step11/dataset/response_styles.example.jsonl
```

Satu fakta dapat dirangkai dengan gaya berbeda:

```text
tegas       → langsung dan singkat
moody       → reflektif dan lebih panjang
profesional → terstruktur dan formal
```

Fakta tetap berasal dari RAG. Gaya hanya mengatur cara base model merangkai fakta tersebut.

## 🟢 Tahap 1 — RAG mencari referensi

User bertanya:

```text
Sebutkan pemain Inter Miami.
```

Embedding model RAG mengubah pertanyaan menjadi query vector:

```text
pertanyaan user
→ query vector
```

Query vector dibandingkan dengan embedding dokumen di database:

```text
query vector · embedding dokumen
→ similarity score
```

Dokumen dengan similarity paling tinggi dipilih:

```text
Pemain Inter Miami yang tercatat antara lain
Lionel Messi, Luis Suárez, dan Rodrigo De Paul.
```

Pada tahap ini belum ada jawaban dari LLM. RAG baru memilih sumber informasi.

## 🟢 Tahap 2 — Referensi dimasukkan ke context model

Aplikasi membuat prompt untuk model:

```text
System:
Gunakan referensi berikut sebagai sumber fakta utama.
Jika referensi tidak cukup, katakan informasi tidak ditemukan.

Referensi RAG:
Pemain Inter Miami yang tercatat antara lain Lionel Messi,
Luis Suárez, dan Rodrigo De Paul.

User:
Sebutkan pemain Inter Miami.

Assistant:
```

Secara teknis, pesan tersebut diubah oleh chat template menjadi token. Token dari system, referensi RAG, dan user semuanya masuk ke context window.

```text
system + referensi RAG + user
→ token context window
```

Model tidak menerima embedding RAG sebagai pengganti token. Model menerima natural text hasil retrieval, lalu membuat embedding/contextual vector-nya sendiri melalui embedding table dan Transformer.

## 🟢 Tahap 3 — Model melakukan inference

Setelah referensi berada di context window, model menjalankan inference:

```text
token context window
→ embedding awal
→ Transformer layer 1
→ Transformer layer 2
→ ...
→ Transformer layer 28
→ contextual vector akhir
→ lm_head
```

`lm_head` memberi skor pada seluruh kandidat token. Misalnya, pada langkah tertentu:

```text
“Lionel”  → probabilitas tinggi
“Pemain”  → probabilitas tinggi
“Jakarta” → probabilitas rendah
```

Model memilih satu token, memasukkannya kembali ke context window, lalu memprediksi token berikutnya:

```text
“Pemain”
→ “Inter”
→ “Miami”
→ “yang”
→ “tercatat”
→ ...
```

Jawaban natural dapat menjadi:

```text
Pemain Inter Miami yang tercatat antara lain Lionel Messi,
Luis Suárez, dan Rodrigo De Paul.
```

Kalimat tersebut tidak harus tersimpan persis di database. Database memberi fakta; base model menyusun kalimat token demi token.

## 🟢 Dua perhitungan yang berbeda

RAG dan LLM sama-sama menggunakan vector, tetapi pekerjaannya berbeda:

```text
RAG retrieval
→ query embedding dibandingkan dengan document embedding
→ memilih natural text yang relevan

LLM inference
→ natural text masuk context window
→ attention membentuk contextual vector
→ lm_head memilih next token
```

Jangan menyamakan similarity RAG dengan probabilitas token LLM:

```text
similarity RAG
→ seberapa mirip pertanyaan dengan dokumen

probabilitas lm_head
→ seberapa mungkin kandidat token menjadi token berikutnya
```

## 🟢 Peran base model dan RAG

```text
Base model
→ tata bahasa
→ kemampuan mengikuti instruksi
→ pola percakapan
→ kemampuan merangkai jawaban

RAG/database/API
→ fakta khusus aplikasi
→ data terbaru
→ data yang tidak perlu dimasukkan ke bobot model
```

Jika RAG mengembalikan daftar pemain, model dapat mengubahnya menjadi kalimat yang lebih natural. Jika RAG tidak menemukan fakta yang cukup, system prompt harus meminta model untuk tidak mengarang.

## 🟢 RAG tidak mengubah bobot model

Saat user bertanya, RAG hanya mengambil data dan memasukkannya ke context window. Bobot model tidak diperbarui:

```text
RAG retrieval
→ context sementara
→ inference
→ response
→ bobot tetap
```

Jika chat dihapus, referensi RAG yang sudah masuk context dan KV cache dapat dilepas. Model tidak kehilangan bobot atau pengetahuan permanennya; model hanya kehilangan konteks sementara percakapan tersebut.

## 🟢 Ringkasan

```text
1. User mengirim pertanyaan.
2. Embedding model mengubah pertanyaan menjadi query vector.
3. RAG mencari dokumen paling relevan di database.
4. Natural text hasil RAG dimasukkan ke context window.
5. LLM memproses context melalui 28 Transformer layer.
6. lm_head memilih token jawaban satu per satu.
7. Jawaban dikirim ke user.
```

Kesimpulan utamanya:

> RAG memberikan fakta yang harus dibaca model, sedangkan base model menentukan bagaimana fakta tersebut dipahami dan dirangkai menjadi jawaban.

## 🟢 Menjalankan script

Step 11 dapat memakai adapter hasil training Step 11 untuk demo RAG. Jika belum ada, script memakai adapter Step 8 sebagai fallback. Step ini tidak melakukan merge.

### Tahap 1 — Training adapter Step 11

Pastikan base model tersedia:

```bash
ls /Users/fajri/Projects/mytestmodel/config.json
```

Jalankan dari root project:

```bash
python3 step11/train_peft.py
```

Output penting:

```text
step11/output/
└── adapter_config.json
```

### Tahap 2 — Menjalankan RAG dan model

Install dependency:

```bash
python3 -m pip install -r step11/requirements.txt
```

Jalankan dari root project:

```bash
python3 step11/test_rag_model_flow.py
```

Script menampilkan tahapnya secara berurutan:

```text
[1/5] memuat embedding model RAG
[2/5] memuat model jawaban
[3/5] membaca dokumen MongoDB
[4/5] mencari referensi dengan similarity
[5/5] memasukkan natural text ke context window dan membuat jawaban
```

Gaya default script adalah `profesional`. Untuk menggantinya saat program berjalan:

```text
/style
→ pilih: tegas, moody, atau profesional
```

Perintah `/style` hanya mengubah instruksi `system` di context window. Perintah tersebut tidak mengubah bobot model.

Script Step 11 menggunakan helper dan koneksi database yang sama dengan Step 9. MongoDB hanya dibaca; script tidak menambahkan atau mengubah dokumen.
