# Step 4 — Training LoRA dan TensorBoard

Step ini digunakan untuk melatih LoRA dan melihat hasil proses training melalui TensorBoard.

## Alur sederhana

```text
Dataset
   ↓
Base Model + LoRA membuat prediksi
   ↓
Prediksi dibandingkan dengan jawaban dataset
   ↓
Loss dan accuracy dihitung
   ↓
Parameter LoRA diperbarui
   ↓
Hasil disimpan ke folder output
```

Base model tidak berubah. Hasil training disimpan sebagai adapter LoRA.

## Dataset

Contoh `dataset/dataset.jsonl`:

```jsonl
{"messages":[{"role":"user","content":"Siapa Alfajri?"},{"role":"assistant","content":"Alfajri adalah programmer Java."}]}
```

```text
role=user      = pertanyaan
role=assistant = jawaban target atau kunci jawaban
```

Dataset yang konsisten dan memiliki beberapa variasi pertanyaan membuat jawaban yang diinginkan lebih mungkin muncul.

Saat model digunakan, model tidak mencari jawaban langsung di file JSONL. Hasil pembelajaran sudah tersimpan pada parameter LoRA.

## Token

Tokenizer mengubah teks menjadi token ID:

```text
"program" → tokenizer → token ID → model
```

Token ID hanya nomor token di dalam kamus tokenizer, bukan nilai kemiripan.

Model membuat jawaban satu token demi satu token:

```text
"Program" → " apa" → " yang" → " kamu" → " maksud" → "?"
```

Setiap token dipilih berdasarkan probabilitas dari konteks, base model, LoRA, dan konfigurasi generation.

Token di dalam kamus tidak dapat habis. Yang memiliki batas adalah jumlah token di dalam context window.

## Menjalankan training

Instal dependensi:

```bash
python3 -m pip install datasets peft trl tensorboardX
```

Pastikan path pada `train_peft.py` mengarah ke step4:

```python
DATASET_PATH = "/Users/fajri/Projects/testmodel/step4/dataset/dataset.jsonl"
OUTPUT_PATH = "/Users/fajri/Projects/testmodel/step4/output"
logging_dir = "/Users/fajri/Projects/testmodel/step4/tensorboard_logs"
```

Jalankan:

```bash
cd /Users/fajri/Projects/testmodel/step4
python3 train_peft.py
```

Hasil generate PEFT/LoRA akan tersimpan di:

```text
step4/output/
```

Log grafik akan tersimpan di:

```text
step4/tensorboard_logs/
```

## Menjalankan TensorBoard

```bash
docker run -d --rm \
  --name testmodel-tensorboard-native \
  -p 127.0.0.1:6006:6006 \
  -v /Users/fajri/Projects/testmodel/step4/tensorboard_logs:/logs:ro \
  python:3.11-slim \
  sh -c 'pip install --no-cache-dir "setuptools<81" tensorboard && exec tensorboard --logdir=/logs --host=0.0.0.0 --port=6006'
```

Buka:

```text
http://localhost:6006
```

Untuk menghentikan:

```bash
docker stop testmodel-tensorboard-native
```

## Accuracy, loss, dan entropy

### Accuracy

Accuracy berasal dari perbandingan:

```text
Prediksi Base Model + LoRA
              dibandingkan dengan
Jawaban target dataset (role=assistant)
```

Contoh:

```text
Target    : Alfajri adalah programmer Java.
Prediksi  : Alfajri adalah programmer Java.
```

Semakin banyak token yang cocok, semakin tinggi `token accuracy`.

Ini adalah akurasi terhadap dataset training, bukan jaminan bahwa jawaban untuk pertanyaan baru selalu benar.

### Analogi loss dan entropy

```text
Loss    = lurus atau melenceng dari jawaban target
Entropy = sempit atau lebar persebaran pilihan model
```

Misalnya jawaban target adalah `Java`:

```text
Loss rendah    = prediksi mengarah lurus ke Java
Loss tinggi    = prediksi melenceng dari Java

Entropy rendah = pilihan sempit, model lebih yakin
Entropy tinggi = pilihan lebar, model lebih ragu
```

Model bisa memiliki entropy rendah tetapi loss tinggi. Artinya, model sangat yakin tetapi yakin pada jawaban yang salah.

## Pembelian dan pemakaian token API

Layanan model terkenal seperti OpenAI, Anthropic Claude, dan Google Gemini pada umumnya menghitung biaya API berdasarkan token yang benar-benar diproses.

```text
Token input  = system prompt + memory + riwayat chat + pertanyaan
Token output = jawaban yang dibuat model

Biaya API = biaya token input + biaya token output
```

Harga input dan output dapat berbeda untuk setiap model. Beberapa layanan juga menghitung cached token, tool, atau fitur tambahan secara terpisah.

### Context window bukan kuota

```text
Context window = kapasitas maksimum dalam satu request
Kuota API      = batas atau saldo penggunaan
Token usage    = jumlah token yang benar-benar diproses
```

Misalnya sebuah model memiliki context window 1.000.000 token. Angka tersebut hanya menunjukkan kapasitas maksimum, bukan berarti setiap request langsung memakai 1.000.000 token.

```text
Input  = 10.000 token
Output =  2.000 token

Pemakaian request = 12.000 token
```

Jika input benar-benar berjumlah 1.000.000 token, hampir tidak ada ruang tersisa untuk output. Total input dan output harus tetap berada dalam batas context window.

### Menghapus history

Menghapus chat history atau memory dapat mengurangi token input pada request berikutnya. Namun pemakaian dan biaya request sebelumnya tidak dikembalikan.

```text
Request lama sudah diproses → tetap ditagihkan
History dihapus             → token request berikutnya lebih sedikit
```

Pada aplikasi chat, pengguna bisa membayar biaya langganan bulanan. Pada API untuk developer, pola yang umum adalah pay-as-you-go berdasarkan pemakaian, walaupun paket gratis, prabayar, dan kontrak perusahaan juga dapat tersedia.

Model yang dijalankan secara lokal seperti proyek ini tidak memiliki biaya API per token. Biayanya berasal dari perangkat, RAM, listrik, penyimpanan, dan waktu komputasi.

Dokumentasi resmi:

- [OpenAI — What are tokens?](https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them)
- [Anthropic — Pricing](https://docs.anthropic.com/en/docs/about-claude/pricing)
- [Google Gemini API — Billing](https://ai.google.dev/gemini-api/docs/billing)

## Cara membaca hasil training

```text
Loss turun               → prediksi semakin mendekati target
Token accuracy naik      → lebih banyak token target diprediksi benar
Gradient norm ekstrem    → training mungkin tidak stabil
Learning rate turun      → scheduler sedang bekerja
Entropy turun            → model semakin yakin terhadap prediksinya
```

TensorBoard hanya membantu melihat proses training. Kualitas jawaban tetap perlu diuji menggunakan pertanyaan baru yang tidak terdapat di dataset training.
