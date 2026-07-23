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

## Cara membaca hasil training

```text
Loss turun               → prediksi semakin mendekati target
Token accuracy naik      → lebih banyak token target diprediksi benar
Gradient norm ekstrem    → training mungkin tidak stabil
Learning rate turun      → scheduler sedang bekerja
Entropy turun            → model semakin yakin terhadap prediksinya
```

TensorBoard hanya membantu melihat proses training. Kualitas jawaban tetap perlu diuji menggunakan pertanyaan baru yang tidak terdapat di dataset training.
