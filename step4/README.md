# Step 4 — Monitoring Training PEFT/LoRA dengan TensorBoard

Step 4 mempelajari cara mencatat dan membaca metrik training PEFT/LoRA melalui TensorBoard. TensorBoard digunakan untuk mengobservasi proses belajar model, bukan untuk menjalankan model atau membuktikan bahwa seluruh jawaban model benar secara faktual.

## Struktur folder

```text
step4/
├── dataset/
│   └── dataset.jsonl       # Pertanyaan dan jawaban target
├── output/                 # Adapter serta checkpoint hasil PEFT/LoRA
├── tensorboard_logs/       # Event log hasil training
├── README.md
├── test_model.py           # Inference/pengujian model
└── train_peft.py           # Training dan pencatatan TensorBoard
```

## Alur Step 4

```text
dataset/dataset.jsonl
          ↓
python train_peft.py
          ↓
┌──────────────────────────────┐
│ output/                      │ → adapter hasil PEFT/LoRA
│ tensorboard_logs/            │ → metrik proses training
└──────────────────────────────┘
          ↓
TensorBoard membaca event log
          ↓
Dashboard di browser
```

Folder `output` dan `tensorboard_logs` sama-sama dihasilkan oleh proses training, tetapi isinya berbeda:

```text
output/
→ bobot adapter, konfigurasi, tokenizer, dan checkpoint

tensorboard_logs/
→ event metrik seperti loss, accuracy, gradient, dan learning rate
```

## Dataset sebagai soal dan jawaban target

Setiap baris JSONL merupakan satu contoh training:

```jsonl
{"messages":[{"role":"user","content":"Siapa Alfajri?"},{"role":"assistant","content":"Alfajri adalah programmer Java."}]}
```

Pemetaan:

```text
role=user      → input/pertanyaan
role=assistant → jawaban target
```

Tidak ada pengguna nyata yang bertanya selama training. `SFTTrainer` mengambil pertanyaan dan jawaban target dari dataset secara otomatis.

```text
SFTTrainer membaca satu contoh
          ↓
Chat template dan tokenizer membuat token
          ↓
Base Model + LoRA memprediksi token berikutnya
          ↓
Prediksi dibandingkan dengan token target
          ↓
Loss dan gradient dihitung
          ↓
Parameter LoRA diperbarui
```

Model tidak membuat pertanyaan baru dan tidak bertanya kepada dirinya sendiri. Model hanya memprediksi token berikutnya dari urutan yang diberikan trainer.

### Dari mana keakuratan training diperoleh?

Keakuratan training diperoleh dengan membandingkan dua hal:

```text
1. Prediksi dari Base Model + metode LoRA
2. Jawaban target dari dataset pada role=assistant
```

Alurnya:

```text
Dataset
├── role=user      → pertanyaan/input
└── role=assistant → jawaban target/kunci jawaban
          ↓
Pertanyaan diberikan kepada Base Model + LoRA
          ↓
Model menghasilkan probabilitas prediksi token
          ↓
Prediksi dibandingkan dengan token jawaban dataset
          ↓
Loss dan token accuracy dihitung
          ↓
Parameter LoRA diperbarui
```

Secara konseptual:

```text
Keakuratan training
= kecocokan prediksi (Base Model + LoRA)
  terhadap target jawaban dataset
```

Dataset tidak digabung menjadi bagian permanen model ketika perbandingan dilakukan. Dataset berperan sebagai sumber input dan kunci jawaban, sedangkan model aktif selama training adalah Base Model yang telah dipasangi parameter tambahan LoRA.

Contoh:

```text
Pertanyaan dataset : Siapa Alfajri?
Target dataset     : Alfajri adalah programmer Java.

Prediksi model     : Alfajri adalah programmer Java.
Hasil              : token banyak yang cocok, accuracy naik
```

Jika prediksinya berbeda:

```text
Prediksi model : Alfajri adalah seorang penyanyi.
Target dataset : Alfajri adalah programmer Java.

Hasil:
- token yang berbeda dianggap salah
- loss meningkat
- gradient mengarahkan parameter LoRA mendekati target
```

Keakuratan ini disebut **training token accuracy**. Nilai tersebut mengukur kecocokan terhadap dataset training, bukan jaminan akurasi faktual atau kemampuan menjawab pertanyaan baru.

## Konfigurasi TensorBoard

### Install writer TensorBoard untuk training

Environment Python yang menjalankan `train_peft.py` memerlukan backend untuk menulis event log TensorBoard. Pada setup proyek ini digunakan `tensorboardX`:

```bash
source /path/ke/virtual-environment/bin/activate
python -m pip install tensorboardX
```

Verifikasi instalasi pada interpreter yang sama dengan training:

```bash
which python
python -c "import tensorboardX; print(tensorboardX.__version__)"
```

Jika virtual environment tidak sedang aktif, package dapat terpasang pada Python yang berbeda dan `TensorBoardCallback` tetap menghasilkan error.

Pembagian fungsinya dalam setup Step 4:

```text
tensorboardX pada environment training
→ menulis events.out.tfevents.* saat train_peft.py berjalan

tensorboard pada container Docker
→ membaca event log dan menampilkan dashboard web
```

`tensorboardX` tidak menyediakan command dashboard berikut:

```bash
tensorboard --logdir ...
```

Karena itu, menginstal `tensorboardX` saja tidak membuat CLI `tensorboard` tersedia di terminal host. Dashboard pada proyek ini dijalankan melalui Docker, sedangkan `tensorboardX` digunakan oleh proses training lokal untuk menghasilkan log.

Sebagai alternatif, package `tensorboard` juga dapat dipasang langsung pada environment training:

```bash
python -m pip install tensorboard
```

Untuk setup yang didokumentasikan di Step 4, cukup gunakan `tensorboardX` pada environment training dan TensorBoard di dalam container Docker.

`train_peft.py` menggunakan:

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
    report_to="tensorboard",
    logging_dir=(
        "/Users/fajri/Projects/testmodel/"
        "step4/tensorboard_logs"
    ),
)
```

Bagian penting:

```text
report_to="tensorboard" → mengaktifkan TensorBoard callback
logging_dir=...          → menentukan lokasi event log
logging_steps=1          → mencatat metrik setiap training step
```

## Menjalankan training

Aktifkan Python environment yang berisi dependensi proyek, lalu jalankan:

```bash
cd /Users/fajri/Projects/testmodel
python step4/train_peft.py
```

Setelah training dimulai, file berikut akan dibuat:

```text
step4/tensorboard_logs/events.out.tfevents.*
```

Periksa:

```bash
find /Users/fajri/Projects/testmodel/step4/tensorboard_logs \
  -name "events.out.tfevents.*" \
  -print
```

## Menjalankan TensorBoard dengan Docker

Image resmi TensorFlow `amd64` sempat berjalan melalui `qemu-x86_64` dan tidak membuka port pada Mac Apple Silicon. Setup yang berhasil menggunakan image Python multi-architecture.

Jalankan:

```bash
docker run -d --rm \
  --name testmodel-tensorboard-native \
  -p 127.0.0.1:6006:6006 \
  -v /Users/fajri/Projects/testmodel/step4/tensorboard_logs:/logs:ro \
  python:3.11-slim \
  sh -c 'pip install --no-cache-dir "setuptools<81" tensorboard && exec tensorboard --logdir=/logs --host=0.0.0.0 --port=6006'
```

Port diikat ke `127.0.0.1` agar dashboard hanya dapat diakses dari komputer lokal.

Pantau startup:

```bash
docker logs -f testmodel-tensorboard-native
```

Buka:

```text
http://localhost:6006
```

Perintah operasional:

```bash
# Melihat status
docker ps --filter name=testmodel-tensorboard-native

# Memeriksa port
docker port testmodel-tensorboard-native

# Menghentikan dashboard
docker stop testmodel-tensorboard-native
```

Karena container memakai `--rm`, container akan terhapus setelah dihentikan. Event log tetap aman di folder lokal.

## Data yang tersedia di dashboard

Event log Step 4 saat ini berisi scalar berikut:

```text
train/loss
train/grad_norm
train/learning_rate
train/entropy
train/num_tokens
train/mean_token_accuracy
train/epoch
train/train_runtime
train/train_samples_per_second
train/train_steps_per_second
train/total_flos
train/train_loss
```

Log saat ini belum berisi:

```text
histograms
distributions
embeddings/projector
model graph
```

Karena itu, dashboard yang langsung tersedia adalah **Time Series/Scalars**. Histogram bobot dan titik embedding memerlukan logging tambahan.

## Interpretasi metrik training

### Loss

Loss mengukur seberapa salah prediksi token model terhadap token target dari dataset.

```text
Loss turun → prediksi semakin mendekati target dataset
Loss tinggi → probabilitas token target masih rendah
```

Cross-entropy loss untuk satu target dapat disederhanakan menjadi:

```text
loss = -log(probabilitas token target)
```

Training loss rendah tidak otomatis berarti model benar secara faktual atau mampu menjawab pertanyaan baru.

### Mean token accuracy

Token accuracy menghitung persentase posisi token ketika prediksi dengan probabilitas tertinggi sama dengan target.

```text
Token accuracy naik
→ lebih banyak token target berhasil diprediksi dengan benar
```

Ini adalah kecocokan terhadap teks target, bukan pengukuran pemahaman atau kebenaran dunia nyata.

Perbandingannya dapat diringkas sebagai:

```text
Input pertanyaan dari dataset
          ↓
Base Model + metode LoRA membuat prediksi
          ↓
Prediksi token dibandingkan dengan role=assistant dari dataset
          ↓
Persentase token yang cocok = mean token accuracy
```

### Entropy

Entropy mengukur seberapa lebar probabilitas model tersebar ke berbagai pilihan token.

```text
Entropy tinggi → probabilitas tersebar, model lebih ragu
Entropy rendah → probabilitas terkumpul, model lebih yakin
```

Analogi:

```text
Loss    → seberapa tepat arah prediksi menuju target
Entropy → seberapa lebar pilihan probabilitas tersebar
```

Dengan kata lain, perbandingan sederhananya adalah:

```text
Loss    = lurus atau melenceng terhadap jawaban target
Entropy = sempit atau lebar persebaran pilihan model
```

Loss membutuhkan jawaban target dari dataset sebagai pembanding. Entropy tidak menilai apakah pilihan tersebut benar; entropy hanya mengukur apakah probabilitas terkumpul pada sedikit pilihan atau tersebar ke banyak pilihan.

| Kondisi prediksi | Loss | Entropy | Interpretasi |
|---|---:|---:|---|
| Probabilitas terkumpul pada target yang benar | Rendah | Rendah | Yakin dan benar |
| Probabilitas terkumpul pada token yang salah | Tinggi | Rendah | Yakin tetapi salah |
| Probabilitas tersebar ke banyak token | Bergantung target | Tinggi | Model masih ragu |

Visual sederhananya:

```text
Entropy rendah/sempit:
             ●●●●●
                ↑
       pilihan terkonsentrasi

Entropy tinggi/lebar:
       ●    ●    ●    ●
                ↑
         pilihan tersebar
```

Contoh target token `Java`:

```text
Java 90%, Python 5%, Kotlin 5%
→ loss rendah dan entropy rendah

Java 1%, Python 98%, Kotlin 1%
→ loss tinggi tetapi entropy rendah

Java 35%, Python 35%, Kotlin 30%
→ model ragu sehingga entropy tinggi
```

Model dapat memiliki entropy rendah tetapi loss tinggi jika sangat yakin pada token yang salah.

```text
Yakin dan benar       → loss rendah, entropy rendah
Yakin tetapi salah    → loss tinggi, entropy rendah
Ragu-ragu             → entropy tinggi
```

### Gradient norm

Gradient norm menunjukkan besarnya sinyal yang akan memperbarui parameter LoRA.

```text
Gradient stabil             → pembaruan relatif terkendali
Gradient sering sangat besar → training mungkin tidak stabil
Gradient mendekati nol terus → pembelajaran mungkin terlalu lemah
```

Satu lonjakan belum tentu masalah. Perhatikan pola yang terjadi berulang kali bersama loss.

### Learning rate

Learning rate menentukan besar langkah pembaruan parameter.

```text
Learning rate turun
→ scheduler sedang mengurangi ukuran langkah training
```

Grafik learning rate lebih banyak ditentukan konfigurasi scheduler daripada kualitas prediksi.

### Number of tokens

`train/num_tokens` menunjukkan jumlah token yang telah diproses.

```text
num_tokens naik → lebih banyak data/token sudah dilalui
```

Ini mengukur volume pekerjaan, bukan kualitas model.

### Epoch

Epoch menunjukkan berapa kali seluruh dataset diproses.

```text
epoch 1.0 → dataset selesai diproses satu kali
epoch 5.0 → dataset selesai diproses lima kali
```

Dataset kecil dengan epoch terlalu banyak dapat meningkatkan risiko menghafal.

### Runtime dan throughput

```text
train_runtime             → total durasi training
train_samples_per_second  → contoh yang diproses per detik
train_steps_per_second    → step yang diselesaikan per detik
total_flos                → perkiraan total operasi komputasi
```

Metrik tersebut mengukur performa dan biaya komputasi, bukan akurasi jawaban.

### `train/loss` dan `train/train_loss`

```text
train/loss       → loss lokal yang dicatat per interval/step
train/train_loss → rata-rata/ringkasan loss seluruh training
```

## Pengaruh dataset, base model, konfigurasi, dan hardware

Semua metrik tidak dipengaruhi oleh satu faktor yang sama.

```text
Dataset            → apa dan seberapa sulit yang dipelajari
Base model         → kemampuan serta distribusi awal
LoRA/configuration → bagaimana parameter tambahan dilatih
Hardware           → seberapa cepat proses berjalan
```

### Dataset

Dataset memengaruhi loss, token accuracy, entropy, gradient, jumlah token, dan jumlah step.

```text
Data sederhana dan berulang
→ lebih mudah dihafal
→ training loss dapat cepat turun

Data beragam atau bertentangan
→ prediksi lebih sulit
→ loss dapat lebih tinggi atau tidak stabil
```

Data lebih banyak hanya membantu jika relevan, konsisten, berkualitas, dan seimbang.

### Base model

Base model menentukan kemampuan awal. Model yang sudah memahami bahasa dan domain target biasanya lebih mudah disesuaikan.

```text
Base model sudah mengenal pola/domain
→ perubahan LoRA yang diperlukan lebih kecil

Base model lemah pada domain
→ membutuhkan sinyal training lebih kuat
```

Ukuran base model juga memengaruhi RAM, runtime, throughput, dan FLOPs.

### Konfigurasi training

Konfigurasi yang berpengaruh antara lain:

```text
learning_rate
num_train_epochs
batch_size
gradient_accumulation_steps
max_length
rank dan alpha LoRA
target_modules
gradient_checkpointing
```

Learning rate dan epoch terutama ditentukan oleh konfigurasi, bukan dipilih otomatis berdasarkan kualitas dataset.

### Hardware

Hardware memengaruhi kecepatan, kapasitas batch, precision, dan ukuran model yang dapat dimuat. Hardware yang lebih cepat tidak otomatis meningkatkan akurasi jika data dan konfigurasi tetap sama.

## Training dan inference berbeda

Metrik TensorBoard di atas dicatat saat `train_peft.py` berjalan:

```text
Training
→ memiliki jawaban target
→ menghitung loss dan gradient
→ memperbarui LoRA
```

Saat `test_model.py` menghasilkan jawaban:

```text
Inference
→ biasanya tidak memiliki jawaban target
→ tidak menghitung training loss
→ tidak melakukan backpropagation
→ tidak memperbarui parameter
```

`torch.inference_mode()` pada script pengujian memastikan inference tidak membuat gradient.

Untuk mengukur kualitas jawaban inference, diperlukan evaluation dataset terpisah yang memiliki pertanyaan dan jawaban acuan.

## Cara membaca dashboard

Pada halaman TensorBoard:

1. Buka tab **Time Series** atau **Scalars**.
2. Pilih tag seperti `train/loss` atau `train/mean_token_accuracy`.
3. Gunakan horizontal axis **Step**.
4. Atur smoothing ke `0` untuk melihat nilai asli setiap step.
5. Naikkan smoothing hanya untuk melihat tren umum.

Smoothing hanya mengubah visualisasi, bukan isi log.

Kondisi training yang relatif sehat:

```text
Loss turun
Token accuracy naik
Entropy turun secara bertahap
Gradient tidak terus-menerus ekstrem
Learning rate mengikuti scheduler
Throughput relatif stabil
```

Kemungkinan menghafal atau overfitting:

```text
Training loss sangat rendah
Training accuracy sangat tinggi
Dataset kecil dan epoch banyak
Hasil pada pertanyaan baru tetap buruk
```

## Batasan TensorBoard

TensorBoard tidak otomatis:

- Menjamin model faktual.
- Menghilangkan halusinasi.
- Menunjukkan kalimat dataset tersimpan di layer tertentu.
- Mencari fakta dalam dataset ketika inference.
- Membuktikan generalisasi hanya dari training loss.

Ringkasnya:

```text
TensorBoard = alat observasi proses training
Dataset     = input dan target pembelajaran
LoRA        = parameter tambahan yang dilatih
Evaluation  = pengujian kualitas hasil model
```

TensorBoard membantu memahami apakah proses training berjalan sesuai harapan. Kualitas model tetap harus dibuktikan melalui evaluasi pada data yang tidak digunakan untuk training.
