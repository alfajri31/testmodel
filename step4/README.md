# Step 4 — Melihat Training LoRA dengan TensorBoard

Pada step ini kita melatih LoRA dan melihat proses training melalui TensorBoard.

## Struktur folder

```text
step4/
├── dataset/
│   └── dataset.jsonl
├── train_peft.py
├── test_model.py
├── output/                 # hasil generate PEFT/LoRA
└── tensorboard_logs/       # data grafik TensorBoard
```

## Cara kerja sederhananya

```text
Dataset
   ↓
Base model + LoRA mencoba memprediksi jawaban
   ↓
Prediksi dibandingkan dengan jawaban dataset
   ↓
Loss dihitung
   ↓
Parameter LoRA diperbarui
   ↓
Hasil disimpan ke folder output
```

Base model tidak ditimpa. Hanya parameter tambahan LoRA yang dilatih dan disimpan.

## Format dataset

Contoh satu baris `dataset.jsonl`:

```jsonl
{"messages":[{"role":"user","content":"Siapa Alfajri?"},{"role":"assistant","content":"Alfajri adalah programmer Java."}]}
```

Artinya:

```text
role=user      = pertanyaan atau input
role=assistant = jawaban target
```

Saat training, model memprediksi jawaban lalu membandingkannya dengan `role=assistant`.

Dataset yang relevan, konsisten, dan memiliki variasi pertanyaan membantu meningkatkan kemungkinan munculnya jawaban yang diinginkan.

```jsonl
{"messages":[{"role":"user","content":"Siapa Alfajri?"},{"role":"assistant","content":"Alfajri adalah programmer Java."}]}
{"messages":[{"role":"user","content":"Apa pekerjaan Alfajri?"},{"role":"assistant","content":"Alfajri bekerja sebagai programmer Java."}]}
```

Saat model digunakan untuk menjawab, model tidak mencari langsung ke file JSONL. Hasil training sudah memengaruhi parameter LoRA sehingga jawaban target menjadi lebih mungkin muncul.

## Token secara sederhana

Tokenizer mengubah teks menjadi token ID:

```text
"program" → tokenizer → token ID → model
```

Token ID hanyalah nomor token dalam kamus tokenizer, bukan nilai kemiripan.

Model kemudian membuat jawaban satu token demi satu token berdasarkan probabilitas:

```text
"Program" → " apa" → " yang" → " kamu" → " maksud" → "?"
```

Probabilitas dipengaruhi oleh:

- pesan user dan riwayat chat;
- base model;
- LoRA hasil training; dan
- konfigurasi generation.

Token dalam kosakata tokenizer tidak dapat habis. Yang memiliki batas adalah jumlah token yang dapat masuk ke context window.

## Persiapan

Aktifkan virtual environment yang digunakan untuk training, lalu instal dependensi:

```bash
python3 -m pip install datasets peft trl tensorboardX
```

`tensorboardX` dipakai oleh proses training untuk menulis log. Dashboard TensorBoard dijalankan melalui Docker.

Pastikan path di `train_peft.py` menunjuk ke folder step4:

```python
DATASET_PATH = "/Users/fajri/Projects/testmodel/step4/dataset/dataset.jsonl"
OUTPUT_PATH = "/Users/fajri/Projects/testmodel/step4/output"
logging_dir = "/Users/fajri/Projects/testmodel/step4/tensorboard_logs"
```

## Menjalankan training

Dari folder `step4`:

```bash
python3 train_peft.py
```

Hasil running `train_peft.py`:

```text
output/
├── adapter_config.json
├── adapter_model.safetensors
└── file tokenizer
```

File `adapter_model.safetensors` berisi parameter LoRA hasil training, bukan seluruh base model.

Log untuk grafik akan muncul di:

```text
step4/tensorboard_logs/
```

## Menjalankan TensorBoard dengan Docker

Dari folder utama proyek:

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

Cek log container:

```bash
docker logs -f testmodel-tensorboard-native
```

Hentikan TensorBoard:

```bash
docker stop testmodel-tensorboard-native
```

## Arti grafik utama

| Grafik | Arti sederhana |
|---|---|
| Loss | Seberapa jauh prediksi dari jawaban target |
| Token accuracy | Persentase token prediksi yang cocok dengan target |
| Entropy | Seberapa lebar pilihan jawaban model |
| Gradient norm | Besar perubahan saat model belajar |
| Learning rate | Kecepatan pembaruan parameter |
| Epoch | Berapa kali seluruh dataset dipelajari |

### Dari mana nilai keakuratan berasal?

Keakuratan training diperoleh dengan membandingkan:

```text
Prediksi Base Model + LoRA
              dibandingkan dengan
Jawaban target dari dataset (role=assistant)
```

Contoh:

```text
Pertanyaan dataset : Siapa Alfajri?
Target dataset     : Alfajri adalah programmer Java.
Prediksi model     : Alfajri adalah programmer Java.

Hasil:
Semakin banyak token prediksi yang cocok dengan target,
semakin tinggi nilai token accuracy.
```

Jika model memprediksi `Alfajri adalah penyanyi`, token yang berbeda dari target dianggap salah sehingga accuracy lebih rendah dan loss dapat meningkat.

Nilai ini adalah **training token accuracy**, yaitu keakuratan prediksi terhadap dataset training. Nilai tersebut bukan jaminan bahwa model selalu akurat ketika menerima pertanyaan baru dari user.

Perbandingan sederhananya:

```text
Loss    = lurus atau melenceng dari jawaban target
Entropy = sempit atau lebar persebaran pilihan jawaban
```

Perumpamaannya:

```text
Jawaban target = Java

Loss rendah:
Prediksi model mengarah lurus ke Java.

Loss tinggi:
Prediksi model melenceng dari Java.

Entropy rendah:
Pilihan model sempit dan model lebih yakin.

Entropy tinggi:
Pilihan model tersebar lebar dan model lebih ragu.
```

Jadi, **loss melihat ketepatan arah terhadap target**, sedangkan **entropy melihat lebar pilihan model**.

Model dapat memiliki entropy rendah tetapi loss tinggi. Artinya, model sangat yakin, tetapi yakin pada jawaban yang salah.

Umumnya proses training terlihat membaik ketika:

```text
Loss turun
Token accuracy naik
Gradient tetap stabil
```

Entropy yang turun berarti pilihan model semakin sempit, tetapi belum tentu jawabannya benar.

## Hal penting

- TensorBoard menampilkan proses training, bukan proses model menjawab user.
- Training accuracy membandingkan prediksi dengan jawaban dataset.
- Training accuracy tinggi belum menjamin model akurat pada pertanyaan baru.
- Dataset sedikit dan epoch terlalu banyak dapat menyebabkan model menghafal.
- Kualitas hasil dipengaruhi dataset, base model, konfigurasi training, dan konfigurasi generation.
- Pengujian tetap perlu dilakukan menggunakan pertanyaan yang tidak ada di dataset training.
