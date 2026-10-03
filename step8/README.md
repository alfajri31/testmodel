# 🟢 Step 8 — Training PEFT Standalone

Step ini menunjukkan contoh training adapter LoRA yang berdiri sendiri. Step ini tidak melakukan merge dan tidak mengambil output dari Step lain.

Sumber modelnya adalah base model langsung:

```text
/Users/fajri/Projects/mytestmodel
```

Saat `train_peft.py` berjalan, base model dimuat, adapter LoRA baru dibuat, lalu adapter tersebut dilatih memakai dua contoh di `dataset/dataset.jsonl`.

```text
base model Qwen3-0.6B
        ↓
adapter LoRA Step 8 dilatih
        ↓
step8/output
```

## 🟢 Apa yang dimaksud standalone?

Standalone berarti Step 8 memiliki jalur input dan output sendiri:

```text
base model
→ adapter LoRA Step 8
→ step8/output
```

Output Step 8 tidak menimpa output Step 5, Step 9, atau Step 11.

## 🟢 Struktur folder

```text
step8/
├── dataset/
│   └── dataset.jsonl       # Contoh data tambahan
├── train_peft.py       # Training adapter LoRA standalone
├── test_model.py           # Uji base model + adapter LoRA
├── output/                 # Dibuat setelah training
```

## 🟢 Dataset contoh

Dataset pada step ini sengaja hanya memiliki dua contoh tentang informasi yang sedang dipelajari Alfajri.

Setelah training, uji dua hal berikut:

```text
Apa pekerjaan Alfajri?
→ diharapkan masih menjawab programmer Java

Alfajri sedang belajar apa?
→ diharapkan menjawab machine learning
```

Hasil dapat berubah karena dataset contoh sangat kecil. Tujuannya adalah melihat alur training standalone dan checkpoint, bukan membuat model pengetahuan yang stabil.

## 🟢 Menjalankan

Pastikan base model tersedia:

```bash
ls /Users/fajri/Projects/mytestmodel/config.json
```

Jalankan training Step 8:

```bash
cd /Users/fajri/Projects/testmodel
python3 step8/train_peft.py
```

Hasilnya adalah adapter baru di:

```text
step8/output/
```

Karena training memakai `save_strategy="epoch"` dan tiga epoch, output biasanya berisi checkpoint dari Step 8 sendiri:

```text
step8/output/
├── checkpoint-1/
├── checkpoint-2/
├── checkpoint-3/
└── adapter_config.json
```

Checkpoint tersebut adalah snapshot adapter selama training Step 8. Base model tetap berada di `/Users/fajri/Projects/mytestmodel`.

Jika folder `step8/output/checkpoint-*` sudah ada, `train_peft.py` akan menampilkan daftar checkpoint dan meminta pilihan:

```text
Checkpoint [0]: 2
```

Masukkan `2` untuk melanjutkan dari `checkpoint-2`. Masukkan `0` atau tekan Enter untuk memulai training dari awal.

Uji model:

```bash
python3 step8/test_model.py
```

```text
/reset → hapus riwayat percakapan
/exit  → tutup program
```

## 🟢 Training standalone dan resume checkpoint berbeda

Ada dua istilah yang mudah tertukar:

```text
Resume checkpoint
→ melanjutkan training yang terhenti
→ memakai adapter, optimizer, scheduler, dan progres step yang sama

Training standalone
→ memakai base model langsung sebagai titik awal
→ membuat adapter LoRA Step 8
```

Step 8 memakai training standalone.

Jika training Step 8 terhenti di tengah jalan, pilih checkpoint yang tersedia ketika `train_peft.py` dijalankan kembali. Script akan meneruskan optimizer, scheduler, dan progres training dari checkpoint tersebut.

## 🟢 Tahap berikutnya

Jika ingin menambah training lagi setelah Step 8, gunakan checkpoint adapter sebagai sumber training berikutnya:

```text
step8/output/checkpoint-*
→ checkpoint sumber untuk training tahap berikutnya
```

Step 8 sengaja hanya menghasilkan adapter LoRA. Proses merge tidak termasuk dalam step ini.
