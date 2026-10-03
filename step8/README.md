# <span style="color: #2563EB;">Step 8 — Continue Training dari Merged Model</span>

Step ini menunjukkan cara menambah training setelah model tahap sebelumnya sudah di-*merge*. Model tidak kembali memulai dari `/Users/fajri/Projects/mytestmodel`.

Sumber model untuk training ini adalah:

```text
step5/merged_model
```

Folder tersebut sudah berisi bobot:

```text
base model Qwen3-0.6B
+ adapter LoRA tahap pertama dari step5
= merged model tahap pertama
```

Saat `train_continue.py` berjalan, model tersebut dimuat sebagai model dasar. Script lalu menambahkan adapter LoRA baru dan melatihnya memakai dataset kecil di `dataset/dataset.jsonl`.

```text
step5/merged_model
        ↓
adapter LoRA tahap kedua dilatih
        ↓
step8/output
        ↓
merge_continue.py
        ↓
step8/merged_model
```

## <span style="color: #2563EB;">Mengapa ini tidak mulai dari nol?</span>

`step5/merged_model` sudah memuat perubahan bobot dari training tahap pertama. Karena `train_continue.py` memuat folder itu, semua pengetahuan dan pola hasil merge sebelumnya tetap menjadi titik awal.

Yang baru dari Step 8 adalah adapter LoRA tahap kedua:

```text
bobot model hasil merge tahap pertama
→ dibekukan

adapter LoRA tahap kedua
→ dibuat dan dilatih dengan dataset baru
```

Setelah `merge_continue.py`, perubahan adapter tahap kedua digabungkan ke model hasil tahap pertama.

```text
merged tahap pertama
+ perubahan LoRA tahap kedua
= merged model lanjutan
```

Jadi adapter tahap kedua memang mulai dengan nilai awal baru, tetapi model yang dipakainya sudah bukan base model awal. Ia melanjutkan dari bobot merged tahap pertama.

## <span style="color: #2563EB;">Struktur folder</span>

```text
step8/
├── dataset/
│   └── dataset.jsonl       # Contoh data tambahan
├── train_continue.py       # Training LoRA tahap kedua
├── merge_continue.py       # Merge model tahap pertama + adapter tahap kedua
├── test_model.py           # Uji model hasil merge lanjutan
├── output/                 # Dibuat setelah training
└── merged_model/           # Dibuat setelah merge
```

## <span style="color: #2563EB;">Dataset contoh</span>

Dataset pada step ini sengaja kecil. Dataset awal Step 5 mengajarkan pola bahwa Alfajri adalah programmer Java. Dataset Step 8 menambahkan informasi baru bahwa Alfajri sedang mempelajari machine learning.

Setelah training, uji dua hal berikut:

```text
Apa pekerjaan Alfajri?
→ diharapkan masih menjawab programmer Java

Alfajri sedang belajar apa?
→ diharapkan menjawab machine learning
```

Hasil dapat berubah karena dataset contoh sangat kecil. Tujuannya adalah melihat alur continue training, bukan membuat model pengetahuan yang stabil.

## <span style="color: #2563EB;">Menjalankan</span>

Pastikan Step 5 sudah mempunyai model hasil merge:

```bash
ls /Users/fajri/Projects/testmodel/step5/merged_model/config.json
```

Jalankan training tahap kedua:

```bash
cd /Users/fajri/Projects/testmodel
python3 step8/train_continue.py
```

Hasilnya adalah adapter baru di:

```text
step8/output/
```

Gabungkan adapter tahap kedua:

```bash
python3 step8/merge_continue.py
```

Hasilnya disimpan terpisah agar model tahap pertama tidak tertimpa:

```text
step8/merged_model/
```

Uji model:

```bash
python3 step8/test_model.py
```

```text
/reset → hapus riwayat percakapan
/exit  → tutup program
```

## <span style="color: #2563EB;">Continue training dan resume checkpoint berbeda</span>

Ada dua istilah yang mudah tertukar:

```text
Resume checkpoint
→ melanjutkan training yang terhenti
→ memakai adapter, optimizer, scheduler, dan progres step yang sama

Continue training setelah merge
→ memakai full model hasil merge sebagai titik awal baru
→ membuat adapter LoRA tahap berikutnya untuk dataset tambahan
```

Step 8 memakai cara kedua.

Jika training tahap kedua terhenti di tengah jalan dan ingin melanjutkan tepat dari step terakhir, gunakan checkpoint yang dibuat di `step8/output/checkpoint-*` dengan `trainer.train(resume_from_checkpoint=...)`. Contoh ini sengaja dibuat sederhana dan tidak mengaktifkan resume otomatis.

## <span style="color: #2563EB;">Tahap berikutnya</span>

Jika ingin menambah training lagi setelah Step 8, jadikan hasil merge Step 8 sebagai sumber model berikutnya:

```text
step8/merged_model
→ model sumber untuk training tahap ketiga
```

Jangan kembali memuat `/Users/fajri/Projects/mytestmodel` bila tujuanmu adalah mempertahankan hasil merge sebelumnya.
