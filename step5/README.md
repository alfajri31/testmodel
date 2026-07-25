# Step 5 — Merge Base Model dan LoRA

Pada step ini, adapter LoRA hasil training digabungkan ke base model.

```text
Base model + LoRA → merged model
```

**Fine-tuning bertujuan membuat model lebih cenderung menjawab sesuai pola dan jawaban target yang terdapat dalam dataset.**

Dataset tidak dicari secara langsung ketika model menjawab. Hasil training tersimpan sebagai perubahan bobot yang meningkatkan probabilitas munculnya jawaban sesuai dataset.

Base model asli tidak ditimpa. Model gabungan disimpan di folder baru.

## Struktur

```text
step5/
├── output/                     # adapter LoRA
├── merge_model.py
├── test_merged_model.py
├── cek_spesifikasi_model.py
└── merged_model/               # model hasil merge
```

## Menjalankan step5

Merge model:

```bash
python3 /Users/fajri/Projects/testmodel/step5/merge_model.py
```

Sumber dan hasil:

```text
Base  : /Users/fajri/Projects/mytestmodel
LoRA  : /Users/fajri/Projects/testmodel/step5/output
Hasil : /Users/fajri/Projects/testmodel/step5/merged_model
```

Uji model gabungan:

```bash
python3 /Users/fajri/Projects/testmodel/step5/test_merged_model.py
```

```text
/reset → menghapus riwayat
/exit  → menutup program
```

Cek spesifikasi:

```bash
python3 /Users/fajri/Projects/testmodel/step5/cek_spesifikasi_model.py
```

Setelah merge, model dapat dimuat langsung dengan `AutoModelForCausalLM` tanpa `PeftModel`.

## Spesifikasi model

```text
Arsitektur       : Qwen3
Layer            : 28
Hidden size      : 1.024
Vocabulary       : 151.936 token
Context model    : 40.960 token
LoRA rank        : 8
```

## Gambaran isi model

Analogi sederhananya:

```text
Model     = gedung
Layer     = lantai
Modul     = mesin di setiap lantai
Tensor    = kumpulan angka
Parameter = pengaturan angka di dalam mesin
```

Strukturnya:

```text
Model
└── 28 layer
    └── beberapa modul
        └── tensor/matriks parameter
            └── angka-angka bobot
```

Setiap layer memiliki susunan modul yang sama, tetapi nilai bobotnya berbeda.

## Bagian 1 — Neural network base model

Qwen3 adalah neural network dengan arsitektur Transformer.

```text
Neural Network
└── Transformer
    └── Qwen3
```

Analoginya:

```text
Kendaraan
└── Mobil
    └── Toyota Avanza
```

Hubungannya:

```text
Neural network = kategori besar seperti kendaraan
Transformer    = jenis neural network seperti mobil
Qwen3          = model tertentu seperti Toyota Avanza
```

Neural network dimuat terlebih dahulu sebagai `base_model`:

```python
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
)
```

Bagian neural network:

```text
modeling_qwen3.py = kode proses neural network Qwen3
config.json       = bentuk dan ukuran arsitektur
model.safetensors = angka bobot pretrained
base_model        = neural network yang sudah aktif
```

Kode neural network Qwen3 berasal dari library `transformers`:

```text
transformers/models/qwen3/modeling_qwen3.py
```

### Token, aktivasi, dan parameter

Setiap token direpresentasikan oleh 1.024 angka.

Jika input menghasilkan tiga token:

```text
Tensor aktivasi [3, 1024]
                  ↑     ↑
                  │     └── 1.024 angka per token
                  └──────── tiga token
```

Jumlah angka aktivasi:

```text
3 × 1.024 = 3.072 angka
```

**Ini bukan jumlah parameter model. Tensor aktivasi adalah data yang sedang mengalir.**

Perbedaannya:

| Bentuk | Jenis | Arti |
|---|---|---|
| `[3, 1024]` | Aktivasi | Tiga token, masing-masing 1.024 angka |
| `[2048, 1024]` | Parameter base | Bobot `q_proj` |

Alurnya:

```text
Teks
        ↓ tokenizer dan embedding
Tensor aktivasi [3, 1024]
        ↓ diproses menggunakan bobot
Base q_proj [2048, 1024]
        ↓
Tensor output [3, 2048]
```

```text
Jumlah token     → berubah sesuai input
Jumlah parameter → tetap selama arsitektur tetap
```

### Ukuran arsitektur Qwen3

Angka `1024`, `2048`, dan `3072` ditentukan oleh arsitektur model ini.

```text
1024 = hidden size utama
2048 = output Query Attention
3072 = intermediate size MLP
```

Angka tersebut bukan jumlah token, jumlah dataset, atau context window. Model lain dapat memiliki ukuran berbeda.

Ukuran modul Qwen3 pada proyek ini:

```text
q_proj    : 1024 → 2048
k_proj    : 1024 → 1024
v_proj    : 1024 → 1024
o_proj    : 2048 → 1024

gate_proj : 1024 → 3072
up_proj   : 1024 → 3072
down_proj : 3072 → 1024
```

Analogi perubahan ukurannya:

```text
1024 = posisi utama atau "standby"
2048 = ruang analisis Query Attention
3072 = ruang pengolahan MLP
1024 = kembali ke posisi utama
```

```text
Attention : 1024 standby → 2048 analisis   → 1024 standby
MLP       : 1024 standby → 3072 pengolahan → 1024 standby
```

Kata `standby` hanya analogi untuk hidden size utama antarlayer. Tensor tetap diproses dan nilainya tetap berubah. Pola ukuran ini sudah ditetapkan oleh arsitektur, bukan dipilih secara dinamis.

### Fungsi modul Qwen

`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, dan `down_proj` adalah **modul neural network milik base model Qwen3**.

Modul tersebut bukan file konfigurasi. Hubungannya:

```text
config.json
→ menentukan bentuk dan ukuran neural network

q_proj dan modul lainnya
→ bagian neural network yang dibangun dari konfigurasi

model.safetensors
→ menyimpan angka bobot setiap modul
```

Contoh:

```text
config.json
└── hidden_size = 1024

Neural network
└── q_proj = Linear(1024 → 2048)

model.safetensors
└── q_proj.weight = tensor [2048, 1024]
```

Jadi, kalimat yang tepat:

> `q_proj` adalah modul neural network base model. Ukurannya ditentukan oleh konfigurasi dan angka bobotnya disimpan di Safetensors.

Attention mencari hubungan antartoken:

```text
q_proj = apa yang dicari?
k_proj = token mana yang memiliki informasi?
v_proj = informasi apa yang diberikan?
o_proj = menggabungkan hasilnya
```

MLP mengolah informasi lebih lanjut:

```text
up_proj   = membuka ruang pengolahan
gate_proj = memilih informasi yang diteruskan
down_proj = mengembalikan hasil ke ukuran utama
```

### Cara membaca ukuran matriks

```text
Input size  = jumlah angka yang masuk per token
Output size = jumlah angka yang keluar per token
```

Bobot `torch.nn.Linear` disimpan sebagai:

```text
[output size, input size]
```

Contoh `q_proj`:

```text
Alur modul  : 1024 → 2048
Shape bobot : [2048, 1024]

Parameter base
= (output size × input size)
= (2.048 × 1.024)
= 2.097.152 parameter
```

Jika tiga token melewati `q_proj`:

```text
Tensor input  : [3, 1024]
Bobot q_proj  : [2048, 1024]
Tensor output : [3, 2048]
```

Jumlah token tetap tiga. Yang berubah adalah jumlah angka per token.

### Perhitungan parameter base model

### Embedding token

Embedding menyimpan 1.024 angka untuk setiap jenis token:

```text
(151.936 vocabulary × 1.024 angka)
= 155.582.464 parameter (0.1556 B)
```

Embedding berada sebelum layer pertama dan belum termasuk dalam parameter 28 layer.

### Parameter satu layer

Attention:

```text
Modul      = (output size × input size)

q_proj    = (2.048 × 1.024) = 2.097.152
k_proj    = (1.024 × 1.024) = 1.048.576
v_proj    = (1.024 × 1.024) = 1.048.576
o_proj    = (1.024 × 2.048) = 2.097.152
q_norm    =                           128
k_norm    =                           128
─────────────────────────────────────────
Total Attention                = 6.291.712
```

MLP:

```text
gate_proj = (3.072 × 1.024) = 3.145.728
up_proj   = (3.072 × 1.024) = 3.145.728
down_proj = (1.024 × 3.072) = 3.145.728
─────────────────────────────────────────
Total MLP                      = 9.437.184
```

Normalization:

```text
input_layernorm          = 1.024
post_attention_layernorm = 1.024
──────────────────────────────
Total normalization      = 2.048
```

Total satu layer:

```text
Attention     =  6.291.712
MLP           =  9.437.184
Normalization =      2.048
──────────────────────────
Total         = 15.730.944 parameter (0.0157 B)
```

Seluruh 28 layer:

```text
(15.730.944 × 28)
= 440.466.432 parameter (0.4405 B)
```

Total base model:

```text
Embedding token       = 155.582.464 (0.1556 B)
Seluruh 28 layer      = 440.466.432 (0.4405 B)
Final normalization   =       1.024
────────────────────────────────────
Total base model      = 596.049.920 (0.596 B ≈ 0.6 B)
```

## Bagian 2 — Adaptasi LoRA

Neural network base model harus tersedia terlebih dahulu. Setelah itu, LoRA dipasang sebagai parameter tambahan yang akan dilatih.

```text
Neural network Qwen3
        ↓
Pasang LoRA
        ↓
Latih LoRA menggunakan dataset
```

Konfigurasi LoRA dibuat:

```python
lora_config = LoraConfig(...)
```

Kemudian dipasang melalui trainer:

```python
trainer = SFTTrainer(
    model=base_model,
    peft_config=lora_config,
)
```

Jadi, **neural network dulu, baru LoRA**.

### LoRA dan rank 8

Base model menyimpan hubungan penuh:

```text
Parameter base = (output size × input size)
```

LoRA menyimpan perubahan melalui jalur kecil:

```text
input size → rank 8 → output size
```

Matriksnya:

```text
LoRA A = [rank, input size]
LoRA B = [output size, rank]

Parameter LoRA
= (rank × input size)
  + (output size × rank)
```

Contoh LoRA `q_proj`:

```text
Input size  = 1.024
Output size = 2.048
Rank        = 8

LoRA A = (8 × 1.024)   =  8.192
LoRA B = (2.048 × 8)   = 16.384
────────────────────────────────
Total                      = 24.576 parameter
```

Alur tensor:

```text
[3, 1024]
     ↓ LoRA A
[3, 8]
     ↓ LoRA B
[3, 2048]
```

```text
3    = jumlah token
1024 = input size per token
8    = rank LoRA
2048 = output size per token
```

Rank bukan jumlah token, jumlah data, atau jumlah fakta.

Analogi:

```text
Base model = jalan utama
Rank 8     = delapan jalur koreksi
Dataset    = contoh arah tujuan
Loss       = ukuran kesalahan
Gradient   = petunjuk untuk memperbaiki jalur
```

Seluruh dataset melatih parameter rank yang sama. Dataset menentukan nilai perubahan yang dipelajari, sedangkan rank menentukan kapasitas jalurnya.

Sebelum LoRA dipasang, base model belum memiliki jalur LoRA:

```text
Input → bobot base → output
```

Setelah LoRA dipasang, terdapat jalur koreksi tambahan:

```text
Input → bobot base ─────────┐
    └→ LoRA rank 8 ─────────┤
                            ↓
                     hasil dijumlahkan
```

Saat training:

```text
Bobot base → dibekukan
Bobot LoRA → diperbarui berdasarkan dataset
```

Rank 8 berarti terdapat delapan dimensi perantara pada setiap modul target. Rank bukan jumlah token, jumlah data, atau jumlah fakta dalam dataset.

### Perhitungan parameter LoRA

Satu layer:

```text
Modul      = (rank × input size) + (output size × rank)

q_proj    = (8 × 1.024) + (2.048 × 8) = 24.576
k_proj    = (8 × 1.024) + (1.024 × 8) = 16.384
v_proj    = (8 × 1.024) + (1.024 × 8) = 16.384
o_proj    = (8 × 2.048) + (1.024 × 8) = 24.576
gate_proj = (8 × 1.024) + (3.072 × 8) = 32.768
up_proj   = (8 × 1.024) + (3.072 × 8) = 32.768
down_proj = (8 × 3.072) + (1.024 × 8) = 32.768
───────────────────────────────────────────
Total                              = 180.224 (0.000180 B)
```

Seluruh 28 layer:

```text
(180.224 × 28)
= 5.046.272 parameter LoRA (0.00505 B)
```

LoRA hanya sekitar:

```text
(5.046.272 ÷ 596.049.920) × 100
≈ 0,85% dari parameter base model
```

## Bagian 3 — Proses input user dengan LoRA

Setelah LoRA dilatih, input user tetap diproses oleh neural network Qwen3. LoRA memberikan koreksi tambahan pada modul yang menjadi target training.

```text
Teks user
    ↓
Tokenizer mengubah teks menjadi token
    ↓
Embedding mengubah token menjadi tensor aktivasi
    ↓
Neural network Qwen3 memproses aktivasi
    +
LoRA memberikan perubahan tambahan
    ↓
Probabilitas token berikutnya
    ↓
Model memilih token jawaban
```

Contoh sederhana:

```text
User: "Siapakah Alfajri?"
        ↓
Base model menghasilkan pola jawaban umum
        +
LoRA membuat prediksi lebih condong ke pola dataset
        ↓
Jawaban model
```

LoRA tidak mencari kalimat langsung dari file dataset ketika user bertanya. Dataset sudah digunakan saat training untuk mengubah parameter LoRA.

```text
Saat training  → dataset mengubah bobot LoRA
Saat menjawab  → input user diproses menggunakan bobot hasil training
```

## Bagian 4 — Sebelum dan sesudah merge

Sebelum merge:

```text
Base model        = 596.049.920 (0.596 B)
Parameter LoRA    =   5.046.272 (0.00505 B)
──────────────────────────────────────
Base + LoRA       = 601.096.192 (0.6011 B)

Parameter dilatih = 5.046.272
Parameter base    = dibekukan
```

Setelah merge:

```text
Bobot merged = bobot base + perubahan LoRA

Parameter merged = 596.049.920 (0.596 B)
```

Matriks LoRA tidak lagi disimpan terpisah. Pengaruhnya sudah dimasukkan ke nilai bobot base model.

Yang tetap dan berubah:

```text
Vocabulary    → tetap
Hidden size   → tetap
Jumlah layer  → tetap
Context model → tetap
Nilai bobot   → berubah
```

### Mengapa jumlah token tetap sama?

LoRA tidak menambahkan token baru ke tokenizer atau vocabulary. LoRA hanya mempelajari perubahan bobot melalui jalur tambahan.

```text
Sebelum fine-tuning → vocabulary tetap
Sesudah training    → vocabulary tetap
Sesudah merge       → vocabulary tetap
```

Pada model ini:

```text
Sebelum LoRA  = 151.936 token vocabulary
Sesudah LoRA  = 151.936 token vocabulary
Sesudah merge = 151.936 token vocabulary
```

Saat merge:

```text
Bobot merged = bobot base + perubahan LoRA
```

Jalur LoRA tidak lagi disimpan sebagai jalur terpisah. Pengaruh yang dipelajari sudah dilebur ke bobot utama.

```text
Yang tetap:
Vocabulary, tokenizer, jumlah layer, hidden size, dan context length

Yang berubah:
Nilai bobot, probabilitas token, dan kecenderungan jawaban
```

Jadi, model dapat menghasilkan jawaban berbeda walaupun jumlah token dan bentuk arsitekturnya tetap sama.

## Mengapa karakter model berbeda?

```text
Arsitektur = bentuk dan susunan otak
Bobot      = pengalaman yang sudah dipelajari
LoRA       = kebiasaan tambahan dari dataset
Prompt     = pertanyaan yang diberikan
```

Model dengan struktur mirip dapat menjawab berbeda karena bobot dan data latihnya berbeda.

## Kesimpulan

```text
Fine-tuning = membuat jawaban lebih condong ke pola dataset
Merge       = melebur perubahan LoRA ke bobot base model
```

Fine-tuning membuat model lebih cenderung menjawab berdasarkan pola dan target dataset, tetapi tidak menjamin jawabannya selalu sama atau selalu benar.

Dataset tidak disimpan seperti database dan tidak dicari kembali ketika model menjawab. Dataset digunakan untuk mengubah nilai parameter LoRA sehingga probabilitas jawaban yang diinginkan menjadi lebih tinggi.

LoRA menambahkan jalur koreksi saat training, bukan menambahkan token. Setelah merge, perubahan dari jalur LoRA dimasukkan ke bobot base model. Karena itu, spesifikasi token tetap sama tetapi kecenderungan jawaban dapat berubah.

Untuk fakta yang harus tepat atau selalu terbaru, gunakan RAG atau database.
