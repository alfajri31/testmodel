# 🟢 Step 10 — Cek Spesifikasi Model

Step ini membaca spesifikasi model hasil continue training Step 8. Script hanya membaca `config.json` dan ukuran file bobot; ia tidak melakukan training dan tidak menjalankan inference.

Script memilih model terbaru yang tersedia dengan urutan berikut:

```text
1. step8/merged_model  → hasil continue training
2. step5/merged_model  → hasil merge tahap pertama
3. mytestmodel         → base model awal
```

## 🟢 Menjalankan

Jalankan:

```bash
python3 step10/cek_spesifikasi_model.py
```

Jika Step 8 belum dijalankan, script tetap dapat membaca spesifikasi model hasil Step 5. Setelah Step 8 selesai di-*merge*, script otomatis memilih `step8/merged_model`.

Script menampilkan:

```text
- arsitektur dan jumlah Transformer layer
- hidden size, ukuran MLP, attention head, dan key/value head
- jumlah token vocabulary dan batas context model
- jumlah parameter hasil hitungan dari config
- jumlah serta ukuran file safetensors
```

## 🟢 Bentuk model yang diperiksa

Model hasil Step 8 tetap memakai arsitektur Qwen3-0.6B. Continue training dan merge mengubah nilai bobot melalui LoRA, tetapi tidak mengubah jumlah layer atau bentuk matrix.

```text
Qwen3ForCausalLM
├── embedding table
├── 28 Transformer layer
│   ├── q_proj, k_proj, v_proj, o_proj
│   ├── gate_proj, up_proj, down_proj
│   └── normalization
├── final normalization
└── lm_head
```

## 🟢 Parameter Qwen3-0.6B

Parameter adalah angka bobot yang tersimpan dalam model. Parameter bukan token dan bukan context window.

Konfigurasi Qwen3 lokal:

```text
Vocabulary        = 151.936 token
Hidden size       = 1.024
Intermediate MLP  = 3.072
Transformer layer = 28
```

### 🟢 Embedding table

Embedding table hanya ada satu kali untuk seluruh model:

```text
151.936 token × 1.024 angka per token
= 155.582.464 parameter
≈ 155,6 juta parameter
```

### 🟢 Attention dalam satu layer

```text
q_proj = 1.024 × 2.048 = 2.097.152
k_proj = 1.024 × 1.024 = 1.048.576
v_proj = 1.024 × 1.024 = 1.048.576
o_proj = 2.048 × 1.024 = 2.097.152
-----------------------------------
attention per layer = 6.291.456
```

### 🟢 MLP dalam satu layer

```text
gate_proj = 1.024 × 3.072 = 3.145.728
up_proj   = 1.024 × 3.072 = 3.145.728
down_proj = 3.072 × 1.024 = 3.145.728
-----------------------------------
MLP per layer = 9.437.184
```

Satu layer mempunyai sekitar 15,7 juta parameter dari attention, MLP, dan normalization. Karena model memiliki 28 layer:

```text
≈ 15,7 juta × 28
≈ 440,5 juta parameter
```

Total model:

```text
embedding table       ≈ 155,6 juta
28 Transformer layer  ≈ 440,5 juta
final normalization   ≈ kecil
---------------------------------
total                  ≈ 596 juta
≈ 0,6B parameter
```

`lm_head` memakai embedding table yang sama (`tie_word_embeddings=true`). Karena bobotnya dipakai bersama, `lm_head` tidak menambah embedding table kedua.

## 🟢 Parameter dan memori inference

Angka `0.6B` menunjukkan jumlah bobot model. Saat model menjawab, RAM atau VRAM juga dipakai untuk context window dan perhitungan inference:

```text
total memori inference
= bobot model 0.6B
+ context window dan KV cache
+ Q, K, V, hidden vector, dan attention score sementara
+ buffer runtime
```

Bobot model tetap dimuat selama model aktif. Context window dan KV cache bertambah ketika percakapan atau jawaban makin panjang. Saat chat di-*clear*, context dan cache lama dapat dilepas, sedangkan bobot model tetap ada di memori.

## 🟢 Mengapa jumlah parameter tidak berubah setelah merge?

LoRA menambahkan perubahan kecil ke matrix bobot yang sudah ada. Saat merge, perubahan LoRA dijumlahkan ke bobot model sebelumnya:

```text
W merged = W model sebelumnya + perubahan LoRA
```

Ukuran matrix `W` tidak berubah. Karena itu jumlah parameter arsitektur tetap sekitar 0.6B, walaupun nilai bobotnya berubah melalui training dan merge.
