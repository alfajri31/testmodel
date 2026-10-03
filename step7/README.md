# 🟢 Step 7 — Visualisasi Struktur Model dan Uji Inference

Step ini berfokus pada dua kegiatan:

1. Merekam struktur base model ke TensorBoard dalam bentuk teks dan graph visual.
2. Menjalankan base model bersama adapter LoRA untuk menguji jawabannya secara interaktif.

```text
train_peft.py
→ visualisasi struktur model
→ opsional: lanjut training LoRA

test_model.py
→ memuat base model + adapter
→ menjalankan inference interaktif
```

## 🟢 Struktur Folder

```text
step7/
├── dataset/              # data untuk training LoRA
├── output/               # adapter hasil training
├── tensorboard_logs/     # graph dan metric training
├── train_peft.py         # visualisasi dan training
└── test_model.py         # uji inference interaktif
```

## 🟢 Visualisasi yang Direkam

`train_peft.py` merekam struktur base model sebelum LoRA dipasang:

```text
add_text
→ menampilkan struktur model sebagai teks pada tab Text

add_graph
→ menampilkan struktur model sebagai diagram pada tab Graphs
```

Graph dibuat menggunakan:

```text
ModelGraphView
└── DecoderLayerGraphView untuk setiap Transformer layer
```

Graph tersebut adalah tampilan ringkas untuk membantu melihat susunan:

```text
Embedding
↓
Decoder layers
├── q_proj
├── k_proj
├── v_proj
├── o_proj
├── gate_proj
├── up_proj
└── down_proj
↓
Final normalization
↓
lm_head
```

`ModelGraphView` hanya digunakan untuk visualisasi. Training dan inference tetap memakai forward asli dari model Qwen.

## 🟢 Pilihan 1 — Hanya Membuat Graph

Gunakan `GRAPH_ONLY=1` jika hanya ingin merekam struktur model tanpa menjalankan training:

```bash
cd /Users/fajri/Projects/testmodel
GRAPH_ONLY=1 python3 step7/train_peft.py
```

Alurnya:

```text
Muat tokenizer dan base model
↓
Jalankan add_text dan add_graph
↓
Simpan event TensorBoard
↓
Program berhenti sebelum training
```

`GRAPH_ONLY=1` hanya berlaku untuk perintah tersebut dan tidak mengubah model.

## 🟢 Pilihan 2 — Membuat Graph dan Menjalankan Training

Jalankan tanpa `GRAPH_ONLY=1`:

```bash
python3 step7/train_peft.py
```

Alurnya:

```text
Rekam teks dan graph struktur model
↓
Muat dataset
↓
Pasang LoRA
↓
Training
↓
Simpan adapter ke step7/output
↓
Simpan metric ke step7/tensorboard_logs
```

## 🟢 Membuka TensorBoard

```bash
tensorboard --logdir step7/tensorboard_logs
```

Buka alamat yang ditampilkan, biasanya:

```text
http://localhost:6006
```

Bagian yang dapat dilihat:

```text
Text        → struktur model dari add_text
Graphs      → graph visual dari add_graph
Time Series → metric training seperti loss dan learning rate
```

## 🟢 Menguji Model

Setelah adapter tersedia di `step7/output`, jalankan:

```bash
python3 step7/test_model.py
```

`test_model.py` memuat:

```text
Base model
+
Adapter LoRA dari step7/output
↓
Model inference
```

Perintah interaktif:

```text
/reset → menghapus riwayat percakapan
/help  → menampilkan bantuan
/exit  → menutup program
```

Inference hanya menggunakan bobot yang sudah tersedia. Percakapan tidak memperbarui bobot model.

## 🟢 Kesimpulan

```text
GRAPH_ONLY=1
→ visualisasi saja

Tanpa GRAPH_ONLY
→ visualisasi lalu training

test_model.py
→ uji inference menggunakan adapter hasil training
```
