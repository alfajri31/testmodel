# Pengantar — Cara LLM Memprediksi Token Berikutnya

Step ini menjelaskan satu pertanyaan utama:

> Mengapa model dapat melanjutkan `"Saya bisa ..."` dengan `"berenang"`, tetapi melanjutkan `"Saya terkena bisa ..."` dengan `"ular"`?

Kita memakai model mainan dengan vector dua dimensi. Semua angka hanya contoh agar perhitungannya mudah dilihat.

## Gambaran Besar

```text
Kalimat
  ↓
Token ID
  ↓
Embedding awal setiap token
  ↓
Attention membaca konteks
  ↓
Contextual vector
  ↓
lm_head menghitung skor seluruh vocabulary
  ↓
Token berikutnya dipilih
```

## 1. Token Menjadi Embedding

Anggap tokenizer mempunyai pemetaan berikut:

```text
"Saya"    → token ID 10
"terkena" → token ID 25
"bisa"    → token ID 31
```

Token ID digunakan untuk mengambil satu baris dari embedding table:

```text
embedding_table[10] → [ 1.0, 0.0]  # "Saya"
embedding_table[25] → [-1.0, 0.0]  # "terkena"
embedding_table[31] → [ 0.0, 1.0]  # "bisa"
```

Token `"bisa"` memiliki embedding awal yang sama pada kedua kalimat:

```text
"Saya bisa"         → "bisa" = [0.0, 1.0]
"Saya terkena bisa" → "bisa" = [0.0, 1.0]
```

Vector mempunyai lebih dari satu dimensi agar model memiliki ruang untuk mempelajari banyak pola. Namun satu dimensi tidak memiliki arti tetap seperti `kemampuan` atau `racun`. Makna muncul dari kombinasi seluruh vector dan bobot model.

## 2. Attention Membaca Konteks

Embedding awal `"bisa"` sama, tetapi konteks sebelumnya berbeda.

Untuk memprediksi token berikutnya, kita fokus pada **posisi token terakhir**. Query dari token terakhir dibandingkan dengan Key milik token itu sendiri dan token-token sebelumnya yang boleh dilihat.

```text
Token terakhir: "bisa"
  ↓
Q_"bisa" dibandingkan dengan Key token-token sebelumnya
  ↓
Attention weight
  ↓
Contextual vector posisi terakhir
  ↓
lm_head memprediksi token berikutnya
```

Transformer sebenarnya menghitung attention untuk setiap posisi token. Namun untuk menghasilkan satu token baru setelah prompt, model menggunakan contextual vector dari posisi terakhir.

```text
"Saya bisa"
→ token terakhir dapat melihat "Saya" dan "bisa"

"Saya terkena bisa"
→ token terakhir dapat melihat "Saya", "terkena", dan "bisa"
```

Attention memberikan bobot relevansi kepada token-token tersebut.

### Konteks `"Saya bisa"`

Anggap attention menghasilkan:

```text
"Saya" → 70%
"bisa" → 30%
```

Untuk contoh sederhana ini, contextual vector dihitung dengan mencampurkan vector:

```text
h₁ = 0.7 × [1.0, 0.0] + 0.3 × [0.0, 1.0]
```

```text
h₁ = [
  (0.7 × 1.0) + (0.3 × 0.0),
  (0.7 × 0.0) + (0.3 × 1.0)
]
```

```text
h₁ = [0.7, 0.3]
```

### Konteks `"Saya terkena bisa"`

Anggap attention menghasilkan:

```text
"Saya"    → 10%
"terkena" → 70%
"bisa"    → 20%
```

Contextual vector-nya:

```text
h₂ =
0.1 × [ 1.0, 0.0]
+ 0.7 × [-1.0, 0.0]
+ 0.2 × [ 0.0, 1.0]
```

```text
h₂ = [
  (0.1 × 1.0) + (0.7 × -1.0) + (0.2 × 0.0),
  (0.1 × 0.0) + (0.7 ×  0.0) + (0.2 × 1.0)
]
```

```text
h₂ = [-0.6, 0.2]
```

Hasilnya:

```text
"Saya bisa"         → contextual vector [ 0.7, 0.3]
"Saya terkena bisa" → contextual vector [-0.6, 0.2]
```

Inilah fungsi contextual vector: merepresentasikan token terakhir beserta konteks yang sudah dibacanya.

## 3. Dari Mana Bobot Relevansi Berasal?

Di dalam attention terdapat beberapa projection:

```text
q_proj
k_proj
v_proj
o_proj
```

Untuk pemahaman awal, pembagiannya cukup seperti ini:

```text
q_proj dan k_proj
→ bagian dari proses perhitungan bobot relevansi antartoken

v_proj
→ menyiapkan informasi dari token yang akan dicampurkan

o_proj
→ mengolah hasil attention agar dapat diteruskan oleh model
```

Pada contoh `"Saya terkena bisa"`, proses yang melibatkan `q_proj` dan `k_proj` dapat menghasilkan bobot relevansi seperti:

```text
"Saya"    → 10%
"terkena" → 70%
"bisa"    → 20%
```

Bobot relevansi tersebut dihitung ulang untuk setiap kalimat. Nilainya bukan persentase kemunculan kata dalam data training.

Perhitungan detail Query, Key, attention score, dan softmax sengaja tidak dibahas pada pengantar ini. Bagian tersebut akan dibahas pada README lanjutan tentang cara kerja attention.

## 4. `lm_head` Memprediksi Token Berikutnya

Contextual vector belum berupa kata. `lm_head` membandingkannya dengan bobot output seluruh token dalam vocabulary.

Untuk contoh ini, kita hanya menampilkan dua kandidat:

```text
bobot output "berenang" → [ 1.0, 0.0]
bobot output "ular"     → [-1.0, 0.0]
```

### Setelah `"Saya bisa"`

```text
h₁ = [0.7, 0.3]
```

```text
skor "berenang"
= [0.7, 0.3] · [1.0, 0.0]
= 0.7
```

```text
skor "ular"
= [0.7, 0.3] · [-1.0, 0.0]
= -0.7
```

Hasil:

```text
"berenang" →  0.7  # lebih tinggi
"ular"     → -0.7
```

Model lebih mungkin menghasilkan:

```text
"Saya bisa berenang"
```

### Setelah `"Saya terkena bisa"`

```text
h₂ = [-0.6, 0.2]
```

```text
skor "berenang"
= [-0.6, 0.2] · [1.0, 0.0]
= -0.6
```

```text
skor "ular"
= [-0.6, 0.2] · [-1.0, 0.0]
= 0.6
```

Hasil:

```text
"berenang" → -0.6
"ular"     →  0.6  # lebih tinggi
```

Model lebih mungkin menghasilkan:

```text
"Saya terkena bisa ular"
```

Model nyata menghitung skor untuk seluruh vocabulary, bukan hanya dua token:

```text
contextual vector
  ↓ lm_head
logit untuk setiap token
  ↓ softmax
probabilitas setiap token
  ↓ decoding
satu token berikutnya
```

## 5. Mengapa Model Bisa Melakukan Ini?

Saat training, model melihat banyak pola bahasa dan mencoba memprediksi token berikutnya:

```text
Input  : "Saya bisa"
Target : "berenang"
```

```text
Input  : "Saya terkena bisa"
Target : "ular"
```

Jika prediksinya salah, loss dihitung. Backpropagation memperbarui embedding, bobot Transformer, dan `lm_head` agar target yang benar mendapatkan probabilitas lebih tinggi.

Saat model selesai dilatih:

```text
Embedding dan bobot model
→ tersimpan dan tetap di dalam checkpoint

Attention weight dan contextual vector
→ dihitung sementara untuk setiap input
```

## Kesimpulan

```text
Token "bisa" sama
→ token ID sama
→ embedding awal sama

Konteks sebelumnya berbeda
→ attention weight berbeda
→ contextual vector berbeda
→ skor dari lm_head berbeda
→ token berikutnya dapat berbeda
```

Perbedaan tugasnya:

```text
Attention
→ menentukan informasi dalam konteks yang relevan
→ membantu membentuk contextual vector

lm_head
→ menggunakan contextual vector terakhir
→ menghasilkan skor kandidat token berikutnya
```

Hal penting yang perlu diingat:

- Embedding awal token yang sama tetap sama dalam checkpoint yang sama.
- Contextual vector berubah mengikuti konteks.
- Positif dan negatif hanyalah koordinat angka, bukan label baik atau buruk.
- Angka dan perhitungan pada dokumen ini adalah model mainan, bukan nilai dari model nyata.
