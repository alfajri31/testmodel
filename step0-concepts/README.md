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

### Cara Bobot Relevansi Membentuk Contextual Vector

Untuk pemahaman dasar, contextual vector dapat dibayangkan dengan rumus:

```text
contextual vector
= jumlah dari (bobot relevansi × vector setiap token)
```

Pada model mainan di pengantar ini, `vector setiap token` menggunakan embedding vector awal. Ini adalah penyederhanaan agar hubungan antara bobot relevansi dan contextual vector mudah dipahami.

Jika konteks berisi tiga token:

```text
"Saya terkena bisa"
```

maka bentuk sederhananya:

```text
contextual vector =
(relevansi "Saya"    × vector "Saya")
+
(relevansi "terkena" × vector "terkena")
+
(relevansi "bisa"    × vector "bisa")
```

Bobot relevansi adalah satu angka, sedangkan vector token mempunyai beberapa dimensi. Satu bobot relevansi mengalikan seluruh dimensi vector pasangannya.

Contoh:

```text
bobot relevansi "terkena" = 0.7
vector "terkena"          = [-1.0, 0.0]
```

```text
0.7 × [-1.0, 0.0]
= [0.7 × -1.0, 0.7 × 0.0]
= [-0.7, 0.0]
```

Setelah setiap vector dikalikan dengan bobot relevansinya, semua hasil dijumlahkan per dimensi. Hasil penjumlahan itulah contextual vector sederhana untuk posisi token yang sedang diproses.

```text
Bobot relevansi
↓ mengatur besar pengaruh setiap token
Vector token
↓ membawa informasi numerik token
Perkalian dan penjumlahan
↓
Contextual vector
```

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

Untuk pemahaman dasar, bagian yang perlu dikenal lebih dulu adalah:

```text
q_proj
k_proj
```

Keduanya memiliki bobot matriks hasil training:

```text
q_proj → mempunyai bobot matriks Wq
k_proj → mempunyai bobot matriks Wk
```

Bobot matriks tersebut digunakan untuk mengolah vector token:

```text
Vector token
↓ q_proj dengan Wq
Query

Vector token
↓ k_proj dengan Wk
Key
```

Query dari token terakhir dibandingkan dengan Key token-token yang dapat dilihat. Proses tersebut menghasilkan bobot relevansi.

Pada contoh `"Saya terkena bisa"`:

```text
Token terakhir: "bisa"
↓
Query "bisa" dibandingkan dengan Key:
- "Saya"
- "terkena"
- "bisa"
↓
Bobot relevansi

"Saya"    → 10%
"terkena" → 70%
"bisa"    → 20%
```

Perlu dibedakan:

```text
Bobot matriks Wq dan Wk
→ parameter hasil training
→ tersimpan dalam checkpoint
→ tetap saat inference

Bobot relevansi 10%, 70%, dan 20%
→ hasil perhitungan untuk kalimat saat ini
→ dihitung ulang ketika konteks berubah
```

Bobot relevansi bukan persentase kemunculan kata dalam data training. Perhitungan detail Query, Key, attention score, dan softmax sengaja belum dibahas pada pengantar ini.

Projection lain seperti `v_proj` dan `o_proj` juga belum dibahas agar fokus awal tetap pada hubungan berikut:

```text
q_proj dan k_proj
→ Query dan Key
→ bobot relevansi
```

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

## 5. Bagaimana Satu Token Menjadi Rangkaian Kalimat?

`lm_head` hanya memilih **satu token berikutnya**, bukan langsung membuat satu kalimat lengkap.

Contoh awal:

```text
Input: "Saya terkena bisa"
```

Misalkan token dengan skor/probabilitas kuat adalah `"ular"`:

```text
"Saya terkena bisa"
        ↓ lm_head
"ular"
```

Token `"ular"` kemudian ditambahkan ke context window:

```text
"Saya terkena bisa ular"
```

Sekarang token terakhir berubah menjadi `"ular"`. Model memproses konteks terbaru untuk memprediksi satu token lagi:

```text
"Saya terkena bisa ular"
        ↓ Transformer + lm_head
"."
```

Prosesnya diulang:

```text
konteks saat ini
→ contextual vector posisi terakhir
→ lm_head menghitung kandidat seluruh vocabulary
→ satu token dipilih
→ token ditambahkan ke konteks
→ ulangi
```

Jadi rangkaian kalimat terbentuk token demi token:

```text
"Saya terkena bisa"
→ "ular"
→ "."
→ token berikutnya
```

Kemampuan menyusun kata agar terasa natural berasal dari pola bahasa yang dipelajari base model saat training:

```text
pola kata yang sering cocok berdampingan
→ pola frasa
→ tata bahasa
→ struktur kalimat dan paragraf
```

Model tidak harus menemukan satu kalimat utuh yang sama persis di data latih. Ia dapat menyusun kalimat baru dari banyak pola kecil yang dipelajari.

### Greedy decoding dan sampling

Setelah `lm_head`, model mempunyai probabilitas beberapa kandidat token:

```text
"Berikut" → 60%
"Untuk"   → 30%
"Secara"  → 10%
```

Ada dua cara umum untuk memilih token:

```text
Greedy decoding
→ selalu memilih probabilitas tertinggi
→ contoh: selalu memilih "Berikut"
→ hasil lebih konsisten, tetapi dapat monoton

Sampling
→ memilih secara acak berbobot probabilitas
→ "Berikut" paling sering dipilih, tetapi "Untuk" kadang dapat dipilih
→ hasil lebih bervariasi
```

Sampling bukan acak bebas. Token dengan probabilitas tinggi tetap jauh lebih mungkin muncul daripada token dengan probabilitas rendah.

```text
probabilitas tinggi
→ peluang lebih tinggi untuk dipilih

probabilitas rendah
→ masih mungkin dipilih jika tidak dibatasi oleh top-k/top-p
```

Pengaturan seperti `temperature`, `top-k`, dan `top-p` menentukan seberapa bervariasi pilihan tersebut. Untuk keluaran ketat seperti klasifikasi, JSON, atau format operasional, greedy decoding atau temperature sangat rendah biasanya lebih sesuai.

## 6. Mengapa Model Bisa Melakukan Ini?

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

Rangkaian kalimat terbentuk karena proses prediksi satu token diulang:

```text
contextual vector terakhir
→ lm_head
→ pilih satu token
→ token masuk ke context window
→ contextual vector baru
→ lm_head lagi
```

Hal penting yang perlu diingat:

- Embedding awal token yang sama tetap sama dalam checkpoint yang sama.
- Contextual vector berubah mengikuti konteks.
- Positif dan negatif hanyalah koordinat angka, bukan label baik atau buruk.
- Angka dan perhitungan pada dokumen ini adalah model mainan, bukan nilai dari model nyata.
