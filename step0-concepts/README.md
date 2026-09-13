# Pengantar — Cara LLM Memprediksi Token Berikutnya

Step ini menjelaskan satu pertanyaan utama:

> Mengapa model dapat melanjutkan `"Saya bisa ..."` dengan `"berenang"`, tetapi melanjutkan `"Saya terkena bisa ..."` dengan `"ular"`?

Dokumen ini menggunakan dua model mainan:

```text
Contoh 2 dimensi
→ untuk menjelaskan embedding dan lm_head

Contoh 4 dimensi dengan 2 head
→ untuk menjelaskan q_proj, k_proj, v_proj, dan o_proj
```

Semua angka hanya ilustrasi agar perhitungannya mudah dilihat, bukan nilai asli dari model tertentu.

## Gambaran Besar

```text
Kalimat
  ↓
Token ID
  ↓
Embedding awal setiap token
  ↓
Setiap Transformer layer memproses Q, K, V, dan O
  ↓
Contextual vector akhir pada posisi token terakhir
  ↓
lm_head menghitung skor seluruh vocabulary
  ↓
Token berikutnya dipilih
  ↓
Token ditambahkan ke context window dan proses diulang
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

Angka `[1.0, 0.0]`, `[-1.0, 0.0]`, dan `[0.0, 1.0]` adalah angka **ilustrasi**, bukan nilai embedding asli dari model nyata. Angka ini dipilih agar titik vector mudah digambar dan perhitungan attention mudah diikuti.

Pada model nyata, prosesnya seperti ini:

```text
Model baru dibuat
→ embedding table diisi angka acak kecil

Training
→ loss, backpropagation, dan optimizer memperbarui angka tersebut

Model pretrained
→ embedding vector hasil training disimpan di checkpoint
```

Jadi ketika inference, model tidak membuat embedding token secara acak lagi. Token ID mengambil vector yang sudah tersimpan:

```text
token ID "Saya"
→ embedding_table[token_ID]
→ misalnya [0.012, -0.043, 0.071, ..., 0.020]
```

Model nyata biasanya menggunakan ratusan atau ribuan dimensi. Contoh dua dimensi pada pengantar ini hanya menyederhanakan cara kerja.

Token `"bisa"` memiliki embedding awal yang sama pada kedua kalimat:

```text
"Saya bisa"         → "bisa" = [0.0, 1.0]
"Saya terkena bisa" → "bisa" = [0.0, 1.0]
```

Vector mempunyai lebih dari satu dimensi agar model memiliki ruang untuk mempelajari banyak pola. Namun satu dimensi tidak memiliki arti tetap seperti `kemampuan` atau `racun`. Makna muncul dari kombinasi seluruh vector dan hubungannya dengan vector lain.

## 2. Attention Membaca Konteks

Embedding awal token `"bisa"` tetap sama, tetapi context window yang berbeda akan menghasilkan contextual vector yang berbeda.

Token terakhir adalah token paling akhir yang saat ini berada di context window:

```text
Context window: "Saya terkena bisa"
Token terakhir: "bisa"
```

Untuk memprediksi satu token setelah prompt, model berfokus pada contextual vector di posisi token terakhir. Token terakhir dapat memperhatikan dirinya sendiri dan token-token sebelumnya yang masih berada dalam context window.

```text
"Saya bisa"
→ posisi "bisa" melihat "Saya" dan "bisa"

"Saya terkena bisa"
→ posisi "bisa" melihat "Saya", "terkena", dan "bisa"
```

Ringkasan prosesnya:

```text
Token terakhir "bisa"
→ Query "bisa" dibandingkan dengan Key setiap token konteks
→ menghasilkan bobot relevansi setiap token
→ bobot relevansi mencampurkan Value setiap token
→ semua head digabungkan
→ contextual vector baru pada posisi "bisa"
→ lm_head memprediksi token berikutnya
```

Karena konteksnya berbeda, hasilnya juga dapat berbeda:

```text
"Saya bisa"
→ contextual vector membawa pola kemampuan
→ next token dapat berupa "berenang"

"Saya terkena bisa"
→ contextual vector membawa pola racun
→ next token dapat berupa "ular"
```

Section 2 hanya memberikan gambaran besar. Perhitungan Q, K, V, setiap head, dan O dijelaskan secara berurutan pada Section 3.

## 3. Proses `q_proj`, `k_proj`, `v_proj`, dan `o_proj`

Setiap projection menggunakan parameter yang sudah tersimpan di base model. Pada bagian Q, kita hanya memperlihatkan `Wq` untuk menjelaskan dari mana Q total berasal. Detail parameter projection lainnya dapat dipelajari setelah alur dasarnya dipahami.

Gunakan konfigurasi model mainan:

```text
Kalimat             = "Saya terkena bisa"
Token terakhir      = "bisa"
Jumlah head         = 2
Dimensi setiap head = 2
Total dimensi       = 4
```

### 3.1 `q_proj`: membuat Query dari token terakhir

Untuk memprediksi token setelah prompt, Query yang menjadi fokus berasal dari posisi token terakhir `"bisa"`:

```text
vector token "bisa"
→ dikalikan dengan bobot Wq milik q_proj
→ menghasilkan Q_bisa total
```

Secara singkat:

```text
Q_bisa total = hidden_bisa × Wq
```

Pada Layer pertama, `hidden_bisa` berawal dari embedding vector token `"bisa"` yang diambil dari embedding table base model. Pada layer berikutnya, `hidden_bisa` adalah hasil pemrosesan layer sebelumnya.

Jika total dimensinya empat:

```text
hidden_bisa [1 × 4]
× Wq       [4 × 4]
→ Q_bisa   [1 × 4]
```

Jadi `Q_bisa total` mempunyai empat dimensi karena merupakan hasil perkalian vector token dengan `Wq`, bukan karena Head 1 atau Head 2 membuat angka acak baru.

Misalkan hasil perkalian tersebut adalah:

```text
Q_bisa total = [1.0, 0.0, 0.0, 1.0]
```

Q tersebut dibagi untuk dua head:

```text
Q_bisa Head 1 = [1.0, 0.0]
Q_bisa Head 2 = [0.0, 1.0]
```

Kedua Query berasal dari token `"bisa"` yang sama. Setiap head hanya memprosesnya melalui bagian vector yang berbeda.

Urutan lengkapnya:

```text
hidden_bisa dari embedding/layer sebelumnya
→ q_proj memakai Wq
→ Q_bisa total [1.0, 0.0, 0.0, 1.0]
→ dibagi menjadi dua head
→ Head 1 [1.0, 0.0]
→ Head 2 [0.0, 1.0]
```

### 3.2 `k_proj`: membuat Key untuk semua token

Setiap token yang dapat dilihat menyediakan Key pada setiap head.

Fokus kegunaan K pada tahap ini:

> K menjadi vector pembanding untuk menghasilkan nilai persentase relevansi setiap token dalam context window. Persentase tidak dihasilkan oleh K sendirian; Q dari token terakhir harus dibandingkan dengan semua K, kemudian hasilnya diproses oleh softmax.

Untuk konteks `"Saya terkena bisa"`:

```text
Q dari token terakhir "bisa"
→ dibandingkan dengan K_Saya
→ dibandingkan dengan K_terkena
→ dibandingkan dengan K_bisa
→ softmax
→ persentase relevansi setiap token
```

Head 1:

```text
K_Saya Head 1    = [1.0, 0.2]
K_terkena Head 1 = [2.0, 0.1]
K_bisa Head 1    = [0.5, 0.4]
```

Head 2:

```text
K_Saya Head 2    = [0.2, 1.5]
K_terkena Head 2 = [0.1, 0.5]
K_bisa Head 2    = [0.4, 1.0]
```

Q dan K selalu dibandingkan di dalam head yang sama:

```text
Q_bisa Head 1 · K token Head 1
Q_bisa Head 2 · K token Head 2
```

Bukan:

```text
Q_bisa Head 1 · K token Head 2 ❌
```

### 3.3 Menghitung relevansi Head 1

```text
Q_bisa Head 1 = [1.0, 0.0]
```

```text
Dengan "Saya":
[1.0, 0.0] · [1.0, 0.2]
= (1.0 × 1.0) + (0.0 × 0.2)
= 1.0

Dengan "terkena":
[1.0, 0.0] · [2.0, 0.1]
= (1.0 × 2.0) + (0.0 × 0.1)
= 2.0

Dengan "bisa":
[1.0, 0.0] · [0.5, 0.4]
= (1.0 × 0.5) + (0.0 × 0.4)
= 0.5
```

Setelah softmax:

```text
Saya    → 23%
terkena → 63%
bisa    → 14%
```

Head 1 paling memperhatikan `"terkena"`.

### 3.4 Menghitung relevansi Head 2

```text
Q_bisa Head 2 = [0.0, 1.0]
```

```text
Dengan "Saya":
[0.0, 1.0] · [0.2, 1.5]
= (0.0 × 0.2) + (1.0 × 1.5)
= 1.5

Dengan "terkena":
[0.0, 1.0] · [0.1, 0.5]
= (0.0 × 0.1) + (1.0 × 0.5)
= 0.5

Dengan "bisa":
[0.0, 1.0] · [0.4, 1.0]
= (0.0 × 0.4) + (1.0 × 1.0)
= 1.0
```

Setelah softmax:

```text
Saya    → sekitar 50.6%
terkena → sekitar 18.6%
bisa    → sekitar 30.7%
```

Head 2 paling memperhatikan `"Saya"`. Dengan demikian, dua head dapat menghasilkan pola relevansi berbeda untuk context window yang sama.

### 3.5 `v_proj`: membuat Value lalu mencampurkan informasi

Misalkan `v_proj` menghasilkan Value berikut. Angkanya adalah ilustrasi hasil projection, bukan vector yang dibuat acak saat inference.

Fokus kegunaan V pada tahap ini:

> V membawa informasi berbentuk vector dari setiap token. Persentase relevansi hasil Q–K menentukan seberapa besar setiap V ikut membentuk output attention.

Head 1:

```text
V_Saya Head 1    = [ 1.0, 0.0]
V_terkena Head 1 = [-1.0, 0.0]
V_bisa Head 1    = [ 0.0, 1.0]
```

Gunakan bobot relevansi Head 1:

```text
23% × [ 1.0, 0.0]
+ 63% × [-1.0, 0.0]
+ 14% × [ 0.0, 1.0]
= [-0.40, 0.14]
```

Head 2:

```text
V_Saya Head 2    = [0.2, 0.8]
V_terkena Head 2 = [0.6, 0.1]
V_bisa Head 2    = [0.1, 0.5]
```

Gunakan bobot relevansi Head 2:

```text
50.6% × [0.2, 0.8]
+ 18.6% × [0.6, 0.1]
+ 30.7% × [0.1, 0.5]
≈ [0.24, 0.58]
```

Hasil setiap head:

```text
Output Head 1 = [-0.40, 0.14]
Output Head 2 = [ 0.24, 0.58]
```

Saat generation menggunakan KV cache, hasil K dan V setiap token disimpan sementara pada layer masing-masing:

```text
k_proj → K → masuk KV cache
v_proj → V → masuk KV cache

q_proj → Q → dipakai untuk langkah saat ini
o_proj → output attention → diteruskan ke proses berikutnya
```

Ketika token baru diproses, Query token baru dapat langsung memakai K dan V token-token lama dari KV cache. KV cache hanya mempercepat proses; token dalam context window tetap menjadi sumber konteksnya.

Jadi hubungan V dengan KV cache adalah:

```text
v_proj menghasilkan V
→ V dipakai dalam attention
→ V juga disimpan sementara dalam KV cache agar dapat dipakai kembali
```

V tidak dibuat khusus untuk menghasilkan KV cache. KV cache hanya menyimpan hasil K dan V yang sudah dibuat.

### 3.6 `o_proj`: menggabungkan semua head

Output kedua head disambungkan, bukan dijumlahkan:

```text
concat(
  [-0.40, 0.14],
  [ 0.24, 0.58]
)

= [-0.40, 0.14, 0.24, 0.58]
```

Hasil gabungan mempunyai empat dimensi:

```text
2 head × 2 dimensi setiap head
= 4 dimensi
```

Kemudian `o_proj` mencampurkan hasil semua head dan menghasilkan output attention yang kembali berukuran empat dimensi. Residual menambahkan output tersebut ke hidden vector sebelumnya:

```text
gabungan semua head
→ o_proj
→ output attention 4 dimensi
→ residual
→ contextual vector baru
```

Angka keluaran `o_proj` belum dihitung pada Pengantar ini. Fokusnya adalah memahami urutan proses:

```text
q_proj → membuat Q dari posisi token terakhir
k_proj → membuat K pembanding dari semua token dalam context window
Q · K + softmax → menghasilkan persentase relevansi setiap token
v_proj → membuat V yang membawa informasi setiap token
bobot relevansi × V → menghasilkan output setiap head
o_proj → mencampurkan gabungan seluruh head
```

## 4. `lm_head` Memprediksi Token Berikutnya

Contextual vector belum berupa kata. `lm_head` membandingkannya dengan vector output seluruh token dalam vocabulary.

Bagian ini menggunakan contoh baru dengan dua dimensi agar perhitungan `lm_head` tetap pendek. Contoh ini terpisah dari contoh dua head dengan empat dimensi pada Section 3.

Misalkan Transformer telah menghasilkan contextual vector akhir pada posisi token terakhir:

```text
"Saya bisa"         → h₁ = [ 0.7, 0.3]
"Saya terkena bisa" → h₂ = [-0.6, 0.2]
```

Untuk contoh ini, kita hanya menampilkan dua kandidat:

```text
vector output "berenang" → [ 1.0, 0.0]
vector output "ular"     → [-1.0, 0.0]
```

### Setelah `"Saya bisa"`

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

Jika prediksinya salah, proses training memperbaiki kemampuan model agar target yang benar mendapatkan probabilitas lebih tinggi. Detail parameter yang diperbarui belum dibahas pada pengantar ini.

Saat inference, Q, K, V, bobot relevansi attention, dan contextual vector dihitung sementara untuk setiap input.

## Kesimpulan

```text
Token "bisa" sama
→ token ID sama
→ embedding awal sama

Konteks sebelumnya berbeda
→ bobot relevansi attention berbeda
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

Cara paling sederhana untuk menyimpulkannya:

> Next token adalah hasil pemrosesan context window sebelumnya. Model seperti sangat hati-hati memilih kata: attention menentukan token konteks yang paling relevan, Transformer membentuk contextual vector, lalu `lm_head` memilih kandidat token yang paling cocok atau mengambil sampel berbobot probabilitas.

Tidak semua token dalam context window selalu memiliki pengaruh yang sama:

```text
context window
→ attention memberi bobot relevansi berbeda
→ token yang lebih relevan memberi pengaruh lebih besar
→ contextual vector terakhir
→ prediksi next token
```

Hal penting yang perlu diingat:

- Embedding awal token yang sama tetap sama dalam checkpoint yang sama.
- Contextual vector berubah mengikuti konteks.
- Positif dan negatif hanyalah koordinat angka, bukan label baik atau buruk.
- Angka dan perhitungan pada dokumen ini adalah model mainan, bukan nilai dari model nyata.
