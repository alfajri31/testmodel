# 🟢 Fondasi LLM — Dari Token sampai Prediksi Token Berikutnya

Step ini menjelaskan satu pertanyaan utama:

> Mengapa model dapat melanjutkan `"Saya bisa ..."` dengan `"berenang"`, tetapi melanjutkan `"Saya terkena bisa ..."` dengan `"ular"`?

Dokumen ini menggunakan satu model mainan dengan empat dimensi dan dua head. Dengan begitu, contoh dari `q_proj` sampai `lm_head` tetap berada dalam satu alur yang sama.

Semua angka hanya ilustrasi agar perhitungannya mudah dilihat, bukan nilai asli dari model tertentu.

Nilai bobot seperti `Wq`, `Wk`, `Wv`, `Wo`, `W_up`, dan `W_down` belum dijelaskan di awal agar alurnya tetap sederhana. Asal nilai bobot tersebut dibahas pada Bagian 6, setelah alur prediksi token selesai.

## 🟢 Gambaran Besar

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

## 🟢 1. Token Menjadi Embedding

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

### 🟢 Hidden size: panjang vector utama setiap token

`Hidden size` adalah jumlah angka dalam vector utama yang dibawa setiap token saat melewati model. Pada Qwen3-0.6B di project ini, hidden size adalah `1.024`.

```text
token "bisa"
→ embedding awal [1.024 angka]
→ contextual vector layer 1 [1.024 angka]
→ contextual vector layer 2 [1.024 angka]
→ ...
→ contextual vector akhir h [1.024 angka]
```

Jadi hidden size bukan ukuran seluruh matrix bobot. Hidden size adalah panjang satu vector token pada jalur utama Transformer.

Matrix bobot memakai hidden size sebagai ukuran input atau outputnya. Contoh pada Qwen3-0.6B:

```text
hidden vector token  [1 × 1.024]
× Wq                 [1.024 × 2.048]
= Q                  [1 × 2.048]

hasil attention
→ o_proj
→ kembali menjadi [1 × 1.024]
```

Hasil perlu kembali ke `1.024` angka supaya dapat ditambahkan ke hidden vector lama melalui residual connection dan diteruskan ke layer berikutnya.

Token `"bisa"` memiliki embedding awal yang sama pada kedua kalimat:

```text
"Saya bisa"         → "bisa" = [0.0, 1.0]
"Saya terkena bisa" → "bisa" = [0.0, 1.0]
```

Vector mempunyai lebih dari satu dimensi agar model memiliki ruang untuk mempelajari banyak pola. Namun satu dimensi tidak memiliki arti tetap seperti `kemampuan` atau `racun`. Makna muncul dari kombinasi seluruh vector dan hubungannya dengan vector lain.

## 🟢 2. Attention Membaca Konteks

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

Bagian 2 hanya memberikan gambaran besar. Perhitungan Q, K, V, setiap head, dan O dijelaskan secara berurutan pada Bagian 3 sampai 7.

## 🟢 3. Gambaran projection dalam Transformer

Setiap projection menggunakan matriks bobot yang sudah tersimpan di base model. Matriks tersebut dipakai berulang pada setiap input, tetapi nilainya tidak berubah saat inference. Nilai bobot dan proses pembelajarannya dibahas pada Bagian 6.

Gunakan konfigurasi model mainan:

```text
Kalimat             = "Saya terkena bisa"
Token terakhir      = "bisa"
Jumlah head         = 2
Dimensi setiap head = 2
Total dimensi       = 4
```

### 🟢 3.1 Peta penggunaan bobot dalam satu layer

Dalam satu Transformer layer, bobot dipakai pada bagian berikut:

```text
hidden vector
→ q_proj memakai Wq → Q
→ k_proj memakai Wk → K
→ v_proj memakai Wv → V
→ Q dan K menghasilkan relevansi
→ relevansi mencampurkan V
→ o_proj memakai Wo → output attention
→ residual + MLP (W_gate, W_up, W_down)
→ hidden vector baru untuk layer berikutnya
```

Pada Qwen3-0.6B, hidden size adalah `1.024`. Contoh bentuk matriksnya:

```text
Wq      [1.024 × 2.048]   → Q
Wk      [1.024 × 1.024]   → K
Wv      [1.024 × 1.024]   → V
Wo      [2.048 × 1.024]   → kembali ke 1.024 angka
W_gate  [1.024 × 3.072]   → ruang MLP
W_up    [1.024 × 3.072]   → ruang MLP
W_down  [3.072 × 1.024]   → kembali ke hidden size
```

Contoh hitungan di bagian berikut memakai empat dimensi agar mudah dihitung. Bentuk matriksnya disederhanakan, tetapi urutan prosesnya sama.

## 🟢 4. `q_proj`: membuat Query dari token terakhir

Bobot yang digunakan di bagian ini adalah `Wq`, yaitu matriks bobot milik `q_proj` pada layer yang sedang diproses. Setiap layer memiliki `Wq` sendiri.

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

Jadi `Wq` bukan hasil perhitungan saat inference. `Wq` sudah tersimpan di model; inference hanya mengalikan hidden vector saat ini dengan `Wq` untuk menghasilkan Q.

Pada Layer pertama, `hidden_bisa` berawal dari embedding vector token `"bisa"` yang diambil dari embedding table base model. Pada layer berikutnya, `hidden_bisa` adalah hasil pemrosesan layer sebelumnya.

Untuk contoh empat dimensi ini, misalkan hidden vector pada posisi `"bisa"` adalah:

```text
hidden_bisa = [0.5, 0.5, 0.0, 0.5]
```

Nilai tersebut adalah contoh input ke `q_proj`. Pada Layer pertama, nilainya berasal dari representasi embedding token beserta pemrosesan awal model. Pada layer berikutnya, nilainya berasal dari output layer sebelumnya.

Pada tahap ini, asal dan cara mendapatkan nilai bobot `Wq` **belum dibahas**. Angka matrix berikut hanya ilustrasi untuk menunjukkan bagaimana hidden vector diubah menjadi Q total.

Misalkan `q_proj` pada layer ini mempunyai matrix:

```text
Wq =
[
  [1.0,  1.0, 0.0, 0.0],
  [0.0, -1.0, 0.0, 1.0],
  [0.0,  0.0, 1.0, 0.0],
  [1.0,  0.0, 0.0, 1.0]
]
```

Perkaliannya:

```text
Q_bisa total
= [0.5, 0.5, 0.0, 0.5] × Wq
```

Hitung setiap koordinat hasil:

```text
koordinat 1
= (0.5 × 1.0) + (0.5 × 0.0) + (0.0 × 0.0) + (0.5 × 1.0)
= 1.0

koordinat 2
= (0.5 × 1.0) + (0.5 × -1.0) + (0.0 × 0.0) + (0.5 × 0.0)
= 0.0

koordinat 3
= (0.5 × 0.0) + (0.5 × 0.0) + (0.0 × 1.0) + (0.5 × 0.0)
= 0.0

koordinat 4
= (0.5 × 0.0) + (0.5 × 1.0) + (0.0 × 0.0) + (0.5 × 1.0)
= 1.0
```

Maka:

```text
hidden_bisa × Wq
→ Q_bisa total = [1.0, 0.0, 0.0, 1.0]
```

Ukuran perkaliannya adalah:

```text
hidden_bisa [1 × 4]
× Wq       [4 × 4]
→ Q_bisa   [1 × 4]
```

Jadi `Q_bisa total` mempunyai empat dimensi karena merupakan hasil perkalian hidden vector dengan `Wq`, bukan karena Head 1 atau Head 2 membuat angka acak baru.

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

## 🟢 5. `k_proj`: membuat Key untuk semua token

Bobot yang digunakan di bagian ini adalah `Wk`, yaitu matriks bobot milik `k_proj` pada layer yang sedang diproses. Setiap layer memiliki `Wk` sendiri.

Setiap token yang dapat dilihat menyediakan Key pada setiap head.

Fokus kegunaan K pada tahap ini:

> K menjadi vector pembanding untuk menghasilkan nilai persentase relevansi setiap token dalam context window. Persentase tidak dihasilkan oleh K sendirian; Q dari token terakhir harus dibandingkan dengan semua K, kemudian hasilnya diproses oleh softmax.

Untuk konteks `"Saya terkena bisa"`:

```text
hidden_Saya    × Wk → K_Saya
hidden_terkena × Wk → K_terkena
hidden_bisa    × Wk → K_bisa

Q dari token terakhir "bisa"
→ dibandingkan dengan K_Saya
→ dibandingkan dengan K_terkena
→ dibandingkan dengan K_bisa
→ softmax
→ persentase relevansi setiap token
```

Contoh perkalian sederhana untuk satu token:

```text
hidden_bisa = [0.5, 1.0]

Wk =
[
  [1.0, 0.0],
  [0.5, 0.5]
]

K_bisa = hidden_bisa × Wk
        = [0.5, 1.0] × Wk
        = [1.0, 0.5]
```

Token lain juga dihitung dengan `Wk` milik layer tersebut. Setelah semua `K` tersedia, `Q` token terakhir dibandingkan dengan setiap `K`.

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

### 🟢 5.1 Menghitung relevansi Head 1

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

### 🟢 5.2 Menghitung relevansi Head 2

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

## 🟢 6. `v_proj`: membuat Value lalu mencampurkan informasi

Bobot yang digunakan di bagian ini adalah `Wv`, yaitu matriks bobot milik `v_proj` pada layer yang sedang diproses. Setiap layer memiliki `Wv` sendiri.

Misalkan `v_proj` menghasilkan Value berikut. Angkanya adalah ilustrasi hasil projection, bukan vector yang dibuat acak saat inference.

Fokus kegunaan V pada tahap ini:

> V membawa informasi berbentuk vector dari setiap token. Persentase relevansi hasil Q–K menentukan seberapa besar setiap V ikut membentuk output attention.

Secara proses, setiap hidden vector token dikalikan dengan `Wv` terlebih dahulu:

```text
hidden_Saya    × Wv → V_Saya
hidden_terkena × Wv → V_terkena
hidden_bisa    × Wv → V_bisa
```

Contoh perkalian `v_proj`:

```text
hidden_bisa = [0.5, 1.0]

Wv =
[
  [0.2, 0.0],
  [0.0, 0.8]
]

V_bisa = hidden_bisa × Wv
        = [0.5, 1.0] × Wv
        = [0.1, 0.8]
```

Vector `[0.1, 0.8]` adalah Value yang dibawa token tersebut. Attention kemudian mengalikan Value ini dengan persentase relevansinya.

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

## 🟢 7. `o_proj`: menggabungkan semua head

Bobot yang digunakan di bagian ini adalah `Wo`, yaitu matriks bobot milik `o_proj` pada layer yang sedang diproses. Setiap layer memiliki `Wo` sendiri.

Output kedua head disambungkan, bukan dijumlahkan:

```text
concat(
  [-0.40, 0.14],
  [ 0.24, 0.58]
)

= [-0.40, 0.14, 0.24, 0.58]
```

Contoh perkalian `o_proj` yang disederhanakan:

```text
gabungan head = [a, b, c, d]

Wo = identity matrix 4 × 4

output attention
= [a, b, c, d] × Wo
= [a, b, c, d]
```

Pada model nyata, `Wo` bukan identity matrix. `Wo` mencampurkan semua koordinat hasil head dan mengembalikannya ke hidden size. Contoh angka berikut dibuat sederhana:

```text
gabungan head
→ [ -0.40, 0.14, 0.24, 0.58 ]
× Wo
→ output attention
```

Hasil gabungan mempunyai empat dimensi:

```text
2 head × 2 dimensi setiap head
= 4 dimensi
```

Kemudian `o_proj` mencampurkan hasil semua head dan menghasilkan output attention yang kembali berukuran empat dimensi. Misalkan hasilnya adalah:

```text
gabungan semua head
× Wo
→ output attention = [-0.40, 0.14, 0.24, 0.58]
```

Output `o_proj` belum menjadi kata dan belum langsung masuk ke `lm_head`. Ia ditambahkan ke hidden vector lama agar menjadi contextual vector **setelah attention**. Setelah itu, MLP dan residual kedua menghasilkan contextual vector baru / output penuh layer:

```text
hidden_bisa sebelum attention
= [0.50, 0.50, 0.00, 0.50]

output o_proj
= [-0.40, 0.14, 0.24, 0.58]

hidden setelah attention
= hidden lama + output o_proj
= [0.10, 0.64, 0.24, 1.08]
```

Vector `[0.10, 0.64, 0.24, 1.08]` belum menjadi output penuh layer. Ia masuk ke MLP untuk mendapatkan perubahan tambahan.

Untuk contoh kecil ini, misalkan MLP memakai matrix berikut. Ukurannya `4 × 2` lalu `2 × 4` hanya agar hitungannya pendek; model nyata biasanya memperbesar dimensi tengah.

```text
W_up
→ mengubah 4 dimensi ke ruang MLP 2 dimensi pada contoh ini

W_down
→ mengembalikan hasil MLP dari 2 dimensi menjadi 4 dimensi
→ agar hasilnya dapat ditambahkan kembali ke x yang juga 4 dimensi
```

```text
W_up =
[
  [1.0, 0.0],
  [0.0, 1.0],
  [1.0, 0.0],
  [0.0, 1.0]
]

W_down =
[
  [0.1, 0.0, 0.2,  0.0],
  [0.0, 0.1, 0.0, -0.1]
]
```

MLP menghitung:

```text
z1 = x × W_up
   = [0.10, 0.64, 0.24, 1.08] × W_up
   = [0.34, 1.72]

z2 = activation(z1)
   = [0.34, 1.72]

output_MLP = z2 × W_down
           = [0.034, 0.172, 0.068, -0.172]
```

Pada contoh ini `x` berarti hidden setelah attention. Fungsi `activation` tidak mengubah `z1` karena kedua nilainya positif. Pada model nyata, fungsi tersebut penting untuk mempelajari pola yang lebih kompleks.

Residual kedua menambahkan output MLP ke `x`:

```text
contextual vector baru
= x + output_MLP
= [0.10, 0.64, 0.24, 1.08]
+ [0.034, 0.172, 0.068, -0.172]
= [0.134, 0.812, 0.308, 0.908]
```

Jadi perhitungan output penuh satu layer adalah:

```text
x = hidden_lama + output_o_proj       ← residual attention
contextual_vector_baru = x + output_MLP  ← residual MLP
```

Contextual vector baru `[0.134, 0.812, 0.308, 0.908]` menjadi `hidden_lama` untuk layer 2. Layer 2 melakukan proses yang sama dan menghasilkan vector baru lagi. Begitu seterusnya.

```text
o_proj layer 1
        ↓
contextual vector layer 1
        ↓ menjadi hidden_lama layer 2

o_proj layer 2
        ↓
contextual vector layer 2
        ↓ menjadi hidden_lama layer 3

... layer terakhir
        ↓
contextual vector akhir = h
        ↓
lm_head
```

Jika ini adalah layer terakhir, outputnya disebut **contextual vector akhir** atau `h`:

```text
output layer terakhir
= h = [0.134, 0.812, 0.308, 0.908]
        ↓
lm_head
```

Ringkasan urutannya:

```text
q_proj → membuat Q dari posisi token terakhir
k_proj → membuat K pembanding dari semua token dalam context window
Q · K + softmax → menghasilkan persentase relevansi setiap token
v_proj → membuat V yang membawa informasi setiap token
bobot relevansi × V → menghasilkan output setiap head
o_proj → mencampurkan gabungan seluruh head
output o_proj + hidden lama → hidden setelah attention
hidden setelah attention × W_up → activation → × W_down → output_MLP
hidden setelah attention + output_MLP → contextual vector baru untuk layer berikutnya
output layer terakhir → contextual vector akhir `h` → lm_head
```

## 🟢 8. `lm_head` Memprediksi Token Berikutnya

Contextual vector akhir belum berupa kata. `lm_head` membandingkan `h` dengan vector output seluruh token dalam vocabulary.

Lanjutkan contoh yang sama dari Bagian 3 sampai 7:

```text
contextual vector akhir pada token "bisa"
h = [0.134, 0.812, 0.308, 0.908]
```

Untuk mempermudah, tampilkan dua kandidat token dengan empat dimensi:

```text
vector output "berenang" = [ 0.50, 0.10, 0.10, -0.10]
vector output "ular"     = [-0.30, 0.70, 0.10,  0.20]
```

Kedua vector di atas adalah **vector output `lm_head` milik model**. Model mempunyai satu vector seperti ini untuk setiap token dalam vocabulary. Nilainya dipelajari saat training dan tersimpan di checkpoint/base model; nilainya tidak dibuat acak lagi ketika model sedang menjawab.

Perannya berbeda dari embedding input:

```text
embedding input
→ mengubah token prompt menjadi vector awal

vector output lm_head
→ menjadi kandidat yang dibandingkan dengan contextual vector akhir
→ menghasilkan skor token berikutnya
```

Pada beberapa model, embedding input dan bobot `lm_head` dapat memakai matrix yang sama. Namun pada penjelasan ini, keduanya dianggap terpisah agar perannya mudah dibedakan.

`lm_head` menghitung skor kecocokan dengan perkalian titik seluruh dimensi:

```text
skor "berenang"
= [0.134, 0.812, 0.308, 0.908] · [0.50, 0.10, 0.10, -0.10]
= (0.134 × 0.50) + (0.812 × 0.10) + (0.308 × 0.10) + (0.908 × -0.10)
= 0.09
```

```text
skor "ular"
= [0.134, 0.812, 0.308, 0.908] · [-0.30, 0.70, 0.10, 0.20]
= (0.134 × -0.30) + (0.812 × 0.70) + (0.308 × 0.10) + (0.908 × 0.20)
= 0.74
```

Hasilnya:

```text
"berenang" → 0.09
"ular"     → 0.74  # skor lebih tinggi
```

Maka untuk konteks contoh `"Saya terkena bisa"`, model lebih mungkin memilih token `"ular"`.

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

## 🟢 9. Bagaimana Satu Token Menjadi Rangkaian Kalimat?

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

### 🟢 Greedy decoding dan sampling

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

## 🟢 10. Mengapa Model Bisa Melakukan Ini?

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

### 🟢 Training vs inference

Training dan inference sama-sama melakukan prediksi token, tetapi tujuannya berbeda.

Saat training, dataset menyediakan input dan target. Model membuat prediksi, membandingkannya dengan target untuk menghitung loss, lalu optimizer memperbarui bobot.

```text
input training
→ prediksi token
→ bandingkan dengan target
→ hitung loss
→ perbarui bobot
```

Inference adalah saat model menggunakan bobot hasil training untuk memproses input dan menghasilkan response. Jadi ketika user mengirim pertanyaan, model sedang melakukan inference. Prediksi tetap dilakukan satu token demi satu token, tetapi tidak ada target training, loss untuk belajar, atau pembaruan bobot.

```text
input user
→ prediksi next token
→ token masuk ke context window
→ prediksi next token berikutnya
→ ulangi sampai response selesai
```

Contoh inference:

```text
Input: “Ibu kota Indonesia adalah”
→ prediksi: “Jakarta”
```

Ringkasnya:

```text
Training  → prediksi + loss + pembaruan bobot
Inference → prediksi saja + bobot tetap
```

### 🟢 Context window: ingatan kerja sementara

Selama percakapan berlangsung, token baru ditambahkan ke context window. Model lalu menghitung contextual vector baru agar jawaban berikutnya dapat mempertimbangkan percakapan sebelumnya.

```text
percakapan bertambah
→ token context window bertambah
→ contextual vector diperbarui
→ jawaban dapat memakai informasi chat sebelumnya
```

Ini adalah **ingatan kerja sementara**, bukan pengetahuan baru yang permanen di base model. Pada inference, bobot model tidak berubah. Yang berubah adalah konteks dan vector sementara untuk percakapan saat ini.

Context window yang lebih panjang juga biasanya membutuhkan lebih banyak memori kerja, termasuk KV cache. Karena itu, ketika chat di-*clear*:

```text
context window lama dihapus
→ KV cache lama tidak dipakai lagi
→ penggunaan memori kerja berkurang
→ proses jawaban berikutnya biasanya lebih ringan
```

Base model tetap sama. Yang dibersihkan adalah konteks percakapan sementara, bukan bobot atau pengetahuan permanen model.

### 🟢 Dari mana nilai bobot berasal?

Saat model baru dibuat, matrix bobot—termasuk `Wq`, `Wk`, `Wv`, `Wo`, `W_up`, dan `W_down`—sudah dibuat dan diisi angka acak kecil. Pada tahap training awal (*pretraining*), model membuat prediksi, membandingkannya dengan token target, lalu memperbarui angka-angka bobot tersebut sedikit demi sedikit agar prediksi berikutnya lebih baik.

```text
angka acak kecil
→ training awal berkali-kali
→ bobot mempelajari pola dari data latih
→ bobot tersimpan dalam checkpoint / base model
```

Setelah training selesai, bobot tersebut dipakai ulang saat inference (model sedang menjawab). Misalnya, `W_up` dan `W_down` dipakai untuk mengolah hidden vector pada setiap layer, tetapi nilai bobotnya tidak berubah ketika model menjawab.

```text
hidden vector saat ini
× W_up dan W_down hasil training
→ contextual vector baru
```

Yang berubah mengikuti prompt adalah Q, K, V, bobot relevansi attention, dan contextual vector—bukan bobot modelnya.

## 🟢 Kesimpulan

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
