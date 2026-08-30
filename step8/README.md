# Step 8 — Dari Vector hingga Prediksi Token

Step ini dibatasi pada bagian berikut:

```text
embedding vector
→ q_proj
→ k_proj
→ v_proj
→ attention head
→ o_proj
→ contextual vector
→ layer berikutnya
→ lm_head
→ token berikutnya
```

Contoh utama:

```text
"Saya bisa ..."
"Saya terkena bisa ..."
```

Tujuannya adalah memahami bagaimana token `"bisa"` dapat mengarah ke pola kemampuan pada kalimat pertama dan pola racun pada kalimat kedua.

## 1. Vector seperti titik koordinat

Misalkan token `"bisa"` mempunyai ID `21` dan model menggunakan empat dimensi:

```text
embedding_table[21]
→ [0.12, -0.87, 0.44, 0.21]
```

Vector tersebut dapat dibayangkan sebagai satu titik dalam ruang empat dimensi:

```text
dimensi 1 →  0.12
dimensi 2 → -0.87
dimensi 3 →  0.44
dimensi 4 →  0.21
```

Dimensi adalah sumbu atau slot koordinat. Dimensi bukan daftar arti:

```text
dimensi 1 = kemampuan ❌
dimensi 2 = racun     ❌
```

Makna dan pola terbentuk dari kombinasi seluruh koordinat serta hubungannya dengan vector lain.

## 2. Untuk apa banyak dimensi?

Dimensi menyediakan ruang agar banyak pola dapat direpresentasikan dan dibedakan.

```text
1 dimensi      → semua titik berada pada satu garis
2 dimensi      → titik dapat tersebar pada satu bidang
banyak dimensi → tersedia lebih banyak arah untuk membedakan pola
```

Dengan banyak dimensi, model mempunyai kapasitas lebih besar untuk membedakan pola seperti:

```text
"Saya bisa berenang"     → pola kemampuan
"Saya terkena bisa ular" → pola racun
"Saham BISA meningkat"   → pola nama perusahaan
```

Tetapi semakin banyak dimensi tidak otomatis semakin akurat. Akurasi juga dipengaruhi oleh data training, arsitektur, jumlah layer, jumlah parameter, dan proses training.

## 3. Ukuran vector pada batas setiap layer

Misalkan:

```text
hidden_size = 4
```

Setiap posisi token masuk ke Transformer layer sebagai vector empat dimensi:

```text
"Saya"    → [ 0.12, -0.43, 0.71,  0.20]
"terkena" → [-0.51,  0.82, 0.14, -0.30]
"bisa"    → [ 0.44, -0.18, 0.63,  0.29]
```

Angka tersebut hanya ilustrasi untuk menunjukkan bahwa setiap token mempunyai empat koordinat, bukan nilai asli dari model.

Pada model Transformer biasa, output lengkap satu layer dikembalikan ke ukuran `hidden_size` yang sama:

```text
input Layer 1  → 4 dimensi
output Layer 1 → 4 dimensi

input Layer 2  → 4 dimensi
output Layer 2 → 4 dimensi
```

Hal ini diperlukan agar output dapat:

1. dijumlahkan melalui residual connection; dan
2. menjadi input untuk layer berikutnya.

Namun, **tidak setiap hasil sementara harus memiliki empat dimensi**. Di dalam layer, jumlah dimensi dapat dibagi per head atau diperbesar sementara. Yang harus kembali sama adalah ukuran hidden vector pada batas output layer.

## 4. Attention head membagi pekerjaan

Attention head membagi ruang kerja attention menjadi beberapa bagian.

Contoh konfigurasi sederhana:

```text
hidden_size = 4
jumlah head = 2
dimensi setiap head = 2
```

Sehingga secara sederhana:

```text
4 dimensi total
→ Head 1 mengolah 2 dimensi
→ Head 2 mengolah 2 dimensi
```

Setiap head dapat menghasilkan pola relevansi berbeda:

```text
Head 1:
Saya 20%, terkena 70%, bisa 10%

Head 2:
Saya 40%, terkena 20%, bisa 40%
```

Catatan penting:

> Empat dimensi tidak otomatis berarti harus ada dua head. Dua head hanyalah konfigurasi contoh. Model dapat memilih jumlah head lain selama ukuran proyeksinya cocok dengan arsitektur.

## 5. Q: apa yang sedang dicari posisi ini?

Untuk konteks:

```text
"Saya terkena bisa"
```

kita fokus pada posisi terakhir, yaitu `"bisa"`:

```text
hidden_bisa × Wq → Q_bisa
```

`Q_bisa` bukan token selanjutnya. Q adalah vector yang digunakan posisi `"bisa"` untuk menanyakan:

```text
"Token dalam konteks mana yang relevan bagi posisi saya?"
```

`Wq` adalah matrix bobot milik `q_proj` yang sudah dipelajari saat training.

## 6. K: membandingkan Q dengan token dalam context window

Setiap token yang dapat dilihat menghasilkan K:

```text
hidden_Saya    × Wk → K_Saya
hidden_terkena × Wk → K_terkena
hidden_bisa    × Wk → K_bisa
```

Jadi Q dan K sama-sama berasal dari perkalian hidden vector dengan matrix bobot model:

```text
hidden_bisa × Wq → Q_bisa

hidden_Saya    × Wk → K_Saya
hidden_terkena × Wk → K_terkena
hidden_bisa    × Wk → K_bisa
```

Kemudian `Q_bisa` dibandingkan dengan setiap K menggunakan dot product:

```text
Q_bisa · K_Saya
Q_bisa · K_terkena
Q_bisa · K_bisa
```

Secara sederhana dapat dibaca:

```text
"bisa" memeriksa hubungannya dengan "Saya"
"bisa" memeriksa hubungannya dengan "terkena"
"bisa" memeriksa hubungannya dengan "bisa" sendiri
```

Contoh berikut sengaja menggunakan dua dimensi agar perhitungannya mudah dilihat:

```text
Q_bisa    = [1.0, 0.0]

K_Saya    = [1.0, 0.2]
K_terkena = [2.0, 0.1]
K_bisa    = [0.5, 0.4]
```

Hitung skor relevansi mentah:

```text
skor Saya
= Q_bisa · K_Saya
= [1.0, 0.0] · [1.0, 0.2]
= (1.0 × 1.0) + (0.0 × 0.2)
= 1.0

skor terkena
= Q_bisa · K_terkena
= [1.0, 0.0] · [2.0, 0.1]
= (1.0 × 2.0) + (0.0 × 0.1)
= 2.0

skor bisa
= Q_bisa · K_bisa
= [1.0, 0.0] · [0.5, 0.4]
= (1.0 × 0.5) + (0.0 × 0.4)
= 0.5
```

Hasil perkalian Q dengan semua K adalah:

```text
Saya    → skor 1.0
terkena → skor 2.0
bisa    → skor 0.5
```

Skor tersebut belum berupa probabilitas. `softmax` mengubahnya menjadi bobot relevansi yang jumlahnya `100%`:

```text
softmax([1.0, 2.0, 0.5])
≈ [0.23, 0.63, 0.14]
```

Hasilnya:

```text
Saya    → 23%
terkena → 63%
bisa    → 14%
```

Persentase tersebut bukan isi K dan bukan langsung keluar dari `Wk`. Persentase muncul setelah:

```text
hidden × Wq/Wk
→ Q dan K
→ Q_bisa · setiap K
→ skor relevansi mentah
→ softmax
→ bobot relevansi dengan total 100%
```

Angka seperti `20%`, `70%`, dan `10%` yang digunakan pada bagian lain hanyalah pembulatan ilustratif agar pencampuran V lebih mudah dibaca.

## 7. Contoh perhitungan dengan dua head

Sekarang gunakan konfigurasi:

```text
hidden_size = 4
jumlah head = 2
dimensi setiap head = 2
```

Vector Q dan K hasil projection mempunyai empat dimensi, kemudian dibagi menjadi dua head:

```text
Q_bisa total = [1.0, 0.0, 0.0, 1.0]

Head 1 memakai → [1.0, 0.0]
Head 2 memakai → [0.0, 1.0]
```

Setiap head mempunyai Q dan K dua dimensi serta menghitung relevansinya sendiri.

### Head 1

```text
Q_bisa_head_1    = [1.0, 0.0]

K_Saya_head_1    = [1.0, 0.2]
K_terkena_head_1 = [2.0, 0.1]
K_bisa_head_1    = [0.5, 0.4]
```

Dot product:

```text
Q_bisa_head_1 · K_Saya_head_1
= [1.0, 0.0] · [1.0, 0.2]
= 1.0

Q_bisa_head_1 · K_terkena_head_1
= [1.0, 0.0] · [2.0, 0.1]
= 2.0

Q_bisa_head_1 · K_bisa_head_1
= [1.0, 0.0] · [0.5, 0.4]
= 0.5
```

Softmax:

```text
softmax([1.0, 2.0, 0.5])
≈ [0.23, 0.63, 0.14]

Head 1:
Saya    → 23%
terkena → 63%
bisa    → 14%
```

### Head 2

```text
Q_bisa_head_2    = [0.0, 1.0]

K_Saya_head_2    = [0.2, 1.5]
K_terkena_head_2 = [0.1, 0.5]
K_bisa_head_2    = [0.4, 1.0]
```

Dot product:

```text
Q_bisa_head_2 · K_Saya_head_2
= [0.0, 1.0] · [0.2, 1.5]
= 1.5

Q_bisa_head_2 · K_terkena_head_2
= [0.0, 1.0] · [0.1, 0.5]
= 0.5

Q_bisa_head_2 · K_bisa_head_2
= [0.0, 1.0] · [0.4, 1.0]
= 1.0
```

Softmax:

```text
softmax([1.5, 0.5, 1.0])
≈ [0.51, 0.19, 0.31]

Head 2:
Saya    → sekitar 50.6%
terkena → sekitar 18.6%
bisa    → sekitar 30.7%
```

Terlihat bahwa kedua head membaca context window yang sama, tetapi menghasilkan pola relevansi berbeda:

```text
Head 1 lebih kuat memperhatikan "terkena"
Head 2 lebih kuat memperhatikan "Saya"
```

Persentase setiap head selalu dinormalisasi secara terpisah:

```text
total asli Head 1 = 100%
total asli Head 2 = 100%
```

Jumlah angka yang sudah dibulatkan dapat terlihat sedikit berbeda dari `100%`.

Dalam model nyata, head tidak diberi tugas secara manual. Perbedaan pola tersebut terbentuk dari bobot hasil training.

## 8. V: membawa informasi setiap token

V dibuat secara terpisah menggunakan `v_proj`:

```text
hidden_Saya    × Wv → V_Saya
hidden_terkena × Wv → V_terkena
hidden_bisa    × Wv → V_bisa
```

Pada contoh dua head, setiap head juga mempunyai bagian V sendiri. Misalkan:

```text
V Head 1:
V_Saya_head_1    = [ 1.0, 0.0]
V_terkena_head_1 = [-1.0, 0.0]
V_bisa_head_1    = [ 0.0, 1.0]

V Head 2:
V_Saya_head_2    = [0.2, 0.8]
V_terkena_head_2 = [0.6, 0.1]
V_bisa_head_2    = [0.1, 0.5]
```

Head 1 menggunakan bobot relevansi milik Head 1:

```text
23% × V_Saya_head_1
+ 63% × V_terkena_head_1
+ 14% × V_bisa_head_1

= 0.23 × [ 1.0, 0.0]
+ 0.63 × [-1.0, 0.0]
+ 0.14 × [ 0.0, 1.0]

= [-0.40, 0.14]
```

Head 2 menggunakan bobot relevansi milik Head 2:

```text
50.6% × V_Saya_head_2
+ 18.6% × V_terkena_head_2
+ 30.7% × V_bisa_head_2

≈ 0.506 × [0.2, 0.8]
+ 0.186 × [0.6, 0.1]
+ 0.307 × [0.1, 0.5]

≈ [0.24, 0.58]
```

Maka setiap head menghasilkan vector dua dimensi:

```text
output Head 1 → [-0.40, 0.14]
output Head 2 → [ 0.24, 0.58]
```

Ringkasnya:

```text
Q dan K → menentukan seberapa besar pengaruh setiap token
V       → menyediakan informasi vector yang dipengaruhi
```

V bukan hasil dari K. Q, K, dan V dibuat secara paralel dari hidden vector menggunakan matrix bobot yang berbeda.

## 9. O: menggabungkan semua head

Misalkan dua head menghasilkan:

```text
Head 1 → [-0.40, 0.14]
Head 2 → [ 0.24, 0.58]
```

Keduanya digabungkan:

```text
[-0.40, 0.14, 0.24, 0.58]
```

Proses penggabungan ini disebut `concatenate`. Output head disambungkan, bukan langsung dijumlahkan:

```text
Head 1 + Head 2                         ❌ bukan penjumlahan vector
concat(Head 1, Head 2)                  ✅ penyambungan vector
```

Jika ada head lain, hasilnya ikut ditambahkan ke **daftar gabungan**:

```text
Head 1 → [-0.40, 0.14]
Head 2 → [ 0.24, 0.58]
Head 3 → [ 0.40, -0.10]

concat(Head 1, Head 2, Head 3)
→ [-0.40, 0.14, 0.24, 0.58, 0.40, -0.10]
```

Jadi kata “ditambahkan” di sini berarti **head tambahan ikut disambungkan**, bukan nilainya dijumlahkan per dimensi.

Gabungan tersebut diproses oleh `o_proj`:

```text
gabungan head × Wo
→ output attention berukuran 4 dimensi
```

Ukuran `Wo` menyesuaikan total dimensi hasil seluruh head dan mengubahnya kembali ke `hidden_size`:

```text
2 head × 2 dimensi = 4 dimensi gabungan
→ o_proj
→ 4 dimensi hidden output
```

Jika contoh arsitekturnya menggunakan tiga head dengan dua dimensi per head:

```text
3 head × 2 dimensi = 6 dimensi gabungan
→ o_proj
→ kembali ke hidden_size yang ditentukan model
```

`o_proj` mempunyai dua fungsi utama:

1. mencampurkan informasi dari seluruh head; dan
2. menghasilkan output dengan ukuran `hidden_size` agar dapat diteruskan.

Setelah itu, residual connection menambahkan hidden vector sebelumnya:

```text
contextual vector baru
= hidden sebelumnya + hasil o_proj
```

Untuk pemahaman dasar pada Step 8, kita menyebut hasil setelah `o_proj` dan residual sebagai **contextual vector baru**.

```text
Q dan K
→ bobot relevansi
→ bobot × V
→ hasil setiap head
→ gabungkan head
→ o_proj
→ residual
→ contextual vector baru
```

## 10. Context window dan contextual vector

Keduanya berhubungan langsung, tetapi bukan hal yang sama:

```text
context window
= token-token yang tersedia untuk dilihat model

contextual vector
= representasi angka hasil pemrosesan token-token tersebut
```

Contoh:

```text
Context window: "Saya bisa"
→ contextual vector pada posisi "bisa" membawa pola kemampuan

Context window: "Saya terkena bisa"
→ contextual vector pada posisi "bisa" membawa pola racun
```

Dalam causal language model:

```text
posisi "Saya"    → dapat melihat "Saya"
posisi "terkena" → dapat melihat "Saya terkena"
posisi "bisa"    → dapat melihat "Saya terkena bisa"
```

Itulah sebabnya percakapan dapat kehilangan konteks ketika riwayat atau memory dihapus:

```text
token konteks tidak lagi diberikan
→ token tersebut tidak tersedia di context window
→ attention tidak mempunyai K dan V dari token tersebut
→ contextual vector berikutnya tidak membawa informasi itu
```

Namun, menghapus memory percakapan tidak menghapus pengetahuan hasil training dari bobot model. Jika teks konteks masih dikirim kembali, contextual vector dapat dihitung ulang.

## 11. Setiap layer memperbarui vector

Output satu layer menjadi input layer berikutnya:

```text
embedding awal
    ↓
Layer 1 → contextual vector versi 1
    ↓
Layer 2 → contextual vector versi 2
    ↓
Layer 3 → contextual vector versi 3
    ↓
...
    ↓
Layer terakhir → contextual vector akhir
```

Setiap layer mempunyai `Wq`, `Wk`, `Wv`, dan `Wo` sendiri. Karena input dan bobot setiap layer berbeda, nilai vector hasilnya biasanya berbeda. Kemungkinan seluruh koordinat pada dua layer sama persis sangat kecil.

Residual connection tetap membawa informasi lama:

```text
vector baru
= vector sebelumnya + hasil pemrosesan layer
```

Vector yang terus diperbarui dan ruang dimensi yang cukup besar memberi model kapasitas untuk membentuk serta memisahkan pola relevansi yang lebih kaya. Namun, dimensi yang lebih besar tetap harus didukung training dan data yang baik.

## 12. `lm_head` membandingkan seluruh vocabulary

Setelah layer terakhir, model mengambil contextual vector pada posisi terakhir:

```text
"Saya terkena bisa"
→ contextual vector akhir pada posisi "bisa"
```

Misalnya, untuk ilustrasi dua dimensi:

```text
h_akhir = [-0.6, 0.2]
```

`lm_head` mempunyai satu vector output untuk setiap token dalam vocabulary model:

```text
"ular"     → [-1.0, 0.0]
"berenang" → [ 1.0, 0.0]
"makan"    → [ 0.2, 0.7]
...
```

Contextual vector akhir dibandingkan dengan **semua token vocabulary**, bukan hanya dua contoh tersebut:

```text
skor "ular"
= [-0.6, 0.2] · [-1.0, 0.0]
= 0.6

skor "berenang"
= [-0.6, 0.2] · [1.0, 0.0]
= -0.6
```

Hasil `lm_head` adalah logit:

```text
contextual vector akhir
→ lm_head
→ satu logit untuk setiap token vocabulary
→ softmax
→ probabilitas
```

Secara analogi, token dengan vector output yang paling dekat atau cocok akan mendapat skor tinggi. Secara teknis, yang dihitung biasanya adalah dot product, bukan jarak geometris biasa.

Model juga tidak selalu mengambil probabilitas tertinggi:

```text
greedy decoding → memilih probabilitas tertinggi
sampling        → memilih berdasarkan distribusi probabilitas
```

Jadi token berskor tinggi adalah kandidat terkuat, tetapi strategi decoding menentukan token yang benar-benar dipilih.

## 13. Prediksi berikutnya dilakukan dengan loop

Misalkan token yang dipilih adalah `"ular"`:

```text
"Saya terkena bisa"
+ "ular"
= "Saya terkena bisa ular"
```

Token `"ular"` ditambahkan ke context window. Proses dimulai kembali untuk menghasilkan token berikutnya:

```text
token baru
→ embedding
→ Layer 1
→ Layer 2
→ ...
→ Layer terakhir
→ contextual vector akhir baru
→ lm_head
→ pilih token berikutnya
```

Proses tersebut berulang satu token demi satu token:

```text
contextual vector
→ lm_head
→ pilih token
→ tambahkan ke context window
→ proses seluruh layer lagi
→ contextual vector baru
→ lm_head lagi
```

Dalam implementasi nyata, KV cache menyimpan K dan V token lama agar tidak semuanya dihitung ulang. Namun secara logika, token baru tetap melewati seluruh Transformer layer dari Layer 1 sampai layer terakhir.

## Gambaran keseluruhan

```text
Input: "Saya terkena bisa"
        ↓
Token ID dan embedding vector
        ↓
Layer 1
├── q_proj: membuat Q
├── k_proj: membuat K
├── Q · K: menghasilkan bobot relevansi
├── v_proj: membuat V
├── bobot relevansi × V
├── gabungkan semua head
└── o_proj + residual: contextual vector baru
        ↓
Layer 2 menerima contextual vector Layer 1
        ↓
...
        ↓
Contextual vector dari layer terakhir
        ↓
lm_head membandingkan seluruh vocabulary
        ↓
logit → softmax/decoding
        ↓
Token berikutnya: "ular"
        ↓
Tambahkan ke context window dan ulangi dari Layer 1
```

## Kesimpulan utama

1. Vector dapat dibayangkan sebagai titik koordinat dalam ruang berdimensi tinggi.
2. Dimensi menyediakan kapasitas untuk merepresentasikan banyak pola, bukan satu makna per dimensi.
3. Pada batas setiap Transformer layer, ukuran vector kembali ke `hidden_size` agar dapat diteruskan dan dijumlahkan melalui residual.
4. Attention head membagi pekerjaan attention; `hidden_size = 4` dan `2 head` berarti `2 dimensi per head`, tetapi jumlah head adalah konfigurasi model.
5. Q dan K menghasilkan bobot relevansi, sedangkan V membawa informasi yang dicampurkan.
6. `o_proj` menggabungkan hasil semua head; setelah residual, terbentuk contextual vector baru untuk layer berikutnya.
7. Context window menyediakan token sumber konteks, sedangkan contextual vector adalah hasil pengolahannya.
8. Setiap layer memperbarui vector menggunakan bobotnya sendiri sehingga vector biasanya berbeda pada setiap layer.
9. `lm_head` menghitung skor contextual vector akhir terhadap seluruh vocabulary.
10. Token terpilih ditambahkan ke konteks, lalu proses diulang untuk memprediksi token berikutnya.
