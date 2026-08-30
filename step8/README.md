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

Kemudian terjadi perbandingan silang:

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

Setelah softmax, misalnya diperoleh bobot relevansi:

```text
Saya    → 20%
terkena → 70%
bisa    → 10%
```

Persentase tersebut bukan isi K. Persentase adalah hasil perbandingan Q dengan semua K yang dapat dilihat.

```text
Q dan K
→ skor relevansi
→ softmax
→ bobot relevansi
```

## 7. V: membawa informasi setiap token

V dibuat secara terpisah menggunakan `v_proj`:

```text
hidden_Saya    × Wv → V_Saya
hidden_terkena × Wv → V_terkena
hidden_bisa    × Wv → V_bisa
```

Bobot relevansi kemudian digunakan untuk mencampurkan V:

```text
20% × V_Saya
+ 70% × V_terkena
+ 10% × V_bisa
= output satu attention head pada posisi "bisa"
```

Ringkasnya:

```text
Q dan K → menentukan seberapa besar pengaruh setiap token
V       → menyediakan informasi vector yang dipengaruhi
```

V bukan hasil dari K. Q, K, dan V dibuat secara paralel dari hidden vector menggunakan matrix bobot yang berbeda.

## 8. O: menggabungkan semua head

Misalkan dua head menghasilkan:

```text
Head 1 → [0.7, 0.3]
Head 2 → [-0.6, 0.2]
```

Keduanya digabungkan:

```text
[0.7, 0.3, -0.6, 0.2]
```

Gabungan tersebut diproses oleh `o_proj`:

```text
gabungan head × Wo
→ output attention berukuran 4 dimensi
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

## 9. Context window dan contextual vector

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

## 10. Setiap layer memperbarui vector

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

## 11. `lm_head` membandingkan seluruh vocabulary

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

## 12. Prediksi berikutnya dilakukan dengan loop

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
