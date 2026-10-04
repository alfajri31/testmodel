# 🟢 Step 12 — Cara Kerja Neural Network

Step ini menjelaskan bagaimana neural network memperbaiki bobot melalui
prediksi, loss, gradient, dan learning rate. Kita juga melihat validation loss
untuk memeriksa hasil pada contoh yang tidak dipakai memperbarui bobot.

## 🟢 Gambaran sederhana

Model menerima input:

```text
1 + 1
```

Lalu model memprediksi hasil:

```text
prediksi → 1
target   → 2
```

Karena prediksi berbeda dari target, bobot model diperbaiki.

Alur training:

```text
input
→ prediksi
→ bandingkan dengan target
→ hitung loss
→ hitung gradient
→ perbarui bobot
```

## 🟢 Bobot awal

Misalkan model sederhana memiliki dua bobot:

```text
W1 = 0.5
W2 = 0.5
```

Kedua angka tersebut dapat ditulis sebagai satu vector bobot:

```text
W = [W1, W2]
W = [0.5, 0.5]
```

`W1` dipasangkan dengan angka pertama, sedangkan `W2` dipasangkan dengan
angka kedua.

Rumus model:

```text
hasil = (angka pertama × W1) + (angka kedua × W2)
```

Untuk input `1 + 1`:

```text
hasil = (1 × 0.5) + (1 × 0.5)
      = 1
```

Target yang benar adalah:

```text
target = 2
```

## 🟢 Prediksi

Prediksi adalah **hasil yang diberikan model dengan bobotnya saat ini**. Pada
contoh di atas, input `1 + 1` menghasilkan prediksi `1`. Target dari data
training adalah `2`, sehingga prediksi itu belum tepat.

Dalam contoh ini prediksi berupa angka hasil rumus sederhana. Pada LLM,
prediksi berikutnya berupa skor untuk kandidat token, misalnya token `"2"`
setelah teks `"1 + 1 ="`.

## 🟢 Loss

Loss mengukur seberapa jauh prediksi dari target. Jika dihitung pada contoh
training, nilainya disebut **training loss**.

Rumus sederhana:

```text
loss = ½ × (prediksi - target)²
```

Dengan prediksi `1` dan target `2`:

```text
loss = ½ × (1 - 2)²
     = 0.5
```

Semakin besar perbedaan prediksi dan target, semakin besar loss.

## 🟢 Validation loss

Validation loss dihitung dengan rumus loss yang sama, tetapi memakai **data
validasi**, yaitu contoh yang tidak dipakai untuk memperbarui bobot. Tujuannya
melihat apakah model juga bekerja pada input lain, bukan hanya pada contoh
training.

Misalnya data validasi berisi `5 + 5` dengan target `10`. Dengan bobot awal
`W = [0.5, 0.5]`:

```text
prediksi validasi = (5 × 0.5) + (5 × 0.5) = 5
validation loss   = ½ × (5 − 10)² = 12.5
```

Angka `12.5` dapat dibandingkan dengan validation loss setelah beberapa
update training. **Perhitungan validasi tidak mengubah bobot**: tidak ada
langkah update dari contoh `5 + 5` ini. Dalam latihan kecil ini, model tetap
belajar dari data training, lalu diuji pada data validasi.

Training loss yang turun menunjukkan model makin cocok dengan contoh latihan.
Jika training loss turun tetapi validation loss tetap tinggi atau naik, model
mungkin terlalu mengikuti data latihan dan kurang mampu menangani contoh baru.

## 🟢 Gradient

Gradient menunjukkan bagaimana loss berubah ketika sebuah bobot digeser.
Kita lihat langsung melalui angka. Kedua input bernilai `1`, targetnya `2`,
dan `W2` tetap `0.5`. Jadi `prediksi = W1 + W2`.

Pegang `W2 = 0.5`, lalu coba tiga nilai `W1`:

| W1 | Prediksi (`W1 + W2`) | Loss `½ × (prediksi − 2)²` |
|---:|---:|---:|
| 0.49 | 0.99 | 0.51005 |
| 0.50 | 1.00 | 0.50000 |
| 0.51 | 1.01 | 0.49005 |

Contoh perhitungan loss pada baris pertama dan terakhir:

```text
W1 = 0.49 → loss = ½ × (0.99 − 2)² = 0.51005
W1 = 0.51 → loss = ½ × (1.01 − 2)² = 0.49005
```

Saat `W1` naik dari `0.49` ke `0.51`, loss turun. Besar kemiringan loss di
sekitar `W1 = 0.50`:

```text
perubahan loss / perubahan W1
= (0.49005 − 0.51005) / (0.51 − 0.49)
= −0.02 / 0.02
= −1
```

Itulah **gradient untuk W1** di titik `W1 = 0.5`. Tanda negatif berarti
menaikkan `W1` akan menurunkan loss. Angka loss pada titik itu tetap `0.5`;
loss dan gradient memang mengukur hal yang berbeda.

Jika giliran `W2` digeser sambil `W1` ditahan di `0.5`, perhitungannya sama
karena kedua input bernilai `1`. Maka:

```text
gradient = [gradient_W1, gradient_W2] = [−1, −1]
```

Cara mencoba pergeseran seperti ini membantu kita melihat arti gradient.
Pada model nyata, gradient dihitung dengan backpropagation tanpa mencoba
setiap bobot satu per satu.

## 🟢 Learning rate

Learning rate menentukan besar langkah perubahan bobot.

Rumus pembaruan:

```text
bobot_baru = bobot_lama - learning_rate × gradient
```

Misalkan:

```text
learning_rate = 0.1
```

Maka:

```text
W1_baru = 0.5 - (0.1 × -1) = 0.6
W2_baru = 0.5 - (0.1 × -1) = 0.6
```

Perbandingannya:

```text
bobot lama → [0.5, 0.5]
gradient   → [-1.0, -1.0]
bobot baru  → [0.6, 0.6]
```

Jika learning rate terlalu kecil, training berjalan lambat. Jika terlalu besar,
perubahan bobot dapat meloncat-loncat dan training menjadi tidak stabil.

## 🟢 Bobot diperbaiki berulang kali

Model mencoba lagi:

```text
hasil = (1 × 0.6) + (1 × 0.6)
      = 1.2
```

Target tetap `2`, sehingga bobot diperbaiki lagi. Setelah banyak iterasi,
bobot dapat mendekati:

```text
W1 = 1.0
W2 = 1.0
```

Sekarang:

```text
1 + 1 = (1 × 1.0) + (1 × 1.0)
      = 2
```

## 🟢 Mengapa 5 + 5 dapat menjadi 10?

Jika model telah mempelajari pola bahwa kedua input dijumlahkan:

```text
hasil = angka pertama + angka kedua
```

maka bobot yang sama dapat digunakan untuk input lain:

```text
5 + 5 = (5 × 1.0) + (5 × 1.0)
      = 10
```

Model tidak harus menghafal setiap pasangan angka secara terpisah. Bobot yang
sama digunakan kembali untuk berbagai input.

Namun, jika model hanya pernah melihat satu contoh `1 + 1 = 2`, kemampuan
`5 + 5 = 10` tidak dijamin. Model memerlukan cukup banyak variasi contoh agar
dapat mempelajari pola umum.

## 🟢 Hubungan dengan Transformer

Contoh di atas sangat sederhana. Transformer sebenarnya menggunakan:

```text
token
→ embedding
→ attention
→ MLP
→ banyak layer
→ lm_head
→ prediksi token
```

Semua layer memiliki bobot. Saat training:

```text
prediksi token dibandingkan dengan target token
→ loss dihitung
→ backpropagation menghitung gradient
→ optimizer memperbarui bobot
```

Pada model bahasa, `1`, `+`, `5`, dan `=` tetap diproses sebagai token.
Kemampuan aritmetika muncul dari pola yang tersimpan secara tersebar pada
bobot Transformer.

## 🟢 Ringkasan

```text
Bobot           → angka yang dipelajari model
Prediksi        → hasil dari input dan bobot saat ini
Target          → jawaban yang diharapkan
Training loss   → kesalahan pada contoh yang dipakai belajar
Validation loss → kesalahan pada contoh terpisah untuk memeriksa hasil
Gradient        → arah dan kemiringan perubahan loss terhadap bobot
Learning rate   → ukuran langkah perubahan bobot
```

Kesimpulannya:

```text
bobot awal
→ prediksi salah
→ loss
→ gradient
→ learning rate
→ bobot diperbaiki
→ prediksi semakin baik
```
