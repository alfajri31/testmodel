# 🟢 Step 12 — Cara Kerja Neural Network

Step ini menjelaskan bagaimana neural network memperbaiki bobot melalui
prediksi, loss, gradient, dan learning rate.

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

Model masih salah karena prediksi `1`, sedangkan targetnya `2`.

## 🟢 Loss

Loss mengukur seberapa jauh prediksi dari target.

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

## 🟢 Gradient

Gradient menunjukkan arah dan besar perubahan bobot yang diperlukan.

Untuk model sederhana:

```text
gradient_W1 = (prediksi - target) × input_1
gradient_W2 = (prediksi - target) × input_2
```

Perhitungannya:

```text
gradient_W1 = (1 - 2) × 1 = -1
gradient_W2 = (1 - 2) × 1 = -1
```

Jadi:

```text
gradient = [-1, -1]
```

Gradient bukan bobot baru. Gradient hanya memberi petunjuk bagaimana bobot
harus diubah.

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
Bobot       → kemampuan yang dipelajari model
Prediksi    → tebakan model
Target      → jawaban yang diharapkan
Loss        → ukuran kesalahan
Gradient    → arah perubahan bobot
Learning rate → ukuran langkah perubahan
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

