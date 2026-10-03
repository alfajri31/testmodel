# <span style="color: #2563EB;">Step 6 — Deploy Model ke Ollama</span>

Pada step ini, merged model dari Step 5 dijalankan di Ollama melalui dua jalur:

```text
Base Model + LoRA
        ↓ merge pada Step 5
Merged Model Safetensors
        ├── import langsung ke Ollama
        └── konversi llama.cpp → GGUF → Ollama
```

Model sumber:

```text
/Users/fajri/Projects/testmodel/step5/merged_model
```

## <span style="color: #2563EB;">Dua versi model</span>

```text
alfajri-qwen3
└── import langsung dari Safetensors

alfajri-qwen3-gguf-f32
└── hasil konversi Safetensors menjadi GGUF F32
```

Keduanya berasal dari merged model yang sama dan belum menggunakan quantization Q4/Q8.

## <span style="color: #2563EB;">Struktur Step 6</span>

```text
step6/
├── deploy_ollama_versi_safetensor_ori.py
├── setup_llama_cpp.py
├── convert_safetensor_ori_ke_cpp.py
├── deploy_ollama_versi_llama_cpp.py
├── llama.cpp/
│   └── convert_hf_to_gguf.py
├── .venv-llama-cpp/
├── Modelfile
├── Modelfile.llama_cpp
├── qwen3-alfajri-f32.gguf
└── README.md
```

Penjelasan setup Python 3.9, Python 3.11, `pip`, dan virtual environment berada di:

```text
/Users/fajri/Projects/testmodel/step0/README.md
```

## <span style="color: #2563EB;">Persiapan Ollama</span>

Cek Ollama:

```bash
ollama --version
```

Cek server:

```bash
curl http://127.0.0.1:11434/api/version
```

Jika server belum aktif:

```bash
ollama serve
```

Jika aplikasi Ollama sudah berjalan, `ollama serve` tidak perlu dijalankan lagi.

---

## <span style="color: #2563EB;">Jalur 1 — Safetensors langsung ke Ollama</span>

Alur:

```text
Merged Safetensors
        ↓ ollama create --experimental
Ollama: alfajri-qwen3
```

Jalankan:

```bash
python3.11 \
  /Users/fajri/Projects/testmodel/step6/deploy_ollama_versi_safetensor_ori.py
```

Script tersebut:

```text
1. Memeriksa file merged model
2. Memeriksa server Ollama
3. Membuat Modelfile
4. Menjalankan ollama create --experimental
5. Mengimpor Safetensors dan tokenizer
6. Mendaftarkan model sebagai alfajri-qwen3
```

Ollama mengimpor file berikut:

```text
model.safetensors
config.json
generation_config.json
tokenizer.json
tokenizer_config.json
special_tokens_map.json
added_tokens.json
vocab.json
```

Jalankan model:

```bash
ollama run alfajri-qwen3:latest
```

Contoh hasil:

```text
>>> siapakah alfajri?
<think>

</think>

salah satu teman Alfajri programmer java.
```

---

## <span style="color: #2563EB;">Jalur 2 — Safetensors menjadi GGUF</span>

Alur:

```text
Merged Safetensors
        ↓ convert_hf_to_gguf.py
qwen3-alfajri-f32.gguf
        ↓ ollama create
Ollama: alfajri-qwen3-gguf-f32
```

### <span style="color: #2563EB;">Apa itu llama.cpp?</span>

`llama.cpp` adalah runtime untuk menjalankan LLM secara lokal dengan dukungan CPU, Apple Metal, dan berbagai environment.

Dalam step ini, repository `llama.cpp` juga menyediakan converter resmi:

```text
convert_hf_to_gguf.py
```

Python menjalankan converter tersebut, sedangkan source code converter tetap berasal dari repository `llama.cpp`.

### <span style="color: #2563EB;">Cara kerja converter secara sederhana</span>

convert_hf_to_gguf.py adalah **alat converter**.

Python di .venv adalah **mesin yang menjalankan alat**.

Library llama.cpp, seperti gguf-py dan conversion, adalah **komponen pendukung alat**.

Safetensors adalah **bahan masukannya**.

GGUF adalah **hasilnya**.

`convert_hf_to_gguf.py` membutuhkan library pendukung dari `llama.cpp`. Karena itu, ambil seluruh repository:

```bash
git clone https://github.com/ggml-org/llama.cpp.git
```

Walaupun namanya `llama.cpp`, proses konversinya dijalankan dengan Python. Bagian C/C++ terutama digunakan untuk menjalankan model GGUF.

### <span style="color: #2563EB;">Setup llama.cpp</span>

Repository yang digunakan berada di:

```text
/Users/fajri/Projects/testmodel/step6/llama.cpp
```

Converter:

```text
/Users/fajri/Projects/testmodel/step6/llama.cpp/convert_hf_to_gguf.py
```

Virtual environment harus dibuat menggunakan Python 3.11, bukan Python 3.9:

```bash
python3.11 -m venv \
  /Users/fajri/Projects/testmodel/step6/.venv-llama-cpp-311
```

Cek versinya:

```bash
/Users/fajri/Projects/testmodel/step6/.venv-llama-cpp-311/bin/python \
  --version
```

Instal dependency converter:

```bash
/Users/fajri/Projects/testmodel/step6/.venv-llama-cpp-311/bin/python \
  -m pip install --upgrade pip setuptools wheel
```

```bash
/Users/fajri/Projects/testmodel/step6/.venv-llama-cpp-311/bin/python \
  -m pip install \
  -r /Users/fajri/Projects/testmodel/step6/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
```

> Pastikan nilai `VENV_PYTHON` dalam script converter mengarah ke `.venv-llama-cpp-311/bin/python`. Environment `.venv-llama-cpp` lama dibuat dengan Python 3.9.6.

### <span style="color: #2563EB;">Konversi ke GGUF F32</span>

Script konversi:

```text
convert_safetensor_ori_ke_cpp.py
```

Jalankan:

```bash
python3.11 \
  /Users/fajri/Projects/testmodel/step6/convert_safetensor_ori_ke_cpp.py
```

Perintah inti yang dijalankan:

```bash
/Users/fajri/Projects/testmodel/step6/.venv-llama-cpp-311/bin/python \
  /Users/fajri/Projects/testmodel/step6/llama.cpp/convert_hf_to_gguf.py \
  /Users/fajri/Projects/testmodel/step5/merged_model \
  --outfile /Users/fajri/Projects/testmodel/step6/qwen3-alfajri-f32.gguf \
  --outtype f32
```

Hasil:

```text
/Users/fajri/Projects/testmodel/step6/qwen3-alfajri-f32.gguf
```

Ukuran file saat ini:

```text
sekitar 2,2 GB
```

`F32` mempertahankan bobot dalam presisi 32-bit. Pada step ini belum dilakukan quantization Q4/Q8.

### <span style="color: #2563EB;">Deploy GGUF ke Ollama</span>

Jalankan:

```bash
python3.11 \
  /Users/fajri/Projects/testmodel/step6/deploy_ollama_versi_llama_cpp.py
```

Script membuat `Modelfile.llama_cpp`, lalu menjalankan:

```text
ollama create alfajri-qwen3-gguf-f32
```

Jalankan model:

```bash
ollama run alfajri-qwen3-gguf-f32:latest
```

Contoh hasil:

```text
>>> siapakah alfajri?
salah satu teman Alfajri programmer java.
```

---

## <span style="color: #2563EB;">Perbedaan `<think>` pada kedua versi</span>

`<think>` bukan tag HTML. Itu adalah special token atau pola keluaran Qwen3 untuk memisahkan proses thinking dan jawaban akhir.

Pemeriksaan konfigurasi efektif:

```bash
ollama show --modelfile alfajri-qwen3:latest
```

```bash
ollama show --modelfile alfajri-qwen3-gguf-f32:latest
```

Hasil versi Safetensors:

```text
TEMPLATE {{ .Prompt }}
RENDERER qwen3-coder
PARSER qwen3
```

Hasil versi GGUF:

```text
TEMPLATE {{ .Prompt }}
```

Pada hasil import saat ini:

```text
Safetensors
└── memakai RENDERER qwen3-coder dan PARSER qwen3
    └── format <think> dikenali dan ditampilkan

GGUF
└── tidak memperoleh renderer/parser tersebut
    └── jawaban tampil langsung tanpa <think>
```

Kedua model tetap terdeteksi memiliki capability:

```text
completion
tools
thinking
```

Jadi, GGUF tidak otomatis menghapus kemampuan thinking. Perbedaan tampilan muncul karena jalur import serta konfigurasi template, renderer, dan parser Ollama tidak sama.

Kesimpulan khusus hasil model ini:

> Versi GGUF berjalan tanpa `PARSER qwen3`, sehingga format thinking tidak diproses dan ditampilkan seperti versi Safetensors.

Hal tersebut bukan aturan untuk semua model GGUF. Model GGUF lain dapat memakai parser atau chat template yang berbeda.

---

## <span style="color: #2563EB;">Mengecek model</span>

Daftar model:

```bash
ollama list
```

Detail Safetensors:

```bash
ollama show alfajri-qwen3:latest
```

Detail GGUF:

```bash
ollama show alfajri-qwen3-gguf-f32:latest
```

Lihat Modelfile efektif:

```bash
ollama show --modelfile alfajri-qwen3:latest
ollama show --modelfile alfajri-qwen3-gguf-f32:latest
```

Lihat model yang sedang dimuat ke memori:

```bash
ollama ps
```

## <span style="color: #2563EB;">Benchmark sederhana</span>

Safetensors:

```bash
ollama run alfajri-qwen3:latest \
  "Siapakah Alfajri?" \
  --verbose
```

GGUF:

```bash
ollama run alfajri-qwen3-gguf-f32:latest \
  "Siapakah Alfajri?" \
  --verbose
```

Metrik:

```text
load duration     → waktu memuat model
prompt eval rate  → kecepatan membaca input token
eval rate         → kecepatan menghasilkan output token
total duration    → total waktu request
```

Gunakan prompt, `num_ctx`, dan parameter generation yang sama agar perbandingannya adil.

## <span style="color: #2563EB;">Isi Modelfile</span>

Pengaturan deployment model ke Ollama disimpan di dalam `Modelfile`.

```text
Modelfile
└── pengaturan deploy versi Safetensors

Modelfile.llama_cpp
└── pengaturan deploy versi GGUF
```

`Modelfile` menentukan model sumber, context, temperature, dan system prompt. Script Python kemudian membaca atau membuat file tersebut dan menjalankan `ollama create`.

Safetensors:

```text
FROM /Users/fajri/Projects/testmodel/step5/merged_model

PARAMETER num_ctx 4096
PARAMETER temperature 0

SYSTEM """Jawab pertanyaan dengan singkat dan jelas berdasarkan informasi yang telah dipelajari."""
```

GGUF:

```text
FROM /Users/fajri/Projects/testmodel/step6/qwen3-alfajri-f32.gguf

PARAMETER num_ctx 4096
PARAMETER temperature 0

SYSTEM """Jawab pertanyaan dengan singkat dan jelas berdasarkan informasi yang telah dipelajari."""
```

Penjelasan:

```text
FROM        → lokasi model sumber
num_ctx     → context yang disediakan saat inference
temperature → 0 agar keluaran lebih konsisten
SYSTEM      → instruksi dasar untuk model
```

`num_ctx 4096` bukan batas RAM. Nilai tersebut adalah jumlah maksimum token context yang disediakan untuk percakapan. Context yang lebih besar biasanya membutuhkan memori lebih banyak.

## <span style="color: #2563EB;">Kesimpulan Step 6</span>

Pada Step 6, merged model dari Step 5 berhasil dimasukkan ke Ollama dengan dua cara:

```text
Merged Safetensors
├── langsung ke Ollama → alfajri-qwen3
└── dikonversi ke GGUF → alfajri-qwen3-gguf-f32
```

Versi Safetensors menggunakan fitur import eksperimental Ollama dan memperoleh `RENDERER qwen3-coder` serta `PARSER qwen3`.

Versi GGUF F32 dapat dijalankan oleh Ollama melalui format yang sesuai dengan `llama.cpp`, tetapi pada hasil import ini tidak memperoleh `PARSER qwen3`. Oleh karena itu, tampilan `<think>` pada kedua versi dapat berbeda.

Kedua versi berasal dari bobot fine-tuning yang sama dan belum menggunakan quantization Q4/Q8.

## <span style="color: #2563EB;">Kesimpulan Keseluruhan Konversi</span>

Tujuan konversi adalah mengubah **format penyimpanan dan runtime model**:

```text
Safetensors
└── format model Hugging Face

GGUF
└── format model untuk runtime seperti llama.cpp dan Ollama
```

Konversi dilakukan agar model:

- kompatibel dengan `llama.cpp` dan Ollama;
- lebih mudah dijalankan secara lokal;
- dapat memakai optimasi CPU dan Apple Metal;
- dapat di-quantization pada tahap berikutnya jika diperlukan;
- lebih mudah dipindahkan sebagai file GGUF.

Konversi bukan proses training:

```text
Konversi tidak menambahkan dataset.
Konversi tidak melakukan fine-tuning ulang.
Konversi tidak menambah vocabulary atau jumlah token.
Konversi tidak membuat model lebih pintar.
Konversi tidak memperbaiki jawaban yang salah.
```

Konversi juga tidak otomatis mengurangi kecerdasan model. Dampaknya bergantung pada presisi bobot yang dipilih:

```text
Safetensors F32 → GGUF F32
└── presisi tetap F32

Safetensors F64 → GGUF F32
└── presisi berkurang dari 64-bit menjadi 32-bit

Safetensors F32/BF16 → GGUF Q8/Q4
└── bobot dibulatkan melalui quantization
```

Model LLM umumnya tidak menggunakan F64 karena membutuhkan ukuran file dan memori yang sangat besar. Merged model pada Step 5 menggunakan F32 dan dikonversi menjadi GGUF F32:

```text
Merged Safetensors F32
        ↓ --outtype f32
GGUF F32
```

Karena itu, konversi pada Step 6 bukan proses pengurangan dari F64 menjadi F32. Kemampuan model seharusnya tetap sangat dekat. Perbedaan tampilan jawaban, seperti hilangnya `<think>`, lebih berkaitan dengan template, renderer, parser, dan runtime Ollama.

Penurunan kemampuan lebih mungkin terjadi jika model di-quantization ke presisi lebih rendah. Semakin agresif quantization, semakin besar kemungkinan detail bobot dan kualitas jawaban berkurang.

Fine-tuning pada Step 2–5 yang membuat bobot model lebih cenderung mengikuti dataset. Step 6 hanya menyiapkan model tersebut agar dapat dijalankan pada runtime yang dipilih.

Kesimpulan sederhananya:

```text
Fine-tuning → mengubah kecenderungan jawaban model
Merge       → menggabungkan bobot LoRA dengan base model
Konversi    → mengubah format model untuk runtime
Deploy      → mendaftarkan dan menjalankan model di Ollama
```
