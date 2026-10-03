# [Step 0 — Python untuk Proyek](#)

## [Memahami nama dan versi Python](#)

Nama perintah Python menunjukkan versi interpreter yang akan dijalankan:

```text
python      → nama umum; versinya bergantung pada konfigurasi komputer
python2     → Python generasi 2, biasanya Python 2.7
python3     → salah satu versi dalam keluarga Python 3
python3.9   → khusus Python 3.9
python3.11  → khusus Python 3.11
```

Angka versi Python terdiri dari:

```text
Python 3.11.15
       │ │  └── patch: perbaikan bug dan keamanan
       │ └───── minor: fitur dan kompatibilitas
       └─────── major: generasi utama Python
```

`python2` bukan berarti Python versi `2.0`. Perintah tersebut biasanya menunjuk ke rilis terakhir keluarga Python 2, yaitu `2.7.x`. Python 2 sudah berakhir masa dukungannya dan tidak digunakan untuk proyek baru.

`python3` juga tidak selalu berarti versi terbaru. Ia hanya nama umum untuk salah satu Python generasi 3. Mac ini memiliki dua instalasi Python:

```text
/usr/bin/python3
└── Python 3.9.6 bawaan macOS

/opt/homebrew/bin/python3.11
└── Python 3.11 yang diinstal melalui Homebrew
```

Keduanya tidak saling menggantikan. Oleh karena itu, hasil berikut adalah normal:

```bash
python3 --version
# Python 3.9.6

python3.11 --version
# Python 3.11.x
```

Jangan menghapus atau mengganti Python bawaan macOS. Gunakan versi Python yang sesuai untuk setiap proyek.

## [Bagaimana terminal memilih Python?](#)

Ketika perintah berikut dijalankan:

```bash
python3
```

terminal mencari program bernama `python3` pada daftar lokasi di variabel `PATH`. Program pertama yang ditemukan akan dijalankan.

Lihat urutan pencarian:

```bash
echo $PATH
```

Lihat Python pertama yang ditemukan:

```bash
which python3
```

Lihat semua Python yang ditemukan:

```bash
which -a python3
which -a python3.11
```

Karena `/usr/bin/python3` ditemukan sebagai `python3`, perintah tersebut tetap menghasilkan Python 3.9.6. Sementara itu, Homebrew menyediakan perintah yang lebih spesifik, yaitu `python3.11`.

## [Mengecek versi Python](#)

Cek versi:

```bash
python3 --version
python3.11 --version
```

Cek lokasi program:

```bash
which python3
which python3.11
```

Hasil yang diharapkan:

```text
/usr/bin/python3
/opt/homebrew/bin/python3.11
```

Jika perintah tidak tersedia, terminal akan menampilkan:

```text
command not found
```

Contohnya, macOS dapat menyediakan `python3` tetapi tidak menyediakan perintah `python`.

## [Proses instalasi Python 3.11](#)

Homebrew mengunduh Python 3.11 beserta dependency-nya, lalu menaruh executable di dalam direktori Homebrew. Instalasi ini tidak menimpa `/usr/bin/python3` milik macOS.

Instal menggunakan Homebrew:

```bash
brew install python@3.11
```

Jika sudah pernah diinstal:

```bash
brew upgrade python@3.11
```

Kemudian periksa:

```bash
/opt/homebrew/bin/python3.11 --version
```

Periksa lokasi executable:

```bash
which python3.11
```

Hasil yang diharapkan:

```text
/opt/homebrew/bin/python3.11
```

Ringkasan prosesnya:

```text
brew install python@3.11
            ↓
Homebrew mengunduh Python 3.11
            ↓
/opt/homebrew/bin/python3.11 dibuat
            ↓
python3.11 dapat digunakan dari terminal
```

Memasang Python 3.11 bukan berarti perintah `python3` harus berubah:

```text
python3     → /usr/bin/python3             → Python 3.9.6
python3.11  → /opt/homebrew/bin/python3.11 → Python 3.11.x
```

## [Python dan package](#)

Setiap instalasi Python mempunyai lokasi package sendiri:

```text
Python 3.9
└── package milik Python 3.9

Python 3.11
└── package milik Python 3.11
```

Package yang pernah diinstal untuk Python 3.9 tidak otomatis tersedia di Python 3.11. Gunakan bentuk berikut agar jelas `pip` milik Python mana yang digunakan:

```bash
python3 -m pip list
python3.11 -m pip list
```

Lebih aman memakai:

```bash
python3.11 -m pip install nama_package
```

daripada hanya:

```bash
pip install nama_package
```

Karena `python3.11 -m pip` menjamin package dipasang untuk interpreter Python 3.11 tersebut.

## [Apa itu virtual environment?](#)

Virtual environment adalah lingkungan Python khusus untuk satu proyek. Package di dalamnya tidak bercampur dengan package Python lain.

```text
Python 3.9 environment
└── package untuk proses training lama

Python 3.11 environment
└── package untuk converter llama.cpp
```

Virtual environment yang dibuat dengan Python 3.9 akan tetap memakai Python 3.9. Menginstal Python 3.11 tidak memperbaruinya secara otomatis.

Versi virtual environment ditentukan oleh Python yang dipakai saat pembuatannya:

```bash
python3 -m venv .venv
# menghasilkan environment Python 3.9 pada Mac ini

python3.11 -m venv .venv
# menghasilkan environment Python 3.11
```

Folder `.venv` bukan Python baru yang diunduh. Folder itu merupakan lingkungan terpisah yang dibuat berdasarkan interpreter yang dipilih.

```text
Interpreter pembuat + modul venv + lokasi tujuan

python3.11       -m venv       .venv
     │              │            │
     │              │            └── folder environment
     │              └──────────── modul pembuat environment
     └─────────────────────────── versi Python yang digunakan
```

## [Membuat environment Python 3.11](#)

Buat environment baru untuk `llama.cpp`:

```bash
python3.11 -m venv \
  /Users/fajri/Projects/testmodel/step6/.venv-llama-cpp-311
```

Pastikan environment tersebut benar-benar menggunakan Python 3.11:

```bash
/Users/fajri/Projects/testmodel/step6/.venv-llama-cpp-311/bin/python \
  --version
```

Aktifkan environment:

```bash
source \
  /Users/fajri/Projects/testmodel/step6/.venv-llama-cpp-311/bin/activate
```

Setelah aktif, cek:

```bash
python --version
which python
```

Hasil yang diharapkan:

```text
Python 3.11.x
/Users/fajri/Projects/testmodel/step6/.venv-llama-cpp-311/bin/python
```

Keluar dari environment:

```bash
deactivate
```

## [Menginstal package converter](#)

Perbarui alat instalasi di environment Python 3.11:

```bash
/Users/fajri/Projects/testmodel/step6/.venv-llama-cpp-311/bin/python \
  -m pip install --upgrade pip setuptools wheel
```

Instal dependency converter resmi `llama.cpp`:

```bash
/Users/fajri/Projects/testmodel/step6/.venv-llama-cpp-311/bin/python \
  -m pip install \
  -r /Users/fajri/Projects/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
```

Package tersebut hanya masuk ke `.venv-llama-cpp-311`, bukan ke Python sistem atau environment training.

Alur lengkap setup converter:

```text
Instal Python 3.11 melalui Homebrew
                  ↓
Buat .venv-llama-cpp-311 dengan python3.11
                  ↓
Aktifkan environment
                  ↓
Upgrade pip di dalam environment
                  ↓
Instal requirements llama.cpp
                  ↓
Jalankan convert_hf_to_gguf.py
```

## [Kesalahan yang perlu dihindari](#)

Environment lama berikut memakai Python 3.9:

```text
/Users/fajri/Projects/testmodel/step6/.venv-llama-cpp/bin/python
```

Environment baru berikut memakai Python 3.11:

```text
/Users/fajri/Projects/testmodel/step6/.venv-llama-cpp-311/bin/python
```

Jika pesan error masih menyebut:

```text
.venv-llama-cpp/bin/python
```

berarti perintah atau kode masih menggunakan environment Python 3.9. Ubah path-nya agar menggunakan:

```text
.venv-llama-cpp-311/bin/python
```

Jika sebuah script memakai `sys.executable`, jalankan script dengan:

```bash
python3.11 nama_script.py
```

Bukan:

```bash
python3 nama_script.py
```

Karena pada Mac ini, `python3` masih merupakan Python 3.9.6 bawaan macOS.

## [Kesimpulan](#)

```text
Python  = program/interpreter yang menjalankan kode
Versi   = menentukan fitur dan kompatibilitas package
pip     = memasang package untuk Python tertentu
.venv   = lingkungan package yang terpisah untuk sebuah proyek
```

Untuk proyek ini:

```text
Python 3.9 lama
└── tetap dapat digunakan oleh proses training sebelumnya

Python 3.11
└── digunakan untuk environment converter llama.cpp
```

Menginstal Python baru tidak menghapus Python lama, tidak memindahkan package lama, dan tidak memperbarui virtual environment lama. Buat virtual environment baru menggunakan versi Python yang memang ingin digunakan.
