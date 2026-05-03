# NLP Text Preprocessing - Indonesian Abusive & Hate Speech Twitter Dataset

**Natural Language Processing (NLP)**

## Penjelasan Project

Project ini berisi implementasi tahapan **Text Preprocessing** menggunakan teknik Natural Language Processing (NLP) pada dataset teks Twitter berbahasa Indonesia. Tujuan dari project ini adalah untuk membersihkan dan menyiapkan data teks mentah agar siap digunakan untuk proses pemodelan Machine Learning lebih lanjut, khususnya untuk mendeteksi ujaran kebencian (*Hate Speech*) dan bahasa kasar (*Abusive Language*).

## Tahapan Preprocessing

Proses preprocessing yang dilakukan dalam project ini meliputi:

1. **Case Folding**: Mengubah semua huruf dalam teks menjadi huruf kecil (*lowercase*).
2. **Cleansing**: Membersihkan teks dari elemen-elemen yang tidak diperlukan atau mengganggu, seperti:
   - Menghapus URL (`http`/`www`)
   - Menghapus *mention* (`@username`)
   - Menghapus *hashtag* (`#`)
   - Menghapus *Retweet* (RT)
   - Menghapus *placeholder* `USER`
   - Menghapus angka
   - Menghapus tanda baca dan karakter spesial
   - Menghapus spasi/whitespace berlebih
3. **Tokenizing**: Memecah kalimat atau teks utuh menjadi potongan kata-kata tunggal (*token*).
4. **Filtering (Stopword Removal)**: Menghapus kata-kata umum yang tidak memiliki makna penting yang signifikan (seperti 'yang', 'dan', 'di'). Selain menggunakan stopword bawaan dari library Sastrawi, ditambahkan juga *custom stopwords* yang sering digunakan di Twitter (seperti 'yg', 'dg', 'kalo', 'wkwk', dll).
5. **Stemming**: Mengubah kata-kata yang berimbuhan menjadi bentuk kata dasarnya (misalnya: "berjalan" menjadi "jalan"). Proses ini menggunakan library **Sastrawi** yang dikhususkan untuk bahasa Indonesia.

## Library yang Diinstall

Project ini menggunakan bahasa pemrograman **Python 3**. Beberapa library utama yang dibutuhkan dan perlu diinstall meliputi:

- **pandas**: Digunakan untuk manipulasi dan analisis data (membaca, memproses, dan menyimpan file CSV).
- **re (RegEx)**: Digunakan untuk pencocokan pola *regular expression* pada tahap *cleansing* teks. (Library bawaan Python)
- **NLTK (Natural Language Toolkit)**: Digunakan untuk tahap *tokenizing* teks.
- **Sastrawi**: Library khusus untuk NLP bahasa Indonesia, digunakan pada tahap *stemming* dan untuk mendapatkan daftar *stopwords* bahasa Indonesia.

Untuk menginstall semua *dependencies*, Anda dapat menjalankan perintah berikut:
```bash
pip install pandas numpy nltk sastrawi scikit-learn
```
## 📊 Penjelasan Dataset

### Nama Dataset
**"Multi-label Hate Speech and Abusive Language Detection in Indonesian Twitter"**

### Sumber / Asal Dataset
Dataset ini diperoleh dari riset oleh **Muhammad Okky Ibrohim dan Indra Budi (2019)** yang dipublikasikan pada acara *ALW3: 3rd Workshop on Abusive Language Online*.
- Link referensi / Paper: [https://www.aclweb.org/anthology/W19-3506.pdf](https://www.aclweb.org/anthology/W19-3506.pdf)

### Detail Dataset
Dataset utama yang digunakan (`original_data.csv`) memuat sekumpulan tweet berbahasa Indonesia yang telah dianotasi untuk keperluan deteksi ujaran kebencian multi-label dan bahasa kasar. Dataset ini memiliki informasi label biner (1 = ya, 0 = tidak) sebagai berikut:
- **HS**: Label *hate speech* (ujaran kebencian) umum
- **Abusive**: Label *abusive language* (bahasa kasar) umum
- **HS_Individual**: *Hate speech* yang ditujukan kepada individu
- **HS_Group**: *Hate speech* yang ditujukan kepada suatu kelompok
- **HS_Religion**: *Hate speech* terkait agama/kepercayaan
- **HS_Race**: *Hate speech* terkait ras/etnis
- **HS_Physical**: *Hate speech* terkait fisik/disabilitas
- **HS_Gender**: *Hate speech* terkait gender/orientasi seksual
- **HS_Other**: *Hate speech* terkait hinaan/fitnah lainnya
- **HS_Weak**: *Hate speech* tingkat lemah
- **HS_Moderate**: *Hate speech* tingkat sedang
- **HS_Strong**: *Hate speech* tingkat kuat

Demi menjaga privasi pengguna sesuai dengan *Terms of Service* dari Twitter, ID tweet tidak disediakan di dataset ini. Selain itu, semua *username* dan URL asli dalam dataset ini telah disamarkan menjadi kata ganti `USER` dan `URL`. 

Dataset pendukung lainnya yang disediakan oleh peneliti di folder ini antara lain `new_kamusalay.csv` (kamus pemetaan kata gaul/typo menjadi kata baku) dan `abusive.csv` (daftar leksikon kata-kata kasar).

---

## 🚀 Cara Menjalankan

1. Pastikan Anda berada di dalam direktori project `preprocessing`.
2. Jika ada, jalankan perintah instalasi dari requirements:
   Atau install manual seperti penjelasan di atas:
   ```bash
   pip install pandas nltk Sastrawi
   ```
3. Running script:
   ```bash
   python main.py
   ```
4. Tunggu hingga proses *stemming* selesai. Hasil akhir teks yang telah diproses akan disimpan dalam file `datasets/clean_data.csv`.

## 📁 Struktur Project

```text
preprocessing/
├── datasets/
│   ├── original_data.csv   # Dataset mentah Twitter
│   ├── clean_data.csv      # Hasil akhir preprocessing (dihasilkan otomatis setelah script jalan)
│   ├── abusive.csv         # Leksikon kata kasar
│   ├── new_kamusalay.csv   # Kamus slang/typo Twitter
│   ├── README.md           # Deskripsi asli dataset dari author
│   └── citation.bib        # Sitasi untuk penggunaan dataset
├── main.py                 # File script utama untuk tahapan preprocessing
├── requirements.txt        # Daftar dependencies
└── README.md               # Dokumentasi project ini
```
