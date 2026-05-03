"""
NLP Text Preprocessing - Indonesian Abusive & Hate Speech Twitter Dataset
Menggunakan Sastrawi Stemmer untuk Bahasa Indonesia

Tahapan Preprocessing:
1. Case Folding
2. Cleansing (hapus URL, mention, hashtag, angka, tanda baca)
3. Tokenizing
4. Filtering (Stopword Removal)
5. Stemming (Sastrawi)
"""

import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Download NLTK data (hanya perlu sekali)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# =============================
# 1. Load Dataset
# =============================
print("=" * 60)
print("LOADING DATASET...")
print("=" * 60)

df = pd.read_csv("datasets/original_data.csv", encoding="latin-1")
print(f"Jumlah data: {len(df)} baris")
print(f"Kolom: {list(df.columns)}")
print(f"\nContoh data awal (5 baris pertama):")
print(df[['Tweet']].head())

# =============================
# 2. Case Folding
# =============================
print("\n" + "=" * 60)
print("TAHAP 1: CASE FOLDING")
print("=" * 60)

def case_folding(text):
    """Mengubah semua teks menjadi huruf kecil"""
    return str(text).lower()

df['case_folding'] = df['Tweet'].apply(case_folding)
print(df[['Tweet', 'case_folding']].head())

# =============================
# 3. Cleansing
# =============================
print("\n" + "=" * 60)
print("TAHAP 2: CLEANSING")
print("=" * 60)

def cleansing(text):
    """Membersihkan teks dari karakter yang tidak diperlukan"""
    # Hapus URL
    text = re.sub(r'http\S+|www\.\S+', '', text)
    # Hapus mention (@username)
    text = re.sub(r'@\w+', '', text)
    # Hapus hashtag
    text = re.sub(r'#\w+', '', text)
    # Hapus RT (retweet)
    text = re.sub(r'\brt\b', '', text)
    # Hapus 'USER' placeholder
    text = re.sub(r'\buser\b', '', text, flags=re.IGNORECASE)
    # Hapus angka
    text = re.sub(r'\d+', '', text)
    # Hapus tanda baca dan karakter spesial
    text = re.sub(r'[^\w\s]', '', text)
    # Hapus whitespace berlebih
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['cleansing'] = df['case_folding'].apply(cleansing)
print(df[['case_folding', 'cleansing']].head())

# =============================
# 4. Tokenizing
# =============================
print("\n" + "=" * 60)
print("TAHAP 3: TOKENIZING")
print("=" * 60)

def tokenizing(text):
    """Memecah teks menjadi token/kata"""
    return word_tokenize(text)

df['tokenizing'] = df['cleansing'].apply(tokenizing)
print(df[['cleansing', 'tokenizing']].head())

# =============================
# 5. Filtering (Stopword Removal)
# =============================
print("\n" + "=" * 60)
print("TAHAP 4: FILTERING (STOPWORD REMOVAL)")
print("=" * 60)

# Menggunakan stopword Sastrawi (Bahasa Indonesia)
stop_factory = StopWordRemoverFactory()
stopwords_id = set(stop_factory.get_stop_words())

# Tambahkan custom stopwords (kata-kata umum di Twitter)
custom_stopwords = {'yg', 'dg', 'dgn', 'ny', 'ga', 'gak', 'gk', 'udh',
                    'udah', 'sdh', 'lg', 'jg', 'jgn', 'tdk', 'blm',
                    'kalo', 'kl', 'org', 'org2', 'sm', 'bgt', 'aja',
                    'sih', 'deh', 'dong', 'kok', 'lah', 'kan', 'tuh',
                    'nih', 'wkwk', 'wkwkwk', 'haha', 'lol', 'amp'}
stopwords_id.update(custom_stopwords)

def filtering(tokens):
    """Menghapus stopwords dari list token"""
    return [word for word in tokens if word not in stopwords_id and len(word) > 1]

df['filtering'] = df['tokenizing'].apply(filtering)
print(df[['tokenizing', 'filtering']].head())

# =============================
# 6. Stemming (Sastrawi)
# =============================
print("\n" + "=" * 60)
print("TAHAP 5: STEMMING (SASTRAWI)")
print("=" * 60)

# Inisialisasi Sastrawi Stemmer
stem_factory = StemmerFactory()
stemmer = stem_factory.create_stemmer()

def stemming(tokens):
    """Mengubah kata ke bentuk dasar menggunakan Sastrawi"""
    return [stemmer.stem(word) for word in tokens]

# Stemming bisa memakan waktu, tampilkan progress
print("Proses stemming sedang berjalan (mungkin memakan waktu beberapa menit)...")
df['stemming'] = df['filtering'].apply(stemming)
print("Stemming selesai!")
print(df[['filtering', 'stemming']].head())

# =============================
# 7. Gabungkan hasil akhir
# =============================
df['hasil_preprocessing'] = df['stemming'].apply(lambda x: ' '.join(x))

# =============================
# 8. Simpan Hasil
# =============================
print("\n" + "=" * 60)
print("MENYIMPAN HASIL...")
print("=" * 60)

# Simpan hasil preprocessing
output_df = df[['Tweet', 'hasil_preprocessing', 'HS', 'Abusive',
                'HS_Individual', 'HS_Group', 'HS_Religion', 'HS_Race',
                'HS_Physical', 'HS_Gender', 'HS_Other',
                'HS_Weak', 'HS_Moderate', 'HS_Strong']]

output_df.to_csv("datasets/clean_data.csv", index=False)
print("Hasil disimpan ke: datasets/hasil_preprocessing.csv")

# =============================
# 9. Ringkasan
# =============================
print("\n" + "=" * 60)
print("RINGKASAN PREPROCESSING")
print("=" * 60)
print(f"Total data diproses : {len(df)} baris")
print(f"Contoh hasil akhir  :")
for i in range(min(3, len(df))):
    print(f"\n--- Data ke-{i+1} ---")
    print(f"  Original : {df['Tweet'].iloc[i][:80]}...")
    print(f"  Hasil    : {df['hasil_preprocessing'].iloc[i][:80]}...")

print("\n✅ Preprocessing selesai!")
