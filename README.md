# Prediksi Jenis Tanaman Berdasarkan Tanah & Iklim

Proyek machine learning untuk memprediksi jenis tanaman yang optimal berdasarkan karakteristik tanah dan iklim, seperti kandungan nitrogen, fosfor, kalium, suhu, pH tanah, dan curah hujan. Dataset digunakan dari India dan relevan dengan konteks agroekologis Indonesia.

---

## Tujuan Proyek
- Membangun model klasifikasi untuk merekomendasikan jenis tanaman berdasarkan data lingkungan.
- Membandingkan beberapa algoritma machine learning untuk menemukan model terbaik.



## Algoritma yang Digunakan
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest (model terbaik, akurasi 99.3%)



## Dataset
- Sumber: [Kaggle](https://www.kaggle.com/datasets/madhuraatmarambhagat/crop-recommendation-dataset)
- 2200 sampel
- 7 fitur numerik + 1 label target

### Fitur Input
- `N`, `P`, `K`: Unsur hara tanah
- `temperature`: Suhu (°C)
- `humidity`: Kelembapan (%)
- `ph`: Keasaman tanah
- `rainfall`: Curah hujan (mm)



## Proses yang Dilakukan
1. Analisis data eksploratif (EDA)
2. Penanganan outlier (Winsorization)
3. Encoding label dan normalisasi fitur
4. Pembagian data (80% train, 20% test)
5. Pelatihan dan evaluasi model klasifikasi
6. Visualisasi performa model dan confusion matrix



## Hasil Evaluasi

| Model          | Akurasi | F1-score |
|----------------|---------|----------|
| Random Forest  | 99.3%   | 99.3%    |
| Decision Tree  | 98.6%   | 98.6%    |
| KNN            | 97.0%   | 97.1%    |



## Lisensi
Proyek ini untuk tujuan edukasi dan latihan. Dataset milik Kaggle, hak cipta sepenuhnya pada pemilik aslinya.
