# **Laporan Proyek Machine Learning - Syahel Rusfi Razaba**
*Prediksi Jenis Tanaman Optimal Berdasarkan Karakteristik Tanah dan Iklim Menggunakan Model Klasifikasi Machine Learning*

## **Domain Proyek**
Sektor pertanian memainkan peran penting dalam mendukung perekonomian nasional di banyak negara berkembang, termasuk Indonesia dan India. Salah satu tantangan utama yang dihadapi oleh petani di kedua negara ini adalah bagaimana menentukan jenis tanaman yang paling sesuai untuk ditanam berdasarkan kondisi tanah dan iklim. Pemilihan jenis tanaman yang tidak sesuai dapat berdampak pada rendahnya produktivitas, kerugian ekonomi, serta pemborosan sumber daya lahan dan air.

Dalam proyek ini, dikembangkan model machine learning berbasis klasifikasi yang dapat memprediksi jenis tanaman optimal untuk ditanam berdasarkan karakteristik tanah (kandungan nitrogen, fosfor, kalium, pH) dan kondisi iklim (suhu, kelembapan, curah hujan). Dataset yang digunakan dalam proyek ini berasal dari India, yang memiliki karakteristik agroklimat yang serupa dengan Indonesia, terutama di wilayah-wilayah tropis.

Meskipun data tidak secara langsung berasal dari Indonesia, pendekatan dan fitur-fitur yang digunakan tetap relevan. Berdasarkan studi oleh [1], sistem pertanian di Sumatera Barat diklasifikasikan ke dalam zona agroekologi (Agro-Ecological Zone/AEZ) menggunakan variabel-variabel seperti iklim, kesuburan tanah, dan elevasi—variabel-variabel yang juga menjadi dasar dalam dataset yang digunakan dalam proyek ini.

Selain itu, studi oleh [2] menunjukkan bahwa India merupakan salah satu mitra dagang utama Indonesia di sektor pertanian, khususnya dalam produk-produk hortikultura seperti buah-buahan dan sayuran. Fakta ini mencerminkan bahwa terdapat kesamaan dalam jenis komoditas unggulan serta kesesuaian lingkungan tanam antara kedua negara. Oleh karena itu, pemanfaatan dataset pertanian dari India untuk studi prediksi tanaman di Indonesia tetap dapat dipertanggungjawabkan, khususnya dalam konteks studi awal berbasis klasifikasi.
> Referensi
- F. Farida, Ansofino, A. Rezki, and Yolamalinda, “Assessing the Climate Change Impact on Farmers Household Welfare According to West Sumatra Agro-Ecological Zone,” International Journal of Applied Business and Economic Research, vol. 15, no. 5, pp. 593–606, 2017, doi: 10.31227/osf.io/gw5x9.
- R. Kustiari and N. Hermanto, “Impacts of Indonesia-India Free Trade Agreements on Agricultural Sector of Indonesia: A CGE Analysis,” Jurnal Agro Ekonomi, vol. 35, no. 1, pp. 33–48, 2017, doi: 10.21082/jae.v35n1.2017.33-48.

## **Business Understanding**

### Problem Statement
1. Bagaimana menentukan jenis tanaman yang paling optimal untuk ditanam berdasarkan karakteristik tanah dan iklim?
2. Bagaimana membangun sistem klasifikasi tanaman yang akurat untuk membantu pengambilan keputusan dalam sektor pertanian?

### Goals
1. Membangun model klasifikasi machine learning yang dapat memprediksi jenis tanaman berdasarkan fitur-fitur input seperti kandungan Nitrogen (N), Fosfor (P), Kalium (K) dalam tanah, suhu udara, kelembapan, pH tanah, dan curah hujan.
2. Mengevaluasi performa berbagai algoritma klasifikasi untuk menemukan model terbaik yang memberikan akurasi prediksi tertinggi.
3. Menyediakan solusi berbasis data yang dapat membantu dalam membuat keputusan yang lebih tepat terkait pemilihan jenis tanaman, sehingga dapat mengoptimalkan hasil panen dan efisiensi penggunaan sumber daya.

### Solution Statement
1. Menerapkan algoritma KNN, Decision Tree, dan Random Forest untuk membandingkan performa klasifikasi.
2. Melakukan proses pembersihan data, termasuk verifikasi missing value, duplikasi, dan penanganan outlier untuk memastikan kualitas data yang baik.
3. Melakukan analisis data eksploratif (EDA) untuk memahami distribusi data, korelasi antar fitur, serta distribusi label.
4. Melakukan pra-pemrosesan data, termasuk encoding label, pembagian data (train-test split), dan normalisasi fitur numerik.
5. Mengevaluasi hasil model menggunakan metrik akurasi, precision, recall, dan F1-score untuk memastikan efektivitas solusi.
6. Menggunakan model dengan performa terbaik sebagai dasar untuk prediksi jenis tanaman yang optimal.

## **Data Understanding**
Dataset yang digunakan adalah Crop Recommendation Dataset yang diperoleh dari [Kaggle](https://www.kaggle.com/datasets/madhuraatmarambhagat/crop-recommendation-dataset).

### Informasi Dataset
- Jumlah sampel: 2200 baris
- Jumlah fitur: 7 fitur input + 1 target output
- Target: Nama tanaman (*multi-class*)

### Kondisi Dataset
- Missing value: Tidak ditemukan nilai kosong dalam dataset (`.isnull().sum() = 0`)
- Duplikat: Tidak ditemukan data duplikat (`df.duplicated().sum() = 0`)
- Outlier: Ditemukan pada beberapa fitur numerik, ditangani menggunakan transformasi Winsorization berbasis IQR
- Multikolinearitas: Tidak terdeteksi adanya korelasi yang sangat tinggi antar fitur numerik. Korelasi antar fitur relatif rendah, sehingga tidak ada indikasi multikolinearitas yang signifikan.

### Fitur
| Fitur | Deskripsi |
| ------ | ------ |
| N | Kandungan Nitrogen dalam tanah |
| P | Kandungan Fosfor dalam tanah |
| K | Kandungan Kalium dalam tanah |
| temperature | Rata-rata suhu udara (°C) |
| humidity | Rata-rata kelembapan relatif (%) |
| ph | Tingkat keasaman tanah (ph tanah) |
| rainfall | Curah hujan dalam mm |
| label | Tanaman (kelas target/output) |

### Exploratory Data Analysis
EDA bertujuan memahami pola dan distribusi dalam data, serta membantu mengenali hubungan antar fitur. Analisis dilakukan melalui:
- Statistik deskriptif: Melihat ringkasan statistik dari fitur-fitur numerik.
- Korelasi antar fitur: Menggunakan heatmap untuk memvisualisasikan hubungan antar fitur numerik.
- Distribusi kelas target: Memeriksa sebaran jumlah sampel untuk setiap jenis tanaman.

Visualisasi dilakukan menggunakan library seaborn dan matplotlib.

## **Data Preparation**
### Pemisahan Fitur dan Label
- Fitur (X): N, P, K, temperature, humidity, ph, rainfall
- Label (y): label

### Penanganan Outlier
Outlier ditangani dengan metode **Winsorization**, di mana nilai ekstrem diubah menjadi batas atas dan bawah berdasarkan IQR (Q1 - 1.5 x IQR dan Q3 + 1.5 x IQR).

### Data Splitting and Transformation
Dataset dibagi menjadi dua bagian:
- 80% Data Latih (*Training Set*): Digunakan untuk melatih model
- 20 % Data Uji (*Test Set*): Digunakan untuk mengevaluasi performa model

Transformasi dilakukan agar data dalam bentuk numerik dan memiliki skala yang seragam:

- **Label Encoding**: Mengubah label tanaman dari bentuk teks menjadi nilai numerik menggunakan LabelEncoder.
- **Normalisasi**: Menggunakan MinMaxScaler untuk mengubah skala semua fitur numerik ke rentang 0–1 agar seimbang dalam perhitungan algoritma.

Alasan:
- **Train-test split**: Digunakan untuk mengevaluasi seberapa baik model yang telah dilatih dapat menggeneralisasi pada data baru yang belum pernah dilihat sebelumnya, serta untuk mendapatkan perkiraan performa model yang tidak bias.
- **Encoding**: Diperlukan karena model machine learning umumnya memerlukan input numerik.
- **Normalisasi**: Membantu algoritma machine learning yang sensitif terhadap skala fitur (seperti KNN dan beberapa algoritma berbasis gradien) untuk bekerja lebih efektif dan konvergen lebih cepat.

## **Modeling**
### Model 1: K-Nearest Neighbors (KNN)
- Cara kerja: Mengklasifikasikan berdasarkan mayoritas label dari *K* tetangga terdekat (berdasarkan jarak Euclidean).
- Parameter: [default parameter](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier)
- Kelebihan: Sederhana, tidak perlu pelatihan awal
- Kekurangan: Sensitif terhadap skala dan outlier, lambat untuk data besar

### Model 2: Decision Tree
- Cara kerja: Decision Tree membagi data secara berulang berdasarkan fitur yang paling membedakan kelas. Pemisahan dilakukan hingga setiap cabang pohon berakhir pada sebuah keputusan akhir berupa kelas target.
- Parameter: [default parameter](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier), `random_state=42`
- Kelebihan: Mudah dipahami, tidak perlu normalisasi
- Kekurangan: Rentan overfitting, hasil sangat tergantung struktur data

### Model 3: Random Forest
- Cara kerja: Kombinasi banyak *decision tree* yang dilatih pada subset data acak, dan voting hasil klasifikasi.
- Parameter: [default parameter](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn.ensemble.RandomForestClassifier), `random_state=42`
- Kelebihan: Akurasi tinggi, robust terhadap overfitting
- Kekurangan: Kurang interpretatif, komputasi lebih tinggi

## **Evaluation**
### Metrik Evaluasi yang digunakan:
- Accuracy: proporsi prediksi benar
- Precision: fokus pada prediksi positif benar
- Recall: fokus pada seberapa banyak kasus positif terdeteksi
- F1-Score: harmonic mean dari precision dan recall

### Formula:
- Accuracy = TP + TN / (TP + FN + TN + FP)
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1 = 2 * (Precision * Recall) / (Precision + Recall)

### Hasil Evaluasi:
| | accuracy | f1_score | precision | recall |
| ------ | ------ | ------ | ------ | ------ |
| Random Forest | 0.993182 | 0.993175 | 0.993735 | 0.993182 |
| Decision Tree | 0.986364 | 0.986315 | 0.986806 | 0.986364 |
| KNN | 0.970455 | 0.970638 | 0.975117 | 0.970455 |

### Evaluation Summary
1. Apakah model menjawab Problem Statement?
Ya. Model berhasil mengklasifikasikan jenis tanaman berdasarkan karakteristik tanah dan iklim dengan akurasi sangat tinggi.
2. Apakah goals tercapai?
Ya. Model klasifikasi berhasil dibangun dan dievaluasi dengan metrik yang sesuai. **Random Forest** sebagai model terbaik mencapai akurasi 99.3%, memenuhi tujuan membangun sistem klasifikasi yang andal.
3. Apakah solusi berdampak?
Ya. Seluruh langkah yang dirumuskan dalam **solution statement** telah diimplementasikan:
   - Proses pembersihan dan EDA dilakukan secara sistematis.
   - Data dipersiapkan dengan pemisahan fitur-label, normalisasi, dan penanganan outlier.
   - Tiga model diuji, dan hasil evaluasi menunjukkan performa tinggi.
   - Model dengan performa terbaik (**Random Forest**) digunakan untuk klasifikasi final.
   
   Dampak dari solusi ini adalah terciptanya sistem prediksi jenis tanaman yang dapat digunakan untuk mendukung pengambilan keputusan pertanian berbasis data, terutama dalam konteks tanah dan iklim tropis.

## Kesimpulan
Proyek ini berhasil membangun model klasifikasi *multi-class* yang mampu memprediksi jenis tanaman paling optimal berdasarkan input karakteristik tanah dan iklim. Model terbaik yang digunakan adalah Random Forest Classifier yang mencapai akurasi 99.3%.

Model ini dapat dikembangkan lebih lanjut dengan data geografis, musim, serta informasi jenis tanah lokal agar lebih aplikatif di dunia nyata.