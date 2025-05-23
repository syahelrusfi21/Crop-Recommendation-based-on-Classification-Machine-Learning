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
Sektor pertanian seringkali dihadapkan pada tantangan dalam menentukan jenis tanaman yang paling sesuai untuk ditanam guna mencapai hasil panen yang optimal. Penentuan ini secara tradisional bergantung pada pengalaman petani atau metode konvensional lainnya yang mungkin kurang akurat dan efisien. Kualitas tanah dan kondisi iklim merupakan faktor krusial yang sangat bervariasi antar lokasi dan waktu, sehingga memerlukan pendekatan yang lebih sistematis dan berbasis data untuk pengambilan keputusan yang tepat. Kebutuhan akan penentuan jenis tanaman yang akurat berdasarkan kondisi spesifik lahan dan iklim menjadi sangat penting untuk meningkatkan produktivitas pertanian dan mengurangi risiko kerugian akibat kesalahan pemilihan jenis tanaman.

### Goals
Tujuan utama dari proyek ini adalah untuk mengembangkan sebuah model machine learning yang akurat dan efisien dalam memprediksi jenis tanaman yang paling optimal untuk ditanam di suatu lokasi berdasarkan data karakteristik tanah dan iklim. Secara spesifik, proyek ini bertujuan untuk:
1. Membangun model klasifikasi machine learning yang dapat memprediksi jenis tanaman berdasarkan fitur-fitur input seperti kandungan Nitrogen (N), Fosfor (P), Kalium (K) dalam tanah, suhu udara, kelembapan, pH tanah, dan curah hujan.
2. Mengevaluasi performa berbagai algoritma klasifikasi untuk menemukan model terbaik yang memberikan akurasi prediksi tertinggi.
3. Menyediakan solusi berbasis data yang dapat membantu dalam membuat keputusan yang lebih tepat terkait pemilihan jenis tanaman, sehingga dapat mengoptimalkan hasil panen dan efisiensi penggunaan sumber daya.

### Solution Statement
Untuk mencapai tujuan tersebut, solusi yang dikembangkan adalah membangun model klasifikasi machine learning menggunakan dataset yang berisi data karakteristik tanah dan iklim beserta jenis tanaman yang sesuai. Proses pengembangan solusi meliputi beberapa tahapan utama: pengumpulan dan pembersihan data, analisis data eksploratif (EDA) untuk memahami pola dan hubungan antar fitur, pra-pemrosesan data termasuk penanganan outlier dan normalisasi, serta pembangunan dan evaluasi beberapa model klasifikasi (K-Nearest Neighbors, Decision Tree, dan Random Forest). Model terbaik yang dihasilkan akan digunakan untuk memberikan prediksi jenis tanaman. Pendekatan ini diharapkan dapat memberikan prediksi yang akurat dan menjadi alat bantu yang efektif dalam meningkatkan produktivitas pertanian.

## **Data Understanding**
Dataset yang digunakan adalah Crop Recommendation Dataset yang diperoleh dari [Kaggle](https://www.kaggle.com/datasets/madhuraatmarambhagat/crop-recommendation-dataset).

### Informasi Dataset:
- Jumlah sampel: 2200 baris
- Jumlah fitur: 7 fitur input + 1 target output
- Target: Nama tanaman (*multi-class*)

### Fitur:
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
### Data Transformation and Splitting
Transformasi dilakukan agar data dalam bentuk numerik dan memiliki skala yang seragam:

- **Label Encoding**: Mengubah label tanaman dari bentuk teks menjadi nilai numerik menggunakan LabelEncoder.
- **Normalisasi**: Menggunakan MinMaxScaler untuk mengubah skala semua fitur numerik ke rentang 0–1 agar seimbang dalam perhitungan algoritma.

Dataset dibagi menjadi dua bagian:
- Data Latih (*Training Set*): Digunakan untuk melatih model
- Data Uji (*Test Set*): Digunakan untuk mengevaluasi performa model

Alasan:
- **Encoding**: Diperlukan karena model machine learning umumnya memerlukan input numerik.
- **Normalisasi**: Membantu algoritma machine learning yang sensitif terhadap skala fitur (seperti KNN dan beberapa algoritma berbasis gradien) untuk bekerja lebih efektif dan konvergen lebih cepat.
- **Train-test split**: Digunakan untuk mengevaluasi seberapa baik model yang telah dilatih dapat menggeneralisasi pada data baru yang belum pernah dilihat sebelumnya, serta untuk mendapatkan perkiraan performa model yang tidak bias.

## **Modeling**
Algoritma yang Dicoba:
- K-Nearest Neighbors (KNN)
- Decision Tree Classifier
- Random Forest Classifier

Model dievaluasi menggunakan akurasi, precision, recall, dan F1-score.

## **Evaluation**
Metrik Evaluasi yang digunakan:
- Accuracy: proporsi prediksi benar
- Precision: fokus pada prediksi positif benar
- Recall: fokus pada seberapa banyak kasus positif terdeteksi
- F1-Score: harmonic mean dari precision dan recall

Formula:
- Accuracy = TP + TN / (TP + FN + TN + FP)
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1 = 2 * (Precision * Recall) / (Precision + Recall)

Hasil Evaluasi:
| | accuracy | f1_score | precision | recall |
| ------ | ------ | ------ | ------ | ------ |
| Random Forest | 0.993182 | 0.993175 | 0.993735 | 0.993182 |
| Decision Tree | 0.986364 | 0.986315 | 0.986806 | 0.986364 |
| KNN | 0.970455 | 0.970638 | 0.975117 | 0.970455 |

Kesimpulan Evaluasi:
- Model Random Forest mencapai akurasi 99.3% pada data uji.
- Nilai F1-score yang tinggi menunjukkan model mampu mengklasifikasikan tanaman secara seimbang pada berbagai kelas.

## Kesimpulan
Proyek ini berhasil membangun model klasifikasi *multi-class* yang mampu memprediksi jenis tanaman paling optimal berdasarkan input karakteristik tanah dan iklim. Model terbaik yang digunakan adalah Random Forest Classifier yang mencapai akurasi 99.3%.

Model ini dapat dikembangkan lebih lanjut dengan data geografis, musim, serta informasi jenis tanah lokal agar lebih aplikatif di dunia nyata.