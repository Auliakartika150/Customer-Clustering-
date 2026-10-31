# Customer Segmentation Dashboard
Dashboard interaktifuntuk melakukan segmentasi pelanggan menggunakan algoritma Machine Learning K-Means Clustering. Aplikasi ini dirancang untuk mengelompokkan pelanggan berdasarkan karakteristik demografis dan sosial ekonomi, meliputi usia, pendapatan, pendidikan, pekerjaan, serta ukuran wilayah tempat tinggal.

## Tujuan
1. Mengelompokkan pelanggan berdasarkan kemiripan karakteristik menggunakan algoritma K-Means Clustering.
2. Mengidentifikasi dan memahami karakteristik setiap kelompok pelanggan berdasarkan lima variabel utama.
3. Menampilkan distribusi, persebaran, dan profil pelanggan melalui visualisasi data interaktif.
4. Mengevaluasi kualitas hasil clustering menggunakan Silhouette Score dan Davies–Bouldin Index.
5. Membantu pengguna memperoleh informasi segmentasi pelanggan sebagai dasar pendukung pengambilan keputusan bisnis.

## Fitur
- **Segmentation Overview:** Menampilkan ringkasan jumlah pelanggan, jumlah segmen, Silhouette Score, dan Davies–Bouldin Index.
- **Customer Segmentation:** Mengelompokkan pelanggan menggunakan algoritma K-Means berdasarkan lima karakteristik, yaitu Age, Income, Education, Occupation, dan Settlement Size.
- **Cluster Visualization:** Menampilkan distribusi dan persebaran pelanggan berdasarkan segmen melalui grafik interaktif.
- **Customer Profiling:** Menyajikan perbandingan rata-rata karakteristik pelanggan pada setiap segmen.
- **Cluster Configuration:** Memungkinkan pengguna mengatur jumlah kelompok pelanggan dan random state untuk menjalankan model clustering.
- **Customer Data Exploration:** Menampilkan data pelanggan berdasarkan segmen yang dipilih.
- **Data Export:** Menyediakan fitur unduh hasil segmentasi dalam format CSV.

## Metodologi
Proses segmentasi pelanggan dilakukan melalui beberapa tahapan:
1. Memuat dan memeriksa dataset pelanggan.
2. Melakukan preprocessing data, termasuk konversi tipe data numerik dan penanganan nilai kosong.
3. Melakukan standardisasi fitur menggunakan StandardScaler.
4. Menerapkan algoritma K-Means Clustering untuk membentuk kelompok pelanggan.
5. Mengevaluasi kualitas clustering menggunakan Silhouette Score dan Davies–Bouldin Index.
6. Menganalisis profil dan distribusi karakteristik setiap segmen.
7. Menampilkan hasil analisis melalui dashboard interaktif.

## Teknologi yang Digunakan

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Plotly

## Dataset

Dataset yang digunakan adalah `segmentation data.csv`, dengan lima variabel utama:
- **Age:** Usia pelanggan.
- **Income:** Pendapatan pelanggan.
- **Education:** Tingkat pendidikan pelanggan.
- **Occupation:** Kategori pekerjaan pelanggan.
- **Settlement Size:** Ukuran wilayah tempat tinggal pelanggan.

## Deployment

Customer Segmentation Dashboard: https://dztnvbb9jlfzqort7xjrjt.streamlit.app/
