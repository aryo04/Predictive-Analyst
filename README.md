# Laporan Proyek Machine Learning - Aryo Daffa Khairuddin

## Domain Proyek

Stroke merupakan penyebab utama disabilitas dan penyebab kematian kedua secara global. Menurut World Health Organization (WHO), risiko seumur hidup seseorang untuk mengalami stroke telah meningkat sebesar 50% dalam 17 tahun terakhir, dengan estimasi bahwa 1 dari 4 orang akan mengalami stroke dalam hidupnya. Dari tahun 1990 hingga 2019, terjadi peningkatan 70% dalam kejadian stroke, 43% peningkatan kematian akibat stroke, 102% peningkatan prevalensi stroke, dan 143% peningkatan dalam Disability Adjusted Life Years (DALY). Setiap tahunnya, sekitar 15 juta orang di seluruh dunia mengalami stroke. Dari jumlah tersebut, 5 juta meninggal dan 5 juta lainnya mengalami disabilitas permanen, memberikan beban besar bagi keluarga dan komunitas.

Pemanfaatan teknologi, khususnya Machine Learning dapat membantu untuk melakukan prediksi risiko stroke sejak dini berdasarkan data riwayat kesehatan, gaya hidup, dan faktor demografis. Penelitian oleh Aulia et al. (2024) membandingkan algoritma Decision Tree, Naïve Bayes, serta Random Forest dan hasilnya menunjukkan bahwa Decision Tree memiliki akurasi tertinggi sebesar 95,13%. Berdasarkan temuan tersebut, penerapan Machine Learning terbukti efektif dalam mendeteksi risiko stroke secara dini dan berpotensi membantu pengambilan keputusan klinis yang lebih cepat dan tepat.

Refrensi :
- [World Stroke Day 2022](https://www.who.int/srilanka/news/detail/29-10-2022-world-stroke-day-2022).
- [Stroke, Cerebrovascular accident](https://www.emro.who.int/health-topics/stroke-cerebrovascular-accident/index.html).
- Aulia, Y., Andriyansyah, A., Suharjito, S., & Nensi, S. W. (2024). Analisis Prediksi Stroke dengan Membandingkan Tiga Metode Klasifikasi Decision Tree, Naïve Bayes, dan Random Forest. Jurnal Ilmu Komputer dan Informatika, 3(2), 89–98.

## Business Understanding

### Problem Statements
* Bagaimana membangun model yang mampu mengidentifikasi risiko stroke secara otomatis berdasarkan data kesehatan pasien?

* Algoritma Machine Learning apa yang memberikan performa terbaik dalam memprediksi risiko stroke berdasarkan data tersebut?

### Goals
* Mengembangkan model prediksi risiko stroke menggunakan Machine Learning yang dapat mengolah data kesehatan pasien secara otomatis.

* Menentukan algoritma Machine Learning yang paling optimal dalam memprediksi risiko stroke melalui analisis performa dari berbagai model.

### Solution Statements
* Mengimplementasikan beberapa algoritma Machine Learning seperti Random Forest, Naïve Bayes, dan XGBoost dalam membangun model prediksi risiko stroke.

* Menggunakan teknik SMOTE (Synthetic Minority Oversampling Technique) untuk mengatasi ketidakseimbangan kelas pada dataset dan meningkatkan kemampuan deteksi kasus stroke.

* Mengevaluasi performa masing-masing model menggunakan metrik akurasi, precision, recall, dan F1-score untuk menentukan algoritma terbaik.

## Data Understanding
Dataset yang digunakan diperoleh dari platform Kaggle dengan judul '[Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)' yang dibuat oleh pengguna bernama fedesoriano. Dataset ini memuat informasi medis dan karakteristik individu pasien yang terdiri dari 5110 baris dan 12 kolom.

### Variabel-variabel pada Stroke Prediction Dataset adalah sebagai berikut:
1. id: Nomor unik (numerik).
2. gender: Jenis kelamin pasien (kategorikal: Male, Female).
3. age: Usia pasien (numerik, dalam tahun).
4. hypertension: Status hipertensi pasien (numerik: 0 = tidak ada, 1 = ada).
5. heart_disease: Status penyakit jantung pasien (numerik: 0 = tidak ada, 1 = ada).
6. ever_married: Status pernikahan pasien (kategorikal: Yes, No).
7. work_type: Jenis pekerjaan pasien (kategorikal: Private, Self-employed, Govt_job, children, Never_worked).
8. Residence_type: Tipe tempat tinggal pasien (kategorikal: Urban, Rural).
9. avg_glucose_level: Rata-rata kadar glukosa darah pasien (numerik).
10. bmi: Indeks Massa Tubuh pasien (numerik).
11. smoking_status: Status merokok pasien (kategorikal: never smoked, formerly smoked, smokes, Unknown).
12. stroke: Variabel target, menunjukkan apakah pasien mengalami stroke (numerik: 0 = tidak, 1 = ya).

### Kondisi Awal Dataset

1. ```
   print(f"Jumlah baris: {df.shape[0]}")
   print(f"Jumlah kolom: {df.shape[1]}")
   ```
   ```
   Jumlah baris: 5110
   Jumlah kolom: 12
   ```
   * Terdiri dari 5110 baris dan 12 kolom.
  
2. ```
   df.info()
   ```
   ```
   <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 5110 entries, 0 to 5109
    Data columns (total 12 columns):
     #   Column             Non-Null Count  Dtype  
    ---  ------             --------------  -----  
     0   id                 5110 non-null   int64  
     1   gender             5110 non-null   object 
     2   age                5110 non-null   float64
     3   hypertension       5110 non-null   int64  
     4   heart_disease      5110 non-null   int64  
     5   ever_married       5110 non-null   object 
     6   work_type          5110 non-null   object 
     7   Residence_type     5110 non-null   object 
     8   avg_glucose_level  5110 non-null   float64
     9   bmi                4909 non-null   float64
     10  smoking_status     5110 non-null   object 
     11  stroke             5110 non-null   int64  
    dtypes: float64(3), int64(4), object(5)
    memory usage: 479.2+ KB
   ```
   * Mayoritas tipe data dalam dataset yaitu numerik, dengan beberapa kolom kategorikal.
3. ```
   df.describe()
   ```
   ```
       id	age	hypertension	heart_disease	avg_glucose_level	bmi	stroke
    count	5110.000000	5110.000000	5110.000000	5110.000000	5110.000000	4909.000000	5110.000000
    mean	36517.829354	43.226614	0.097456	0.054012	106.147677	28.893237	0.048728
    std	21161.721625	22.612647	0.296607	0.226063	45.283560	7.854067	0.215320
    min	67.000000	0.080000	0.000000	0.000000	55.120000	10.300000	0.000000
    25%	17741.250000	25.000000	0.000000	0.000000	77.245000	23.500000	0.000000
    50%	36932.000000	45.000000	0.000000	0.000000	91.885000	28.100000	0.000000
    75%	54682.000000	61.000000	0.000000	0.000000	114.090000	33.100000	0.000000
    max	72940.000000	82.000000	1.000000	1.000000	271.740000	97.600000	1.000000
   ```
   * Terdapat outlier pada kolom bmi dan avg_glucose_lvl berdasarkan hasil statistik deskriptif
4. ```
   duplicate_count = df.duplicated().sum()
   print(f"Jumlah data duplikat : {duplicate_count}")
   ```
   ```
   Jumlah data duplikat : 0
   ```
   * Tidak terdapat duplikat data
5. ```
   df.isnull().sum()
   ```
   ```
    id                   0
    gender               0
    age                  0
    hypertension         0
    heart_disease        0
    ever_married         0
    work_type            0
    Residence_type       0
    avg_glucose_level    0
    bmi                201
    smoking_status       0
    stroke               0
    dtype: int64
   ```
   * Terdapat missing value pada kolom bmi sebanyak 201.

### Exploratory Data Analysis
1. Distribusi Kelas

   ![image](https://github.com/user-attachments/assets/d79d2a42-c3c5-4275-b828-cdc3a6375d34)

   **Insight**: Distribusi kelas tidak seimbang, dimana jumlah pasien dengan stroke (1) jauh lebih sedikit dibandingkan yang tidak mengalami stroke (0), sehingga     diperlukan teknik penanganan ketidakseimbangan data, seperti SMOTE untuk menyeimbangkan distrbusinya.

2. Distibusi Variabel Numerik

    ![image](https://github.com/user-attachments/assets/1f5a634c-1ea2-484d-9285-f2504b090835)

   **Insight**:
   * `age`: Distribusi usia cenderung merata, dengan sedikit puncak pada usia 50–60 tahun.

   * `hypertension` & `heart_disease`: Mayoritas data bernilai 0, menunjukkan sebagian besar individu tidak memiliki hipertensi atau penyakit jantung.

   * `avg_glucose_level`: Distribusi condong ke kanan (right-skewed), menunjukkan ada beberapa individu dengan kadar glukosa sangat tinggi.

   * `bmi`: Sebagian besar nilai berada di kisaran 20–40, tetapi ada outlier dengan nilai ekstrem mendekati 100

4. Korelasi Antar Variabel Numerik

   ![image](https://github.com/user-attachments/assets/ba3de57e-fb34-4140-a736-848019a45815)

   **Insight** :
    * `age` memiliki korelasi sedang dengan `bmi` (0.33), menunjukkan sedikit hubungan antara kedua fitur tersebut. `Age` juga berkorelasi positif dengan                 `hypertension` (0.28), `heart_disease` (0.26), `avg_glucose_level` (0.24), dan `stroke` (0.25), yang berarti makin bertambah usia cenderung meningkatkan           risiko terhadap variabel-variabel tersebut.

    * `hypertension`, `heart_disease`, dan `avg_glucose_level` memiliki korelasi rendah satu sama lain, tetapi semua memiliki korelasi positif kecil dengan              `stroke` (~0.13), menunjukkan kontribusi lemah terhadap risiko stroke.

    * `bmi` berkorelasi sedang dengan age (0.33), namun sangat lemah dengan variabel lain, termasuk stroke (0.04), sehingga pengaruhnya sangat kecil.

6. Distrbusi Variabel Kategorikal

   ![image](https://github.com/user-attachments/assets/debb0060-1eb7-4fe4-8a8a-1be903df417c)

   **Insight** :
    * `gender`: Jenis kelamin perempuan lebih banyak dibandingkan laki-laki.

    * `ever_married`: Sebagian besar individu sudah pernah menikah,yang menunjukkan usia individu cenderung dewasa.

    * `work_type`: Mayoritas individu tidak menyebutkan pekerjaannya (Private), kemudian Self-employed, childern, Govt_job, dan yang paling sedikit adalah yang          belum pernah bekerja.

    * `Residence_type`: Individu dari daerah Urban sedikit lebih banyak daripada Rural.

    * `smoking_status`: Mayoritas individu tidak pernah merokok, diikuti oleh status Unknown, lalu formerly smoked, dan paling sedikit smokes.
7. Korelasi Antar Fitur Kategorikal

   ![image](https://github.com/user-attachments/assets/911a2577-a999-4efc-af40-f9dfc6223401)

   **Insight** :
    * `gender` tidak memiliki korelasi yang signifikan dengan fitur lain. Nilai korelasinya mendekati nol terhadap semua fitur.

    * `ever_married` memiliki korelasi sedang dengan `work_type` (0.38) dan `smoking_status` (0.30), menunjukkan bahwa status pernikahan sedikit berhubungan 
      dengan jenis pekerjaan dan status merokok.

    * `work_type` juga menunjukkan korelasi sedang dengan `smoking_status` (0.31),yang mengindikasikan bahwa jenis pekerjaan sedikit berhubungan dengan status            merokok.

    * `Residence_type` tidak berkorelasi dengan fitur manapun, yang berarti tipe tempat tinggal tidak berpengaruh terhadap fitur kategorikal lainnya.

    * `smoking_status` memiliki korelasi sedang dengan `ever_married`(0.30) dan `work_type`(0.31), namun korelasinya dengan fitur lain sangat rendah.
 
9. Pengecekan Outlier

   ![image](https://github.com/user-attachments/assets/092e5ce4-3288-4f16-865b-b71076b54287)

   **Insight** :
   variabel `avg_glucose_level` dan `bmi` mengandung banyak outlier. `avg_glucose_level` memiliki data ekstrem di atas 250, sementara `bmi` mencapai hampir 100.      Keberadaan outlier ini menyebabkan distribusi data menjadi tidak seimbang

## Data Preparation
Proses ini dilakukan untuk memastikan dataset siap digunakan dalam pemodelan prediksi stroke. Tahapan ini dilakukan sebagai berikut:
1. Drop Kolom id

   ![image](https://github.com/user-attachments/assets/0955d020-3992-48a5-972a-0c645aca02f9)

   Kolom id dihapus dari dataset karena hanya nomor unik  yang tidak memiliki nilai prediktif terhadap risiko stroke. 
   
2. Menghapus Missing Value

   ![image](https://github.com/user-attachments/assets/31a0ab10-6e08-47ef-8295-30bf80bbee14)

   Missing value pada kolom bmi dihapus karena persentase hanya sedikit  missing value-nya dan menghapusnya tidak akan signifikan mengurangi jumlah data.
   
3. Menangani Outlier

   ![image](https://github.com/user-attachments/assets/08cecc68-2e84-49eb-9e26-d0080cd3add2)

   Setelah menangani outlier menggunakan metode Z-score dengan threshold 3, terlihat bahwa nilai-nilai ekstrem pada fitur avg_glucose_level dan bmi masih tetap 
   muncul dalam visualisasi boxplot, namun jumlahnya telah berkurang. Hal ini menunjukkan bahwa penerapan Z-score dengan ambang ±3 berhasil mengidentifikasi dan 
   menghapus sebagian besar data yang menyimpang secara signifikan dari distribusi normalnya.
   
4. Encoding 

   ![image](https://github.com/user-attachments/assets/a3f17d5d-707e-4d54-a2c0-aa88e83d5520)

   Proses label encoding dilakukan untuk mengubah nilai kategorikal menjadi format numerik agar bisa diproses oleh algoritma machine learning. Setiap nilai 
   pada kolom gender, ever_married, work_type, Residence_type, dan smoking_status dikonversi menjadi angka.
   
5. Normalisasi

   ![image](https://github.com/user-attachments/assets/f68cbe82-cffe-4d01-a1e1-fa60de53790b)

   Fitur numerik (age, avg_glucose_level, bmi) dinormalisasi menggunakan MinMaxScaler untuk mengubah nilai ke rentang 0-1. Normalisasi ini penting supaya rentang 
   nilainya tidak berjauhan.

6. Penyeimbangan Data dan Split Data
   ```
   X = df_normalized.drop('stroke', axis=1)
   y = df_normalized['stroke']
    
   smote = SMOTE(random_state=42)
   X_resampled, y_resampled = smote.fit_resample(X, y)
   X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)
   ```
   Data diseimbangkan menggunakan teknik SMOTE (Synthetic Minority Oversampling Technique) untuk mengatasi ketidakseimbangan kelas antara pasien dengan stroke 
   (minoritas) dan tanpa stroke (mayoritas). Setelah itu, dataset dibagi menjadi data latih (70%) dan data uji (30%). Penyeimbangan diperlukan untuk meningkatkan 
   sensitivitas model terhadap kasus stroke.

## Modeling
Pada tahap ini, dilakukan proses pemodelan untuk memprediksi risiko stroke menggunakan tiga algoritma machine learning, yaitu Random Forest, Naïve Bayes, dan XGBoost.

1. Random Forest

![image](https://github.com/user-attachments/assets/3f045d36-aafa-434a-92a9-23eea5b5909d)

   * Pengertian: Random Forest adalah algoritma ensemble yang menggabungkan banyak pohon keputusan (decision tree) untuk meningkatkan akurasi dan mengurangi 
                 risiko overfitting.
    
   * Cara Kerja: Membangun beberapa pohon keputusan dari subset data dan subset fitur yang dipilih secara acak (metode bagging), kemudian menggabungkan hasil 
                 prediksi dari semua pohon dengan menggunakan voting mayoritas (klasifikasi).
    
   * Parameter yang digunakan: RandomForestClassifier(random_state=42, n_estimators=100, max_depth=None)
    
      * random_state=42: Menjamin hasil yang konsisten melalui seed pengacakan.
    
      * n_estimators=100: Jumlah pohon dalam hutan.
    
      * max_depth=None: Kedalaman maksimum pohon tidak dibatasi.
    
   * Kelebihan: Tahan terhadap overfitting, cocok untuk data dengan fitur campuran (numerik dan kategorikal), mampu memberikan informasi pentingnya fitur (feature 
                importance).
    
   * Kekurangan: Proses pelatihan bisa lambat pada dataset besar, kurang efisien tanpa seleksi fitur awal.
    
   * Hasil: Akurasi 0.937 (93,7%).

2. Naïve Bayes

   ![image](https://github.com/user-attachments/assets/7c7e2361-106b-4458-ab5c-297e0049e18a)

   * Pengertian: Naïve Bayes adalah algoritma berbasis probabilitas yang menerapkan teorema Bayes dengan asumsi bahwa fitur-fitur saling independen.
    
   * Cara Kerja: Menghitung probabilitas setiap kelas berdasarkan distribusi fitur (menggunakan distribusi Gaussian untuk fitur numerik), lalu memilih kelas 
                 dengan probabilitas tertinggi.
    
    * Parameter yang digunakan: GaussianNB(var_smoothing=1e-9)
    
      * var_smoothing=1e-9: Menambahkan nilai kecil pada varians untuk menghindari pembagian dengan nol, meningkatkan stabilitas numerik.
    
    * Kelebihan: Cepat, efisien, dan bekerja baik pada dataset kecil atau sederhana.
    
    * Kekurangan: Asumsi independensi fitur sering tidak terpenuhi dalam data medis, sehingga akurasi bisa menurun.
    
    * Hasil: Akurasi 0.708 (70,8%).
  
3. XGBoost

   ![image](https://github.com/user-attachments/assets/187c01bb-d223-47f8-8382-663822bc195d)

   * Pengertian: XGBoost (Extreme Gradient Boosting) adalah algoritma gradient boosting yang membangun pohon keputusan secara bertahap untuk memperbaiki kesalahan 
                 dari prediksi sebelumnya.

   * Cara Kerja: Model membangun pohon secara iteratif, fokus pada kesalahan (residual) dari model sebelumnya, dan mengatur bobot untuk menangani 
                 ketidakseimbangan kelas.

   * Parameter yang digunakan: XGBClassifier(objective='binary:logistic', random_state=42)

      * objective='binary:logistic': Fungsi objektif untuk klasifikasi biner, menghasilkan probabilitas sebagai output.
 
      * random_state=42: Menjamin konsistensi hasil dengan seed acak.
        
      * Kelebihan: Akurasi tinggi, efektif untuk data tidak seimbang, mendukung banyak teknik optimasi dan tuning parameter.

      * Kekurangan: Kompleks dan membutuhkan penyesuaian parameter yang hati-hati agar tidak overfitting.

      * Hasil: Akurasi 0.951 (95,1%).
    
Berdasarkan hasil akurasi dari ketiga algoritma, XGBoost dipilih sebagai model terbaik karena memperoleh akurasi tertinggi sebesar 95,13%, mengungguli Random Forest (93,76%) dan Naïve Bayes (70,08%). Selain itu, XGBoost memiliki kemampuan yang baik dalam menangani data tidak seimbang dan mendukung pengaturan parameter secara fleksibel, sehingga berpotensi memberikan hasil yang lebih optimal dalam prediksi risiko stroke.

## Evaluation
Dalam mengevaluasi performa model klasifikasi untuk prediksi risiko stroke, digunakan metrik evaluasi yang khusus untuk kasus klasifikasi, yaitu Accuracy, Precision, Recall, dan F1-score. Sebelum membahas metrik-metrik tersebut, memahami terlebih dahulu konsep dasar dalam evaluasi model klasifikasi, yaitu:

* True Positive (TP): Kasus di mana model memprediksi positif (stroke) dan kenyataannya memang positif.

* True Negative (TN): Kasus di mana model memprediksi negatif (tidak stroke) dan kenyataannya memang negatif.

* False Positive (FP): Kasus di mana model memprediksi positif (stroke) padahal kenyataannya negatif.

* False Negative (FN): Kasus di mana model memprediksi negatif (tidak stroke) padahal kenyataannya positif.

Nilai-nilai ini didapat dari confusion matrix, dan digunakan untuk menghitung berbagai metrik evaluasi sebagai berikut:

* Accuracy: Mengukur proporsi prediksi yang benar dari keseluruhan prediksi.
  
  Rumus:
  
  `Accuracy = (TP + TN) / (TP + TN + FP + FN)`

* Precision: Mengukur proporsi prediksi positif yang benar-benar positif.
  
  Rumus:
  
  `Precision = TP / (TP + FP)`

* Recall: Mengukur proporsi kasus positif yang berhasil dikenali model.
  
  Rumus:
  
  `Recall = TP / (TP + FN)`

* F1-Score: Rata-rata harmonik dari Precision dan Recall, memberikan keseimbangan antara keduanya.
  
  Rumus:
  
  `F1-Score = 2 × (Precision × Recall) / (Precision + Recall)`

### Hasil Evaluasi Model
1. Random Forest
   * Akurasi: 93.7%

   * Precision: 0.96 (class 0), 0.92 (class 1)

   * Recall: 0.91 (class 0), 0.96 (class 1)

   * F1-score: 0.94 (class 1 dan 2)

   * Confusion Matrix:
     ```
     [[1276  119]
     [  53 1309]]
     ```
   * Model ini cukup seimbang dalam mendeteksi kedua kelas. Tingkat kesalahan rendah dan kinerjanya stabil baik untuk kasus stroke maupun bukan.
     
2. Naïve Bayes
   * Akurasi: 70.8%

   * Precision: 0.92 (class 0), 0.64 (class 1)

   * Recall: 0.46 (class 0), 0.96 (class 1)

   * F1-score: 0.62 (class 0), 0.76 (class 1)

   * Confusion Matrix:
     ```
     [[ 646  749]
     [  55 1307]]
     ```
    * Naïve Bayes memiliki recall tinggi pada kelas stroke (class 1), tetapi sangat lemah pada kelas non-stroke (class 0). Hal ini menandakan model cenderung 
      memberikan banyak false positive.

3. XGBoost
   * Akurasi: 95.1%

   * Precision: 0.97 (class 0), 0.94 (class 1)

   * Recall: 0.94 (class 0), 0.97 (class 1)

   * F1-score: 0.95 (class 1 dan 2)

   * Confusion Matrix:
     ```
     [[1308   87]
     [  47 1315]]
     ```
   * XGBoost menunjukkan performa terbaik di semua metrik. Nilai precision dan recall tinggi serta seimbang, menjadikannya model yang sangat andal dalam 
     mendeteksi pasien stroke dan non-stroke secara akurat.

### Kesimpulan Evaluasi
* XGBoost menjadi model terbaik dengan akurasi tertinggi (95.1%) serta precision dan recall yang sangat seimbang.

* Random Forest memberikan performa baik dan cukup kompetitif, meskipun sedikit di bawah XGBoost.

* Naïve Bayes memiliki performa terendah karena ketidakseimbangan prediksi, terutama pada kelas negatif (non-stroke).      
