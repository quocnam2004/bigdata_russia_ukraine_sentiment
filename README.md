# 🇺🇦 Big Data Sentiment Analysis – Russia vs Ukraine Twitter

> **Phân tích cảm xúc và xu hướng dư luận về xung đột Nga-Ukraine sử dụng Apache Spark & Machine Learning**

[![PySpark](https://img.shields.io/badge/PySpark-3.1.1-orange.svg)](https://spark.apache.org/)
[![Hadoop](https://img.shields.io/badge/Hadoop-3.3.6-blue.svg)](https://hadoop.apache.org/)
[![Python](https://img.shields.io/badge/Python-3.13-green.svg)](https://www.python.org/)

---

## 📥 Tải Dataset (BẮT BUỘC)

> ⚠️ **Dataset KHÔNG có sẵn trong source code!** Bạn cần tải từ Kaggle.

### 1. Download từ Kaggle

📦 **Link:** [Ukraine Conflict Twitter Dataset](https://www.kaggle.com/datasets/bwandowando/ukraine-russian-crisis-twitter-dataset-1-2-m-rows)

- **Size:** ~10GB (giải nén ra ~44GB)
- **Format:** CSV (292 files)
- **Số lượng:** 57.8 triệu tweets

### 2. Upload lên HDFS
   ```powershell
```powershell
# Tạo thư mục trên HDFS
hdfs dfs -mkdir -p /data/raw /data/processed /data/results

# Upload tất cả file CSV vào HDFS
hdfs dfs -put "C:\Downloads\ukraine-dataset\*.csv" /data/raw/

# Kiểm tra đã upload thành công
hdfs dfs -ls /data/raw
hdfs dfs -du -h /data/raw
```

> 💡 **Lưu ý:**
> - Dữ liệu thật phải nằm trên HDFS (`hdfs://localhost:9000/data/raw/`) mới chạy được
> - Quá trình upload có thể mất 10-30 phút tùy tốc độ ổ cứng/mạng

---

## 📁 Cấu trúc thư mục dự án

```
bigdata_russia_ukraine_sentiment/
│
├── 📂 data/
│   ├── raw/                    # (HDFS) Dữ liệu gốc CSV
│   ├── processed/              # (HDFS) Dữ liệu Parquet đã làm sạch
│   └── results/                # (HDFS) Kết quả dự đoán & phân tích
│
├── 📂 src/
│   ├── etl_preprocess.py       # 🔧 Bước 1: Làm sạch, định nghĩa Schema, lưu Parquet
│   ├── ml_sentiment_model.py   # 🤖 Bước 2: Feature Eng (TF-IDF), Train (LogisticRegression)
│   ├── generate_submission.py  # 📄 Bước 3: Tạo file submission.csv chuẩn định dạng
│   ├── model_evaluation.py     # 📊 Bước 4: Đánh giá độ chính xác (F1, Accuracy)
│   └── trend_analysis.py       # 📈 Bước 5: Phân tích xu hướng & Vẽ biểu đồ
│
└── 📄 README.md
```

---
## ⚙️ Cấu hình môi trường (Windows Local)

| Thành phần | Version | Ghi chú |
|------------|---------|---------||
| **Hadoop** | 3.3.6 | Đã cài `winutils.exe` trong `bin/` |
| **Spark** | 3.1.1 | PySpark với Hadoop |
| **Python** | 3.13.x | Đã xử lý tương thích Pickling/UDF |
| **Java** | JDK 8/11 | Cần thiết cho Hadoop & Spark |

### Cài đặt thư viện Python

```powershell
pip install pyspark numpy pandas matplotlib seaborn scikit-learn
```

### Thiết lập biến môi trường

```powershell
setx HADOOP_HOME "C:\hadoop-3.3.6"
setx SPARK_HOME "C:\spark-3.1.1-bin-hadoop2.7"
setx PATH "%HADOOP_HOME%\bin;%SPARK_HOME%\bin;%PATH%"
```

---
## 🚀 Hướng dẫn chạy dự án

### ⚠️ TRƯỜNG HỢP 1: Chạy lần đầu (Chưa có dữ liệu sạch)

Thực hiện **tuần tự** 5 bước sau:

#### **Bước 1️⃣: Làm sạch dữ liệu (ETL)**

Xử lý dữ liệu thô, lọc nhiễu và chuyển sang định dạng Parquet tối ưu.

```powershell
spark-submit src\etl_preprocess.py
```

**Output:** `hdfs://localhost:9000/data/processed/clean_tweets`

---
#### **Bước 2️⃣: Huấn luyện & Dự đoán (Machine Learning)**

Trích xuất đặc trưng (TF-IDF) và huấn luyện mô hình Logistic Regression.

```powershell
spark-submit src\ml_sentiment_model.py
```

**Output:** `hdfs://localhost:9000/data/results/sentiment_tweets_ml`

---

#### **Bước 3️⃣: Tạo file nộp bài (Submission)**

Trích xuất các cột ID và xác suất, gộp thành 1 file CSV.

```powershell
spark-submit src\generate_submission.py

# Tải file submission từ HDFS về máy local (nếu cần)
hdfs dfs -getmerge /data/results/submission_csv submission.csv
```

**Output:** `submission.csv` với format:
```
tweet_id | predicted_sentiment | prob_positive | prob_negative | prob_neutral
```

---

#### **Bước 4️⃣: Đánh giá mô hình**

Tính toán các chỉ số Accuracy, F1-Score, Precision, Recall.

```powershell
spark-submit src\model_evaluation.py
```

**Output:** `data/results/model_evaluation_report.txt`

---

#### **Bước 5️⃣: Phân tích & Vẽ biểu đồ**

Phân tích xu hướng theo thời gian, vị trí và xuất ra file ảnh `.png`.

```powershell
spark-submit src\trend_analysis.py
```

**Output:** 6 biểu đồ PNG trong `data/results/`:
- 📊 `1_sentiment_distribution.png` - Phân bố cảm xúc tổng quan
- 📈 `2_sentiment_time_trend.png` - Xu hướng theo thời gian
- 🌍 `3_sentiment_by_location.png` - Top 15 vị trí
- #️⃣ `4_sentiment_by_hashtag.png` - Top 20 hashtags
- 🔥 `5_peak_discussion_periods.png` - Giai đoạn cao điểm
- 🕐 `6_sentiment_by_hour.png` - Phân bố theo giờ

---
### ✅ TRƯỜNG HỢP 2: Chạy lại (Dữ liệu gốc đã có trên HDFS)

Nếu bạn cần chạy lại code (ví dụ: sau khi sửa logic), hãy xóa các thư mục output cũ trên HDFS để tránh lỗi "File already exists".

#### 1. Khởi động HDFS (nếu chưa chạy)

```powershell
start-dfs.cmd
```

#### 2. Xóa kết quả cũ

```powershell
hdfs dfs -rm -r /data/processed/clean_tweets
hdfs dfs -rm -r /data/results/sentiment_tweets_ml
hdfs dfs -rm -r /data/results/submission_csv
```

#### 3. Chạy lại Pipeline

Thực hiện lại các lệnh `spark-submit` như ở **Trường hợp 1**.

---

## 📊 Kết quả mong đợi

| Metric | Score |
|--------|-------|
| **F1-Score (Macro)** | 0.85+ |
| **Accuracy** | 0.87+ |
| **AP Score** | 0.83+ |

---

