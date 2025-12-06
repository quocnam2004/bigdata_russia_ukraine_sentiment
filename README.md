
# 🧠 Big Data Sentiment Analysis – Russia vs Ukraine Twitter

## 📥 Tải Dataset (BẮT BUỘC)

**Dataset KHÔNG có sẵn trong source code!** Bạn cần tải từ Kaggle:

1. **Download từ Kaggle:**
   - Link: https://www.kaggle.com/datasets/bwandowando/ukraine-russian-crisis-twitter-dataset-1-2-m-rows
   - Size: ~10GB (292 files CSV)
   - Đăng nhập Kaggle → Download → Giải nén

2. **Upload lên HDFS:**
   ```powershell
   # Tạo thư mục trên HDFS
   hdfs dfs -mkdir -p /data/raw /data/processed /data/results
   
   # Upload tất cả file CSV vào HDFS (thay đường dẫn thực tế)
   hdfs dfs -put "C:\Downloads\ukraine-dataset\*.csv" /data/raw/
   
   # Kiểm tra đã upload thành công
   hdfs dfs -ls /data/raw
   hdfs dfs -du -h /data/raw
   ```

3. **Lưu ý:**
   - Thư mục `data/raw/` local chỉ để tham khảo cấu trúc, KHÔNG chứa data thật
   - Data thật phải nằm trên HDFS mới chạy được
   - Mất ~10-30 phút để upload tùy tốc độ mạng

## 📁 Cấu trúc thư mục
```
bigdata_russia_ukraine_sentiment/
│
├── data/
│   ├── raw/                # Dữ liệu gốc (CSV/JSON từ Kaggle)
│   ├── processed/          # Dữ liệu đã xử lý
│   └── results/            # Kết quả phân tích & biểu đồ
│
├── src/
│   ├── etl_preprocess.py   # Tiền xử lý dữ liệu (ETL)
│   ├── sentiment_model.py  # Phân loại cảm xúc (Spark ML hoặc rule-based)
│   ├── trend_analysis.py   # Phân tích xu hướng & thống kê
│
└── README.md
```

## ⚙️ Cấu hình môi trường
- **Hadoop**: 3.3.6 (đã cài winutils.exe trong `bin/`)
- **Spark**: 3.1.1
- **Python**: >= 3.10
- **PySpark**: 3.1.2

Thêm vào PATH:
```powershell
setx HADOOP_HOME "C:\hadoop-3.3.6"
setx SPARK_HOME "C:\spark-3.1.1-bin-hadoop2.7"
setx PATH "%HADOOP_HOME%\bin;%SPARK_HOME%\bin;%PATH%"
```

## 🚀 Hướng dẫn chạy dự án

### 📋 Yêu cầu hệ thống
- Hadoop HDFS 3.3.6+
- Apache Spark 3.1.1+
- Python 3.10+ với PySpark
- RAM: Tối thiểu 8GB (khuyến nghị 16GB)

---

## 🔀 Chọn trường hợp của bạn

### ⚠️ TRƯỜNG HỢP 1: Chưa có dữ liệu trên HDFS (Lần đầu chạy)

**Bước 1: Tải dataset từ Kaggle**
```powershell
# Truy cập và download: https://www.kaggle.com/datasets/bwandowando/ukraine-russian-crisis-twitter-dataset-1-2-m-rows
# Giải nén file zip vào thư mục tạm (ví dụ: C:\Downloads\ukraine-dataset\)
```

**Bước 2: Khởi động HDFS**
```powershell
jps  # Kiểm tra HDFS
start-dfs.cmd  # Nếu chưa chạy

# Kiểm tra HDFS Web UI tại: http://localhost:9870
```

**Bước 3: Tạo thư mục và upload data lên HDFS**
```powershell
# Tạo cấu trúc thư mục
hdfs dfs -mkdir -p /data/raw /data/processed /data/results

# Upload tất cả file CSV (thay đường dẫn thực tế)
hdfs dfs -put "C:\Downloads\ukraine-dataset\*.csv" /data/raw/

# Kiểm tra (phải thấy 292 files, ~10GB)
hdfs dfs -ls /data/raw
hdfs dfs -du -h /data/raw
```

**Bước 4: Chạy pipeline**
```powershell
# Chạy lần lượt từng bước
spark-submit src\etl_preprocess.py
spark-submit src\sentiment_model.py
spark-submit src\trend_analysis.py
```

---

### ✅ TRƯỜNG HỢP 2: Đã có dữ liệu trên HDFS (Chạy lại)

**Bước 1: Kiểm tra HDFS & Data**
```powershell
# Khởi động HDFS (nếu chưa chạy)
jps
start-dfs.cmd

# Kiểm tra HDFS Web UI: http://localhost:9870

# Xác nhận có data (phải thấy 292 files)
hdfs dfs -ls /data/raw
```

**Bước 2: Xóa kết quả cũ (nếu cần chạy lại)**
```powershell
hdfs dfs -rm -r /data/processed/clean_tweets
hdfs dfs -rm -r /data/results/sentiment_tweets
```

**Bước 3: Chạy pipeline**
```powershell
spark-submit src\etl_preprocess.py
spark-submit src\sentiment_model.py
spark-submit src\trend_analysis.py
```

---

## 📊 Xem kết quả

**Kiểm tra trên HDFS:**
```powershell
hdfs dfs -ls /data/results/sentiment_tweets
hdfs dfs -count /data/results/sentiment_tweets
```

**Xuất mẫu ra local:**
```powershell
hdfs dfs -get /data/results/sentiment_tweets/part-00000*.parquet data\results\
```

**Biểu đồ:** File `sentiment_overview.png` trong thư mục hiện tại

---

## 🔧 Xử lý sự cố

**Lỗi Out of Memory:**
```powershell
spark-submit --driver-memory 6g --executor-memory 6g src\sentiment_model.py
```

**Spark Web UI:** http://localhost:4040 (khi job đang chạy)

---

### 📊 Kết quả cuối cùng

Sau khi hoàn thành, bạn sẽ có:
```
HDFS:
├── /data/raw/                    # Dữ liệu gốc (~10GB, 57M tweets)
├── /data/processed/clean_tweets  # Dữ liệu đã làm sạch
└── /data/results/sentiment_tweets # Kết quả phân tích cảm xúc

Local:
└── sentiment_overview.png        # Biểu đồ tổng quan
```

## 📊 Kết quả
- Biểu đồ cảm xúc tổng thể
- Biểu đồ biến động cảm xúc theo thời gian


## 🧩 Ghi chú
- Dataset lớn (~44GB), nên ưu tiên chạy local cluster `--master local[*]`.
- Có thể export sample nhỏ để vẽ biểu đồ nhẹ hơn.

---
🧑‍💻 Project thực hiện bởi: TRÀ QUỐC NAM
