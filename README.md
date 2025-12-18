# 🧠 Big Data Sentiment Analysis – Russia vs Ukraine Twitter

## 📥 Tải Dataset (BẮT BUỘC)

**Dataset KHÔNG có sẵn trong source code!** Bạn cần tải từ Kaggle:

1. **Download từ Kaggle:**
   - Link: [Ukraine Conflict Twitter Dataset](https://www.kaggle.com/datasets/bwandowando/ukraine-russian-crisis-twitter-dataset-1-2-m-rows/versions/508)
   - Size: ~18GB
   - Đăng nhập Kaggle → Download → Giải nén.

2. **Upload lên HDFS:**
   ```powershell
   # Tạo thư mục trên HDFS
   hdfs dfs -mkdir -p /data/raw /data/processed /data/results
   
   # Upload tất cả file CSV vào HDFS (Lưu ý đường dẫn file local của bạn)
   # Ví dụ:
   hdfs dfs -put "C:\Downloads\ukraine-dataset\*" /data/raw/
   
   # Kiểm tra đã upload thành công
   hdfs dfs -ls /data/raw
   hdfs dfs -du -h /data/raw
   ```
# Truy cập http://localhost:9870/ sau khi dùng lệnh start-dfs.cmd
# Truy cập http://localhost:18080/ thì dùng lệnh .\bin\spark-class.cmd org.apache.spark.deploy.history.HistoryServer

**Lưu ý:**

- Dữ liệu thật phải nằm trên HDFS (`hdfs://localhost:9000/data/raw/`) mới chạy được.
- Quá trình upload có thể mất 10-30 phút tùy tốc độ ổ cứng/mạng.
```

## ⚙️ Cấu hình môi trường (Windows Local)
- **Hadoop:** 3.3.6 (đã cài `winutils.exe` trong `bin/`)
- **Spark:** 3.1.1
- **Python:** 3.9.x  
- **Java:** JDK 8 hoặc 11

**Cài đặt thư viện Python cần thiết:**
```powershell
pip install pyspark numpy pandas matplotlib seaborn
```

**Thiết lập biến môi trường (PowerShell):**
```powershell
setx HADOOP_HOME "C:\hadoop-3.3.6"
setx SPARK_HOME "C:\spark-3.1.1-bin-hadoop2.7"
setx PATH "%HADOOP_HOME%\bin;%SPARK_HOME%\bin;%PATH%"
```

## 🚀 Hướng dẫn chạy dự án

### ⚠️ TRƯỜNG HỢP 1: Chạy lần đầu (Chưa có dữ liệu sạch)
Thực hiện tuần tự 5 bước sau:

**Bước 1: Làm sạch dữ liệu (ETL)**  
Xử lý dữ liệu thô, lọc nhiễu và chuyển sang định dạng Parquet tối ưu.
```powershell
spark-submit src\etl_preprocess.py
```

**Bước 2: Huấn luyện & Dự đoán (Machine Learning)**  
Trích xuất đặc trưng (TF-IDF) và huấn luyện mô hình Logistic Regression (có Sampling 10% để tránh OOM).
```powershell
spark-submit src\ml_sentiment_model.py
```

**Bước 3: Đánh giá mô hình**  
Tính toán các chỉ số Accuracy, F1-Score, Precision, Recall.
```powershell
spark-submit --conf spark.eventLog.enabled=true --conf spark.eventLog.dir=hdfs://localhost:9000/spark-logs --conf spark.history.fs.logDirectory=hdfs://localhost:9000/spark-logs src/model_evaluation.py
```

**Bước 4: Phân tích & Vẽ biểu đồ**  
Phân tích xu hướng theo thời gian, vị trí và xuất ra file ảnh .png.
```powershell
spark-submit --driver-memory 4g --executor-memory 4g src/trend_analysis.py
```

### ✅ TRƯỜNG HỢP 2: Chạy lại (Dữ liệu gốc đã có trên HDFS)
Nếu bạn cần chạy lại code (ví dụ: sau khi sửa logic), hãy xóa các thư mục output cũ trên HDFS để tránh lỗi "File already exists".

**1. Khởi động HDFS (nếu chưa chạy):**
```powershell
start-dfs.cmd
```

**2. Xóa kết quả cũ:**
```powershell
hdfs dfs -rm -r /data/processed/clean_tweets
hdfs dfs -rm -r /data/results/sentiment_predictions
```

**3. Chạy lại Pipeline:**  
Thực hiện lại các lệnh `spark-submit` như ở Trường hợp 1.

---

## 🧑‍💻 Project thực hiện bởi:
- **TRÀ QUỐC NAM**


