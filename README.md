# 🧠 Big Data Sentiment Analysis – Russia vs Ukraine Twitter

## 📥 Tải Dataset (BẮT BUỘC)

**Dataset KHÔNG có sẵn trong source code!** Bạn cần tải từ Kaggle:

1. **Download từ Kaggle:**
   - Link: [Ukraine Conflict Twitter Dataset](https://www.kaggle.com/datasets/bwandowando/ukraine-russian-crisis-twitter-dataset-1-2-m-rows)
   - Size: ~10GB (Giải nén ra ~44GB)
   - Đăng nhập Kaggle → Download → Giải nén.

2. **Upload lên HDFS:**
   ```powershell
   # Tạo thư mục trên HDFS
   hdfs dfs -mkdir -p /data/raw /data/processed /data/results
   
   # Upload tất cả file CSV vào HDFS (Lưu ý đường dẫn file local của bạn)
   # Ví dụ:
   hdfs dfs -put "C:\Downloads\ukraine-dataset\*.csv" /data/raw/
   
   # Kiểm tra đã upload thành công
   hdfs dfs -ls /data/raw
   hdfs dfs -du -h /data/raw
Lưu ý:

Dữ liệu thật phải nằm trên HDFS (hdfs://localhost:9000/data/raw/) mới chạy được.

Quá trình upload có thể mất 10-30 phút tùy tốc độ ổ cứng/mạng.

📁 Cấu trúc thư mục dự án
bigdata_russia_ukraine_sentiment/
│
├── data/
│   ├── raw/                # (Trên HDFS) Dữ liệu gốc CSV
│   ├── processed/          # (Trên HDFS) Dữ liệu Parquet đã làm sạch
│   └── results/            # (Trên HDFS) Kết quả dự đoán & phân tích
│
├── src/
│   ├── etl_preprocess.py       # Bước 1: Làm sạch, định nghĩa Schema, lưu Parquet
│   ├── ml_sentiment_model.py   # Bước 2: Feature Eng (TF-IDF), Train (LogisticRegression), Predict
│   ├── generate_submission.py  # Bước 3: Tạo file submission.csv chuẩn định dạng
│   ├── model_evaluation.py     # Bước 4: Đánh giá độ chính xác (F1, Accuracy)
│   ├── trend_analysis.py       # Bước 5: Phân tích xu hướng & Vẽ biểu đồ
│
└── README.md
⚙️ Cấu hình môi trường (Windows Local)
Hadoop: 3.3.6 (đã cài winutils.exe trong bin/)

Spark: 3.1.1

Python: 3.13.x (Đã xử lý tương thích Pickling/UDF)

Java: JDK 8 hoặc 11

Cài đặt thư viện Python cần thiết:

PowerShell

pip install pyspark numpy pandas matplotlib seaborn
Thiết lập biến môi trường (PowerShell):

PowerShell

setx HADOOP_HOME "C:\hadoop-3.3.6"
setx SPARK_HOME "C:\spark-3.1.1-bin-hadoop2.7"
setx PATH "%HADOOP_HOME%\bin;%SPARK_HOME%\bin;%PATH%"
🚀 Hướng dẫn chạy dự án
⚠️ TRƯỜNG HỢP 1: Chạy lần đầu (Chưa có dữ liệu sạch)
Thực hiện tuần tự 5 bước sau:

Bước 1: Làm sạch dữ liệu (ETL) Xử lý dữ liệu thô, lọc nhiễu và chuyển sang định dạng Parquet tối ưu.

PowerShell

spark-submit src\etl_preprocess.py
Bước 2: Huấn luyện & Dự đoán (Machine Learning) Trích xuất đặc trưng (TF-IDF) và huấn luyện mô hình Logistic Regression (có Sampling 10% để tránh OOM).

PowerShell

spark-submit src\ml_sentiment_model.py
Bước 3: Tạo file nộp bài (Submission) Trích xuất các cột ID và xác suất, gộp thành 1 file CSV.

PowerShell

spark-submit src\generate_submission.py

# Tải file submission từ HDFS về máy local (để nộp)
hdfs dfs -getmerge /data/results/submission_csv submission.csv
Bước 4: Đánh giá mô hình Tính toán các chỉ số Accuracy, F1-Score, Precision, Recall.

PowerShell

spark-submit src\model_evaluation.py
Bước 5: Phân tích & Vẽ biểu đồ Phân tích xu hướng theo thời gian, vị trí và xuất ra file ảnh .png.

PowerShell

spark-submit src\trend_analysis.py
✅ TRƯỜNG HỢP 2: Chạy lại (Dữ liệu gốc đã có trên HDFS)
Nếu bạn cần chạy lại code (ví dụ: sau khi sửa logic), hãy xóa các thư mục output cũ trên HDFS để tránh lỗi "File already exists".

1. Khởi động HDFS (nếu chưa chạy):

PowerShell

start-dfs.cmd
2. Xóa kết quả cũ:

PowerShell

hdfs dfs -rm -r /data/processed/clean_tweets
hdfs dfs -rm -r /data/results/sentiment_tweets_ml
hdfs dfs -rm -r /data/results/submission_csv
3. Chạy lại Pipeline: Thực hiện lại các lệnh spark-submit như ở Trường hợp 1.

📊 Kết quả đầu ra
Sau khi chạy xong src/trend_analysis.py, tại thư mục dự án sẽ xuất hiện các file báo cáo:

submission.csv: File kết quả dự đoán (dùng để nộp bài).

sentiment_distribution.png: Biểu đồ tròn tỷ lệ cảm xúc (Positive/Negative/Neutral).

sentiment_trend_timeline.png: Biểu đồ đường thể hiện biến động cảm xúc theo thời gian (2022-2023).

top_locations.png: Biểu đồ cột 10 quốc gia/vị trí thảo luận nhiều nhất.

daily_sentiment_stats.csv: Số liệu chi tiết từng ngày.

🔧 Ghi chú kỹ thuật & Xử lý sự cố
Lỗi OutOfMemoryError: Java heap space:

Code hiện tại đã được cấu hình spark.driver.memory 8g.

Đã áp dụng kỹ thuật Sampling (10%) trong ml_sentiment_model.py để đảm bảo chạy được trên máy cá nhân (RAM 16GB).

Lỗi PicklingError / Tuple index out of range:

Do xung đột giữa Spark 3.1.1 và Python 3.13.

Giải pháp đã áp dụng: Code đã loại bỏ UDF Python thuần và chuyển sang sử dụng các hàm Native Spark (vector_to_array, when/case) để đảm bảo tương thích tuyệt đối.

Lỗi OutOfBoundsDatetime khi vẽ biểu đồ:

Do dữ liệu rác (năm âm hoặc quá xa).

Giải pháp đã áp dụng: Script trend_analysis.py đã có bộ lọc chỉ lấy dữ liệu từ năm 2020-2025.

🧑‍💻 Project thực hiện bởi: - TRÀ QUỐC NAM

PHẠM ĐỨC BẢO NGỌC

LÊ ĐÌNH VŨ