# -*- coding: utf-8 -*-
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from pyspark.sql import SparkSession

# Khởi tạo Spark
spark = SparkSession.builder \
    .appName("Generate_Submission") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()

# Đọc kết quả từ bước Train Model
in_path = "hdfs://localhost:9000/data/results/sentiment_tweets_ml"
print(f">>> Đang đọc dữ liệu từ: {in_path}")
df = spark.read.parquet(in_path)

# Chọn các cột theo đúng yêu cầu định dạng nộp bài
# tweet_id | predicted_sentiment | prob_positive | prob_negative | prob_neutral
df_submission = df.select(
    "tweet_id", 
    "predicted_sentiment", 
    "prob_positive", 
    "prob_negative", 
    "prob_neutral"
)

# Lưu thành 1 file CSV duy nhất (Coalesce 1) để dễ nộp
out_path = "hdfs://localhost:9000/data/results/submission_csv"
print(f">>> Đang lưu file submission vào: {out_path}")

# header=True để có dòng tiêu đề
df_submission.coalesce(1).write.mode("overwrite").option("header", "true").csv(out_path)

print(">>> Hoàn tất! Bạn có thể tải file CSV từ HDFS về máy bằng lệnh:")
print(f"    hdfs dfs -getmerge {out_path} submission.csv")

spark.stop()