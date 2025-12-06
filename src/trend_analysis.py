# -*- coding: utf-8 -*-
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from pyspark.sql import SparkSession
from pyspark.sql.functions import to_timestamp, col, count
import matplotlib.pyplot as plt

# ===============================
# 1. Khởi tạo SparkSession
# ===============================
spark = SparkSession.builder \
    .appName("Trend_Analysis_Twitter") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "200") \
    .getOrCreate()

# ===============================
# 2. Đọc dữ liệu có sentiment
# ===============================
in_path = "hdfs://localhost:9000/data/results/sentiment_tweets"
df = spark.read.parquet(in_path)

# ===============================
# 3. Chuyển cột thời gian & lọc dữ liệu hợp lệ
# ===============================
df = df.withColumn("tweetcreatedts", to_timestamp(col("tweetcreatedts"), "yyyy-MM-dd HH:mm:ss")) \
       .filter(col("tweetcreatedts").isNotNull())

# ===============================
# 4. Thống kê cảm xúc theo thời gian
# ===============================
df_trend = df.groupBy("sentiment").agg(count("*").alias("count")).toPandas()

# ===============================
# 5. Vẽ biểu đồ tổng quan cảm xúc
# ===============================
plt.figure(figsize=(6, 4))
plt.bar(df_trend["sentiment"], df_trend["count"], color=["green", "gray", "red"])
plt.title("Tổng quan cảm xúc các tweet về chiến tranh Nga - Ukraine")
plt.xlabel("Cảm xúc")
plt.ylabel("Số lượng tweet")
plt.tight_layout()
plt.savefig("sentiment_overview.png")
print(">>> Biểu đồ đã lưu: sentiment_overview.png")

spark.stop()
