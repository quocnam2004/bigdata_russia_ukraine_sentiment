# -*- coding: utf-8 -*-
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col, lower

# ===============================
# 1. Khởi tạo SparkSession
# ===============================
spark = SparkSession.builder \
    .appName("Sentiment_Analysis_RuleBased") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "200") \
    .getOrCreate()

# ===============================
# 2. Đọc dữ liệu đã làm sạch
# ===============================
in_path = "hdfs://localhost:9000/data/processed/clean_tweets"

df = spark.read.parquet(in_path)
print(">>> Đã load dữ liệu sạch:", in_path)

# ===============================
# 3. Áp dụng quy tắc phân loại cảm xúc đơn giản
# ===============================
positive_words = ["peace", "hope", "love", "support", "brave", "hero", "win"]
negative_words = ["war", "attack", "kill", "bomb", "dead", "crisis", "hate"]

df_sentiment = df.withColumn(
    "sentiment",
    when(
        (
            col("text").contains("peace") |
            col("text").contains("hope") |
            col("text").contains("love") |
            col("text").contains("support") |
            col("text").contains("brave") |
            col("text").contains("hero") |
            col("text").contains("win")
        ),
        "positive"
    ).when(
        (
            col("text").contains("war") |
            col("text").contains("attack") |
            col("text").contains("kill") |
            col("text").contains("bomb") |
            col("text").contains("dead") |
            col("text").contains("crisis") |
            col("text").contains("hate")
        ),
        "negative"
    ).otherwise("neutral")
)

# ===============================
# 4. Ghi kết quả phân tích cảm xúc ra HDFS
out_path = "hdfs://localhost:9000/data/results/sentiment_tweets"

df_sentiment.select("text", "tweetcreatedts", "sentiment").write.mode("overwrite").parquet(out_path)

print(">>> Đã lưu kết quả phân tích cảm xúc vào:", out_path)

spark.stop()
