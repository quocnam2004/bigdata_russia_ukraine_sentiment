# -*- coding: utf-8 -*-
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace, trim, lower

# ===============================
# 1. Khởi tạo SparkSession
# ===============================
spark = SparkSession.builder \
    .appName("ETL_Preprocess_Twitter") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "200") \
    .getOrCreate()

# ===============================
# 2. Đọc dữ liệu từ HDFS
# ===============================
raw_path = "hdfs://localhost:9000/data/raw/"
print(">>> Đang đọc dữ liệu từ HDFS:", raw_path)

df = spark.read.option("header", True).csv(raw_path)

print("Số dòng ban đầu:", df.count())
df.printSchema()

# ===============================
# 3. Làm sạch dữ liệu
# ===============================
df_clean = (
    df.dropna(subset=["text"])
      .withColumn("text", lower(trim(col("text"))))
      .withColumn("text", regexp_replace(col("text"), r"http\S+", ""))     # Xóa link
      .withColumn("text", regexp_replace(col("text"), r"@\w+", ""))        # Xóa mention
      .withColumn("text", regexp_replace(col("text"), r"[^a-zA-ZÀ-ỹ\s]", ""))  # Chỉ giữ chữ
      .select("text", "tweetcreatedts")  # Giữ lại cột timestamp
)

# ===============================
# 4. Lưu dữ liệu sạch ra HDFS
# ===============================
out_path = "hdfs://localhost:9000/data/processed/clean_tweets"
df_clean.write.mode("overwrite").parquet(out_path)

print(">>> Dữ liệu sạch đã lưu tại:", out_path)
spark.stop()
