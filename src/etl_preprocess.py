# -*- coding: utf-8 -*-
import sys
import io
# Đảm bảo in console tiếng Việt đúng định dạng
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace, trim, lower, coalesce, lit
from pyspark.sql.types import StructType, StructField, LongType, StringType, DoubleType, IntegerType, BooleanType

# ===============================
# 1. Định nghĩa Schema (Fix lỗi AnalysisException)
# ===============================
# Dựa trên bộ dữ liệu Kaggle và các trường bạn đang sử dụng.
# Sử dụng StringType cho hầu hết các cột có thể chứa giá trị Null hoặc bị lỗi format, 
# sau đó sẽ cast (chuyển đổi) sang kiểu số trong bước xử lý sau (Feature Engineering).

TWEET_SCHEMA = StructType([
    StructField("tweetid", LongType(), True),
    StructField("userid", LongType(), True),
    StructField("user_name", StringType(), True),
    StructField("tweetcreatedts", StringType(), True),
    StructField("text", StringType(), True),
    StructField("language", StringType(), True),
    StructField("location", StringType(), True),
    StructField("country", StringType(), True),
    StructField("hashtags", StringType(), True),
    StructField("retweetcount", LongType(), True),
    StructField("favorite_count", LongType(), True),
    StructField("followers", LongType(), True),
    # Thêm các cột khác từ file CSV gốc nếu cần cho phân tích sau
    # Ví dụ: user_verified, is_reply, ...
])

# ===============================
# 2. Khởi tạo SparkSession
# ===============================
spark = SparkSession.builder \
    .appName("ETL_Preprocess_Twitter") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000") \
    .config("spark.driver.memory", "6g") \
    .config("spark.executor.memory", "6g") \
    .config("spark.sql.shuffle.partitions", "200") \
    .getOrCreate()

# ===============================
# 3. Đọc dữ liệu từ HDFS
# ===============================
# SỬA: Thêm tùy chọn recursiveFileLookup="true" để tìm kiếm đệ quy
# trong các thư mục con (raw/ và thư mục khác)
raw_path = "hdfs://localhost:9000/data/raw/"
print(">>> Đang đọc dữ liệu từ HDFS:", raw_path)

df = spark.read.option("header", True).option("recursiveFileLookup", "true").schema(TWEET_SCHEMA).csv(raw_path)

# Lấy số dòng chỉ để kiểm tra (có thể mất thời gian với data lớn)
# Chúng ta sẽ sử dụng df.count() để kiểm tra nhanh.
initial_count = df.count()
print(f"Số dòng ban đầu (có thể bao gồm header trùng): {initial_count}")
df.printSchema()

# ===============================
# 4. Làm sạch dữ liệu và Feature Engineering cơ bản
# ===============================

# Thêm cột bị thiếu nếu có (ví dụ: favorite_count có thể bị thiếu trong một số file)
df_filled = df.withColumn(
    "favorite_count", 
    coalesce(col("favorite_count"), lit(0))
).withColumn(
    "retweetcount", 
    coalesce(col("retweetcount"), lit(0))
)

df_clean = (
    df_filled.dropna(subset=["text"]) # Xóa các tweet không có nội dung
      # Đảm bảo các cột số là kiểu LongType hoặc DoubleType
      .withColumn("retweet_count", col("retweetcount").cast(LongType()))
      .withColumn("like_count", col("favorite_count").cast(LongType()))
      .withColumn("followers", col("followers").cast(LongType()))
      
      # Tiền xử lý văn bản (Chuẩn hóa chữ thường, xóa khoảng trắng dư thừa)
      .withColumn("text_clean", lower(trim(col("text"))))
      .withColumn("text_clean", regexp_replace(col("text_clean"), r"http\S+", ""))    # Xóa link
      .withColumn("text_clean", regexp_replace(col("text_clean"), r"@\w+", ""))        # Xóa mention
      # Giữ lại chữ cái (bao gồm chữ cái tiếng Việt), khoảng trắng, và số.
      .withColumn("text_clean", regexp_replace(col("text_clean"), r"[^a-zA-ZÀ-ỹ0-9\s]", " ")) 
      .withColumn("text_clean", regexp_replace(col("text_clean"), r"\s+", " ")) # Thay thế nhiều khoảng trắng thành một

    .select(
          col("tweetid").alias("tweet_id"), 
          col("text"),                                        # Text gốc (để tham chiếu)
          col("text_clean"),                                  # Text đã làm sạch (cho mô hình ML)
          col("tweetcreatedts").alias("created_at"),          # Thời gian
          col("location"),                                    # Vị trí địa lý
          col("hashtags"),                                    # Hashtags
          col("retweet_count"), 
          col("like_count"), 
          col("followers").alias("user_followers")             # Độ ảnh hưởng user
      )
)

# ===============================
# 5. Lưu dữ liệu sạch ra HDFS
# ===============================
out_path = "hdfs://localhost:9000/data/processed/clean_tweets"
df_clean.write.mode("overwrite").parquet(out_path)

print(">>> Dữ liệu sạch đã lưu tại:", out_path)
print(f"Số dòng đã xử lý và lưu: {df_clean.count()}")

spark.stop()