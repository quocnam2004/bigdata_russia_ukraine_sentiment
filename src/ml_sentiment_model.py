# -*- coding: utf-8 -*-
import sys
import io
# Đảm bảo in console tiếng Việt đúng định dạng
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit
from pyspark.sql.types import StringType, DoubleType, LongType
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
# QUAN TRỌNG: Import hàm biến đổi vector thành mảng của Spark ML
from pyspark.ml.functions import vector_to_array

# ===============================
# 1. Khởi tạo SparkSession (Khắc phục OOM và Lỗi I/O Shuffle)
# ===============================
spark = SparkSession.builder \
    .appName("ML_Sentiment_Analysis") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .config("spark.driver.maxResultSize", "4g") \
    .config("spark.driver.extraJavaOptions", "-Xms7g -Xmx7g") \
    .config("spark.sql.shuffle.partitions", "100") \
    .config("spark.local.dir", "C:/tmp/spark-local") \
    .getOrCreate()

# ===============================
# 2. Đọc dữ liệu đã làm sạch (ÁP DỤNG SAMPLING KHẮC PHỤC OOM)
# ===============================
in_path = "hdfs://localhost:9000/data/processed/clean_tweets"
df_full = spark.read.parquet(in_path)

# Lấy mẫu 10% dữ liệu
df = df_full.sample(False, 0.1, seed=42) 

print(f">>> Đã load (sampled 10%) {df.count()} tweets từ {in_path}")

# ===============================
# 3. Tạo label từ Rule-Based (Pseudo-Labeling)
# ===============================
df_labeled = df.withColumn(
    "label_text",
    when(
        (
            col("text_clean").contains("peace") | col("text_clean").contains("hope") |
            col("text_clean").contains("love") | col("text_clean").contains("support") |
            col("text_clean").contains("brave") | col("text_clean").contains("hero") |
            col("text_clean").contains("win")
        ),
        "positive"
    ).when(
        (
            col("text_clean").contains("war") | col("text_clean").contains("attack") |
            col("text_clean").contains("kill") | col("text_clean").contains("bomb") |
            col("text_clean").contains("dead") | col("text_clean").contains("crisis") |
            col("text_clean").contains("hate")
        ),
        "negative"
    ).otherwise("neutral")
).filter(col("text_clean").isNotNull() & (col("text_clean") != ""))

# ===============================
# 4. Feature Engineering: Giảm số lượng Feature
# ===============================
print(">>> Bắt đầu Feature Engineering với TF-IDF...")

tokenizer = Tokenizer(inputCol="text_clean", outputCol="words")
# Giữ 2000 features để tránh OOM
hashingTF = HashingTF(inputCol="words", outputCol="raw_features", numFeatures=2000)
idf = IDF(inputCol="raw_features", outputCol="features")
label_indexer = StringIndexer(inputCol="label_text", outputCol="label")

# ===============================
# 5. Train ML Model: Logistic Regression
# ===============================
print(">>> Training Logistic Regression model...")

lr = LogisticRegression(
    maxIter=20,
    regParam=0.01,
    elasticNetParam=0.0,
    family="multinomial"
)

pipeline_lr = Pipeline(stages=[tokenizer, hashingTF, idf, label_indexer, lr])
model_lr = pipeline_lr.fit(df_labeled)

print(">>> Logistic Regression model đã train xong!")

# ===============================
# 6. Dự đoán và Trích xuất Xác suất (KHÔNG DÙNG UDF - Native Spark)
# ===============================
print(">>> Bắt đầu dự đoán sentiment với probability...")

df_predictions = model_lr.transform(df_labeled)

label_mapping = model_lr.stages[3].labels
print(f">>> Label mapping: {label_mapping}")

# Tìm index
try:
    idx_positive = label_mapping.index("positive")
    idx_negative = label_mapping.index("negative")
    idx_neutral = label_mapping.index("neutral")
except ValueError:
    idx_positive = 0
    idx_negative = 1
    idx_neutral = 2
    print("CẢNH BÁO: Lỗi tìm kiếm nhãn. Dùng index mặc định [0, 1, 2].")

# --- GIẢI PHÁP NATIVE SPARK (Tránh lỗi Python 3.13) ---
# 1. Chuyển đổi cột Vector 'probability' thành cột Array<Double>
# Hàm vector_to_array chạy bằng Scala, không bị ảnh hưởng bởi phiên bản Python
df_predictions = df_predictions.withColumn("prob_array", vector_to_array("probability"))

# 2. Truy cập phần tử mảng bằng cách thông thường (getItem)
df_result = df_predictions.withColumn(
    "prob_positive", col("prob_array")[idx_positive]
).withColumn(
    "prob_negative", col("prob_array")[idx_negative]
).withColumn(
    "prob_neutral", col("prob_array")[idx_neutral]
)

# 3. Ánh xạ nhãn bằng WHEN/CASE (Native Spark)
df_result = df_result.withColumn(
    "predicted_sentiment",
    when(col("prediction") == lit(idx_neutral), "neutral")
    .when(col("prediction") == lit(idx_negative), "negative")
    .when(col("prediction") == lit(idx_positive), "positive")
    .otherwise("unknown")
)

# ===============================
# 7. Lưu kết quả
# ===============================
out_path = "hdfs://localhost:9000/data/results/sentiment_tweets_ml"

df_final = df_result.select(
    "tweet_id",
    "text",
    "text_clean",
    "created_at",
    "location",
    "hashtags",
    "retweet_count",
    "like_count",
    "user_followers",
    "predicted_sentiment",
    "prob_positive",
    "prob_negative",
    "prob_neutral"
)

print(f">>> Đang lưu kết quả vào {out_path}...")
df_final.write.mode("overwrite").parquet(out_path)

final_count = df_final.count()
print(">>> Hoàn tất! Kết quả ML sentiment đã lưu.")
print(f">>> Tổng số tweets: {final_count}")

# Sample kết quả
print("\n>>> Sample 5 kết quả:")
df_final.select("tweet_id", "predicted_sentiment", "prob_positive", "prob_negative", "prob_neutral").show(5, truncate=False)

spark.stop()