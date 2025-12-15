# -*- coding: utf-8 -*-
import sys
import io

# Cấu hình encoding để in tiếng Việt không lỗi
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, CountVectorizer, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, when
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# ===============================
# 1. Khởi tạo Spark (CHẾ ĐỘ TIẾT KIỆM TỐI ĐA)
# ===============================
# .master("local[2]"): Chỉ dùng đúng 2 nhân CPU. Đây là CHÌA KHÓA để không bị tràn RAM.
# driver/executor memory: Để mức 4GB là mức an toàn nhất cho máy 16GB.
spark = SparkSession.builder \
    .appName("Sentiment_Analysis_Model_Final") \
    .master("local[2]") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.memory.fraction", "0.6") \
    .config("spark.sql.shuffle.partitions", "20") \
    .getOrCreate()

# ===============================
# 2. Đọc dữ liệu sạch
# ===============================
data_path = "hdfs://localhost:9000/data/processed/clean_tweets"
print(f">>> Đang đọc dữ liệu từ: {data_path}")

df = spark.read.parquet(data_path)
df = df.dropna(subset=["text_clean"])

# ===============================
# 3. Gán nhãn (Labeling)
# ===============================
print(">>> Đang gán nhãn dữ liệu...")

df_labeled = df.withColumn("sentiment", 
    when(col("text_clean").rlike("good|peace|support|love|save|help|hope|thank"), "positive")
    .when(col("text_clean").rlike("war|kill|die|hate|attack|bomb|sad|pain|destroy"), "negative")
    .otherwise("neutral")
)

# LẤY MẪU 0.1% (Khoảng 66,000 dòng)
print(">>> Đang lấy mẫu 0.1%...")
df_sample = df_labeled.sample(False, 0.001, seed=42)

# QUAN TRỌNG: Cache dữ liệu vào RAM ngay lập tức để ổn định luồng xử lý
df_sample.cache()
print(f">>> Số lượng mẫu thực tế: {df_sample.count()}")

# ===============================
# 4. Xây dựng Pipeline (SIÊU NHẸ)
# ===============================
print(">>> Đang xây dựng Pipeline...")

# 4.1. Tách từ
tokenizer = Tokenizer(inputCol="text_clean", outputCol="words")

# 4.2. CountVectorizer (Chỉ giữ 500 từ phổ biến nhất)
vectorizer = CountVectorizer(inputCol="words", outputCol="features", vocabSize=500, minDF=2)

# 4.3. Chuyển đổi nhãn
labelIndexer = StringIndexer(inputCol="sentiment", outputCol="label")

# 4.4. Mô hình Logistic Regression
lr = LogisticRegression(maxIter=5, regParam=0.1)

# 4.5. Pipeline
pipeline = Pipeline(stages=[tokenizer, vectorizer, labelIndexer, lr])

# ===============================
# 5. Huấn luyện & Đánh giá
# ===============================
(train_data, test_data) = df_sample.randomSplit([0.8, 0.2], seed=42)

print(">>> Đang huấn luyện (Training)...")
model = pipeline.fit(train_data)

print(">>> Đang dự đoán (Predicting)...")
predictions = model.transform(test_data)

evaluator = MulticlassClassificationEvaluator(
    labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)

print(f"\n{'='*40}")
print(f"ĐỘ CHÍNH XÁC (ACCURACY): {accuracy:.2%}")
print(f"{'='*40}\n")

# ===============================
# 6. Lưu kết quả
# ===============================
out_path = "hdfs://localhost:9000/data/results/sentiment_predictions"
print(f">>> Đang lưu kết quả vào: {out_path}")

predictions.select("tweet_id", "text", "sentiment", "prediction") \
    .write.mode("overwrite").parquet(out_path)

print(">>> HOÀN TẤT TOÀN BỘ!")
spark.stop()