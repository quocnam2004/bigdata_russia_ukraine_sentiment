# -*- coding: utf-8 -*-
import sys
import io

# Cấu hình encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, CountVectorizer, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline, PipelineModel
from pyspark.sql.functions import col, when
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# ====================================================
# 1. KHỞI TẠO SPARK (CÓ LOG HISTORY SERVER)
# ====================================================
spark = SparkSession.builder \
    .appName("Sentiment_Model_MultiLang") \
    .master("local[2]") \
    .config("spark.eventLog.enabled", "true") \
    .config("spark.eventLog.dir", "hdfs://localhost:9000/spark-logs") \
    .config("spark.history.fs.logDirectory", "hdfs://localhost:9000/spark-logs") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.memory.fraction", "0.6") \
    .config("spark.sql.shuffle.partitions", "20") \
    .getOrCreate()

# ====================================================
# 2. ĐỌC DỮ LIỆU SẠCH TỪ PARQUET
# ====================================================
data_path = "hdfs://localhost:9000/data/processed/clean_tweets"
print(f">>> Đang đọc dữ liệu từ: {data_path}")

try:
    df = spark.read.parquet(data_path)
    # Chỉ lấy các cột cần thiết để tiết kiệm RAM
    df = df.select("tweet_id", "text_clean", "lang")
    df = df.dropna(subset=["text_clean"])
except Exception as e:
    print(f"LỖI: Không đọc được file Parquet. Kiểm tra lại đường dẫn! {e}")
    sys.exit(1)

# ====================================================
# 3. GÁN NHÃN ĐA NGÔN NGỮ (RULE-BASED LABELING)
# ====================================================
print(">>> Đang gán nhãn dữ liệu (Hỗ trợ: En, De, Pl, Ru, Uk)...")

# Từ khóa Tích cực (Positive) - 5 Ngôn ngữ
# En: good, peace... | De: gut, frieden | Pl: dobry, pokój | Ru/Uk: мир (hòa bình), любовь (yêu), добре (tốt)
pos_keywords = r"good|peace|support|love|save|help|hope|thank|gut|frieden|liebe|danke|dobry|pokój|miłość|мир|любовь|спасибо|дякую|слава|перемога"

# Từ khóa Tiêu cực (Negative) - 5 Ngôn ngữ
# En: war, kill... | De: krieg, töten | Pl: wojna, zabić | Ru/Uk: война (chiến tranh), убить (giết), krieg
neg_keywords = r"war|kill|die|hate|attack|bomb|sad|pain|destroy|krieg|töten|angriff|wojna|zabić|atak|война|війна|убить|вбити|атака|бомба"

df_labeled = df.withColumn("sentiment", 
    when(col("text_clean").rlike(pos_keywords), "positive")
    .when(col("text_clean").rlike(neg_keywords), "negative")
    .otherwise("neutral")
)

# --- LẤY MẪU (SAMPLING) ---
# Dữ liệu 48 triệu dòng -> Lấy 0.5% (khoảng 240,000 dòng)
# Tăng lên 0.5% để mô hình học tốt hơn, máy 16GB vẫn chịu được tốt.
print(">>> Đang lấy mẫu 0.5% cho Training...")
df_sample = df_labeled.sample(False, 0.005, seed=42)
df_sample.cache()

count = df_sample.count()
print(f">>> Số lượng mẫu huấn luyện thực tế: {count} dòng")

# ====================================================
# 4. XÂY DỰNG PIPELINE (Feature Engineering)
# ====================================================
print(">>> Đang xây dựng Pipeline...")

# 4.1. Tách từ
tokenizer = Tokenizer(inputCol="text_clean", outputCol="words")

# 4.2. Vector hóa (Tăng vocabSize lên 1000 vì có 5 ngôn ngữ)
vectorizer = CountVectorizer(inputCol="words", outputCol="features", vocabSize=1000, minDF=5)

# 4.3. Chuyển đổi nhãn (positive/negative/neutral -> 0, 1, 2)
labelIndexer = StringIndexer(inputCol="sentiment", outputCol="label")

# 4.4. Mô hình (Logistic Regression)
lr = LogisticRegression(maxIter=10, regParam=0.01, elasticNetParam=0.8)

# 4.5. Pipeline
pipeline = Pipeline(stages=[tokenizer, vectorizer, labelIndexer, lr])

# ====================================================
# 5. HUẤN LUYỆN & ĐÁNH GIÁ
# ====================================================
(train_data, test_data) = df_sample.randomSplit([0.8, 0.2], seed=42)

print(">>> Đang huấn luyện (Training)...")
model = pipeline.fit(train_data)

print(">>> Đang dự đoán (Predicting)...")
predictions = model.transform(test_data)

# Đánh giá Accuracy
evaluator_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator_acc.evaluate(predictions)

# Đánh giá F1-Score (Quan trọng hơn Accuracy khi dữ liệu lệch)
evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
f1_score = evaluator_f1.evaluate(predictions)

print(f"\n{'='*40}")
print(f"KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH (ĐA NGÔN NGỮ)")
print(f"{'='*40}")
print(f" - Accuracy: {accuracy:.2%}")
print(f" - F1-Score: {f1_score:.2%}")
print(f"{'='*40}\n")

# ====================================================
# 6. LƯU KẾT QUẢ & MODEL
# ====================================================
# Lưu kết quả dự đoán (Dữ liệu)
res_path = "hdfs://localhost:9000/data/results/sentiment_predictions"
print(f">>> Đang lưu dữ liệu dự đoán vào: {res_path}")
predictions.select("tweet_id", "sentiment", "prediction", "label") \
    .write.mode("overwrite").parquet(res_path)

# [QUAN TRỌNG] Lưu Model để dùng lại cho bước phân tích xu hướng
model_path = "hdfs://localhost:9000/data/models/sentiment_logreg_model"
print(f">>> Đang lưu Model đã train vào: {model_path}")
model.write().overwrite().save(model_path)

print(">>> HOÀN TẤT TOÀN BỘ!")
spark.stop()