# -*- coding: utf-8 -*-
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from pyspark.sql import SparkSession
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql.types import FloatType
from pyspark.sql.functions import col

# ========================================================
# 1. KHỞI TẠO SPARK
# ========================================================
spark = SparkSession.builder \
    .appName("Model_Evaluation") \
    .master("local[2]") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

# ========================================================
# 2. ĐỌC DỮ LIỆU DỰ ĐOÁN
# ========================================================
pred_path = "hdfs://localhost:9000/data/results/sentiment_predictions"
print(f">>> Đang đọc kết quả dự đoán từ: {pred_path}")
df_pred = spark.read.parquet(pred_path)

# ========================================================
# 3. CHUẨN BỊ DỮ LIỆU ĐÁNH GIÁ
# ========================================================
# Spark MLlib Metrics cần RDD dạng (prediction, label) là số thực (float)
# Ta cần map label chữ (sentiment) sang số tương ứng với prediction
# Dựa trên kết quả trước: Neutral=0.0, Negative=1.0, Positive=2.0

predictionAndLabels = df_pred.select("prediction", "sentiment") \
    .rdd.map(lambda row: (
        float(row.prediction), 
        0.0 if row.sentiment == "neutral" else (1.0 if row.sentiment == "negative" else 2.0)
    ))

# ========================================================
# 4. TÍNH TOÁN CHỈ SỐ (METRICS)
# ========================================================
metrics = MulticlassMetrics(predictionAndLabels)

print("\n" + "="*50)
print("   BÁO CÁO ĐÁNH GIÁ MÔ HÌNH (EVALUATION REPORT)")
print("="*50)

# 1. Ma trận nhầm lẫn (Confusion Matrix)
print("\n>>> [1] Ma trận nhầm lẫn (Confusion Matrix):")
print("   (Dòng: Thực tế | Cột: Dự đoán)")
print(metrics.confusionMatrix().toArray())

# 2. Các chỉ số tổng quát
print(f"\n>>> [2] Chỉ số tổng quát:")
print(f"   - Độ chính xác (Accuracy):  {metrics.accuracy:.2%}")
print(f"   - Precision (Weighted):     {metrics.weightedPrecision:.2%}")
print(f"   - Recall (Weighted):        {metrics.weightedRecall:.2%}")
print(f"   - F1 Score (Weighted):      {metrics.weightedFMeasure():.2%}")

# 3. Chỉ số chi tiết từng lớp
labels = [0.0, 1.0, 2.0]
label_names = {0.0: "Neutral", 1.0: "Negative", 2.0: "Positive"}

print(f"\n>>> [3] Chỉ số chi tiết từng lớp:")
print(f"{'Lớp (Label)':<15} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
print("-" * 55)

for label in labels:
    print(f"{label_names[label]:<15} | {metrics.precision(label):<10.2%} | {metrics.recall(label):<10.2%} | {metrics.fMeasure(label):<10.2%}")

print("\n" + "="*50)
spark.stop()