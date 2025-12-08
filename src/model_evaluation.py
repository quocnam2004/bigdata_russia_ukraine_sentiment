# -*- coding: utf-8 -*-
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from pyspark.sql import SparkSession
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col, when, lit
from pyspark.ml.feature import StringIndexer

spark = SparkSession.builder \
    .appName("Model_Evaluation") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000") \
    .getOrCreate()

# 1. Đọc dữ liệu kết quả
data_path = "hdfs://localhost:9000/data/results/sentiment_tweets_ml"
print(f">>> Đang đọc dữ liệu đánh giá từ: {data_path}")
df = spark.read.parquet(data_path)

# 2. Tái tạo cột label (dạng số) để so sánh
# Vì MulticlassClassificationEvaluator cần cột label dạng số (double), 
# ta cần map lại 'predicted_sentiment' và label gốc (từ rule-based) sang số.

# Tạo lại label gốc từ rule-based (giống bước train) để làm 'ground truth' giả định
df_eval = df.withColumn(
    "label_str",
    when(
        (col("text_clean").contains("peace") | col("text_clean").contains("hope") | col("text_clean").contains("win")), 
        "positive"
    ).when(
        (col("text_clean").contains("war") | col("text_clean").contains("kill") | col("text_clean").contains("crisis")), 
        "negative"
    ).otherwise("neutral")
)

# Chuyển đổi String Label sang Index (Số)
indexer = StringIndexer(inputCol="label_str", outputCol="label_idx")
indexer_model = indexer.fit(df_eval)
df_indexed = indexer_model.transform(df_eval)

# Chuyển đổi Predicted Sentiment sang Index (Số) tương ứng
# Lưu ý: Cần đảm bảo mapping giống nhau. Cách tốt nhất là dùng lại indexer model
# Nhưng ở đây ta dùng StringIndexer thứ 2 cho cột prediction string
pred_indexer = StringIndexer(inputCol="predicted_sentiment", outputCol="prediction_idx")
# Fit trên chính cột prediction để lấy mapping
df_final = pred_indexer.fit(df_indexed).transform(df_indexed)

# 3. Tính toán Metrics
print(">>> Đang tính toán các chỉ số đánh giá...")

# Accuracy
evaluator_acc = MulticlassClassificationEvaluator(
    labelCol="label_idx", predictionCol="prediction_idx", metricName="accuracy")
accuracy = evaluator_acc.evaluate(df_final)

# F1-Score
evaluator_f1 = MulticlassClassificationEvaluator(
    labelCol="label_idx", predictionCol="prediction_idx", metricName="f1")
f1_score = evaluator_f1.evaluate(df_final)

# Weighted Precision
evaluator_prec = MulticlassClassificationEvaluator(
    labelCol="label_idx", predictionCol="prediction_idx", metricName="weightedPrecision")
precision = evaluator_prec.evaluate(df_final)

# Weighted Recall
evaluator_rec = MulticlassClassificationEvaluator(
    labelCol="label_idx", predictionCol="prediction_idx", metricName="weightedRecall")
recall = evaluator_rec.evaluate(df_final)

print("="*40)
print(f"KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH (Trên tập dữ liệu Sample)")
print("="*40)
print(f"Accuracy:  {accuracy:.4f}")
print(f"F1-Score:  {f1_score:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print("="*40)

spark.stop()