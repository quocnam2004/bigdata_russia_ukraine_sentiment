# -*- coding: utf-8 -*-
import sys
import io
import pyspark.sql.functions as F

# Cấu hình encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from pyspark.sql import SparkSession
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql.types import FloatType

# ========================================================
# 1. KHỞI TẠO SPARK (CÓ LOG HISTORY)
# ========================================================
spark = SparkSession.builder \
    .appName("Model_Evaluation_Final") \
    .master("local[2]") \
    .config("spark.eventLog.enabled", "true") \
    .config("spark.eventLog.dir", "hdfs://localhost:9000/spark-logs") \
    .config("spark.history.fs.logDirectory", "hdfs://localhost:9000/spark-logs") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

# ========================================================
# 2. ĐỌC DỮ LIỆU DỰ ĐOÁN
# ========================================================
pred_path = "hdfs://localhost:9000/data/results/sentiment_predictions"
print(f">>> Đang đọc kết quả dự đoán từ: {pred_path}")

try:
    df_pred = spark.read.parquet(pred_path)
    
    # Kiểm tra xem có đủ cột không
    if "label" not in df_pred.columns:
        print("LỖI: File kết quả thiếu cột 'label'. Hãy chạy lại file training!")
        sys.exit(1)
        
    # Cache để xử lý nhanh hơn
    df_pred.cache()
    total_rows = df_pred.count()
    print(f">>> Đã tải {total_rows} dòng kết quả dự đoán.")

except Exception as e:
    print(f"LỖI: Không đọc được file. {e}")
    sys.exit(1)

# ========================================================
# 3. TỰ ĐỘNG MAP NHÃN (SỐ -> CHỮ)
# ========================================================
# Lấy mapping thực tế từ dữ liệu để in báo cáo cho đúng
print(">>> Đang xác định mapping nhãn...")
label_map_rows = df_pred.select("label", "sentiment").distinct().collect()
label_map = {row.label: row.sentiment for row in label_map_rows}

print(f"Mapping tìm thấy: {label_map}")

# ========================================================
# 4. CHUẨN BỊ DỮ LIỆU ĐÁNH GIÁ
# ========================================================
# Chuyển đổi sang RDD (prediction, label)
# Lưu ý: prediction và label trong dataframe là Double, cần ép kiểu về Float cho Metrics
predictionAndLabels = df_pred.select("prediction", "label") \
    .rdd.map(lambda row: (float(row.prediction), float(row.label)))

# ========================================================
# 5. TÍNH TOÁN CHỈ SỐ (METRICS)
# ========================================================
metrics = MulticlassMetrics(predictionAndLabels)

print("\n" + "="*60)
print("             BÁO CÁO ĐÁNH GIÁ MÔ HÌNH CHI TIẾT")
print("="*60)

# 1. Chỉ số tổng quát
print(f"\n>>> [1] CHỈ SỐ TỔNG QUÁT (OVERALL METRICS):")
print(f"   - Độ chính xác (Accuracy):      {metrics.accuracy:.2%}")
print(f"   - Precision (Weighted):         {metrics.weightedPrecision:.2%}")
print(f"   - Recall (Weighted):            {metrics.weightedRecall:.2%}")
print(f"   - F1 Score (Weighted):          {metrics.weightedFMeasure():.2%}")

# 2. Ma trận nhầm lẫn
print("\n>>> [2] MA TRẬN NHẦM LẪN (CONFUSION MATRIX):")
print("   (Hàng: Thực tế | Cột: Dự đoán)")
print(metrics.confusionMatrix().toArray())

# 3. Chỉ số chi tiết từng lớp
print(f"\n>>> [3] CHI TIẾT TỪNG LỚP (BY CLASS):")
print(f"{'ID':<5} | {'Nhãn (Label)':<15} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
print("-" * 65)

# Lặp qua các label có trong map
sorted_labels = sorted(label_map.keys())

for label in sorted_labels:
    label_name = label_map.get(label, "Unknown")
    print(f"{label:<5.1f} | {label_name:<15} | {metrics.precision(label):<10.2%} | {metrics.recall(label):<10.2%} | {metrics.fMeasure(label):<10.2%}")

print("\n" + "="*60)
print(">>> HOÀN TẤT ĐÁNH GIÁ!")
spark.stop()