# -*- coding: utf-8 -*-
import sys
import io
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Đảm bảo in console tiếng Việt đúng định dạng
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, count, date_format, to_timestamp, when, desc, year

# ===============================
# 1. Khởi tạo SparkSession
# ===============================
spark = SparkSession.builder \
    .appName("Trend_Analysis") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000") \
    .config("spark.driver.memory", "4g") \
    .getOrCreate()

# ===============================
# 2. Đọc dữ liệu kết quả từ HDFS
# ===============================
in_path = "hdfs://localhost:9000/data/results/sentiment_tweets_ml"
print(f">>> Đang đọc dữ liệu phân tích từ: {in_path}")
df = spark.read.parquet(in_path)

# Chuyển đổi cột created_at sang kiểu Date
# VÀ QUAN TRỌNG: Lọc bỏ các năm bị lỗi (chỉ lấy từ năm 2020 đến 2025)
df_trend = df.withColumn("event_date", to_date(col("created_at"))) \
             .filter(col("event_date").isNotNull()) \
             .filter((year(col("event_date")) >= 2020) & (year(col("event_date")) <= 2025)) # <-- FIX LỖI OutOfBounds

print(">>> Đang tổng hợp dữ liệu...")

# ===============================
# 3. Phân tích 1: Tỷ lệ Cảm xúc Tổng quan
# ===============================
sentiment_dist = df_trend.groupBy("predicted_sentiment").count().toPandas()
print("\n[Phân bố Cảm xúc]")
print(sentiment_dist)

if not sentiment_dist.empty:
    plt.figure(figsize=(8, 8))
    plt.pie(sentiment_dist['count'], labels=sentiment_dist['predicted_sentiment'], autopct='%1.1f%%', startangle=140, colors=['#ff9999','#66b3ff','#99ff99'])
    plt.title('Tỷ lệ Cảm xúc về Xung đột Nga - Ukraine')
    plt.savefig('sentiment_distribution.png')
    print(">>> Đã lưu biểu đồ: sentiment_distribution.png")

# ===============================
# 4. Phân tích 2: Xu hướng Cảm xúc theo Thời gian (Timeline)
# ===============================
# Group by Date và Sentiment
trend_df = df_trend.groupBy("event_date", "predicted_sentiment") \
    .count() \
    .orderBy("event_date") \
    .toPandas()

if not trend_df.empty:
    # Chuyển đổi cột event_date sang datetime của pandas, ép kiểu errors='coerce' để an toàn
    trend_df['event_date'] = pd.to_datetime(trend_df['event_date'], errors='coerce')
    
    # Loại bỏ các dòng bị NaT (Not a Time) nếu vẫn còn sót
    trend_df = trend_df.dropna(subset=['event_date'])

    if not trend_df.empty:
        # Pivot table để vẽ biểu đồ đường
        trend_pivot = trend_df.pivot(index='event_date', columns='predicted_sentiment', values='count').fillna(0)
        
        plt.figure(figsize=(12, 6))
        trend_pivot.plot(kind='line', marker='o', ax=plt.gca())
        plt.title('Xu hướng Cảm xúc theo Thời gian')
        plt.xlabel('Ngày')
        plt.ylabel('Số lượng Tweet')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('sentiment_trend_timeline.png')
        print(">>> Đã lưu biểu đồ: sentiment_trend_timeline.png")
        
        # Lưu kết quả thống kê ra CSV (để làm báo cáo)
        trend_df.to_csv("daily_sentiment_stats.csv", index=False)
        print(">>> Đã lưu file thống kê: daily_sentiment_stats.csv")
    else:
        print(">>> CẢNH BÁO: Dữ liệu thời gian không hợp lệ sau khi lọc.")
else:
    print(">>> CẢNH BÁO: Không thể phân tích xu hướng thời gian (Dữ liệu rỗng).")

# ===============================
# 5. Phân tích 3: Top Quốc gia/Vị trí quan tâm (nếu có dữ liệu location)
# ===============================
# Lọc bỏ location null hoặc rỗng
top_locations = df_trend.filter((col("location").isNotNull()) & (col("location") != "")) \
    .groupBy("location") \
    .count() \
    .orderBy(desc("count")) \
    .limit(10) \
    .toPandas()

if not top_locations.empty:
    plt.figure(figsize=(12, 6))
    sns.barplot(x='count', y='location', data=top_locations, palette='viridis')
    plt.title('Top 10 Vị trí thảo luận nhiều nhất')
    plt.xlabel('Số lượng Tweet')
    plt.ylabel('Vị trí')
    plt.tight_layout()
    plt.savefig('top_locations.png')
    print(">>> Đã lưu biểu đồ: top_locations.png")

spark.stop()