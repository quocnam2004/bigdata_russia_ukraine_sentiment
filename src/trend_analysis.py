# -*- coding: utf-8 -*-
import sys
import io
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.dates as mdates # Thư viện xử lý ngày tháng

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, count, sum as spark_sum, when, explode, split, desc, trim, lower, regexp_replace, length

# ========================================================
# 1. KHỞI TẠO SPARK
# ========================================================
spark = SparkSession.builder \
    .appName("Trend_Analysis_Final_Polished") \
    .master("local[2]") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "20") \
    .getOrCreate()

spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

# ========================================================
# 2. ĐỌC DỮ LIỆU
# ========================================================
print(">>> Đang đọc dữ liệu...")
df_pred = spark.read.parquet("hdfs://localhost:9000/data/results/sentiment_predictions")
df_clean = spark.read.parquet("hdfs://localhost:9000/data/processed/clean_tweets")

df_meta = df_clean.select(
    "tweet_id", 
    "created_at", 
    col("location").alias("user_location_raw"), 
    "hashtags",
    "retweet_count"
)

df_merged = df_pred.join(df_meta, "tweet_id")

# Chuẩn hóa Location
df_final = df_merged.withColumn("user_location", 
    when(lower(col("user_location_raw")).rlike("ukrain|україна|украина|kyiv|kiev"), "Ukraine")
    .when(lower(col("user_location_raw")).rlike("usa|united states|america|washington|ny|ca"), "USA")
    .when(lower(col("user_location_raw")).rlike("india|bharat"), "India")
    .when(lower(col("user_location_raw")).rlike("uk|united kingdom|london|england|britain"), "UK")
    .when(lower(col("user_location_raw")).rlike("germany|deutschland|berlin"), "Germany")
    .when(lower(col("user_location_raw")).rlike("france|paris"), "France")
    .otherwise(trim(col("user_location_raw")))
)

df_final = df_final.withColumn("sentiment_label", 
    when(col("prediction") == 0.0, "Neutral")
    .when(col("prediction") == 1.0, "Negative")
    .when(col("prediction") == 2.0, "Positive")
)

df_final = df_final.withColumn("date", to_date(col("created_at")))
df_final.cache()

# ========================================================
# [FIX 1] VẼ BIỂU ĐỒ HASHTAG (LỌC RÁC)
# ========================================================
print("\n>>> [FIX] Đang xử lý Hashtag (Lọc bỏ 'text', 'indices')...")

# Loại bỏ ký tự thừa
hashtag_clean = df_final.withColumn("hashtags_str", regexp_replace(col("hashtags"), r"[\[\]',\{\}:]", " "))

# Tách từ
hashtag_df = hashtag_clean.select("sentiment_label", explode(split(col("hashtags_str"), " ")).alias("hashtag"))

# LỌC RÁC KỸ CÀNG:
# 1. Bỏ từ 'text', 'indices' (do cấu trúc JSON)
# 2. Bỏ các con số thuần túy (ví dụ '100', '114')
# 3. Bỏ các từ quá ngắn (< 3 ký tự)
hashtag_exploded = hashtag_df.filter(
    (length(col("hashtag")) > 2) &
    (col("hashtag") != "text") &
    (col("hashtag") != "indices") &
    (~col("hashtag").rlike("^[0-9]+$")) # Regex: Không phải là số
)

# Lấy Top 10
top_hashtags = hashtag_exploded.groupBy("hashtag").count().orderBy(desc("count")).limit(10).select("hashtag")
top_hash_list = [row.hashtag for row in top_hashtags.collect()]

plot_data = hashtag_exploded.filter(col("hashtag").isin(top_hash_list)) \
    .groupBy("hashtag", "sentiment_label").count().toPandas()

if not plot_data.empty:
    plt.figure(figsize=(12, 8))
    # Thêm dấu # hiển thị cho đẹp
    plot_data['hashtag'] = plot_data['hashtag'].apply(lambda x: "#" + x if not x.startswith("#") else x)
    
    sns.barplot(x='count', y='hashtag', hue='sentiment_label', data=plot_data, palette='viridis')
    plt.title('Top 10 Hashtag nổi bật nhất')
    plt.xlabel('Số lượng')
    plt.ylabel('Hashtag')
    plt.tight_layout()
    plt.savefig('chart_3_hashtag_sentiment.png')
    print("   -> Đã lưu: chart_3_hashtag_sentiment.png")

# ========================================================
# [FIX 2] BIỂU ĐỒ CAO ĐIỂM (LÀM GỌN NHÃN)
# ========================================================
print("\n>>> [FIX] Vẽ lại biểu đồ cao điểm (Làm gọn nhãn)...")

df_eng = df_final.withColumn("total_engagement", col("retweet_count"))
daily_eng = df_eng.groupBy("date").agg(spark_sum("total_engagement").alias("daily_interaction")).orderBy("date").toPandas()

# Convert cột date sang datetime để matplotlib hiểu
daily_eng['date'] = pd.to_datetime(daily_eng['date'])

threshold = daily_eng['daily_interaction'].mean() + (1.5 * daily_eng['daily_interaction'].std())

fig, ax = plt.subplots(figsize=(14, 7))

# Vẽ đường line
sns.lineplot(data=daily_eng, x='date', y='daily_interaction', label='Tổng Retweet', color='blue', ax=ax)
ax.axhline(threshold, color='red', linestyle='--', label=f'Ngưỡng đột biến')

# Lấy các điểm cao điểm
peaks = daily_eng[daily_eng['daily_interaction'] > threshold]
ax.scatter(peaks['date'], peaks['daily_interaction'], color='red', s=100, zorder=5)

# CHỈ HIỂN THỊ NHÃN CHO TOP 3 ĐỈNH CAO NHẤT (Để không bị rối)
top_3_peaks = peaks.sort_values(by='daily_interaction', ascending=False).head(3)

for _, row in top_3_peaks.iterrows():
    # Format ngày tháng đẹp (dd-mm-yyyy)
    date_str = row['date'].strftime('%d-%m-%Y')
    ax.annotate(f"{date_str}", 
                xy=(row['date'], row['daily_interaction']), 
                xytext=(10, 10), textcoords='offset points',
                arrowprops=dict(arrowstyle="->", color='black'),
                fontsize=10, fontweight='bold', color='darkred')

# Định dạng trục ngày tháng
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1)) # Mỗi tháng 1 vạch
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
plt.xticks(rotation=45) # Xoay nghiêng chữ ngày tháng

plt.title('Biểu đồ Tương tác & Các sự kiện Cao điểm (Top 3)')
plt.ylabel('Tổng lượng Retweet')
plt.xlabel('Thời gian')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('chart_4_peaks_detection.png')
print("   -> Đã lưu: chart_4_peaks_detection.png")

# ========================================================
# CÁC BIỂU ĐỒ CÒN LẠI (VẼ LẠI CHO ĐỒNG BỘ)
# ========================================================
# 1. TIME TREND
print("\n>>> Vẽ lại biểu đồ xu hướng thời gian...")
trend_df = df_final.groupBy("date", "sentiment_label").count().orderBy("date").toPandas()
pivot_trend = trend_df.pivot(index='date', columns='sentiment_label', values='count').fillna(0)
plt.figure(figsize=(12, 6))
sns.lineplot(data=pivot_trend, markers=True, dashes=False)
plt.title('Xu hướng thay đổi cảm xúc theo thời gian')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('chart_1_time_trend.png')

# 2. LOCATION
print("\n>>> Vẽ lại biểu đồ vị trí...")
loc_df = df_final.filter((col("user_location") != "") & (col("user_location") != "Unknown") & (col("user_location").isNotNull()))
top_locs = loc_df.groupBy("user_location").count().orderBy(desc("count")).limit(10).select("user_location")
top_loc_list_final = [row.user_location for row in top_locs.collect()]
loc_sentiment = loc_df.filter(col("user_location").isin(top_loc_list_final)).groupBy("user_location", "sentiment_label").count().toPandas()
plt.figure(figsize=(12, 6))
sns.barplot(x='count', y='user_location', hue='sentiment_label', data=loc_sentiment)
plt.title('Phân bố cảm xúc tại Top 10 vị trí')
plt.xticks(rotation=15) # Xoay nhẹ tên nước cho dễ đọc
plt.tight_layout()
plt.savefig('chart_2_location_sentiment.png')

print("\n>>> HOÀN TẤT! Đã tối ưu hóa tất cả biểu đồ.")
spark.stop()