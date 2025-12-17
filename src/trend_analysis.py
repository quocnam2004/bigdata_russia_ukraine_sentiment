# -*- coding: utf-8 -*-
import sys
import io
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.dates as mdates

# Cấu hình encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# CẤU HÌNH FONT CHỮ CHO MATPLOTLIB (QUAN TRỌNG)
# Để hiển thị được tiếng Việt và ký tự đặc biệt
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Tahoma']

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, count, sum as spark_sum, when, explode, split, desc, trim, lower, regexp_replace, length

# ========================================================
# 1. KHỞI TẠO SPARK (ĐÃ SỬA LỖI NAMENODE)
# ========================================================
spark = SparkSession.builder \
    .appName("Trend_Analysis_Final") \
    .master("local[2]") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "20") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000") \
    .config("spark.eventLog.enabled", "true") \
    .config("spark.eventLog.dir", "hdfs://localhost:9000/spark-logs") \
    .config("spark.history.fs.logDirectory", "hdfs://localhost:9000/spark-logs") \
    .getOrCreate()
    
# Tăng tốc độ chuyển đổi sang Pandas bằng Apache Arrow
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

# ========================================================
# 2. ĐỌC DỮ LIỆU KẾT QUẢ DỰ ĐOÁN
# ========================================================
# Đọc trực tiếp file kết quả (đã có nhãn sentiment) từ bước ML
data_path = "hdfs://localhost:9000/data/results/sentiment_predictions"
print(f">>> Đang đọc dữ liệu dự đoán từ: {data_path}")

try:
    # File này đã chứa: tweet_id, sentiment, prediction, label
    df_pred = spark.read.parquet(data_path)
    
    # Đọc thêm file gốc để lấy ngày tháng, location (nếu file dự đoán thiếu)
    # Tuy nhiên, để tối ưu, ta nên join với clean_tweets nhưng chỉ lấy cột cần thiết
    meta_path = "hdfs://localhost:9000/data/processed/clean_tweets"
    df_meta = spark.read.parquet(meta_path).select("tweet_id", "created_at", "location", "hashtags", "retweet_count", "lang")
    
    # Join dữ liệu (Dùng Inner Join để chỉ lấy các tweet đã dự đoán)
    print(">>> Đang ghép dữ liệu metadata...")
    df_final = df_pred.join(df_meta, "tweet_id")
    
except Exception as e:
    print(f"LỖI: Không đọc được dữ liệu. {e}")
    sys.exit(1)

# ========================================================
# 3. TIỀN XỬ LÝ CHO VẼ BIỂU ĐỒ
# ========================================================
print(">>> Đang chuẩn hóa dữ liệu...")

# 3.1. Chuẩn hóa Location (Gộp các tên khác nhau về 1 mối)
df_final = df_final.withColumn("user_location", 
    when(lower(col("location")).rlike("ukrain|україна|украина|kyiv|kiev|odesa"), "Ukraine")
    .when(lower(col("location")).rlike("usa|united states|america|washington|ny|ca|texas|florida"), "USA")
    .when(lower(col("location")).rlike("india|bharat|delhi|mumbai"), "India")
    .when(lower(col("location")).rlike("uk|united kingdom|london|england|britain"), "UK")
    .when(lower(col("location")).rlike("germany|deutschland|berlin|munich"), "Germany")
    .when(lower(col("location")).rlike("france|paris"), "France")
    .when(lower(col("location")).rlike("poland|polska|warsaw|krakow"), "Poland")
    .when(lower(col("location")).rlike("russia|moscow|kremlin|россия|москва"), "Russia")
    .otherwise("Other") # Gom nhóm còn lại thành Other để biểu đồ đỡ rối
)

# 3.2. Chuyển đổi ngày tháng
df_final = df_final.withColumn("date", to_date(col("created_at")))

# Cache lại để dùng chung cho 4 biểu đồ
df_final.cache()
print(f">>> Sẵn sàng phân tích trên {df_final.count()} dòng dữ liệu.")

# ========================================================
# BIỂU ĐỒ 1: XU HƯỚNG CẢM XÚC THEO THỜI GIAN (TIME SERIES)
# ========================================================
print("\n>>> [1/4] Vẽ biểu đồ Xu hướng thời gian...")
trend_df = df_final.groupBy("date", "sentiment").count().orderBy("date").toPandas()

# Pivot table để vẽ line chart
pivot_trend = trend_df.pivot(index='date', columns='sentiment', values='count').fillna(0)

plt.figure(figsize=(12, 6))
# Vẽ các đường với màu sắc chuẩn tâm lý học
try:
    sns.lineplot(data=pivot_trend['positive'], label='Positive', color='green', linewidth=2)
    sns.lineplot(data=pivot_trend['negative'], label='Negative', color='red', linewidth=2)
    sns.lineplot(data=pivot_trend['neutral'], label='Neutral', color='gray', linestyle='--', linewidth=1.5)
except KeyError:
    pass # Bỏ qua nếu thiếu lớp nào đó

plt.title('Xu hướng cảm xúc theo thời gian (2022-2023)', fontsize=14)
plt.xlabel('Thời gian')
plt.ylabel('Số lượng Tweet')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('chart_1_time_trend.png')
print("   -> Đã lưu: chart_1_time_trend.png")

# ========================================================
# BIỂU ĐỒ 2: PHÂN BỐ CẢM XÚC THEO QUỐC GIA (STACKED BAR)
# ========================================================
print("\n>>> [2/4] Vẽ biểu đồ Vị trí địa lý...")

# Lọc bỏ 'Other' và 'Unknown' để biểu đồ tập trung vào các nước lớn
loc_df = df_final.filter(col("user_location") != "Other")

# Tính toán
loc_sentiment = loc_df.groupBy("user_location", "sentiment").count().toPandas()

# Pivot để vẽ Stacked Bar Chart (Dễ so sánh tỷ lệ hơn)
loc_pivot = loc_sentiment.pivot(index='user_location', columns='sentiment', values='count').fillna(0)
# Sắp xếp theo tổng số lượng giảm dần
loc_pivot['total'] = loc_pivot.sum(axis=1)
loc_pivot = loc_pivot.sort_values('total', ascending=False).drop(columns='total')

loc_pivot.plot(kind='bar', stacked=True, figsize=(12, 7), color=['red', 'gray', 'green'])
plt.title('Phân bố cảm xúc tại các Quốc gia chủ chốt', fontsize=14)
plt.xlabel('Quốc gia')
plt.ylabel('Số lượng Tweet')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('chart_2_location_sentiment.png')
print("   -> Đã lưu: chart_2_location_sentiment.png")

# ========================================================
# BIỂU ĐỒ 3: TOP HASHTAG THEO CẢM XÚC (ĐÃ SỬA LỖI LABEL)
# ========================================================
print("\n>>> [3/4] Vẽ biểu đồ Top Hashtag...")

# 1. Loại bỏ các ký tự JSON thừa ({text: , }, [, ])
hashtag_df = df_final.withColumn("clean_tags", regexp_replace(col("hashtags"), r"\{text:|'text':|'indices':|\{|\[|\]|\}|'|\d+|:", " ")) \
    .withColumn("tag", explode(split(col("clean_tags"), ","))) \
    .withColumn("tag", trim(lower(col("tag"))))

# 2. Lọc rác kỹ càng hơn
hashtag_df = hashtag_df.filter(
    (length(col("tag")) > 2) & 
    (col("tag") != "text") & 
    (col("tag") != "indices") &
    (col("tag") != "")
)

# Lấy Top 10 Hashtag phổ biến nhất
top_tags = hashtag_df.groupBy("tag").count().orderBy(desc("count")).limit(10).select("tag")
top_tags_list = [r.tag for r in top_tags.collect()]

# Lọc dữ liệu chỉ lấy top tags để vẽ
plot_data = hashtag_df.filter(col("tag").isin(top_tags_list)) \
    .groupBy("tag", "sentiment").count().toPandas()

plt.figure(figsize=(12, 8))
# Thêm dấu # vào trước tên tag cho đúng chuẩn Twitter
plot_data['tag'] = plot_data['tag'].apply(lambda x: "#" + x if not x.startswith("#") else x)

sns.barplot(x='count', y='tag', hue='sentiment', data=plot_data, 
            palette={'positive': 'green', 'negative': 'red', 'neutral': 'gray'})
plt.title('Top 10 Hashtag và Cảm xúc liên quan', fontsize=14)
plt.xlabel('Số lượng')
plt.ylabel('Hashtag')
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('chart_3_hashtag_sentiment.png')
print("   -> Đã lưu: chart_3_hashtag_sentiment.png")

# ========================================================
# BIỂU ĐỒ 4: PHÁT HIỆN SỰ KIỆN NÓNG (ANOMALY DETECTION)
# ========================================================
print("\n>>> [4/4] Vẽ biểu đồ Phát hiện sự kiện nóng...")

# Tính tổng retweet theo ngày
daily_eng = df_final.groupBy("date").agg(spark_sum("retweet_count").alias("interactions")).orderBy("date").toPandas()
daily_eng['date'] = pd.to_datetime(daily_eng['date'])

# Tính ngưỡng bất thường (Mean + 2*StdDev)
mean_val = daily_eng['interactions'].mean()
std_val = daily_eng['interactions'].std()
threshold = mean_val + (2 * std_val)

fig, ax = plt.subplots(figsize=(14, 7))
sns.lineplot(data=daily_eng, x='date', y='interactions', label='Lượng tương tác (Retweets)', color='blue', ax=ax)
ax.axhline(threshold, color='red', linestyle='--', label=f'Ngưỡng đột biến ({int(threshold):,})')

# Đánh dấu các đỉnh
peaks = daily_eng[daily_eng['interactions'] > threshold]
ax.scatter(peaks['date'], peaks['interactions'], color='red', s=50, zorder=5)

# Gắn nhãn ngày cho 3 đỉnh cao nhất
top_peaks = peaks.sort_values('interactions', ascending=False).head(3)
for _, row in top_peaks.iterrows():
    ax.annotate(f"{row['date'].strftime('%d-%m-%Y')}", 
                xy=(row['date'], row['interactions']), 
                xytext=(10, 15), textcoords='offset points',
                arrowprops=dict(arrowstyle="->", color='black'),
                fontsize=9, fontweight='bold', color='darkred')

# Định dạng trục X
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))

plt.title('Phát hiện Sự kiện nóng dựa trên lượng Retweet', fontsize=14)
plt.ylabel('Tổng Retweet')
plt.xlabel('Thời gian')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('chart_4_peaks_detection.png')
print("   -> Đã lưu: chart_4_peaks_detection.png")

print("\n>>> HOÀN TẤT TOÀN BỘ!")
spark.stop()