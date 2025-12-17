# -*- coding: utf-8 -*-
import sys
import io
import csv
import gzip

# Cấu hình encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, LongType
from pyspark.sql import Row
from pyspark.sql.functions import col, lower, trim, regexp_replace

# ============================================================
# CẤU HÌNH NGÔN NGỮ MỤC TIÊU
# ============================================================
# en: Anh, ru: Nga, uk: Ukraine, pl: Ba Lan, de: Đức
TARGET_LANGS = {'en', 'ru', 'uk', 'pl', 'de'}

# ============================================================
# 1. HÀM XỬ LÝ: GIẢI NÉN & PARSE (Worker Node)
# ============================================================
def process_binary_file(filename_content_tuple):
    filename, content = filename_content_tuple
    rows = []
    try:
        # Giải nén GZIP
        with gzip.GzipFile(fileobj=io.BytesIO(content)) as f:
            text_data = f.read().decode('utf-8', errors='ignore')
            
        # Parse CSV
        csv_file = io.StringIO(text_data)
        reader = csv.reader(csv_file)
        
        for parts in reader:
            if len(parts) < 15: continue 
            
            def get_safe(idx):
                return parts[idx] if idx < len(parts) else None
            
            # --- [SỬA ĐỔI 1] LỌC ĐA NGÔN NGỮ ---
            lang = get_safe(14)
            # Nếu ngôn ngữ không nằm trong danh sách 5 nước trên thì bỏ qua
            if lang not in TARGET_LANGS: continue

            # Lấy Tweet ID
            tid = get_safe(9)
            if not tid or not tid.isdigit(): continue

            def safe_long(val):
                try: return int(float(val)) if val else 0
                except: return 0

            rows.append(Row(
                tweet_id      = int(tid),
                text          = get_safe(12),
                created_at    = get_safe(10),
                location      = get_safe(4),
                hashtags      = get_safe(13),
                retweet_count = safe_long(get_safe(11)),
                user_followers = safe_long(get_safe(6)),
                lang          = lang # (Tùy chọn) Giữ lại cột lang để sau này thống kê
            ))
    except Exception:
        return [] 
    return rows

# ============================================================
# 2. MAIN PROGRAM
# ============================================================
if __name__ == "__main__":
    
    spark = SparkSession.builder \
        .appName("ETL_MultiLang_Process") \
        .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000") \
        .config("spark.eventLog.enabled", "true") \
        .config("spark.eventLog.dir", "hdfs://localhost:9000/spark-logs") \
        .config("spark.history.fs.logDirectory", "hdfs://localhost:9000/spark-logs") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "100") \
        .getOrCreate()

    sc = spark.sparkContext
    fs = sc._gateway.jvm.org.apache.hadoop.fs.FileSystem.get(sc._jsc.hadoopConfiguration())
    Path = sc._gateway.jvm.org.apache.hadoop.fs.Path

    # --- QUÉT FILE ---
    print(">>> Đang quét toàn bộ file trên HDFS...")
    raw_path_str = "/data/raw/"
    
    all_files = []
    try:
        remote_iterator = fs.listFiles(Path(raw_path_str), True)
        while remote_iterator.hasNext():
            file_status = remote_iterator.next()
            path = file_status.getPath().toString()
            if path.endswith(".gzip") and file_status.getLen() > 0:
                all_files.append(path)
    except Exception:
        print("ERROR: Không đọc được HDFS.")
        spark.stop(); sys.exit(1)

    total_files = len(all_files)
    print(f">>> Đã tìm thấy: {total_files} file.")

    # --- CHUẨN BỊ LƯU TRỮ ---
    out_path = "hdfs://localhost:9000/data/processed/clean_tweets"
    if fs.exists(Path(out_path)):
        fs.delete(Path(out_path), True)

    BATCH_SIZE = 5
    # Thêm cột lang vào schema
    schema = StructType([
        StructField("tweet_id", LongType(), True),
        StructField("text", StringType(), True),
        StructField("created_at", StringType(), True),
        StructField("location", StringType(), True),
        StructField("hashtags", StringType(), True),
        StructField("retweet_count", LongType(), True),
        StructField("user_followers", LongType(), True),
        StructField("lang", StringType(), True) 
    ])

    print(f">>> Bắt đầu xử lý đa ngôn ngữ (En, Ru, Uk, Pl, De)...")

    for i in range(0, total_files, BATCH_SIZE):
        batch_files = all_files[i : i + BATCH_SIZE]
        
        try:
            files_str = ",".join(batch_files)
            binary_rdd = sc.binaryFiles(files_str, minPartitions=4)
            row_rdd = binary_rdd.flatMap(process_binary_file)
            
            if row_rdd.isEmpty(): continue

            df = spark.createDataFrame(row_rdd, schema=schema)
            
            # --- [SỬA ĐỔI 2] REGEX CHO ĐA NGÔN NGỮ ---
            # Sử dụng \p{L} để giữ lại mọi chữ cái (kể cả tiếng Nga, Đức, Ba Lan...)
            # Regex giữ lại: a-z, 0-9, khoảng trắng, Tiếng Nga/Ukraine (\u0400-\u04FF), Tiếng Đức/Ba Lan (\u00C0-\u024F)
            multilang_regex = r"[^a-zA-Z0-9\s\u0400-\u04FF\u00C0-\u024F]"

            df_clean = (
                df
                .withColumn("text_clean", lower(trim(col("text"))))
                .withColumn("text_clean", regexp_replace(col("text_clean"), r"http\S+", "")) 
                .withColumn("text_clean", regexp_replace(col("text_clean"), r"@\w+", ""))
                # Thay regex a-z bằng regex đa ngôn ngữ
                .withColumn("text_clean", regexp_replace(col("text_clean"), multilang_regex, " ")) 
                .withColumn("text_clean", regexp_replace(col("text_clean"), r"\s+", " "))
                .filter(col("text_clean") != "")
            )
            
            print(f"   -> [Batch {i//BATCH_SIZE + 1}] Ghi xuống HDFS...")
            df_clean.coalesce(1).write.mode("append").parquet(out_path)
            
            del df, df_clean
            
        except Exception as e:
            print(f"   -> Lỗi: {e}")
            continue

    print("\n>>> HOÀN TẤT ETL ĐA NGÔN NGỮ!")
    
    # Kiểm tra kết quả
    try:
        final_df = spark.read.parquet(out_path)
        print(f"TỔNG SỐ DÒNG: {final_df.count()}")
        print("Phân bố ngôn ngữ:")
        final_df.groupBy("lang").count().show() # Hiện bảng thống kê ngôn ngữ
        final_df.select("text_clean", "lang").show(5, truncate=False)
    except: pass

    spark.stop()