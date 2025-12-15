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
# 1. HÀM XỬ LÝ: GIẢI NÉN & PARSE (Worker)
# ============================================================
def process_binary_file(filename_content_tuple):
    filename, content = filename_content_tuple
    rows = []
    try:
        # Giải nén GZIP
        with gzip.GzipFile(fileobj=io.BytesIO(content)) as f:
            text_data = f.read().decode('utf-8', errors='ignore')
            
        # Parse CSV (Hỗ trợ multiline)
        csv_file = io.StringIO(text_data)
        reader = csv.reader(csv_file)
        
        for parts in reader:
            if len(parts) < 15: continue 
            
            def get_safe(idx):
                return parts[idx] if idx < len(parts) else None

            # Lấy Tweet ID tại Index 9
            tid = get_safe(9)
            if not tid or not tid.isdigit(): continue

            def safe_long(val):
                try: return int(float(val)) if val else 0
                except: return 0

            rows.append(Row(
                tweet_id     = int(tid),
                text         = get_safe(12),
                created_at   = get_safe(10),
                location     = get_safe(4),
                hashtags     = get_safe(13),
                retweet_count = safe_long(get_safe(11)),
                user_followers = safe_long(get_safe(6))
            ))
    except Exception:
        return [] 
    return rows

# ============================================================
# 2. MAIN PROGRAM (RECURSIVE BATCH)
# ============================================================
if __name__ == "__main__":
    
    spark = SparkSession.builder \
        .appName("ETL_Recursive_Batch_V6") \
        .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "100") \
        .getOrCreate()

    sc = spark.sparkContext
    
    # --- CẤU HÌNH HADOOP FILE SYSTEM ---
    fs = sc._gateway.jvm.org.apache.hadoop.fs.FileSystem.get(sc._jsc.hadoopConfiguration())
    Path = sc._gateway.jvm.org.apache.hadoop.fs.Path

    # --- QUÉT FILE ĐỆ QUY (RECURSIVE SCAN) ---
    print(">>> Đang quét toàn bộ file (bao gồm thư mục con)...")
    raw_path_str = "/data/raw/"
    
    # listFiles(path, recursive=True) -> Trả về Iterator
    remote_iterator = fs.listFiles(Path(raw_path_str), True)
    
    all_files = []
    while remote_iterator.hasNext():
        file_status = remote_iterator.next()
        path = file_status.getPath().toString()
        size = file_status.getLen()
        
        # Lọc file gzip có dữ liệu
        if path.endswith(".gzip") and size > 0:
            all_files.append(path)
            
    total_files = len(all_files)
    print(f">>> Đã tìm thấy: {total_files} file .gzip hợp lệ.")

    if total_files == 0:
        print("ERROR: Không tìm thấy file nào! Kiểm tra lại đường dẫn HDFS.")
        spark.stop()
        sys.exit(1)

    # --- CHUẨN BỊ LƯU TRỮ ---
    out_path = "hdfs://localhost:9000/data/processed/clean_tweets"
    
    # Xóa dữ liệu cũ nếu có
    if fs.exists(Path(out_path)):
        print(">>> Đang dọn dẹp thư mục đích cũ...")
        fs.delete(Path(out_path), True)

    # --- CHẠY BATCH (5 FILE/LẦN) ---
    BATCH_SIZE = 5
    schema = StructType([
        StructField("tweet_id", LongType(), True),
        StructField("text", StringType(), True),
        StructField("created_at", StringType(), True),
        StructField("location", StringType(), True),
        StructField("hashtags", StringType(), True),
        StructField("retweet_count", LongType(), True),
        StructField("user_followers", LongType(), True)
    ])

    print(f">>> Bắt đầu xử lý {total_files} file theo từng đợt (Batch size: {BATCH_SIZE})...")

    for i in range(0, total_files, BATCH_SIZE):
        batch_files = all_files[i : i + BATCH_SIZE]
        current_batch = i//BATCH_SIZE + 1
        total_batches = (total_files + BATCH_SIZE - 1) // BATCH_SIZE
        
        print(f"\n[Batch {current_batch}/{total_batches}] Đang xử lý {len(batch_files)} file...")
        
        try:
            # Gom danh sách file thành chuỗi
            files_str = ",".join(batch_files)
            
            # Đọc Binary
            binary_rdd = sc.binaryFiles(files_str, minPartitions=10)
            
            # Giải nén & Parse
            row_rdd = binary_rdd.flatMap(process_binary_file)
            
            if row_rdd.isEmpty():
                print("   -> (Trống) Batch này không có dữ liệu hợp lệ.")
                continue

            # Tạo DataFrame & Làm sạch
            df = spark.createDataFrame(row_rdd, schema=schema)
            
            df_clean = (
                df
                .withColumn("text_clean", lower(trim(col("text"))))
                .withColumn("text_clean", regexp_replace(col("text_clean"), r"http\S+", ""))
                .withColumn("text_clean", regexp_replace(col("text_clean"), r"@\w+", ""))
                .withColumn("text_clean", regexp_replace(col("text_clean"), r"[^a-zA-Z0-9\s]", " "))
                .withColumn("text_clean", regexp_replace(col("text_clean"), r"\s+", " "))
                .filter(col("text_clean") != "")
            )
            
            # Ghi nối đuôi (Append)
            print("   -> Đang ghi xuống HDFS...")
            df_clean.write.mode("append").parquet(out_path)
            
            # Dọn dẹp bộ nhớ
            df.unpersist()
            
        except Exception as e:
            print(f"   -> [CẢNH BÁO] Lỗi tại batch này: {e}")
            continue

    print("\n>>> HOÀN TẤT TOÀN BỘ QUÁ TRÌNH ETL!")
    
    # --- KIỂM TRA KẾT QUẢ ---
    try:
        final_count = spark.read.parquet(out_path).count()
        print(f"{'='*40}")
        print(f"TỔNG SỐ DÒNG SẠCH: {final_count}")
        print(f"{'='*40}")
        if final_count > 0:
            spark.read.parquet(out_path).select("tweet_id", "text").show(5, truncate=False)
    except:
        print("Không thể đọc file kết quả (có thể do lỗi hoặc rỗng).")

    spark.stop()