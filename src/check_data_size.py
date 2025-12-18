# -*- coding: utf-8 -*-
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from pyspark.sql import SparkSession

# ============================================================
# HÀM TÍNH DUNG LƯỢNG HDFS (RECURSIVE)
# ============================================================
def get_hdfs_size(fs, Path, path_str, suffix_filter=None):
    """
    Trả về:
        total_size (bytes), file_count
    """
    total_size = 0
    file_count = 0

    try:
        if not fs.exists(Path(path_str)):
            return 0, 0

        # True = recursive (quét cả thư mục con)
        iterator = fs.listFiles(Path(path_str), True)
        
        while iterator.hasNext():
            status = iterator.next()
            path = status.getPath().toString()
            size = status.getLen()
            
            # Bỏ qua các file hệ thống của Spark/HDFS
            if "_SUCCESS" in path or path.endswith(".crc"):
                continue

            # Logic lọc đuôi file (linh hoạt hơn)
            if suffix_filter:
                # Nếu suffix_filter là list/tuple (ví dụ: ['.gz', '.gzip'])
                if isinstance(suffix_filter, (list, tuple)):
                    if not any(path.endswith(s) for s in suffix_filter):
                        continue
                # Nếu là string đơn
                elif not path.endswith(suffix_filter):
                    continue

            total_size += size
            file_count += 1
    except Exception as e:
        print(f"Lỗi khi quét đường dẫn {path_str}: {e}")
        return 0, 0

    return total_size, file_count


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    # --- SỬA LỖI NAMENODE: THÊM CẤU HÌNH EVENT LOG ---
    spark = SparkSession.builder \
        .appName("Check_Data_Size_Final") \
        .master("local[2]") \
        .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000") \
        .config("spark.eventLog.enabled", "true") \
        .config("spark.eventLog.dir", "hdfs://localhost:9000/spark-logs") \
        .config("spark.history.fs.logDirectory", "hdfs://localhost:9000/spark-logs") \
        .getOrCreate()

    sc = spark.sparkContext

    # Lấy đối tượng FileSystem từ JVM
    fs = sc._gateway.jvm.org.apache.hadoop.fs.FileSystem.get(
        sc._jsc.hadoopConfiguration()
    )
    Path = sc._gateway.jvm.org.apache.hadoop.fs.Path

    # --------------------------------------------------------
    # CẤU HÌNH ĐƯỜNG DẪN
    # --------------------------------------------------------
    RAW_PATH = "/data/raw/"
    PROCESSED_PATH = "/data/processed/clean_tweets"

    print("\n>>> ĐANG KIỂM TRA DUNG LƯỢNG DỮ LIỆU <<<\n")

    # --------------------------------------------------------
    # 1. DỮ LIỆU TRƯỚC ETL (RAW)
    # --------------------------------------------------------
    # Chấp nhận cả .gzip và .gz cho an toàn
    raw_size, raw_files = get_hdfs_size(
        fs, Path, RAW_PATH, suffix_filter=[".gzip", ".gz"]
    )

    # --------------------------------------------------------
    # 2. DỮ LIỆU SAU ETL (CLEAN)
    # --------------------------------------------------------
    processed_size, processed_files = get_hdfs_size(
        fs, Path, PROCESSED_PATH
    )

    # --------------------------------------------------------
    # IN KẾT QUẢ
    # --------------------------------------------------------
    GB = 1024 ** 3
    MB = 1024 ** 2

    print("=" * 70)
    print("DUNG LƯỢNG DỮ LIỆU TRƯỚC & SAU PREPROCESS")
    print("=" * 70)

    print(f"📥 RAW DATA")
    print(f"   • Số file       : {raw_files:,}")
    print(f"   • Tổng dung lượng: {raw_size / GB:,.2f} GB ({raw_size / MB:,.2f} MB)")

    print("-" * 70)

    print(f"📤 PROCESSED DATA (Parquet - Cleaned)")
    print(f"   • Số file       : {processed_files:,}")
    print(f"   • Tổng dung lượng: {processed_size / GB:,.2f} GB ({processed_size / MB:,.2f} MB)")

    print("-" * 70)

    if raw_size > 0:
        ratio = processed_size / raw_size * 100
        print(f"📊 TỶ LỆ DỮ LIỆU CÒN LẠI: {ratio:.2f}%")
        print(f"📉 ĐÃ GIẢM (Lọc rác + Nén): {100 - ratio:.2f}%")
    else:
        print("⚠️ Không tìm thấy dữ liệu Raw (hoặc sai đuôi file).")

    print("=" * 70)

    spark.stop()