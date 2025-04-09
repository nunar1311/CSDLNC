#!/usr/bin/env python3
import os
import sys
import argparse
import shutil
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit, mean, stddev, percentile_approx, expr, when

def create_spark_session():
    """Create and return a Spark session."""
    return SparkSession.builder \
        .appName("Anomaly Transaction Detection") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()

def load_data(spark, student_id):
    """Load the processed transaction data."""
    file_path = f"processed_transactions_{student_id}.csv"

    if not os.path.exists(file_path):
        print(f"Lỗi: File {file_path} không tìm thấy.")
        print("Hãy chạy script tiền xử lý dữ liệu trước.")
        sys.exit(1)

    # Load the data with appropriate type conversion
    df = spark.read.option("header", "true") \
        .option("inferSchema", "true") \
        .csv(file_path)

    return df

def detect_anomalies_zscore(df, threshold=3.0):
    """
    Detect anomalies using Z-score method.
    A transaction is anomalous if its Z-score is above the threshold.
    """
    print("\n--- Phát hiện giao dịch bất thường (Z-score) ---")

    # Calculate statistics
    stats = df.select(
        mean("total_amount").alias("mean"),
        stddev("total_amount").alias("stddev")
    ).collect()[0]

    mean_val = stats["mean"]
    stddev_val = stats["stddev"]

    print(f"Giá trị trung bình total_amount: {mean_val:.2f}")
    print(f"Độ lệch chuẩn total_amount: {stddev_val:.2f}")
    print(f"Ngưỡng Z-score sử dụng: {threshold}")

    # Calculate Z-score for each transaction
    df_with_zscore = df.withColumn(
        "zscore",
        (col("total_amount") - lit(mean_val)) / lit(stddev_val)
    )

    # Identify anomalies
    anomalies = df_with_zscore.filter(col("zscore").abs() > threshold)

    print(f"Số lượng giao dịch bất thường (theo Z-score): {anomalies.count()}")

    # Show some examples
    print("Các ví dụ về giao dịch bất thường:")
    anomalies.orderBy(col("zscore").desc()).select(
        "transaction_id", "transaction_date", "customer_id", "total_amount", "zscore"
    ).show(10)

    return anomalies

def detect_anomalies_iqr(df, multiplier=1.5):
    """
    Detect anomalies using IQR method.
    A transaction is anomalous if:
    total_amount > Q3 + multiplier * IQR or total_amount < Q1 - multiplier * IQR
    """
    print("\n--- Phát hiện giao dịch bất thường (IQR) ---")

    # Calculate quartiles (using an approximation for better performance)
    quartiles = df.select(
        percentile_approx("total_amount", 0.25).alias("Q1"),
        percentile_approx("total_amount", 0.5).alias("median"),
        percentile_approx("total_amount", 0.75).alias("Q3")
    ).collect()[0]

    q1 = quartiles["Q1"]
    median = quartiles["median"]
    q3 = quartiles["Q3"]
    iqr = q3 - q1

    # Calculate thresholds
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    print(f"Q1 (25th percentile): {q1:.2f}")
    print(f"Median: {median:.2f}")
    print(f"Q3 (75th percentile): {q3:.2f}")
    print(f"IQR: {iqr:.2f}")
    print(f"Ngưỡng dưới: {lower_bound:.2f}")
    print(f"Ngưỡng trên: {upper_bound:.2f}")

    # Identify anomalies and add flags
    df_with_bounds = df.withColumn(
        "is_anomaly",
        (col("total_amount") < lower_bound) | (col("total_amount") > upper_bound)
    ).withColumn(
        "lower_bound",
        lit(lower_bound)
    ).withColumn(
        "upper_bound",
        lit(upper_bound)
    )

    # Filter to get only anomalies
    anomalies = df_with_bounds.filter(col("is_anomaly"))

    print(f"Số lượng giao dịch bất thường (theo IQR): {anomalies.count()}")

    # Show some examples of high anomalies
    print("Ví dụ về giao dịch bất thường (giá trị cao):")
    high_anomalies = anomalies.filter(col("total_amount") > upper_bound)
    high_anomalies.orderBy(col("total_amount").desc()).select(
        "transaction_id", "transaction_date", "customer_id", "total_amount"
    ).show(5)

    # Show some examples of low anomalies if any
    if anomalies.filter(col("total_amount") < lower_bound).count() > 0:
        print("Ví dụ về giao dịch bất thường (giá trị thấp):")
        low_anomalies = anomalies.filter(col("total_amount") < lower_bound)
        low_anomalies.orderBy(col("total_amount")).select(
            "transaction_id", "transaction_date", "customer_id", "total_amount"
        ).show(5)

    return anomalies

def detect_anomalies_by_median(df, multiplier=5.0):
    """
    Detect anomalies using median method.
    A transaction is anomalous if its value is > multiplier * median.
    """
    print("\n--- Phát hiện giao dịch bất thường (Phương pháp trung vị) ---")

    # Calculate median
    median_row = df.select(percentile_approx("total_amount", 0.5).alias("median")).collect()[0]
    median_val = median_row["median"]

    # Calculate threshold
    threshold = median_val * multiplier

    print(f"Giá trị trung vị: {median_val:.2f}")
    print(f"Hệ số nhân: {multiplier}")
    print(f"Ngưỡng phát hiện: {threshold:.2f}")

    # Identify anomalies
    anomalies = df.withColumn(
        "threshold",
        lit(threshold)
    ).withColumn(
        "is_anomaly",
        col("total_amount") > threshold
    ).filter(col("is_anomaly"))

    print(f"Số lượng giao dịch bất thường (theo trung vị): {anomalies.count()}")

    # Show some examples
    print("Các ví dụ về giao dịch bất thường:")
    anomalies.orderBy(col("total_amount").desc()).select(
        "transaction_id", "transaction_date", "customer_id", "total_amount", "threshold"
    ).show(10)

    return anomalies

def save_anomalies(df, student_id, method_name):
    """Save anomalies to CSV file."""
    output_dir = f"suspect_transactions_{student_id}_temp"
    output_file = f"suspect_transactions_{student_id}.csv"

    print(f"\nĐang lưu kết quả phát hiện bất thường ({method_name})...")

    # Select columns to save
    columns_to_save = [
        "transaction_id", "transaction_date", "transaction_time",
        "customer_id", "product_id", "quantity", "unit_price",
        "total_amount", "payment_method", "store_location"
    ]

    # Add method-specific columns if they exist
    for col_name in df.columns:
        if col_name not in columns_to_save and col_name not in ["is_anomaly"]:
            if col_name in ["zscore", "threshold", "lower_bound", "upper_bound"]:
                columns_to_save.append(col_name)

    # Save to a temporary directory with a single partition
    df.select(*columns_to_save).coalesce(1).write \
        .option("header", "true") \
        .mode("overwrite") \
        .csv(output_dir)

    # Copy the part file to final output
    copy_part_file(output_dir, output_file)

    return output_file

def copy_part_file(directory, output_file):
    """Copy the first part file found to the output file."""
    part_file = None
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith("part-") and file.endswith(".csv"):
                part_file = os.path.join(root, file)
                break
        if part_file:
            break

    if part_file:
        shutil.copyfile(part_file, output_file)
        print(f"✅ Kết quả đã được lưu vào: {output_file}")

        # Clean up temporary directory
        try:
            shutil.rmtree(directory)
        except Exception as e:
            print(f"⚠️ Cảnh báo: Không thể xóa thư mục tạm {directory}: {e}")
    else:
        print(f"⚠️ Cảnh báo: Không tìm thấy file part trong {directory}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Phát hiện giao dịch bất thường.")
    parser.add_argument("student_id", help="ID học sinh để xác định file đầu vào/đầu ra.")
    parser.add_argument("--method", choices=["zscore", "iqr", "median"], default="median",
                        help="Phương pháp phát hiện bất thường: zscore, iqr, hoặc median (mặc định: median)")
    parser.add_argument("--threshold", type=float, default=3.0,
                        help="Ngưỡng Z-score (mặc định: 3.0)")
    parser.add_argument("--multiplier", type=float, default=1.5,
                        help="Hệ số nhân IQR (mặc định: 1.5) hoặc hệ số nhân trung vị (mặc định: 5.0 cho phương pháp median)")
    args = parser.parse_args()

    # Adjust multiplier for median method if not explicitly set
    if args.method == "median" and args.multiplier == 1.5:
        args.multiplier = 5.0

    # Create Spark session
    spark = create_spark_session()

    try:
        # Load data
        df = load_data(spark, args.student_id)

        # Print basic statistics of total_amount
        print("\n--- Thống kê cơ bản về total_amount ---")
        df.select("total_amount").summary().show()

        # Detect anomalies using the specified method
        if args.method == "zscore":
            anomalies = detect_anomalies_zscore(df, args.threshold)
            method_name = f"Z-score (ngưỡng = {args.threshold})"
        elif args.method == "iqr":
            anomalies = detect_anomalies_iqr(df, args.multiplier)
            method_name = f"IQR (hệ số = {args.multiplier})"
        else:  # median
            anomalies = detect_anomalies_by_median(df, args.multiplier)
            method_name = f"Trung vị (hệ số = {args.multiplier})"

        # Save results
        output_file = save_anomalies(anomalies, args.student_id, method_name)

        print("\n--- Hoàn thành phát hiện giao dịch bất thường ---")
        print(f"Phương pháp: {method_name}")
        print(f"Số lượng giao dịch bất thường: {anomalies.count()}")
        print(f"Kết quả đã được lưu vào: {output_file}")

    finally:
        # Stop Spark session
        spark.stop()

if __name__ == "__main__":
    main()
    