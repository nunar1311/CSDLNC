#!/usr/bin/env python3
import sys
import os
import shutil
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, to_date
from pyspark.sql.types import DateType, FloatType, IntegerType
import argparse

def create_spark_session(app_name="Data Preprocessing"):
    """Create and return a Spark session."""
    return SparkSession.builder \
        .appName(app_name) \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()

def read_csv_to_dataframe(spark, file_path):
    """Read CSV file into a Spark DataFrame."""
    print(f"📥 Reading CSV file: {file_path}")
    df = spark.read.option("header", "true") \
        .option("inferSchema", "true") \
        .csv(file_path)
    df.printSchema()
    return df

def convert_data_types(df, student_id):
    """Convert columns to appropriate data types and detect formatting errors."""
    print("🔄 Converting and validating data types...")

    # Define condition to detect errors
    error_condition = (
        col("transaction_date").isNull() |
        (~col("quantity").cast(IntegerType()).isNotNull()) |
        (col("quantity").cast(IntegerType()) < 0) |
        (~col("unit_price").cast(FloatType()).isNotNull()) |
        (col("unit_price").cast(FloatType()) <= 0) |
        (~col("total_amount").cast(FloatType()).isNotNull()) |
        (col("customer_id") != f"STD_{student_id}")
    )

    df_with_error_flag = df.withColumn("has_error", error_condition)

    clean_df = df_with_error_flag.filter(~col("has_error"))
    error_df = df_with_error_flag.filter(col("has_error"))

    # Cast to correct types in clean data
    typed_df = clean_df.select(
        col("transaction_id"),
        to_date(col("transaction_date"), "yyyy-MM-dd").alias("transaction_date"),
        col("transaction_time"),
        col("customer_id"),
        col("product_id"),
        col("quantity").cast(IntegerType()).alias("quantity"),
        col("unit_price").cast(FloatType()).alias("unit_price"),
        col("total_amount").cast(FloatType()).alias("total_amount"),
        col("payment_method"),
        col("store_location")
    )

    return typed_df, error_df.drop("has_error")

def create_total_amount_column(df):
    """Ensure total_amount column is correct."""
    print("➕ Calculating total_amount if needed...")
    df_with_calc = df.withColumn(
        "calculated_total",
        col("quantity") * col("unit_price")
    )

    result_df = df_with_calc.withColumn(
        "total_amount",
        when(
            (col("total_amount").isNull()) |
            (col("total_amount") != col("calculated_total")),
            col("calculated_total")
        ).otherwise(col("total_amount"))
    ).drop("calculated_total")

    return result_df

def save_dataframes(clean_df, error_df, student_id):
    """Save cleaned and error data to CSV."""
    clean_output_dir = f"processed_transactions_{student_id}_temp"
    error_output_dir = f"bad_rows_{student_id}_temp"
    clean_final_file = f"processed_transactions_{student_id}.csv"
    error_final_file = f"bad_rows_{student_id}.csv"

    print("💾 Saving clean data...")
    clean_df.coalesce(1).write.option("header", "true").mode("overwrite").csv(clean_output_dir)

    print("💾 Saving error data...")
    error_df.coalesce(1).write.option("header", "true").mode("overwrite").csv(error_output_dir)

    copy_part_file(clean_output_dir, clean_final_file)
    copy_part_file(error_output_dir, error_final_file)

    try:
        shutil.rmtree(clean_output_dir)
        shutil.rmtree(error_output_dir)
    except Exception as e:
        print(f"⚠️  Warning: Could not remove temporary directories: {e}")

    return clean_final_file, error_final_file

def copy_part_file(directory, output_file):
    """Copy the output part file to a single named file."""
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
        print(f"✅ Output saved to: {output_file}")
    else:
        print(f"⚠️  Warning: No part file found in {directory}")

def analyze_results(spark, clean_path, error_path):
    """Print basic analysis of clean vs. error data."""
    clean_df = spark.read.option("header", "true").csv(clean_path)
    error_df = spark.read.option("header", "true").csv(error_path)

    clean_count = clean_df.count()
    error_count = error_df.count()
    total_count = clean_count + error_count
    error_percent = (error_count / total_count) * 100 if total_count > 0 else 0

    print("\n📊 --- Data Summary ---")
    print(f"🟢 Clean records : {clean_count}")
    print(f"🔴 Error records : {error_count}")
    print(f"📦 Total records : {total_count}")
    print(f"❗ Error Rate     : {error_percent:.2f}%")

def main():
    parser = argparse.ArgumentParser(description="Spark-based data preprocessor.")
    parser.add_argument("student_id", help="Student ID to identify input/output files.")
    args = parser.parse_args()

    student_id = args.student_id
    input_file = f"transactions_{student_id}.csv"

    if not os.path.exists(input_file):
        print(f"❌ Input file {input_file} not found.")
        print("💡 Run generate_data.py <student_id> to generate input first.")
        sys.exit(1)

    spark = create_spark_session()

    try:
        df = read_csv_to_dataframe(spark, input_file)
        clean_df, error_df = convert_data_types(df, student_id)
        clean_df = create_total_amount_column(clean_df)
        clean_path, error_path = save_dataframes(clean_df, error_df, student_id)
        analyze_results(spark, clean_path, error_path)
    finally:
        spark.stop()

if __name__ == "__main__":
    main()
