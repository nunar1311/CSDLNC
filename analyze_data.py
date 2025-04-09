#!/usr/bin/env python3
import sys
import os
import argparse
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import (
    col, sum as sql_sum, count, countDistinct,
    weekofyear, year, month, row_number, desc,
    lag, when, lit, date_format, concat
)
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

def create_spark_session():
    """Create and return a Spark session."""
    return SparkSession.builder \
        .appName("Transaction Data Analysis") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()

def load_data(spark, student_id):
    """Load the processed transaction data."""
    file_path = f"processed_transactions_{student_id}.csv"

    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        print("Please run the data preprocessing script first.")
        sys.exit(1)

    # Load the data with appropriate type conversion
    df = spark.read.option("header", "true") \
        .option("inferSchema", "true") \
        .csv(file_path)

    # Register the DataFrame as a SQL table
    df.createOrReplaceTempView("transactions")

    return df

def analyze_weekly_spending(spark, df, student_id):
    """
    Analyze total spending per week for each customer_id.
    """
    print("\n--- Weekly Spending Analysis ---")

    # Add year and week columns
    df_with_date = df.withColumn("year", year(col("transaction_date"))) \
                     .withColumn("week", weekofyear(col("transaction_date")))

    # Calculate weekly spending by customer
    weekly_spending = df_with_date.groupBy("customer_id", "year", "week") \
                                 .agg(sql_sum("total_amount").alias("weekly_spending")) \
                                 .orderBy("customer_id", "year", "week")

    # Show a sample of the results
    print("Sample of weekly spending by customer:")
    weekly_spending.show(10)

    # Count number of weeks with transactions for each customer
    customer_weeks = weekly_spending.groupBy("customer_id") \
                                   .count() \
                                   .withColumnRenamed("count", "weeks_with_transactions")

    # Show customer activity summary
    print("\nCustomer activity summary (by number of active weeks):")
    customer_weeks.groupBy("weeks_with_transactions") \
                 .count() \
                 .orderBy("weeks_with_transactions") \
                 .show(20)

    # Save the weekly spending data
    output_file = f"weekly_spending_{student_id}.csv"
    weekly_spending.coalesce(1).write \
        .option("header", "true") \
        .mode("overwrite") \
        .csv(f"weekly_spending_{student_id}_temp")

    # Copy the part file to the output location
    copy_part_file(f"weekly_spending_{student_id}_temp", output_file)

    return weekly_spending

def cluster_customer_behavior(spark, df, student_id):
    """
    Cluster customers based on their purchasing behavior:
    - Total number of orders
    - Total spending
    - Number of unique products purchased
    """
    print("\n--- Customer Behavior Clustering ---")

    # Calculate metrics for each customer
    customer_metrics = df.groupBy("customer_id").agg(
        count("transaction_id").alias("total_orders"),
        sql_sum("total_amount").alias("total_spending"),
        countDistinct("product_id").alias("unique_products")
    )

    # Show sample of customer metrics
    print("Sample of customer metrics:")
    customer_metrics.show(5)

    # Check for null values
    print("Checking for null values in customer metrics:")
    customer_metrics.select(
        [count(when(col(c).isNull(), c)).alias(c) for c in customer_metrics.columns]
    ).show()

    # Generate statistics to spot potential issues
    print("Statistics for clustering features:")
    customer_metrics.describe(["total_orders", "total_spending", "unique_products"]).show()

    # Remove any rows with null values
    clean_metrics = customer_metrics.na.drop()

    # Check if we lost any rows
    original_count = customer_metrics.count()
    clean_count = clean_metrics.count()
    if original_count != clean_count:
        print(f"Warning: Dropped {original_count - clean_count} rows with null values")

    # Scale features to handle potential outliers
    # This step is important for KMeans to work well
    from pyspark.ml.feature import StandardScaler

    # First assemble the features
    assembler = VectorAssembler(
        inputCols=["total_orders", "total_spending", "unique_products"],
        outputCol="raw_features"
    )

    # Assemble raw features
    assembled_data = assembler.transform(clean_metrics)

    # Apply standard scaling
    scaler = StandardScaler(
        inputCol="raw_features",
        outputCol="features",
        withStd=True,
        withMean=True
    )

    try:
        # Fit the scaler and transform the data
        scaler_model = scaler.fit(assembled_data)
        scaled_data = scaler_model.transform(assembled_data)

        print("Sample of scaled features:")
        scaled_data.select("customer_id", "raw_features", "features").show(5, truncate=False)

        # Try with a smaller range of K values first
        k_values = list(range(2, 6))  # Start with fewer clusters for testing

        # Try a single K value first as a sanity check
        test_k = 3
        print(f"Testing with a single K value = {test_k}")

        kmeans = KMeans().setK(test_k).setSeed(42).setFeaturesCol("features")
        model = kmeans.fit(scaled_data)
        print("Initial test clustering succeeded. Proceeding to find optimal K...")

        # Now try different K values to find optimal number
        silhouette_scores = []
        evaluator = ClusteringEvaluator(featuresCol="features")

        for k in k_values:
            print(f"Testing with k={k}")
            kmeans = KMeans().setK(k).setSeed(42).setFeaturesCol("features")
            model = kmeans.fit(scaled_data)
            predictions = model.transform(scaled_data)

            # Evaluate clustering
            silhouette = evaluator.evaluate(predictions)
            silhouette_scores.append(silhouette)
            print(f"Silhouette score with {k} clusters: {silhouette}")

        # Find best K (highest silhouette score)
        best_k = k_values[silhouette_scores.index(max(silhouette_scores))]
        print(f"Best number of clusters: {best_k} (Silhouette score: {max(silhouette_scores):.4f})")

        # Final clustering with best K
        kmeans = KMeans().setK(best_k).setSeed(42).setFeaturesCol("features")
        model = kmeans.fit(scaled_data)

        # Add cluster predictions to data
        predictions = model.transform(scaled_data)

        # Get cluster centers (convert to original scale if needed)
        centers = model.clusterCenters()
        print("\nCluster Centers (in standardized scale):")
        for i, center in enumerate(centers):
            print(f"Cluster {i}: {center}")

        # Analyze clusters
        cluster_analysis = predictions.groupBy("prediction").agg(
            count("customer_id").alias("customer_count"),
            sql_sum("total_orders").alias("sum_orders"),
            sql_sum("total_spending").alias("sum_spending"),
            sql_sum("unique_products").alias("sum_products")
        ).orderBy("prediction")

        print("\nCluster analysis:")
        cluster_analysis.show()

        # Add more detailed metrics for each cluster
        detailed_clusters = predictions.groupBy("prediction").agg(
            count("customer_id").alias("customer_count"),
            sql_sum("total_orders").alias("sum_orders"),
            sql_sum("total_spending").alias("sum_spending"),
            sql_sum("unique_products").alias("sum_products"),
            (sql_sum("total_orders") / count("customer_id")).alias("avg_orders"),
            (sql_sum("total_spending") / count("customer_id")).alias("avg_spending"),
            (sql_sum("unique_products") / count("customer_id")).alias("avg_products")
        ).orderBy("prediction")

        print("\nDetailed cluster metrics:")
        detailed_clusters.show()

        # Save clustering results
        output_file = f"customer_clusters_{student_id}.csv"
        predictions.select("customer_id", "total_orders", "total_spending", "unique_products", "prediction") \
            .coalesce(1) \
            .write \
            .option("header", "true") \
            .mode("overwrite") \
            .csv(f"customer_clusters_{student_id}_temp")

        # Copy the part file to the output location
        copy_part_file(f"customer_clusters_{student_id}_temp", output_file)

        return predictions

    except Exception as e:
        print(f"Error during clustering: {str(e)}")
        print("Attempting to diagnose the issue...")

        # Check for extreme values/outliers
        print("\nChecking for potential outliers:")
        for col_name in ["total_orders", "total_spending", "unique_products"]:
            customer_metrics.select(col_name).summary(
                "min", "1%", "5%", "25%", "50%", "75%", "95%", "99%", "max"
            ).show()

        # Try a very simple alternative approach
        print("\nAttempting simplified clustering with 2 clusters only...")
        try:
            # Try with just 2 clusters and default parameters
            simple_kmeans = KMeans().setK(2).setSeed(42)
            simple_model = simple_kmeans.fit(assembled_data)
            simple_predictions = simple_model.transform(assembled_data)

            # Save these simplified results
            output_file = f"customer_clusters_{student_id}.csv"
            simple_predictions.select("customer_id", "total_orders", "total_spending", "unique_products", "prediction") \
                .coalesce(1) \
                .write \
                .option("header", "true") \
                .mode("overwrite") \
                .csv(f"customer_clusters_{student_id}_temp")

            # Copy the part file to the output location
            copy_part_file(f"customer_clusters_{student_id}_temp", output_file)

            print("Simplified clustering completed successfully!")
            return simple_predictions

        except Exception as e2:
            print(f"Simplified clustering also failed: {str(e2)}")
            print("Saving customer metrics without clustering...")

            # Save just the metrics without clustering
            output_file = f"customer_metrics_{student_id}.csv"
            customer_metrics.coalesce(1).write \
                .option("header", "true") \
                .mode("overwrite") \
                .csv(f"customer_metrics_{student_id}_temp")

            # Copy the part file to the output location
            copy_part_file(f"customer_metrics_{student_id}_temp", output_file)

            print(f"Customer metrics saved to: {output_file}")
            return customer_metrics

def find_declining_customers(spark, df, student_id):
    """
    Find customers with consistently declining orders in the last 3 months.
    """
    print("\n--- Customers with Declining Orders ---")

    # Create a SQL query to find customers with declining orders
    query = """
    WITH monthly_orders AS (
        SELECT
            customer_id,
            YEAR(transaction_date) AS year,
            MONTH(transaction_date) AS month,
            COUNT(transaction_id) AS order_count
        FROM transactions
        GROUP BY customer_id, YEAR(transaction_date), MONTH(transaction_date)
    ),

    ordered_months AS (
        SELECT
            customer_id,
            year,
            month,
            order_count,
            ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY year DESC, month DESC) AS month_rank
        FROM monthly_orders
    ),

    last_three_months AS (
        SELECT
            customer_id,
            year,
            month,
            order_count,
            month_rank,
            LAG(order_count, 1) OVER (PARTITION BY customer_id ORDER BY year, month) AS prev_month_orders,
            LAG(order_count, 2) OVER (PARTITION BY customer_id ORDER BY year, month) AS prev_prev_month_orders
        FROM ordered_months
        WHERE month_rank <= 3
    )

    SELECT
        customer_id,
        CONCAT(year, '-', month) AS year_month,
        order_count,
        month_rank
    FROM last_three_months
    WHERE month_rank = 1
    AND order_count < prev_month_orders
    AND prev_month_orders < prev_prev_month_orders
    ORDER BY customer_id
    """

    # Execute the query
    declining_customers = spark.sql(query)

    # Show results
    print("Customers with declining orders in the last 3 months:")
    declining_customers.show()

    # Count the number of customers with declining orders
    declining_count = declining_customers.count()
    print(f"Number of customers with declining orders: {declining_count}")

    # Alternative approach using DataFrame API
    print("\nVerifying with DataFrame API:")

    # Add year and month columns
    df_with_date = df.withColumn("year", year(col("transaction_date"))) \
                     .withColumn("month", month(col("transaction_date")))

    # Calculate monthly orders by customer
    monthly_orders = df_with_date.groupBy("customer_id", "year", "month") \
                               .agg(count("transaction_id").alias("order_count"))

    # Create window for sorting by date (most recent first)
    window_desc = Window.partitionBy("customer_id").orderBy(col("year").desc(), col("month").desc())

    # Rank months and get last 3 months
    ranked_months = monthly_orders.withColumn("month_rank", row_number().over(window_desc))
    last_three_months = ranked_months.filter(col("month_rank") <= 3)

    # Create window for analyzing order trend (oldest to newest)
    window_asc = Window.partitionBy("customer_id").orderBy("year", "month")

    # Calculate previous months' orders
    with_prev_months = last_three_months.withColumn("prev_month_orders",
                                                 lag("order_count", 1).over(window_asc)) \
                                      .withColumn("prev_prev_month_orders",
                                                 lag("order_count", 2).over(window_asc))

    # Find customers with declining orders in last 3 months
    declining = with_prev_months.filter(
        (col("month_rank") == 1) &
        (col("order_count") < col("prev_month_orders")) &
        (col("prev_month_orders") < col("prev_prev_month_orders"))
    )

    # Show results from DataFrame API approach
    declining.withColumn("year_month", concat(col("year"), lit("-"), col("month"))) \
            .select("customer_id", "year_month", "order_count", "month_rank") \
            .show()

    # Save the declining customers data
    output_file = f"declining_customers_{student_id}.csv"
    declining.select("customer_id", "year", "month", "order_count",
                   "prev_month_orders", "prev_prev_month_orders") \
            .coalesce(1) \
            .write \
            .option("header", "true") \
            .mode("overwrite") \
            .csv(f"declining_customers_{student_id}_temp")

    # Copy the part file to the output location
    copy_part_file(f"declining_customers_{student_id}_temp", output_file)

    return declining

def copy_part_file(directory, output_file):
    """Copy the part file to the desired output file."""
    # Find the part file (should be only one since we used coalesce(1))
    part_file = None
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.startswith("part-") and file.endswith(".csv"):
                part_file = os.path.join(root, file)
                break
        if part_file:
            break

    if part_file:
        # Copy the file to the output location
        import shutil
        shutil.copyfile(part_file, output_file)
        print(f"Saved data to: {output_file}")

        # Clean up temporary directory
        try:
            shutil.rmtree(directory)
        except Exception as e:
            print(f"Warning: Could not remove temporary directory {directory}: {e}")
    else:
        print(f"Warning: No part file found in {directory}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Analyze transaction data.")
    parser.add_argument("student_id", help="Student ID to identify the input files.")
    args = parser.parse_args()

    # Create Spark session
    spark = create_spark_session()

    try:
        # Load data
        df = load_data(spark, args.student_id)

        # Part 1: Weekly spending analysis
        weekly_spending = analyze_weekly_spending(spark, df, args.student_id)

        # Part 2: Customer behavior clustering
        clusters = cluster_customer_behavior(spark, df, args.student_id)

        # Part 3: Find customers with declining orders
        declining = find_declining_customers(spark, df, args.student_id)

        print("\n--- Analysis Complete ---")
        print(f"Weekly spending data saved to: weekly_spending_{args.student_id}.csv")
        print(f"Customer clusters saved to: customer_clusters_{args.student_id}.csv")
        print(f"Declining customers saved to: declining_customers_{args.student_id}.csv")

    finally:
        # Stop Spark session
        spark.stop()

if __name__ == "__main__":
    main()