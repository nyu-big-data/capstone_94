from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, collect_list, lit, udf
from pyspark.sql.types import ArrayType, IntegerType, DoubleType
import os

def main(spark, userID):
    # Load the data
    path = f'hdfs:/user/{userID}/'
    train = spark.read.csv(path + 'train.csv', header=True, inferSchema=True)
    movies = spark.read.csv(path + 'movies.csv', header=True, inferSchema=True)
    test_df = spark.read.csv(path + "test.csv", header=True, inferSchema=True)
    print("DF retrieved")

    # Merge ratings with movie information on 'movieId'
    train_ratings_movies = train.join(movies.select("movieId", "title"), on="movieId", how="inner")
    
    # Function to calculate the top N movies based on average ratings
    def get_top_n_movies(n=10):
        return train_ratings_movies.groupBy("movieId", "title") \
                                   .agg(avg("rating").alias("avg_rating")) \
                                   .orderBy(col("avg_rating").desc()) \
                                   .limit(n)

    # Get top 30 popular movies
    top_30_movies = get_top_n_movies(30)
    top_30_movieIds = [row['movieId'] for row in top_30_movies.collect()]
    print("Top ID Retrived")

    # Create a DataFrame of recommended items for each user
    recommended_items_per_user = test_df.select("userId").distinct()
    recommended_items_per_user = recommended_items_per_user.withColumn("recommended_items", lit(str(top_30_movieIds)))

    # Define a UDF to convert string back to list (since lit cannot handle lists directly)
    def to_list(s):
        return eval(s)

    to_list_udf = udf(to_list, ArrayType(IntegerType()))
    print("To LIst")

    recommended_items_per_user = recommended_items_per_user.withColumn("recommended_items", to_list_udf(col("recommended_items")))

    # Function to calculate Average Precision (AP)
    def average_precision(relevant_items, recommended_items):
        relevant_set = set(relevant_items)
        recommended_set = set(recommended_items)
        hit_count = 0
        sum_precisions = 0.0
        for i, item in enumerate(recommended_items):
            if item in relevant_set:
                hit_count += 1
                precision_at_i = hit_count / (i + 1)
                sum_precisions += precision_at_i
        return sum_precisions / len(relevant_set) if relevant_set else 0.0

    average_precision_udf = udf(average_precision, DoubleType())
    print("Calculating Precision")

    # Collecting relevant items per user
    relevant_items_per_user = test_df.filter(col("rating") >= 4) \
                                     .groupBy("userId") \
                                     .agg(collect_list("movieId").alias("relevant_items"))

    # Join to compare relevant and recommended items
    precision_recall_df = relevant_items_per_user.join(recommended_items_per_user, "userId", "inner")
    precision_recall_df = precision_recall_df.withColumn(
        "average_precision",
        average_precision_udf(col("relevant_items"), col("recommended_items"))
    )

    # Calculate Mean Average Precision
    map_value = precision_recall_df.agg(avg("average_precision")).first()[0]
    print(f"Mean Average Precision (MAP): {map_value}")

if __name__ == "__main__":
    spark = SparkSession.builder.appName('Popularity Based Recommendation Model').getOrCreate()
    userID = os.getenv('USER', 'default_user')  # Default user if not set
    main(spark, userID)
