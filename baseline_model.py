# from pyspark.sql import SparkSession
# from pyspark.sql.functions import col, expr
# from pyspark.ml.feature import MinHashLSH
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import MinHashLSH, VectorAssembler
from pyspark.sql.functions import lit, col, when, least, greatest, avg
from pyspark.sql.functions import col, collect_list, lit, udf, expr
from pyspark.sql.types import ArrayType, IntegerType, FloatType, DoubleType, StructType, StructField

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import BucketedRandomProjectionLSH, MinHashLSHModel
from pyspark.ml.feature import HashingTF, IDF, Tokenizer

import random
from pyspark.sql.functions import monotonically_increasing_id

import os

def main(spark, userID):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    userID : string, userID of student to find files in HDFS
    '''
    print('Dataframe loading and SQL query')

      # Load the data
    path = f'hdfs:/user/{userID}/'
    train = spark.read.csv(path + 'train.csv', header=True, inferSchema=True)
    print("Got Train")
    movies = spark.read.csv(path + 'movies.csv', header=True, inferSchema=True)
    print("Schema of train DataFrame:")
    train.printSchema()
    
    print("Schema of movies DataFrame:")
    movies.printSchema()
    
    print("Bouta Join")

    # Merge ratings with movie information on 'movieId'
    train_ratings_movies = train.join(movies.select("movieId", "title"), on="movieId", how="inner")
    print("Joined bout to get top 30")
    
    # Popularity-Based Recommendation: Top N Movies
    def get_top_n_movies(n=10):
        # Group by movie titles, calculate average rating, and order by rating descending
        top_movies = train_ratings_movies.groupBy("title") \
                                         .agg(avg("rating").alias("avg_rating")) \
                                         .orderBy(col("avg_rating").desc()) \
                                         .limit(n).orderBy(col("title"))
        return top_movies
    
    # Example usage
    top_30_movies = get_top_n_movies(30)
    print("Top 30 Popular Movies:")
    top_30_movies.show()

    # Evaluation of Popularity Based Model using MAP/Mean Average Prediction
    test_df = spark.read.csv(path + "test.csv", header=True, inferSchema=True)
    test_df.printSchema()

    # Get top N movieIds for recommendations
    top_30_movieIds = [row.movieId for row in top_30_movies.collect()]

    # Function to calculate Average Precision (AP)
    def average_precision(relevant_items, recommended_items):
        if not relevant_items:
            return 0.0
            
        relevant_set = set(relevant_items)
        recommended_set = set(recommended_items)
    
        if not recommended_set:
            return 0.0

        hit_count = 0
        sum_precisions = 0.0
        for i, item in enumerate(recommended_items):
            if item in relevant_set:
                hit_count += 1
                precision_at_i = hit_count / (i + 1)
                sum_precisions += precision_at_i
    
        return sum_precisions / len(relevant_set)

    average_precision_udf = udf(average_precision, DoubleType())

    # Create DataFrames with relevant and recommended items per user
    relevant_items_per_user = test_df.filter(col("rating") >= 4) \
                                     .groupBy("userId") \
                                     .agg(collect_list("movieId").alias("relevant_items"))

    recommended_items_per_user = test_df.select("userId").distinct() \
                                        .withColumn("recommended_items", lit(top_30_movieIds))

    # Join relevant and recommended items
    precision_recall_df = relevant_items_per_user.join(recommended_items_per_user, on="userId", how="inner")

    # Calculate Average Precision for each user
    precision_recall_df = precision_recall_df.withColumn("average_precision", average_precision_udf(col("relevant_items"), col("recommended_items")))

    # Calculate MAP by averaging the Average Precision of all users
    map_value = precision_recall_df.agg(avg("average_precision")).first()[0]

    print(f"Mean Average Precision (MAP): {map_value}")
    
  

if __name__ == "__main__":
    spark = SparkSession.builder.appName('capstone').getOrCreate()
    # spark = SparkSession.builder \
    # .appName('capstone') \
    # .config('spark.executor.memory', '4g') \
    # .config('spark.driver.memory', '4g') \
    # .config('spark.sql.shuffle.partitions', '100') \
    # .config('spark.executor.memoryOverhead', '512m') \
    # .getOrCreate()


    
    userID = os.getenv('USER', 'default_user')  # Default user if not set
    main(spark, userID)
