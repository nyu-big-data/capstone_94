# from pyspark.sql import SparkSession
# from pyspark.sql.functions import col, expr
# from pyspark.ml.feature import MinHashLSH
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import MinHashLSH, VectorAssembler
from pyspark.sql.functions import lit, col, when, least, greatest

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
    print("Joined bout to get top 10")
    
    # Popularity-Based Recommendation: Top N Movies
    def get_top_n_movies(n=10):
        # Group by movie titles, calculate average rating, and order by rating descending
        top_movies = train_ratings_movies.groupBy("title") \
                                         .agg(avg("rating").alias("avg_rating")) \
                                         .orderBy(col("avg_rating").desc()) \
                                         .limit(n)
        return top_movies
    
    # Example usage
    top_10_movies = get_top_n_movies(10)
    print("Top 10 Popular Movies:")
    top_10_movies.show()
  
    
    
  

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
