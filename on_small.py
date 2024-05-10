#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Starter PySpark Script for students to complete for their Lab 3 Assignment.
Usage:
    $ spark-submit lab_3_starter_code.py
'''
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import collect_list
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.ml.feature import MinHashLSH
from pyspark.sql.types import ArrayType, IntegerType
from pyspark.sql.functions import udf


def main(spark, userID):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    userID : string, userID of student to find files in HDFS
    '''
    print('Dataframe loading and SQL query')

    # Load the data
    ratings = spark.read.csv(f'hdfs:/user/{userID}/ratings.csv', 
                             schema='userId INT, movieId INT, rating FLOAT, timestamp INT')
    
    # Obtain a distinct list of movies and map each movieId to an index
    movies = ratings.select("movieId").distinct().collect()
    movie_index = {row.movieId: idx for idx, row in enumerate(movies)}
    
    # UDF to convert list of movieIds to sparse vector
    # def movies_to_vector(movies):
    #     indices = [movie_index[movie] for movie in movies if movie in movie_index]
    #     values = [1] * len(indices)  # All elements are 1 (presence of the movie)
    #     return Vectors.sparse(len(movie_index), indices, values)
    
    # movies_to_vector_udf = udf(movies_to_vector, Vectors.typeName())
    # UDF to convert list of movieIds to sparse vector
    def movies_to_vector(movies):
        indices = [movie_index[movie] for movie in movies if movie in movie_index]
        values = [1] * len(indices)  # All elements are 1 (presence of the movie)
        return Vectors.sparse(len(movie_index), indices, values)
    
    # Register the UDF with the correct return type
    movies_to_vector_udf = udf(movies_to_vector, VectorUDT())

    # Group by userId and aggregate movieIds into a list, then convert to vector
    user_movies = ratings.groupBy("userId").agg(collect_list("movieId").alias("movies"))
    user_movies = user_movies.withColumn("features", movies_to_vector_udf("movies"))

    # Prepare MinHashLSH with the new vector column
    mh = MinHashLSH(inputCol="features", outputCol="hashes", numHashTables=3)
    model = mh.fit(user_movies)

    # Find similarity using MinHashLSH
    similar_users = model.approxSimilarityJoin(user_movies, user_movies, threshold=0.6, distCol="JaccardDistance")
    top_pairs = similar_users.orderBy("JaccardDistance").select("datasetA.userId", "datasetB.userId", "JaccardDistance").limit(100)
    top_pairs.show()

    print('Printing ratings with specified schema')
    ratings.printSchema()
    ratings.createOrReplaceTempView('ratings')
    query = spark.sql('SELECT * FROM ratings')
    query.show()

# Only enter this block if we're in main
if __name__ == "__main__":
    # Create the spark session object
    spark = SparkSession.builder.appName('capstone').getOrCreate()

    # Get user userID from the command line/environment
    userID = os.getenv('USER', 'default_user')  # Default user if not set

    # Call our main routine
    main(spark, userID)
