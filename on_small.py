#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Starter Pyspark Script for students to complete for their Lab 3 Assignment.
Usage:
    $ spark-submit lab_3_starter_code.py
'''
import os

# And pyspark.sql to get the spark session
##### spark-submit --deploy-mode client on_small.py #######
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, countDistinct, first, count, max, avg
from pyspark.sql.functions import collect_set
from pyspark.sql.functions import collect_list
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import MinHashLSH
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
# Group data by user and collect rated movieIds into a set
import pyspark.sql.functions as F

def main(spark, userID):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    userID : string, userID of student to find files in HDFS
    '''
    print('Dataframe loading and SQL query')

    # Load the boats.txt and sailors.json data into DataFrame
    
    
    # Load the data
    ratings = spark.read.csv(f'hdfs:/user/{userID}/ratings.csv', 
                             schema='userId INT, movieId INT, rating FLOAT, timestamp INT')
    
    # Aggregate the movieIds into a list for each user
    user_movies = ratings.groupBy("userId").agg(collect_list("movieId").alias("movies"))
    
    # Use CountVectorizer to convert the list of movieIds to feature vectors
    cv = CountVectorizer(inputCol="movies", outputCol="features")
    cv_model = cv.fit(user_movies)
    features_df = cv_model.transform(user_movies)
    
    # Apply MinHashLSH
    mh = MinHashLSH(inputCol="features", outputCol="hashes", numHashTables=5)
    model = mh.fit(features_df)
    similar_users = model.approxSimilarityJoin(features_df, features_df, threshold=0.6, distCol="JaccardDistance")
    
    top_pairs = similar_users.orderBy("JaccardDistance").select("datasetA.userId", "datasetB.userId", "JaccardDistance").limit(100)
    top_pairs.show()


  
  
    print('Printing ratings with specified schema')
    ratings.printSchema()
    ratings.createOrReplaceTempView('boats')

    query = spark.sql('SELECT * FROM ratings')

    # Print the results to the console
    query.show()



# Only enter this block if we're in main
if __name__ == "__main__":
    # Create the spark session object
    spark = SparkSession.builder.appName('capstone').getOrCreate()
    # Get user userID from the command l
    # We need this to access the user's folder in HDFS
    userID = os.environ['USER']

    # Call our main routine
    main(spark, userID)
