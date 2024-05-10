#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''Starter Pyspark Script for students to complete for their Lab 3 Assignment.
Usage:
    $ spark-submit lab_3_starter_code.py
'''
import os

# And pyspark.sql to get the spark session
# spark-submit --deploy-mode client on_small.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, countDistinct, first, count, max, avg
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
    ratings = spark.read.csv(f'hdfs:/user/{userID}/ratings.csv', 
                             schema='userId INT, movieId INT, rating FLOAT, timestamp INT')
    # ratings = spark.read.csv(f'hdfs:/user/{userID}/ratings.csv', header=True, inferSchema=True)

  
  
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
