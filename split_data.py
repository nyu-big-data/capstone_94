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
    ratings = spark.read.csv(path + 'ratings.csv', header=True, inferSchema=True)
    print("Splitting Train")
    train, test = ratings.randomSplit([0.7, 0.3], seed=0)
    print("Splitting Validation")
    train, validation = train.randomSplit([0.8, 0.2], seed=0)
    
    train.write.option("header", "true").csv(path + "train.csv", mode="overwrite")
    validation.write.option("header", "true").csv(path + "validation.csv", mode="overwrite")
    test.write.option("header", "true").csv(path + "test.csv", mode="overwrite")

    
  

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
