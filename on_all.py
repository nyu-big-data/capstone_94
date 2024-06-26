# from pyspark.sql import SparkSession
# from pyspark.sql.functions import col, expr
# from pyspark.ml.feature import MinHashLSH
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import MinHashLSH, VectorAssembler
from pyspark.sql.functions import lit, col, when, least, greatest, avg

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import BucketedRandomProjectionLSH, MinHashLSHModel
from pyspark.ml.feature import HashingTF, IDF, Tokenizer

from pyspark.sql.functions import rand


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
    t_path = path + "top_100_pairs.csv"
    print(t_path)
    ratings = spark.read.csv(path + 'ratings.csv', header=True, inferSchema=True)
    tags = spark.read.csv(path + 'tags.csv', header=True, inferSchema=True)
    movies = spark.read.csv(path + 'movies.csv', header=True, inferSchema=True)
    links = spark.read.csv(path + 'links.csv', header=True, inferSchema=True)

    print("Data Loaded")
  
    # Merging DataFrames
    movie_ratings = ratings.join(movies.select("movieId", "title"), "movieId", "inner")
    movie_tags = tags.join(movies.select("movieId", "title"), "movieId", "inner")

    print("Finished joining")

    rate_history = movie_ratings.union(movie_tags)

    print("Joining")
  
    # Pivot table to get user-movie matrix
    rate_history_pt = rate_history.groupBy("userId").pivot("title").agg(lit(1)).na.fill(0)
    tokenizer = Tokenizer(inputCol="title", outputCol="tokens")
    print("Finished Tokenizing")
    hashingTF = HashingTF(numFeatures=1024, inputCol="tokens", outputCol="features")

    print("Transform")
    rate_history_tf = hashingTF.transform(tokenizer.transform(rate_history))

    print("miHash Fitting")
    
    mh = MinHashLSH(inputCol="features", outputCol="hashes", numHashTables=5)
    model = mh.fit(rate_history_tf)

    
    print("Finished fitting start transforming")
    # Transform features into binary hash buckets
    rate_history_hashed = model.transform(rate_history_tf)
    
    # Find similar pairs
    print("Approximate Similarity Join")
    similar = model.approxSimilarityJoin(rate_history_hashed, rate_history_hashed, 0.6, distCol="JaccardDistance")

    print("Filtering on Similar")
    similar = similar.withColumn("userId1", least(col("datasetA.userId"), col("datasetB.userId"))).withColumn("userId2", greatest(col("datasetA.userId"), col("datasetB.userId")))

    similar = similar.filter("datasetA.userId != datasetB.userId")
    similar = similar.dropDuplicates(["userId1", "userId2"])
    similar = similar.withColumnRenamed("userId", "userIdA").withColumnRenamed("userId", "userIdB")
    print("Continue filtering")
    top_100 = similar.select(col("datasetA.userId").alias("userId1"), 
                                        col("datasetB.userId").alias("userId2"), 
                                        col("JaccardDistance")).orderBy("JaccardDistance", ascending=False).limit(100)
    # top_100.show()
    # top_100.write.csv(path + "top_100_pairs.csv", mode="overwrite")
    # top_100.write.csv("top_100_pairs.csv", mode="overwrite")

    print("Start selecting random pairs")
    # Sample 200 users (since some might be duplicates, we sample slightly more than needed)
    random_100 = similar.orderBy(rand()).limit(100)

    print("100 pairs selected")
    
    avg_jaccard_top_100 = top_100.agg(avg("JaccardDistance")).first()[0]
    print("Average Jaccard Distance for Top 100", avg_jaccard_top_100)
    avg_jaccard_random_100 = random_100.agg(avg("JaccardDistance")).first()[0]
    print("Average Jaccard Distance for Random 100", avg_jaccard_random_100)
    print("Done")
        
    
    # Create DataFrame from these pairs

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
