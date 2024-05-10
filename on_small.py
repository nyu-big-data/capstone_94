# from pyspark.sql import SparkSession
# from pyspark.sql.functions import col, expr
# from pyspark.ml.feature import MinHashLSH
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import MinHashLSH, VectorAssembler

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
    # Assuming 'ratings' has been loaded
    # Filter out movies rated by fewer users to reduce dimensionality
    ratings = spark.read.csv(f'hdfs:/user/{userID}/ratings.csv', 
                             schema='userId INT, movieId INT, rating FLOAT, timestamp INT')
    
    movie_counts = ratings.groupBy("movieId").count()
    popular_movies = movie_counts.filter(col("count") > 20)  # Define a suitable threshold
    
    # Join with ratings to filter out less popular movies
    filtered_ratings = ratings.join(popular_movies, "movieId", "inner")
    
    # Continue with your pivot and vector assembly as before
    user_movie_matrix = filtered_ratings.groupBy("userId").pivot("movieId").count().na.fill(0)

    

    # Create a user-movie matrix where each entry is 1 if the user rated the movie
    # user_movie_matrix = ratings.groupBy("userId").pivot("movieId").count().na.fill(0)

    print('Post groupby')

    # Convert to vector. Each row is a user's vector with entries for each movie
    # Using '1' if rated, otherwise '0'
    assembler = VectorAssembler(inputCols=[col for col in user_movie_matrix.columns if col != "userId"],
                                outputCol="features")
    user_features = assembler.transform(user_movie_matrix)

    print('Assembler transformed')

    # Prepare MinHashLSH with the new vector column
    mh = MinHashLSH(inputCol="features", outputCol="hashes", numHashTables=3)
    model = mh.fit(user_features)

    print('Finished fitting for MinHashLSH')
    
    # Find similarity using MinHashLSH
    print("LSH dataset with hash tables:")
    similar_users = model.approxSimilarityJoin(user_features, user_features, threshold=1.0, distCol="JaccardDistance").filter("datasetA.userId != datasetB.userId")
    top_pairs = similar_users.orderBy("JaccardDistance").select("datasetA.userId", "datasetB.userId", "JaccardDistance").limit(100)
    top_pairs.show()

    print('Printing ratings with specified schema')
    ratings.printSchema()
    ratings.show()

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
