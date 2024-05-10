from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr
from pyspark.ml.feature import MinHashLSH
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
    ratings = spark.read.csv(f'hdfs:/user/{userID}/ratings.csv', 
                             schema='userId INT, movieId INT, rating FLOAT, timestamp INT')

    # Create a user-movie matrix where each entry is 1 if the user rated the movie
    user_movie_matrix = ratings.groupBy("userId").pivot("movieId").count().na.fill(0)

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
    similar_users = model.approxSimilarityJoin(user_features, user_features, threshold=1.0, distCol="JaccardDistance")
    top_pairs = similar_users.orderBy("JaccardDistance").select("datasetA.userId", "datasetB.userId", "JaccardDistance").limit(100)
    top_pairs.show()

    print('Printing ratings with specified schema')
    ratings.printSchema()
    ratings.show()

if __name__ == "__main__":
    spark = SparkSession.builder.appName('capstone').getOrCreate()
    userID = os.getenv('USER', 'default_user')  # Default user if not set
    main(spark, userID)
