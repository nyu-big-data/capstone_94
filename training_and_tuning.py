# from pyspark.sql import SparkSession
# from pyspark.sql.functions import col, expr
# from pyspark.ml.feature import MinHashLSH
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import MinHashLSH, VectorAssembler
from pyspark.sql.functions import lit, col, when, least, greatest

from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import Row

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import BucketedRandomProjectionLSH, MinHashLSHModel
from pyspark.ml.feature import HashingTF, IDF, Tokenizer

import random
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.functions import col

import os

def main(spark, userID):
    '''Main routine for Lab Solutions
    Parameters
    ----------
    spark : SparkSession object
    userID : string, userID of student to find files in HDFS
    '''
    print('Dataframe loading and SQL query')
  
    # Load training and validation data
    path = f'hdfs:/user/{userID}/'

    train_df = spark.read.csv(path + "train.csv")
    validation_df = spark.createDataFrame(path + "validation.csv")
    
    # Configure and train ALS model
    als = ALS(userCol='userId', itemCol='movieId', ratingCol='rating', nonnegative=True, implicitPrefs=False, coldStartStrategy="drop")
    als_model = als.fit(train_df)
    
    # Make predictions on validation set
    validation_predictions = als_model.transform(validation_df)
    
    # Evaluate the model on validation set
    evaluator = RegressionEvaluator(metricName='rmse', labelCol='rating', predictionCol='prediction')
    rmse = evaluator.evaluate(validation_predictions)
    print(f"Validation Root-Mean-Square Error (RMSE): {rmse}")
    
    # Tune ALS model parameters (rank and regularization)
    def tune_als(train_df, validation_df):
        best_model = None
        best_rmse = float("inf")
        best_rank = None
        best_reg_param = None
        ranks = [5, 10, 15, 20]
        reg_params = [0.01, 0.1, 1.0, 10.0]
    
        for rank in ranks:
            for reg_param in reg_params:
                als = ALS(userCol='userId', itemCol='movieId', ratingCol='rating', rank=rank, regParam=reg_param, nonnegative=True, implicitPrefs=False, coldStartStrategy="drop")
                model = als.fit(train_df)
                predictions = model.transform(validation_df)
                rmse = evaluator.evaluate(predictions)
                print(f"Rank: {rank}, Regularization: {reg_param}, Validation RMSE: {rmse}")
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_model = model
                    best_rank = rank
                    best_reg_param = reg_param
    
        print(f"Best Rank: {best_rank}, Best Regularization Parameter: {best_reg_param}, Best RMSE: {best_rmse}")
        return best_model
    
    # Tune ALS model parameters and obtain the best model
    best_als_model = tune_als(train_df, validation_df)

    # Load test data
    # test = pd.read_csv('test.csv')
    # test_data = [Row(userId=int(row['userId']), movieId=int(row['movieId']), rating=float(row['rating'])) for _, row in test.iterrows()]
    test_df = spark.read.csv(path + "test.csv")
    
    # Make predictions on test set using the best ALS model
    test_predictions = best_als_model.transform(test_df)
    
    # Evaluate the best ALS model on test set
    test_rmse = evaluator.evaluate(test_predictions)
    print(f"Test Root-Mean-Square Error (RMSE): {test_rmse}")
    
    def precision_recall_at_k(predictions, k):
        predictions = predictions.orderBy(col('prediction').desc())
        total_relevant = predictions.filter(predictions.rating >= 4).count()
        recommended_and_relevant_count = predictions.filter((predictions.rating >= 4) & (predictions.prediction >= 4)).count()
        precision = recommended_and_relevant_count / total_relevant if total_relevant != 0 else 0
        recall = recommended_and_relevant_count / total_relevant if total_relevant != 0 else 0
        return precision, recall
    
    precision, recall = precision_recall_at_k(test_predictions, 5)
    
    print(f"Precision@5: {precision}")
    print(f"Recall@5: {recall}")
    
        


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
