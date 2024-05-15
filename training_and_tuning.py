# from pyspark.sql import SparkSession
# from pyspark.sql.functions import col, expr
# from pyspark.ml.feature import MinHashLSH
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import expr
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

    train_df = spark.read.csv(path + "train.csv", header=True, inferSchema=True)
    validation_df = spark.read.csv(path + "validation.csv", header=True, inferSchema=True)
    
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
    test_df = spark.read.csv(path + "test.csv", header=True, inferSchema=True)
    
    # Make predictions on test set using the best ALS model
    test_predictions = best_als_model.transform(test_df)
    
    # Evaluate the best ALS model on test set
    test_rmse = evaluator.evaluate(test_predictions)
    print(f"Test Root-Mean-Square Error (RMSE): {test_rmse}")
    
    # Function to evaluate MAP, Precision@k, and Recall@k
    def evaluate_ranking_metrics(predictions, k=10):
        # Generate ranking for each user
        windowSpec = Window.partitionBy('userId').orderBy(col('prediction').desc())
        per_user_predicted = predictions.withColumn("rank", F.rank().over(windowSpec))

        # Get top k predictions
        per_user_predicted = per_user_predicted.filter(col('rank') <= k)

        # True Positives at k
        true_positives_at_k = per_user_predicted.filter((col('rating') >= 4) & (col('prediction') >= 4)).groupBy('userId').count()

        # Relevant items in test set
        relevant_items = predictions.filter(col('rating') >= 4).groupBy('userId').count()

        # Precision at k: (True Positives at k) / k
        precision_at_k = true_positives_at_k.join(relevant_items, 'userId', 'inner') \
                                            .selectExpr('userId', 'count / {} as precision_at_k'.format(k))

        # Recall at k: (True Positives at k) / (Relevant items)
        recall_at_k = true_positives_at_k.join(relevant_items, 'userId', 'inner') \
                                         .selectExpr('userId', 'count / count as recall_at_k')

        # Mean Average Precision
        # Need to consider all predictions, not just top k
        per_user_predictions = predictions.withColumn("rank", F.rank().over(windowSpec))
        average_precision_expr = "SUM(CASE WHEN rating >= 4 THEN 1.0 ELSE 0 END) / MAX(rank)"
        per_user_map = per_user_predictions.groupBy("userId").agg(expr(average_precision_expr).alias("AP"))
        mean_average_precision = per_user_map.selectExpr("AVG(AP) as MAP").first()['MAP']

        return precision_at_k, recall_at_k, mean_average_precision

    # Use the evaluation function
    precision_at_k, recall_at_k, mean_average_precision = evaluate_ranking_metrics(test_predictions, k=10)
    print(f"Precision@10: {precision_at_k.collect()}")
    print(f"Recall@10: {recall_at_k.collect()}")
    print(f"Mean Average Precision (MAP): {mean_average_precision}")
    
        


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
