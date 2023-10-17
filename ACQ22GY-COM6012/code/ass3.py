from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.ml.clustering import KMeans
from pyspark.ml.clustering import KMeansModel
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import avg
from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType

# import numpy as np
import matplotlib.pyplot as plt     
import matplotlib 
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!

spark = SparkSession.builder \
        .master("local[4]") \
        .appName("Assignment 2") \
        .config("spark.local.dir","/fastdata/acq22gy") \
        .getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("WARN") 
#main code
###A(1)
lines = spark.read.text("/home/acq22gy/com6012/ScalableML/Data/ratings.csv").rdd
sample_lines = lines.sample(False, 0.01)
parts = sample_lines.map(lambda row: row.value.split(","))
ratingsRDD = parts.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]),rating=float(p[2]), timestamp=int(p[3])))
ratings = spark.createDataFrame(ratingsRDD).cache()
sorted_ratings = ratings.orderBy('timestamp')
train_size = [0.4, 0.6, 0.8]  # train sizes
#split 0.8
myseed= 37
(training, test) = ratings.randomSplit([0.8, 0.2], myseed)
training_08 = training.cache()
test_08 = test.cache()
#split 0.6
(training, test) = ratings.randomSplit([0.6, 0.4], myseed)
training_06 = training.cache()
test_06 = test.cache()
#split 0.4
(training, test) = ratings.randomSplit([0.4, 0.6], myseed)
training_04 = training.cache()
test_04 = test.cache()
training_04.show()

predictions_list = []
dfWithFeatures_list = []
###A(2)
##0.8
#setting one: Applu ALS model
als = ALS(userCol="userId", itemCol="movieId", seed=myseed, coldStartStrategy="drop")
model = als.fit(training_08)
predictions = model.transform(test_08)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("0.8 training data setting1, Root-mean-square error = " + str(rmse))# Root-mean-square error = 1.7715677562740955
mse = evaluator.evaluate(predictions, {evaluator.metricName: "mse"})
print("0.8 training data setting1, MSE = " + str(mse))
mae = evaluator.evaluate(predictions, {evaluator.metricName: "mae"})
print("0.8 training data setting1, MAE = " + str(mae))
userRecs = model.recommendForAllUsers(10)  #Generate top 10 movie recommendations for each user:
print("Generate top 10 movie recommendations for each user by using 0.8 split setting1:")
userRecs.show(10,  False)

#setting two： 
als_2= ALS(rank= 15, maxIter= 15, regParam=0.01, userCol="userId", itemCol="movieId", seed=myseed, coldStartStrategy="drop")
model_2 = als_2.fit(training_08)
predictions = model_2.transform(test_08)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("0.8 training data adjusting settings2, Root-mean-square error = " + str(rmse))
#0.8 training data adjusting settings, Root-mean-square error = 3.5118559601896844
mse = evaluator.evaluate(predictions, {evaluator.metricName: "mse"})
print("0.8 training data adjusting settings2, MSE = " + str(mse))
mae = evaluator.evaluate(predictions, {evaluator.metricName: "mae"})
print("0.8 training data adjusting settings2, MAE = " + str(mae))
userRecs = model_2.recommendForAllUsers(10)  #Generate top 10 movie recommendations for each user:
print("Generate top 10 movie recommendations for each user by using 0.8 split setting2:")
userRecs.show(10,  False)

#k-means 0.8 training data
print("top 5 clusters with training data split 0.8:\n")
def transData(data):
    return data.rdd.map(lambda r: [Vectors.dense(r[:-1])]).toDF(['features'])
dfFeatureVec_2= transData(training_08).cache()
# dfFeatureVec_2.show(5, False)
from pyspark.sql.functions import monotonically_increasing_id
training_08 = training_08.withColumn("index", monotonically_increasing_id())
# Add the same index to the dataframe of features
dfFeatureVec_2 = dfFeatureVec_2.withColumn("index", monotonically_increasing_id())
dfWithFeatures_2 = training_08.join(dfFeatureVec_2, "index")

#k-means
k=25
kmeans = KMeans().setK(k).setSeed(37)
model_k2 = kmeans.fit(dfFeatureVec_2)    
predictions_2 = model_k2.transform(dfFeatureVec_2)
predictions_2.groupBy('prediction').count().orderBy('count', ascending=False).show(5) #sorting the largest cluster
predictions_list.append(predictions_2)
dfWithFeatures_list.append(dfWithFeatures_2)
print("====================================================")

##0.6
als_3 = ALS(userCol="userId", itemCol="movieId", seed=myseed, coldStartStrategy="drop")
model_3 = als_3.fit(training_06)
predictions = model_3.transform(test_06)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("0.6 training data settings1, Root-mean-square error = " + str(rmse)) #rmse = 2.1740345087686994
mse = evaluator.evaluate(predictions, {evaluator.metricName: "mse"})
print("0.6 training data settings1, MSE = " + str(mse))
mae = evaluator.evaluate(predictions, {evaluator.metricName: "mae"})
print("0.6 training data settings1, MAE = " + str(mae))
# Root-mean-square error = 1.7715677562740955
userRecs = model_3.recommendForAllUsers(10)  #Generate top 10 movie recommendations for each user:
print("Generate top 10 movie recommendations for each user by using 0.6 split setting1:")
userRecs.show(10,  False)

#setting two： 
als_4= ALS(rank= 15, maxIter= 15, regParam=0.01, userCol="userId", itemCol="movieId", seed=myseed, coldStartStrategy="drop")
model_4 = als_4.fit(training_06)
predictions = model_4.transform(test_06)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("0.6 training data adjusting settings2, Root-mean-square error = " + str(rmse))#rmse = 3.543085533107331
mse = evaluator.evaluate(predictions, {evaluator.metricName: "mse"})
print("0.6 training data adjusting settings2, MSE = " + str(mse))
mae = evaluator.evaluate(predictions, {evaluator.metricName: "mae"})
print("0.6 training data adjusting settings2, MAE = " + str(mae))
userRecs = model_4.recommendForAllUsers(10)  #Generate top 10 movie recommendations for each user:
print("Generate top 10 movie recommendations for each user by using 0.6 split setting2:")
userRecs.show(10,  False)

print("top 5 clusters with training data split 0.6:\n")
def transData(data):
    return data.rdd.map(lambda r: [Vectors.dense(r[:-1])]).toDF(['features'])
dfFeatureVec_4= transData(training_04).cache()
from pyspark.sql.functions import monotonically_increasing_id
training_06 = training_06.withColumn("index", monotonically_increasing_id())
# Add the same index to the dataframe of features
dfFeatureVec_4= dfFeatureVec_4.withColumn("index", monotonically_increasing_id())
dfWithFeatures_4 = training_06.join(dfFeatureVec_4, "index")
# dfWithFeatures_2.show(5, False)  #我的userid 和feature
k=25
kmeans = KMeans().setK(k).setSeed(37)
model_k4 = kmeans.fit(dfFeatureVec_4)    
predictions_4 = model_k4.transform(dfFeatureVec_4)
predictions_4.groupBy('prediction').count().orderBy('count', ascending=False).show(5) #sorting the largest cluster
predictions_list.append(predictions_4)
dfWithFeatures_list.append(dfWithFeatures_4)


#0.4
als_5 = ALS(userCol="userId", itemCol="movieId", seed=myseed, coldStartStrategy="drop")
model_5= als_5.fit(training_04)
predictions = model_5.transform(test_04)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("0.4 training data settings1, Root-mean-square error = " + str(rmse)) #rmse = 2.1740345087686994
mse = evaluator.evaluate(predictions, {evaluator.metricName: "mse"})
print("0.4 training data settings1, MSE = " + str(mse))
mae = evaluator.evaluate(predictions, {evaluator.metricName: "mae"})
print("0.4 training data settings1, MAE = " + str(mae))
# Root-mean-square error = 1.7715677562740955
userRecs = model_5.recommendForAllUsers(10)  #Generate top 10 movie recommendations for each user:
print("Generate top 10 movie recommendations for each user by using 0.4 split setting1:")
userRecs.show(10,  False)

#setting two： 
als_6= ALS(rank= 15, maxIter= 15, regParam=0.01, userCol="userId", itemCol="movieId", seed=myseed, coldStartStrategy="drop")
model_6 = als_6.fit(training_04)
predictions = model_6.transform(test_04)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("0.4 training data adjusting settings2, Root-mean-square error = " + str(rmse))#rmse = 3.543085533107331
mse = evaluator.evaluate(predictions, {evaluator.metricName: "mse"})
print("0.4 training data adjusting settings2, MSE = " + str(mse))
mae = evaluator.evaluate(predictions, {evaluator.metricName: "mae"})
print("0.4 training data adjusting settings2, MAE = " + str(mae))
userRecs = model_6.recommendForAllUsers(10)  #Generate top 10 movie recommendations for each user:
print("Generate top 10 movie recommendations for each user by using 0.4 split setting2:")
userRecs.show(10,  False)
# dfItemFactors_6=model_6.itemFactors
# dfItemFactors_6.show()

print("top 5 clusters with training data split 0.4:\n")
def transData(data):
    return data.rdd.map(lambda r: [Vectors.dense(r[:-1])]).toDF(['features'])
dfFeatureVec_6= transData(training_04).cache()
from pyspark.sql.functions import monotonically_increasing_id
training_04 = training_04.withColumn("index", monotonically_increasing_id())
# Add the same index to the dataframe of features
dfFeatureVec_6 = dfFeatureVec_6.withColumn("index", monotonically_increasing_id())
dfWithFeatures_6 = training_04.join(dfFeatureVec_6, "index")
dfWithFeatures_6.show(5, False)  #我的userid 和feature
k=25
kmeans = KMeans().setK(k).setSeed(37)
model_k6 = kmeans.fit(dfFeatureVec_6)    
predictions_6 = model_k6.transform(dfFeatureVec_6)
predictions_6.show()
predictions_6.groupBy('prediction').count().orderBy('count', ascending=False).show(5)#sorting the largest cluster
predictions_list.append(predictions_6)
dfWithFeatures_list.append(dfWithFeatures_6)

print("=============PART B 2=====================")
# #B2:
ratings= spark.read.load('Data/ratings.csv', format = 'csv', inferSchema = "true", header = "true").cache() 
ratings = ratings.withColumnRenamed("userId", "Id")
movies = spark.read.load('Data/movies.csv', format = 'csv', inferSchema = "true", header = "true")  
movies_largest_cluster_list=[]
top_movies_list=[]
top10_genres_list=[]
for i,_ in enumerate(dfWithFeatures_list):
    name = ['movieId','rating','timestamp']
    predictions_id_df = predictions_list[i].join(dfWithFeatures_list[i], 'index', 'outer').orderBy('index').drop('index','features') 
    # first find userid, second find movieid and last find rating for each movie id
    largest_cluster = predictions_6.groupBy('prediction').count().orderBy('count', ascending=False).first()[0] #find the largerst cluster
    UsersID_df = predictions_id_df.filter(predictions_id_df['prediction'] == largest_cluster).drop(*name)
    UsersID_df.show(10) #有prediction 和userid
    UsersID_df.printSchema()
    #convert rdd
    joined_df = UsersID_df.join(ratings, UsersID_df['userId'] == ratings['Id'], 'inner')
    filtered_df = joined_df.select(ratings.columns)
    movies_largest_cluster=filtered_df.select('movieID','rating').groupBy('movieID').agg(avg("rating").alias("avg_ratings"))
    movies_largest_cluster_list.append(movies_largest_cluster)   #get top largest movie cluster list
    top_movies=movies_largest_cluster.filter(col('avg_ratings')>=4).withColumnRenamed("movieID", "ID")  
    top_movies_list.append(top_movies)  #get top movie list
    joined_movies = top_movies.join(movies, top_movies['ID'] == movies['movieId'], 'inner')
    filtered_genres = joined_movies.select(movies.columns) 
    genres_list =filtered_genres.select('genres').rdd.flatMap(lambda x: x).collect()
    genres_dict={}  
    for row in genres_list:
        genres_row_list=row.split('|')  
        for genres in genres_row_list:  
            if genres not in genres_dict:
                genres_dict[genres]=1  
            else:
                genres_dict[genres]+=1 
    sorted_genres_dict = sorted(genres_dict.items(), key=lambda x: x[1], reverse=True)
    top10_genres = [x[0] for x in sorted_genres_dict[:10]]
    top10_genres_list.append(top10_genres)  #get top ten movie genres

#creat chart
from pyspark.sql.types import StructType,IntegerType,StructField,StringType
schema = StructType([StructField("spilts", IntegerType(), True),     
    StructField("1", StringType(), True),
    StructField("2", StringType(), True),
    StructField("3", StringType(), True),
    StructField("4", StringType(), True),
    StructField("5", StringType(), True),
    StructField("6", StringType(), True),
    StructField("7", StringType(), True),
    StructField("8", StringType(), True),
    StructField("9", StringType(), True),
    StructField("10", StringType(), True)

])
top10_genres_df = spark.createDataFrame(spark.sparkContext.emptyRDD(), schema)
for i,top10_genres in enumerate(top10_genres_list):
    split = [0.8,0.6,0.4]
    new_row = Row(split[i], top10_genres[0],top10_genres[1], top10_genres[2],top10_genres[3], top10_genres[4],top10_genres[5],top10_genres[6],top10_genres[7],top10_genres[8],top10_genres[9])   #add each case to the dataframe
    top10_genres_df = top10_genres_df.union(sc.parallelize([new_row]).toDF())


print('For the 0.8 spilt:')
print('\nmovies_largest_cluster:')
movies_largest_cluster_list[0].show(20)
print('\ntop_movies:')
top_movies_list[0].show(20)
print('\n\nFor the 0.6 spilt:')
print('\nmovies_largest_cluster:')
movies_largest_cluster_list[1].show(20)
print('\ntop_movies:')
top_movies_list[1].show(20)
print('\n\nFor the 0.4 spilt:')
print('\nmovies_largest_cluster:')
movies_largest_cluster_list[2].show(20)
print('\ntop_movies:')
top_movies_list[2].show(20)
print('\nThe top ten most popular genres for each of the spilts are:')
top10_genres_df.show()
print("======================================================")
print('\n\n')

spark.stop()