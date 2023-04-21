from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .master("local[2]") \
    .appName("COM6012 Spark Intro") \
    .config("spark.local.dir","/fastdata/acq22gy") \
    .appName("OneHotEncoderEstimator")\
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")  # This can only affect the log level after it is executed.


from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row


#Read the data in and split words (tab separated):
lines = spark.read.text("Data/MovieLens100k.data").rdd
parts = lines.map(lambda row: row.value.split("\t"))

#convert RDD to DataFrame:
ratingsRDD = parts.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]),rating=float(p[2]), timestamp=int(p[3])))
ratings = spark.createDataFrame(ratingsRDD).cache()

#Check data:
ratings.show(5)

#Check data type:
ratings.printSchema()

#Prepare the training/test data with seed 6012:
myseed=6012
(training, test) = ratings.randomSplit([0.8, 0.2], myseed)
training = training.cache()
test = test.cache()

#Build the recommendation model using ALS on the training data.
als = ALS(userCol="userId", itemCol="movieId", seed=myseed, coldStartStrategy="drop")
model = als.fit(training)

#Evaluate the model by computing the RMSE on the test data:
predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))
# Root-mean-square error = 0.920957307650525

#Generate top 10 movie recommendations for each user:
userRecs = model.recommendForAllUsers(10)
userRecs.show(5,  False)

#Generate top 10 user recommendations for each movie
movieRecs = model.recommendForAllItems(10)
movieRecs.show(5, False)

#Generate top 10 movie recommendations for a specified set of users:
users = ratings.select(als.getUserCol()).distinct().limit(3)
userSubsetRecs = model.recommendForUserSubset(users, 10)
users.show()

#Generate top 10 user recommendations for a specified set of movies
movies = ratings.select(als.getItemCol()).distinct().limit(3)
movieSubSetRecs = model.recommendForItemSubset(movies, 10)
movies.show()

#Let us take a look at the learned factors:
dfItemFactors=model.itemFactors
dfItemFactors.show()