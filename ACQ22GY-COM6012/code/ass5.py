import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql.functions import rand,col
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
spark = SparkSession.builder \
    .master("local[4]") \
    .appName("NIPS_PCA") \
    .config("spark.local.dir","/fastdata/acq22gy") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "16g") \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN") 
#main code
# Load the HIGGS dataset into a Spark DataFrame
data = spark.read.csv('/home/acq22gy/com6012/ScalableML/Data/HIGGS.csv', header=False, inferSchema=True)
data = data.withColumn("_c8", col("_c8").cast("double"))
# Split the data into training and testing sets
training_data, test_data = data.filter(data['_c0'] == 1).sample(0.01, seed=42).union(data.filter(data['_c0'] == 0).sample(0.01, seed=42)), data.sampleBy('_c0', {0: 0.01, 1: 0.01}, seed=42)

training_data.show(10) #有prediction 和userid
training_data.printSchema()
print(f"There are {training_data.cache().count()} rows in the training set, and {test_data.cache().count()} in the test set")
# Define the pipeline for Random Forests
rf_assembler = VectorAssembler(inputCols=data.columns[1:], outputCol='features')
vecTrainingData = rf_assembler.transform(training_data)
vecTrainingData.select("features","_c0").show(5)



# Combine stages into pipeline
print("==================A=====================")
rf = RandomForestClassifier(labelCol="_c0", featuresCol="features", seed=42)
stages = [rf_assembler, rf]
pipeline = Pipeline(stages=stages)

paramGrid = ParamGridBuilder() \
    .addGrid(rf.maxDepth, [1, 5, 10]) \
    .addGrid(rf.maxBins, [2, 10, 20]) \
    .build()
evaluator = MulticlassClassificationEvaluator\
      (labelCol="_c0", predictionCol="prediction", metricName="accuracy")
crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=5)
cvModel = crossval.fit(training_data)
prediction = cvModel.transform(test_data)
accuracy = evaluator.evaluate(prediction)
print("The 0.01 data Accuracy for best rf model = %g " % accuracy)
paramDict = {param[0].name: param[1] for param in cvModel.bestModel.stages[-1].extractParamMap().items()}
print(json.dumps(paramDict, indent = 4))


# GBT
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
gbt = GBTClassifier(labelCol='_c0', featuresCol="features", maxDepth=5, maxIter=10, stepSize=0.1)
# Create the pipeline
stages = [rf_assembler, gbt]
pipeline = Pipeline(stages=stages)
paramGrid = ParamGridBuilder() \
    .addGrid(gbt.maxDepth, [1, 10, 30]) \
    .addGrid(gbt.maxIter, [1, 10, 30]) \
    .build()
evaluator = BinaryClassificationEvaluator(labelCol='_c0', rawPredictionCol="prediction", metricName="areaUnderROC")
crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=5)
cvModel = crossval.fit(training_data)
prediction = cvModel.transform(test_data)

auc_roc = evaluator.evaluate(prediction)
print("The 0.01 data AUC-ROC:", auc_roc)
paramDict = {param[0].name: param[1] for param in cvModel.bestModel.stages[-1].extractParamMap().items()}
print(json.dumps(paramDict, indent = 4))




print("==================B=====================")
trainingData, testData = data.filter(data['_c0'] == 1).sample(0.1, seed=42).union(data.filter(data['_c0'] == 0).sample(0.1, seed=42)), data.sampleBy('_c0', {0: 0.1, 1: 0.1}, seed=42)
trainingData.show(5)
rf_assembler = VectorAssembler(inputCols=data.columns[1:], outputCol='features')
vecTrainingData = rf_assembler.transform(trainingData)
vecTrainingData.select("features","_c0").show(5)


print("choose the best parameter to finish the rf model")
rf = RandomForestClassifier(labelCol="_c0", featuresCol="features", maxDepth=5, maxBins = 20, impurity='gini')
stages = [rf_assembler, rf]
pipeline = Pipeline(stages=stages)
pipeline_model = pipeline.fit(trainingData)
prediction = pipeline_model.transform(testData)
evaluator = MulticlassClassificationEvaluator\
      (labelCol="_c0", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(prediction)
print("The ALL data Accuracy for best rf model = %g " % accuracy)


#GBT
print("choose the best parameter to finish the gbt model")
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
gbt = GBTClassifier(labelCol='_c0', featuresCol="features", maxDepth=1, maxIter=30,maxBins=32, stepSize=0.1)
# Create the pipeline
stages = [rf_assembler, gbt]
pipeline = Pipeline(stages=stages)
evaluator = BinaryClassificationEvaluator(labelCol='_c0', rawPredictionCol="prediction", metricName="areaUnderROC")
pipelineModel = pipeline.fit(trainingData)
prediction = pipelineModel.transform(testData)
auc_roc = evaluator.evaluate(prediction)
print("The ALL data AUC-ROC:", auc_roc)
