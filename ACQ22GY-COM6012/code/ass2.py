from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import numpy as np
from pyspark.sql.functions import log,col
from pyspark.sql.functions import when,lit
from pyspark.ml.feature import StandardScaler, OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
spark = SparkSession.builder \
        .master("local[4]") \
        .appName("Assignment 2") \
        .config("spark.local.dir","/fastdata/acq22gy") \
        .getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("WARN") 

#main code:
#inport data
from pyspark.sql.types import DoubleType
raw_df = spark.read.csv("/home/acq22gy/com6012/ScalableML/Data/freMTPL2freq.csv", header=True, inferSchema=True).cache()
schemaNames = raw_df.schema.names
ncolumns = len(raw_df.columns)
selected_cols = ["ClaimNb","Exposure", "VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]
for col_name in selected_cols:
    raw_df = raw_df.withColumn(col_name, col(col_name).cast(DoubleType()))

#A:Create two new columns: LogClaimNb and NZClaim, where LogClaimNb = Log(ClaimNb),  
#and NZClaim is a binary value for indicating a non-zero number of claims, 
#the value equals 1 if ClaimNb>0, and 0 otherwise
raw_df = raw_df.withColumn("NZClaim", when(col("ClaimNb") > 0,1).otherwise(0))
fractions = raw_df.select('NZClaim').distinct().withColumn('fraction', lit(0.7)).rdd.collectAsMap() 
print('============================================================================')
print(fractions)
print('=================================================================================')
raw_df = raw_df.withColumn("ClaimNb", when(raw_df["ClaimNb"] == 0, 0.0001).otherwise(raw_df["ClaimNb"]))# add 0.0001bias to 0
raw_df = raw_df.withColumn("LogClaimNb", log(raw_df["ClaimNb"]))  #already transfer data to log 

#B:
#a: 
# One-hot encode categorical features
categorical_cols = ["Area", "VehBrand", "VehGas", "Region"]
indexer = StringIndexer(inputCols=categorical_cols, outputCols=["AreaIndex", "VehBrandIndex", "VehGasIndex", "RegionIndex"])
indexed_df = indexer.fit(raw_df).transform(raw_df).drop(*categorical_cols)
Index_cols = ['AreaIndex','VehBrandIndex','VehGasIndex','RegionIndex']
encoder1 = OneHotEncoder(inputCol='AreaIndex', outputCol='AreaEncoded')
encoded_df1 = encoder1.fit(indexed_df).transform(indexed_df)

encoder2 = OneHotEncoder(inputCol='VehBrandIndex', outputCol='VehBrandEncoded')
encoded_df2 = encoder2.fit(encoded_df1).transform(encoded_df1)

encoder3 = OneHotEncoder(inputCol='VehGasIndex', outputCol='VehGasEncoded')
encoded_df3 = encoder3.fit(encoded_df2).transform(encoded_df2)

encoder4 = OneHotEncoder(inputCol='RegionIndex', outputCol='RegionEncoded')
encoded = encoder4.fit(encoded_df3).transform(encoded_df3).drop(*Index_cols)

# Standardize numeric features
numeric_cols = ["Exposure", "VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]
numeric_assembler = VectorAssembler(inputCols=numeric_cols, outputCol="numeric_features")
numeric_df = numeric_assembler.transform(encoded)
scalers = StandardScaler(inputCol="numeric_features", outputCol="std_features", withStd=True, withMean=True)  
scaler_df = scalers.fit(numeric_df).transform(numeric_df).drop(*numeric_cols).drop("numeric_features")


#split trainning data and testing data
(trainingData, testData) = scaler_df.randomSplit([0.7, 0.3], 37)
trainingData.cache()
testData.cache()
trainingData.show(30)

# #b1:poisson regression:
# # Combine features into a single column
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml import Pipeline
assembler = VectorAssembler(inputCols=["std_features"] + [col+"Encoded" for col in categorical_cols], outputCol="features")
scaler_df_new = assembler.transform(scaler_df)
glm_poisson = GeneralizedLinearRegression(featuresCol='features', labelCol="ClaimNb", maxIter=50, regParam=0.01,\
                                          family='poisson', link='log')
stages = [assembler, glm_poisson]
pipeline = Pipeline(stages=stages)
pipelineModel = pipeline.fit(trainingData)
predictions = pipelineModel.transform(testData)

evaluator = RegressionEvaluator\
      (labelCol="ClaimNb", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("RMSE = %g " % rmse)
coefficients = pipelineModel.stages[-1].coefficients
intercept = pipelineModel.stages[-1].intercept
print("Coefficients: %s" % coefficients)
print("Intercept: %s" % intercept)


#b2:linear regression :
l1_qn = LinearRegression(featuresCol='features', labelCol='LogClaimNb', maxIter=50, regParam=0.01,
                          elasticNetParam=1, solver='normal')
l2_qn = LinearRegression(featuresCol='features', labelCol='LogClaimNb', maxIter=50, regParam=0.01,
                          elasticNetParam=0, solver='normal')
l1_qn_pipe = Pipeline(stages=[assembler, l1_qn])
l2_qn_pipe = Pipeline(stages=[assembler, l2_qn])
l1_qn_model = l1_qn_pipe.fit(trainingData)
l2_qn_model= l2_qn_pipe.fit(trainingData)
l1_qn_preds = l1_qn_model.transform(testData)
l2_qn_preds = l2_qn_model.transform(testData)
evaluator_lr = RegressionEvaluator\
      (labelCol="LogClaimNb", predictionCol="prediction", metricName="rmse")
rmse_l1_qn = evaluator_lr.evaluate(l1_qn_preds)
rmse_l2_qn = evaluator_lr.evaluate(l2_qn_preds)
print("L1 (OWL-QN optimisation) RMSE = " ,rmse_l1_qn)
print("L2 (OWL-QN optimisation) RMSE = " ,rmse_l2_qn)
coefficients_l1 = l1_qn_model.stages[-1].coefficients
print("L1 Coefficients: %s" % coefficients_l1)

coefficients_l2 = l2_qn_model.stages[-1].coefficients
print("L2 Coefficients: %s" % coefficients_l2)


#b3:logistic regression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import LogisticRegression
lr_l1 = LogisticRegression(featuresCol="features", labelCol="NZClaim", maxIter=50, regParam=0.01, elasticNetParam=1.0,family ='multinomial')
lr_l2 = LogisticRegression(featuresCol="features", labelCol="NZClaim", maxIter=50, regParam=0.01, elasticNetParam=0.0,family = 'multinomial')
# Create pipeline for L1 model and fit on training data
pipeline_l1 = Pipeline(stages=[assembler, lr_l1])
model_l1 = pipeline_l1.fit(trainingData)

# Create pipeline for L2 model and fit on training data
pipeline_l2 = Pipeline(stages=[assembler, lr_l2])
model_l2 = pipeline_l2.fit(trainingData)

# Make predictions on test data and evaluate the models
predictions_l1 = model_l1.transform(testData)
predictions_l2 = model_l2.transform(testData)

# Evaluate model performance with ROC AUC metric
evaluator_logistic = MulticlassClassificationEvaluator(labelCol="NZClaim") 
auc_l1 = evaluator_logistic.evaluate(predictions_l1)
auc_l2 = evaluator_logistic.evaluate(predictions_l2)
# lrModel1 = predictions_l1.stages[-1]
print("AUC for L1 model:" ,auc_l1)
print("AUC for L2 model:" , auc_l2)
coefficients_lo1 = model_l1.stages[-1].coefficientMatrix.values
print("L1 Coefficients: %s" % coefficients_lo1)
coefficients_lo2 = model_l2.stages[-1].coefficientMatrix.values
print("L2 Coefficients: %s" % coefficients_lo2)

#c: use cross validator to choose the best model and plot the validation loss performance via applying these parameter
#poisson regression:
regParam = [0.001, 0.01, 0.1,1,10]
train_subset = trainingData.sample(False, 0.1, seed=37)
paramGrid1 = ParamGridBuilder() \
    .addGrid(glm_poisson.regParam, [0.001, 0.01, 0.1,1,10]) \
    .build()
cv1 = CrossValidator(estimator=pipeline, estimatorParamMaps=paramGrid1, evaluator=evaluator)
cvModel1 = cv1.fit(train_subset)
cvResults1 = cvModel1.avgMetrics
Best_regParam_log1 = regParam[cvResults1.index(min(cvResults1))]
print('the best regparameter for poisson distribution regression is {}'.format(Best_regParam_log1))
# Plot the validation curve
plt.plot([0.001, 0.01, 0.1, 1, 10], cvResults1)
plt.xscale('log')
plt.xlabel('regParam')
plt.ylabel('Score')
plt.savefig("/home/acq22gy/com6012/ScalableML/Output/poisson_validation_curve.png")
plt.clf()

#linear regression first one use l1 and second one usel2
regParam = [0.001, 0.01, 0.1,1,10]
train_subset = trainingData.sample(False, 0.1, seed=37)
paramGrid2 = ParamGridBuilder() \
    .addGrid(l1_qn.regParam, [0.001, 0.01, 0.1,1,10]) \
    .build()
#l1:
cv2 = CrossValidator(estimator=l1_qn_pipe , estimatorParamMaps=paramGrid2, evaluator=evaluator_lr)
cvModel2 = cv2.fit(train_subset)
cvResults2 = cvModel2.avgMetrics
Best_regParam_log2 = regParam[cvResults2.index(min(cvResults2))]
print('the best regparameter for l1 linear regression is {}'.format(Best_regParam_log2))
# Plot the validation curve
plt.plot([0.001, 0.01, 0.1, 1, 10], cvResults2)
plt.xscale('log')
plt.xlabel('regParam')
plt.ylabel('Score')
plt.savefig("/home/acq22gy/com6012/ScalableML/Output/linearl1_validation_curve.png")
plt.clf()
#l2
paramGrid3 = ParamGridBuilder() \
    .addGrid(l2_qn.regParam, [0.001, 0.01, 0.1,1,10]) \
    .build()
cv3 = CrossValidator(estimator=l2_qn_pipe , estimatorParamMaps=paramGrid3, evaluator=evaluator_lr)
cvModel3 = cv3.fit(train_subset)
cvResults3 = cvModel3.avgMetrics
Best_regParam_log3 = regParam[cvResults3.index(min(cvResults3))]
print('the best regparameter for l2 linear regression is {}'.format(Best_regParam_log3))
# Plot the validation curve
plt.plot([0.001, 0.01, 0.1, 1, 10], cvResults3)
plt.xscale('log')
plt.xlabel('regParam')
plt.ylabel('Score')
plt.savefig("/home/acq22gy/com6012/ScalableML/Output/linearl2_validation_curve.png")
plt.clf()


#logistic regression: 
# l1
regParam = [0.001, 0.01, 0.1,1,10]
train_subset = trainingData.sample(False, 0.1, seed=37)
train_subset.show(5)
paramGrid4 = ParamGridBuilder() \
    .addGrid(lr_l1.regParam, [0.001, 0.01, 0.1,1,10]) \
    .build()
cv4 = CrossValidator(estimator=pipeline_l1, estimatorParamMaps=paramGrid4, evaluator=evaluator_logistic)
cvModel4 = cv4.fit(train_subset)
cvResults4 = cvModel4.avgMetrics
Best_regParam_log4 = regParam[cvResults4.index(min(cvResults4))]
print('the best regparameter for l1 logistic regression is {}'.format(Best_regParam_log4))
# Plot the validation curve
plt.plot([0.001, 0.01, 0.1, 1, 10], cvResults4)
plt.xscale('log')
plt.xlabel('regParam')
plt.ylabel('Score')
plt.savefig("/home/acq22gy/com6012/ScalableML/Output/lrl1_validation_curve.png")
plt.clf()
#l2
paramGrid5 = ParamGridBuilder() \
    .addGrid(lr_l2.regParam, [0.001, 0.01, 0.1,1,10]) \
    .build()
cv5 = CrossValidator(estimator=pipeline_l2, estimatorParamMaps=paramGrid5, evaluator=evaluator_logistic)
cvModel5 = cv5.fit(train_subset)
cvResults5 = cvModel5.avgMetrics
Best_regParam_log5 = regParam[cvResults5.index(min(cvResults5))]
print('the best regparameter for l2 logistic regression is {}'.format(Best_regParam_log5))
# Plot the validation curve
plt.plot([0.001, 0.01, 0.1, 1, 10], cvResults5)
plt.xscale('log')
plt.xlabel('regParam')
plt.ylabel('Score')
plt.savefig("/home/acq22gy/com6012/ScalableML/Output/lrl2_validation_curve.png")

spark.stop()