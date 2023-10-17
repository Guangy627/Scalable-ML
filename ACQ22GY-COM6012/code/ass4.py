import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml.feature import PCA
from pyspark.sql import Row
from pyspark.ml.linalg import Vectors

from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA
from pyspark.mllib.stat import Statistics
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
#dataframe read the file, delete the first line, save it as a new column, 
#then take out each column and pack it into element (zip), convert it into a list and then rdd


paper_sample = spark.read.load('/home/acq22gy/com6012/ScalableML/Data/NIPS_1987-2015.csv', format='csv', inferSchema='true').cache()

#paper_sample#check
# Get the first row values
firstrow = [row[0] for row in paper_sample.collect()]
firstrow[0]='id_year'

# Drop the first column from the DataFrame
paper_sample = paper_sample.drop(paper_sample.schema.names[0])
# Convert DataFrame to a list of lists and transpose it

paper_zip = list(zip(*paper_sample.collect()))
# Create a new RDD from the transposed list and convert it back to a DataFrame
paper_rdd = sc.parallelize(paper_zip)
paper_transpose = paper_rdd.toDF()
#transform features to dense vector matrix
def transData(data):
    return data.rdd.map(lambda r: [Vectors.dense(r[1:]), r[0]]).toDF(['features','id'])

dfFeatureVec= transData(paper_transpose).cache()
# dfFeatureVec.show(5)
# normalization
from pyspark.ml.feature import StandardScaler
scaler = StandardScaler(withMean = True, withStd = True, inputCol = 'features',outputCol = 'scaled_features')
scaler_model = scaler.fit(dfFeatureVec)
dfFeatureVec = scaler_model.transform(dfFeatureVec).drop('features').withColumnRenamed('scaled_features','features')

from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.mllib.linalg import Vectors
#apply SVD
paper_rm = RowMatrix(dfFeatureVec.rdd.map(lambda x: Vectors.dense(x[1].tolist())))
svd = paper_rm.computeSVD(2, computeU=True)
U = svd.U       # The U factor is a RowMatrix.
s = svd.s       # The singular values are stored in a local dense vector.
V = svd.V  
pca1,pca2 = V.toArray()[:10,0],V.toArray[:10,1]
print("The first 10 entries of the PC1:",pca1)
print("The second 10 entries of the PC1:",pca2)
evs=s*s
print('the pca model two corresponding evs value is:',evs)
print('pca model explain variance:',evs/sum(evs))

# print(evs)
# print("RDD PC:")
# print(rdd_pc)
# projected = paper_rm.multiply(rdd_pc)
# print("RDD PCA projected features")
# print(projected.rows.collect())
# pca = PCA(k = 2, inputCol = 'features').setOutputCol('pca_features')
# pca_model = pca.fit(dfFeatureVec)
# label = ['features']
# pca_feature = pca_model.transform(dfFeatureVec)
# pca_feature.drop(*label)
# # Add the ID column back to the PCA-transformed DataFrame
# pca_feature_with_id = pca_feature.select("id", "pca_features")
# # Show the first 10 entries of the 2 PCs
# pca_feature_with_id.show(10,False)
# Varia = pca_model.explainedVariance     
# print('pca model explain variance:', Varia)
# # Extract the values of pca_1 and pca_2 from the DataFrame
# pca_vals = pca_feature_with_id.select('pca_features').rdd.map(lambda row: row[0]).collect()
# pca_1 = [x[0] for x in V]
# pca_2 = [x[1] for x in V]


#get pca1 and pca2 separately
V_np = np.array(V.toArray())
pca_1 = []
pca_2 = []
for row in V_np:
    pca_1.append(row[0])
    pca_2.append(row[1])


# Plot the scatter plot
plt.scatter(pca_1, pca_2)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.savefig("Output/PCA_Scatter.png")

spark.stop()












