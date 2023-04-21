import matplotlib 
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab! 
from pyspark.ml.clustering import KMeans
from pyspark.ml.clustering import KMeansModel
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.linalg import Vectors
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession

data = [(Vectors.dense([0.0, 0.0]),), (Vectors.dense([1.0, 1.0]),),
        (Vectors.dense([9.0, 8.0]),), (Vectors.dense([8.0, 9.0]),)]
df = spark.createDataFrame(data, ["features"])
kmeans = KMeans(k=2, seed=1)  # Two clusters with seed = 1
model = kmeans.fit(df)

centers = model.clusterCenters()
len(centers)
# 2
for center in centers:
    print(center)
# [0.5 0.5]
# [8.5 8.5]
model.predict(df.head().features)
# 0
transformed = model.transform(df)
transformed.show()