from pyspark.sql import SparkSession
import pyspark.sql.functions as F


spark = SparkSession.builder \
        .master("local[4]") \
        .appName("Lab 2 Exercise") \
        .config("spark.local.dir","/fastdata/acq22gy") \
        .getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("WARN") 

import numpy as np
from pyspark.mllib.linalg import Vectors

dv1 = np.array([1.0, 0.0, 3.0])  # Use a NumPy array as a dense vector.
dv2 = [1.0, 0.0, 3.0]  # Use a Python list as a dense vector.
sv1 = Vectors.sparse(3, [0, 2], [1.0, 3.0])  # Create a SparseVector.
sv1.toArray()
# array([1., 0., 3.])

from pyspark.mllib.linalg import SparseVector
from pyspark.mllib.regression import LabeledPoint

pos = LabeledPoint(1.0, [1.0, 0.0, 3.0])
neg = LabeledPoint(0.0, SparseVector(3, [0, 2], [1.0, 3.0]))

neg
# LabeledPoint(0.0, (3,[0,2],[1.0,3.0]))
neg.label
# 0.0
neg.features
# SparseVector(3, {0: 1.0, 2: 3.0})
neg.features.toArray()
# array([1., 0., 3.])

from pyspark.mllib.linalg import Matrix, Matrices

dm2 = Matrices.dense(3, 2, [1, 3, 5, 2, 4, 6]) 
sm = Matrices.sparse(3, 2, [0, 1, 3], [0, 2, 1], [9, 6, 8])
print(dm2)
# DenseMatrix([[1., 2.],
#              [3., 4.],
#              [5., 6.]])
print(sm)
# 3 X 2 CSCMatrix
# (0,0) 9.0
# (2,1) 6.0
# (1,1) 8.0

dsm=sm.toDense()
print(dsm)
# DenseMatrix([[9., 0.],
#              [0., 8.],
#              [0., 6.]])

from pyspark.mllib.linalg.distributed import RowMatrix

rows = sc.parallelize([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
mat = RowMatrix(rows)

m = mat.numRows()  # Get its size: m=4, n=3
n = mat.numCols()  

rowsRDD = mat.rows  # Get the rows as an RDD of vectors again.

rowsRDD.collect()
# [DenseVector([1.0, 2.0, 3.0]), DenseVector([4.0, 5.0, 6.0]), DenseVector([7.0, 8.0, 9.0]), DenseVector([10.0, 11.0, 12.0])]

from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors

data = [(Vectors.sparse(5, [(1, 1.0), (3, 7.0)]),),
        (Vectors.dense([2.0, 0.0, 3.0, 4.0, 5.0]),),
        (Vectors.dense([4.0, 0.0, 0.0, 6.0, 7.0]),)]
df = spark.createDataFrame(data, ["features"])
df.show()
# +--------------------+
# |            features|
# +--------------------+
# | (5,[1,3],[1.0,7.0])|
# |[2.0,0.0,3.0,4.0,...|
# |[4.0,0.0,0.0,6.0,...|
# +--------------------+

pca = PCA(k=2, inputCol="features", outputCol="pcaFeatures")
model = pca.fit(df)

result = model.transform(df).select("pcaFeatures")
result.show(truncate=False)
# +----------------------------------------+
# |pcaFeatures                             |
# +----------------------------------------+
# |[1.6485728230883807,-4.013282700516296] |
# |[-4.645104331781534,-1.1167972663619026]|
# |[-6.428880535676489,-5.337951427775355] |
# +----------------------------------------+

model.explainedVariance
# DenseVector([0.7944, 0.2056])
print(model.pc)
# DenseMatrix([[-0.44859172, -0.28423808],
#              [ 0.13301986, -0.05621156],
#              [-0.12523156,  0.76362648],
#              [ 0.21650757, -0.56529588],
#              [-0.84765129, -0.11560341]])

from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg.distributed import RowMatrix

rows = sc.parallelize([
    Vectors.sparse(5, {1: 1.0, 3: 7.0}),
    Vectors.dense(2.0, 0.0, 3.0, 4.0, 5.0),
    Vectors.dense(4.0, 0.0, 0.0, 6.0, 7.0)
])
rows.collect()
# [SparseVector(5, {1: 1.0, 3: 7.0}), DenseVector([2.0, 0.0, 3.0, 4.0, 5.0]), DenseVector([4.0, 0.0, 0.0, 6.0, 7.0])]

mat = RowMatrix(rows)

pc = mat.computePrincipalComponents(2)
print(pc)
# DenseMatrix([[-0.44859172, -0.28423808],
#              [ 0.13301986, -0.05621156],
#              [-0.12523156,  0.76362648],
#              [ 0.21650757, -0.56529588],
#              [-0.84765129, -0.11560341]])

projected = mat.multiply(pc)
projected.rows.collect()
# [DenseVector([1.6486, -4.0133]), DenseVector([-4.6451, -1.1168]), DenseVector([-6.4289, -5.338])]

# from pyspark.mllib.linalg import DenseVector

# denseRows = rows.map(lambda vector: DenseVector(vector.toArray()))
# denseRows.collect()
# # [DenseVector([0.0, 1.0, 0.0, 7.0, 0.0]), DenseVector([2.0, 0.0, 3.0, 4.0, 5.0]), DenseVector([4.0, 0.0, 0.0, 6.0, 7.0])]

# svd = mat.computeSVD(2, computeU=True)
# U = svd.U       # The U factor is a RowMatrix.
# s = svd.s       # The singular values are stored in a local dense vector.
# V = svd.V       # The V factor is a local dense matrix.

# print(V)
# # DenseMatrix([[-0.31278534,  0.31167136],
# #              [-0.02980145, -0.17133211],
# #              [-0.12207248,  0.15256471],
# #              [-0.71847899, -0.68096285],
# #              [-0.60841059,  0.62170723]])

from pyspark.mllib.feature import StandardScaler

standardizer = StandardScaler(True, False)
model = standardizer.fit(rows)
centeredRows = model.transform(rows)
centeredRows.collect()
# [DenseVector([-2.0, 0.6667, -1.0, 1.3333, -4.0]), DenseVector([0.0, -0.3333, 2.0, -1.6667, 1.0]), DenseVector([2.0, -0.3333, -1.0, 0.3333, 3.0])]
centeredmat = RowMatrix(centeredRows)

svd = centeredmat.computeSVD(2, computeU=True)
U = svd.U       # The U factor is a RowMatrix.
s = svd.s       # The singular values are stored in a local dense vector.
V = svd.V       # The V factor is a local dense matrix.

print(V)
print(s)
evs = s*s
print(evs)

evs/sum(evs)

