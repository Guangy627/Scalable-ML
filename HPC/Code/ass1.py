import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pyspark.pandas as ps


spark = SparkSession.builder \
    .master("local[2]") \
    .appName("COM6012 Spark Intro") \
    .config("spark.local.dir","/fastdata/acq22gy") \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN")  
#Q1
logFile=spark.read.text("../ScalableML/Data/NASA_access_log_Jul95.gz").cache()

# split into 5 columns using regex and split
data = logFile.withColumn('host', F.regexp_extract('value', '^(.*) - -.*', 1)) \
                .withColumn('timestamp', F.regexp_extract('value', '.* - - \[(.*)\].*',1)) \
                .withColumn('request', F.regexp_extract('value', '.*\"(.*)\".*',1)) \
                .withColumn('HTTP reply code', F.split('value', ' ').getItem(F.size(F.split('value', ' ')) -2).cast("int")) \
                .withColumn('bytes in the reply', F.split('value', ' ').getItem(F.size(F.split('value', ' ')) - 1).cast("int")).drop("value").cache()
# data.show(20,False)

hostsGerman = data.filter(logFile.value.contains(".de")).cache()#show German
hostsCanada = data.filter(logFile.value.contains(".ca")).cache()#show Canada
hostsSingapore = data.filter(logFile.value.contains(".sg")).cache()#show Singapore
print("\n\nHello Spark: There are %i hosts from German.\n" % (hostsGerman.count()))
print("Hello Spark: There are %i hosts from Canada.\n" % (hostsCanada.count()))
print("Hello Spark: There are %i hosts from Singapore.\n\n" % (hostsSingapore.count()))

import matplotlib.pyplot as plt  
import matplotlib
matplotlib.use('Agg')
rdd = spark.sparkContext.parallelize([
    (hostsGerman.count(),'German'),
    (hostsCanada.count(),'Canada'),
    (hostsSingapore.count(),'Singapore')
])
data = {
    'bins': ['German','Canada','Singapore'],
    'freq': [hostsGerman.count(),hostsCanada.count(),hostsSingapore.count()]
}
plt.bar(data['bins'],data['freq'])

# df = spark.createDataFrame(rdd,schema=['value','index'])
#df.show()
#pandas_df = df.toPandas().set_index('_2')
#matplotlib inline
#ax=pandas_df.plot(kind='bar',title='VALUE')
#ax.show()


# ps_df = ps.DataFrame({'value':[hostsGerman.count(),hostsCanada.count(),hostsSingapore.count()]},
#                    index=['German','Canada','Singapore'])
# ps_df.iplot.hist() 

##Q2

# German_hosts = data.select('host').filter(logFile.value.contains(".de")).distinct().count() # number of German hosts
# print(f"There are {German_hosts} unique hosts")
# Canada_hosts = data.select('host').filter(logFile.value.contains(".ca")).distinct().count() # number of German hosts
# print(f"There are {Canada_hosts} unique hosts")
# Singapore_hosts = data.select('host').filter(logFile.value.contains(".sg")).distinct().count() # number of German hosts
# print(f"There are {Singapore_hosts} unique hosts")

# host_count = data.select('host').groupBy('host').count().sort('count', ascending=False) # most visited host
# host_max = host_count.select("host").first()['host' ]
# print(f"The most frequently visited host is {host_max}")
