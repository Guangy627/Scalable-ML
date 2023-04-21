from pyspark.sql import SparkSession
import pyspark.sql.functions as F


spark = SparkSession.builder \
        .master("local[4]") \
        .appName("Lab 2 Exercise") \
        .config("spark.local.dir","/fastdata/acq22gy") \
        .getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("WARN") 

logFile = spark.read.text("../ScalableML/Data/NASA_access_log_Aug95.gz").cache()  # add it to cache, so it can be used in the following steps efficiently      
logFile.show(20, False)

# split into 5 columns using regex and split
data = logFile.withColumn('host', F.regexp_extract('value', '^(.*) - -.*', 1)) \
                .withColumn('timestamp', F.regexp_extract('value', '.* - - \[(.*)\].*',1)) \
                .withColumn('request', F.regexp_extract('value', '.*\"(.*)\".*',1)) \
                .withColumn('HTTP reply code', F.split('value', ' ').getItem(F.size(F.split('value', ' ')) -2).cast("int")) \
                .withColumn('bytes in the reply', F.split('value', ' ').getItem(F.size(F.split('value', ' ')) - 1).cast("int")).drop("value").cache()
data.show(20,False)

# number of unique hosts

n_hosts = data.select('host').distinct().count()
print("==================== Question 2 ====================")
print(f"There are {n_hosts} unique hosts")
print("====================================================")

# most visited host

host_count = data.select('host').groupBy('host').count().sort('count', ascending=False).freqItems("host")
#host_max = host_count.select("host").first()['host']
print("==================== Question 3 ====================")
print(f"The most frequently visited host is {host_count}")
print("====================================================")


