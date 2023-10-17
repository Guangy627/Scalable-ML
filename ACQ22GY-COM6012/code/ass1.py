import pandas as pd
import numpy as np
from pyspark.sql import SparkSession
import pyspark.sql.functions as F

from pyspark.sql.functions import sum
import pandas as pd
from pyspark.sql.functions import col
from pyspark.sql.functions import format_number
from pyspark.sql import Row

spark = SparkSession.builder \
    .master("local[4]") \
    .appName("COM6012 Spark Intro") \
    .config("spark.local.dir","/fastdata/acq22gy") \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("WARN") 
import matplotlib.pyplot as plt     
import matplotlib 
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!


#Q1 main code
logFile=spark.read.text("/home/acq22gy/com6012/ScalableML/Data/NASA_access_log_Jul95.gz").cache()
hostsGerman = logFile.filter(logFile.value.contains(".de ")).cache()
# split into 5 columns using regex and split
hostsGerman_graph = hostsGerman.withColumn('host', F.regexp_extract('value', '^(.*) - -.*', 1)) \
                .withColumn('timestamp', F.regexp_extract('value', '.* - - \[(.*)\].*',1)) \
                .withColumn('request', F.regexp_extract('value', '.*\"(.*)\".*',1)) \
                .withColumn('HTTP reply code', F.split('value', ' ').getItem(F.size(F.split('value', ' ')) -2).cast("int")) \
                .withColumn('bytes in the reply', F.split('value', ' ').getItem(F.size(F.split('value', ' ')) - 1).cast("int")).drop("value").cache()

hostsCanada= logFile.filter(logFile.value.contains(".ca ")).cache()
# split into 5 columns using regex and split
hostsCanada_graph =hostsCanada.withColumn('host', F.regexp_extract('value', '^(.*) - -.*', 1)) \
                .withColumn('timestamp', F.regexp_extract('value', '.* - - \[(.*)\].*',1)) \
                .withColumn('request', F.regexp_extract('value', '.*\"(.*)\".*',1)) \
                .withColumn('HTTP reply code', F.split('value', ' ').getItem(F.size(F.split('value', ' ')) -2).cast("int")) \
                .withColumn('bytes in the reply', F.split('value', ' ').getItem(F.size(F.split('value', ' ')) - 1).cast("int")).drop("value").cache()

hostsSingapore= logFile.filter(logFile.value.contains(".sg ")).cache()
# split into 5 columns using regex and split
hostsSingapore_graph = hostsSingapore.withColumn('host', F.regexp_extract('value', '^(.*) - -.*', 1)) \
                .withColumn('timestamp', F.regexp_extract('value', '.* - - \[(.*)\].*',1)) \
                .withColumn('request', F.regexp_extract('value', '.*\"(.*)\".*',1)) \
                .withColumn('HTTP reply code', F.split('value', ' ').getItem(F.size(F.split('value', ' ')) -2).cast("int")) \
                .withColumn('bytes in the reply', F.split('value', ' ').getItem(F.size(F.split('value', ' ')) - 1).cast("int")).drop("value").cache()

print(f"There are {hostsGerman.count()} requests for all hosts from Germany in total.") 
print(f"There are {hostsCanada.count()} requests for all hosts from Germany in total.")
print(f"There are {hostsSingapore.count()} requests for all hosts from Germany in total.")  
print('The visualization of all hosts from Germany is as follows :')
hostsGerman_graph.show(20,False)
print('The visualization of all hosts from Canada is as follows :')
hostsCanada_graph.show(20,False)
print('The visualization of all hosts from Singapore is as follows :')
hostsSingapore_graph.show(20,False)
#plot
hosts = ["German", "Canada", "Singapore"]
values = [hostsGerman.count(), hostsCanada.count(), hostsSingapore.count()]
fig, ax = plt.subplots()
ax.bar(hosts, values)
ax.set_title("Hosts Bar Chart")
ax.set_xlabel("Hosts")
ax.set_ylabel("Number of Visits")
plt.savefig("/home/acq22gy/com6012/ScalableML/Output/Q1_A1.png")

#Q2：
hostsGerman_uni = hostsGerman_graph.select('host').distinct().count()
hostsGerman_uni_sort = hostsGerman_graph.select('host').groupBy('host').count().sort('count', ascending=False).cache() 
hostsGerman_uni_row = hostsGerman_uni_sort.select("host").limit(9).collect()
hostsGerman_uni_max= [row.host for row in hostsGerman_uni_row]

hostsCanada_uni= hostsCanada_graph.select('host').distinct().count()
hostsCanada_uni_sort = hostsCanada_graph.select('host').groupBy('host').count().sort('count', ascending=False).cache()
hostsCanada_uni_row = hostsCanada_uni_sort.select("host").limit(9).collect()
hostsCanada_uni_max= [row.host for row in hostsCanada_uni_row]

hostsSingapore_uni = hostsSingapore_graph.select('host').distinct().count()
hostsSingapore_uni_sort = hostsSingapore_graph.select('host').groupBy('host').count().sort('count', ascending=False).cache()
hostsSingapore_uni_row = hostsSingapore_uni_sort.select("host").limit(9).collect()
hostsSingapore_uni_max= [row.host for row in hostsSingapore_uni_row]
print("\n\nHello Spark: There are %i unique hosts from German.\n" % (hostsGerman_uni))
print("Hello Spark: There are %i unique hosts from Canada.\n" % (hostsCanada_uni))
print("Hello Spark: There are %i unique hosts from Singapore.\n\n" % (hostsSingapore_uni))

print(f"The top 9 most frequent hosts in Germany are " + ", ".join(i for i in hostsGerman_uni_max))
print(f"The top 9 most frequent hosts in Canada are " + ", ".join(i for i in hostsCanada_uni_max))
print(f"The top 9 most frequent hosts in Singapore are " + ", ".join(i for i in hostsSingapore_uni_max))

# Q3
German_percentage=hostsGerman_uni_sort.limit(9)\
                 .withColumn("percentage", col("count") / hostsGerman.count())\
                .withColumn("percentage", format_number(col("percentage") * 100, 2))\
                .drop('count')   #first nine cols  
rest_row_German = Row("the rest", 100-German_percentage.select(sum("percentage")).collect()[0][0])
qC_host_German_percentage = German_percentage.union(spark.createDataFrame([rest_row_German], German_percentage.schema))

Canada_percentage=hostsCanada_uni_sort.limit(9)\
                .withColumn("percentage", col("count") / hostsCanada.count())\
                .withColumn("percentage", format_number(col("percentage") * 100, 2))\
                .drop('count')   #first nine cols 
rest_row_Canada = Row("the rest", 100-Canada_percentage.select(sum("percentage")).collect()[0][0])
qC_host_Canada_percentage = Canada_percentage.union(spark.createDataFrame([rest_row_Canada], Canada_percentage.schema))

Singa_percentage=hostsSingapore_uni_sort.limit(9)\
                .withColumn("percentage", col("count") / hostsSingapore.count())\
                .withColumn("percentage", format_number(col("percentage") * 100, 2))\
                .drop('count')   #first nine cols 
rest_row_Singa = Row("the rest", 100-Singa_percentage.select(sum("percentage")).collect()[0][0])
qC_host_Singa_percentage = Singa_percentage.union(spark.createDataFrame([rest_row_Singa], Singa_percentage.schema))
print('The visualization of the percentage of requests from German is:')
qC_host_German_percentage.show()
data = [("host62.ascend.int...", 3.90),
        ("aibn32.astro.uni-...", 3.01),
        ("ns.scn.de", 2.45),
        ("www.rrz.uni-koeln.de", 1.98),
        ("ztivax.zfe.siemen...", 1.81),
        ("sun7.lrz-muenchen.de", 1.31),
        ("relay.ccs.muc.deb...", 1.29),
        ("dws.urz.uni-magde...", 1.14),
        ("relay.urz.uni-hei...", 1.12),
        ("the rest", 81.99)]
df = spark.createDataFrame(data, ["host", "percentage"])
# create a matplotlib pie chart
labels = df.select("host").rdd.flatMap(lambda x: x).collect()
sizes = df.select("percentage").rdd.flatMap(lambda x: x).collect()
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
plt.axis('equal')
plt.title('German Host Distribution')
plt.savefig("/home/acq22gy/com6012/ScalableML/Output/German distribution Q3.png")
plt.clf()


print('The visualization of the percentage of requests from Canada is:')
qC_host_Canada_percentage.show()
# create a Pandas DataFrame from the given data
data = [("ottgate2.bnr.ca", 2.95),
        ("freenet.edmonton....", 1.34),
        ("bianca.osc.on.ca", 0.88),
        ("alize.ere.umontre...", 0.82),
        ("pcrb.ccrs.emr.ca", 0.79),
        ("srv1.freenet.calg...", 0.62),
        ("ccn.cs.dal.ca", 0.60),
        ("oncomdis.on.ca", 0.52),
        ("cobain.arcs.bcit....", 0.50),
        ("the rest", 90.98)]
df = spark.createDataFrame(data, ["host", "percentage"])

# create a matplotlib pie chart
labels = df.select("host").rdd.flatMap(lambda x: x).collect()
sizes = df.select("percentage").rdd.flatMap(lambda x: x).collect()
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
plt.axis('equal')
plt.title('Canada Host Distribution')
plt.savefig("/home/acq22gy/com6012/ScalableML/Output/Canada distribution Q3.png")
plt.clf()

print('The visualization of the percentage of requests from Singapore is:')
qC_host_Singa_percentage.show()
df = spark.createDataFrame([(u'merlion.singnet.com.sg', 29.14),
                            (u'sunsite.nus.sg', 3.78),
                            (u'ts900-1314.singnet.com.sg', 2.84),
                            (u'ssc25.iscs.nus.sg', 2.84),
                            (u'scctn02.sp.ac.sg', 2.37),
                            (u'ts900-1305.singnet.com.sg', 2.37),
                            (u'ts900-406.singnet.com.sg', 2.37),
                            (u'ts900-402.singnet.com.sg', 2.27),
                            (u'einstein.technet.sg', 2.18),
                            (u'the rest', 49.84)], ['host', 'percentage'])
pdf = df.select('host', 'percentage').toPandas()
plt.pie(pdf['percentage'], labels=pdf['host'], autopct='%1.1f%%')
plt.title('Singapore Host Distribution')
plt.axis('equal')
plt.savefig("/home/acq22gy/com6012/ScalableML/Output/Singapore distribution Q3.png")
plt.clf()

#Q4
German_max= hostsGerman_uni_sort.select("host").first()['host']    #step1. derive a new dataframe with 3 cols: day, hour, number of visits
German = logFile.filter(logFile.value.contains(German_max))  
German_graph = German.withColumn('timestamp', F.regexp_extract('value', '.* - - \[(.*)\].*',1))\
                .withColumn('day', F.regexp_extract('timestamp', '^(.*)/.*/.*', 1))\
                .withColumn('hour', F.regexp_extract('timestamp', '.*:(.*):.*:.* -.*', 1))\
                .drop('value')\
                .drop('timestamp')
German_graph_sort=German_graph.groupBy('day', 'hour').count().sort('day','hour')
heatmap_German = pd.pivot_table(German_graph_sort.toPandas(), values="count", index="hour", columns="day")   
fig, ax = plt.subplots()
img_Germany = plt.imshow(heatmap_German, cmap="YlGnBu", interpolation="nearest")
plt.xticks(range(len(heatmap_German.columns)), heatmap_German.columns)
plt.yticks(range(len(heatmap_German.index)), heatmap_German.index)
plt.xlabel('day',color='k') 
plt.ylabel('hour', color='k') 
cbar = ax.figure.colorbar(img_Germany, ax=ax)
cbar.ax.set_ylabel("number of visits", rotation=-90, va="bottom")
plt.title('Heatmap German frequent hosts',color='k')
plt.savefig("/home/acq22gy/com6012/ScalableML/Output/Q1_heatmapGerman.png")

Canada_max= hostsCanada_uni_sort.select("host").first()['host']    #step1. derive a new dataframe with 3 cols: day, hour, number of visits
Canada = logFile.filter(logFile.value.contains(Canada_max))  
Canada_graph = Canada.withColumn('timestamp', F.regexp_extract('value', '.* - - \[(.*)\].*',1))\
                .withColumn('day', F.regexp_extract('timestamp', '^(.*)/.*/.*', 1))\
                .withColumn('hour', F.regexp_extract('timestamp', '.*:(.*):.*:.* -.*', 1))\
                .drop('value')\
                .drop('timestamp')
Canada_graph_sort=Canada_graph.groupBy('day', 'hour').count().sort('day','hour')
heatmap_Canada = pd.pivot_table(Canada_graph_sort.toPandas(), values="count", index="hour", columns="day")   
fig, ax = plt.subplots()
img_Canada = plt.imshow(heatmap_Canada, cmap="YlGnBu", interpolation="nearest")
plt.xticks(range(len(heatmap_Canada.columns)), heatmap_Canada.columns)
plt.yticks(range(len(heatmap_Canada.index)), heatmap_Canada.index)
plt.xlabel('day',color='k') 
plt.ylabel('hour', color='k') 
cbar = ax.figure.colorbar(img_Canada, ax=ax)
cbar.ax.set_ylabel("number of visits", rotation=-90, va="bottom")
plt.title('Heatmap Canada frequent hosts',color='k')
plt.savefig("/home/acq22gy/com6012/ScalableML/Output/Q1_heatmapCanada.png")

#step1. derive a new dataframe with 3 cols: day, hour, number of visits
Sing_max= hostsSingapore_uni_sort.select("host").first()['host']   
Singa = logFile.filter(logFile.value.contains(Sing_max))  
Sing_graph = Singa.withColumn('timestamp', F.regexp_extract('value', '.* - - \[(.*)\].*',1))\
                .withColumn('day', F.regexp_extract('timestamp', '^(.*)/.*/.*', 1))\
                .withColumn('hour', F.regexp_extract('timestamp', '.*:(.*):.*:.* -.*', 1))\
                .drop('value')\
                .drop('timestamp')
Sing_graph_sort=Sing_graph.groupBy('day', 'hour').count().sort('day','hour')
heatmap_Sing = pd.pivot_table(Sing_graph_sort.toPandas(), values="count", index="hour", columns="day")   
fig, ax = plt.subplots()
img_Sing= plt.imshow(heatmap_Sing, cmap="YlGnBu", interpolation="nearest")
plt.xticks(range(len(heatmap_Sing.columns)), heatmap_Sing.columns)
plt.yticks(range(len(heatmap_Sing.index)), heatmap_Sing.index)
plt.xlabel('day',color='k') 
plt.ylabel('hour', color='k') 
cbar = ax.figure.colorbar(img_Sing, ax=ax)
cbar.ax.set_ylabel("number of visits", rotation=-90, va="bottom")
plt.title('Heatmap Singapore frequent hosts',color='k')
plt.savefig("/home/acq22gy/com6012/ScalableML/Output/Q1_heatmapSingapore.png")


spark.stop()