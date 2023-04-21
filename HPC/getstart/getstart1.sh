qrshx -P rse-com6012 -pe smp 4 # request 4 CPU cores using our reserved queue

module load apps/java/jdk1.8.0_102/binary
module load apps/python/conda

conda activate myspark

# cd /data/*abc1de*/ScalableML  # *abc1de* should be replaced by your username
# pyspark

#conda install -y numpy # install numpy, to be used in Task 3. This ONLY needs to be done ONCE. NOT every time.
cd com6012/ScalableML # our main working directory
pyspark --master local[4] # start pyspark with 4 cores requested above.
