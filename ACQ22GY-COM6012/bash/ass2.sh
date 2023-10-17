#qrshx -P rse-com6012 -pe smp 4 -l rmem=32G # request 4 CPU cores using our reserved queue
module load apps/java/jdk1.8.0_102/binary
module load apps/python/conda
#conda activate myspark
cd com6012/ScalableML # our main working directory
spark-submit --driver-memory 32g --executor-memory 32g Code/ass2.py
