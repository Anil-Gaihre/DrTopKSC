#!/bin/bash

COUNTER=0
bucket=256
N=27
k=8192
#k=4194304
alpha=11
beta=2


# declare -a arr=("UK_DrBucket.csv" "kmer_V1r_DrBucket.csv" "KNN_DrBucket.csv" "UK_log" "kmer_V1r_log" "KNN_log" "CWEB_DrBucket.csv" "CWEB_log" "U_K256" "U_K512")
declare -a arr=("Twitter_DrBucketN30.csv" "kmer_V1r_DrBucket.csv" "KNN_DrBucket.csv" "TwitterN30_log" "kmer_V1r_log" "KNN_log" "CWEB_DrBucket.csv" "CWEB_log" "U_K256" "U_K512")
#UK: 105153953: N =26
#kmer_V1r : V=214005017: N= 27

#declare -a k=(1 2 4 8 16 31 64 128 256 512)
#for (( beta=2; beta<3; beta=beta+1 ))
# for (( k=1; k<33554431; k=k*2 ))
# do
# 	for (( N=26; N<=26; N=N+1 ))
# 	do
# 		./topk.bin $N $k $bucket /scratch/datasets/topkData/graph_project_start/graph_reader/degreeUK.dat ${arr[0]} >> ${arr[3]}
# 		#./topk.bin  29 33554431 8 2
# 		echo -n "Finihed processing for a k "
# 	done
# done
# printf "\n"

# for (( k=1; k<33554431; k=k*2 ))
# do
# 	for (( N=27; N<=27; N=N+1 ))
# 	do
# 		./topk.bin $N $k $bucket /scratch/datasets/topkData/graph_project_start/graph_reader/degreeKmer.dat ${arr[1]} >> ${arr[4]}
# 		#./topk.bin  29 33554431 8 2
# 		echo -n "Finihed processing for a k "
# 	done
# done
# printf "\n"

for (( k=1; k<33554431; k=k*2 ))
do
	for (( N=30; N<=30; N=N+1 ))
	do
		./topk.bin $N $k $bucket /scratch/datasets/topkData/twitter/Twitter-COVID-dataset---Jan-2021/Scores_Twitter.dat ${arr[0]} >> ${arr[3]}
		#./topk.bin  29 33554431 8 2
		echo -n "Finihed processing for a k "
	done
done
printf "\n"

