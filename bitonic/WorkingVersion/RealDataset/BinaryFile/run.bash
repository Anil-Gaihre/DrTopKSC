#!/bin/bash
alpha=5

N=30
#k=33554431
beta=2
declare -a arr=("UK_DrBitonic.csv" "kmer_V1r_DrBitonic.csv" "KNN_DrBitonic.csv" "UK_log" "kmer_V1r_log" "KNN_log" "U_K64" "U_K128" "U_K256" "U_K512")
#UK: 105153953: N =26
#kmer_V1r : V=214005017: N= 27

#declare -a k=(1 2 4 8 16 31 64 128 256 512)
#for (( beta=2; beta<3; beta=beta+1 ))
# N=26
# for (( k=1; k<33554431; k=k*2 ))
# do
# 	for (( alpha=10;alpha<=16; alpha=alpha+1 ))
# 	do
# 		./topk.bin $N $k $alpha /scratch/datasets/topkData/graph_project_start/graph_reader/degreeUK.dat ${arr[0]} >> ${arr[3]}
# 		#./topk.bin  29 33554431 8 2
# 		echo -n "Finihed processing for a k "
# 	done
# done
# printf "\n"

# N=27
# for (( k=1; k<33554431; k=k*2 ))
# do
# 	for (( alpha=10;alpha<=16; alpha=alpha+1 ))
# 	do
# 		./topk.bin $N $k $alpha /scratch/datasets/topkData/graph_project_start/graph_reader/degreeKmer.dat ${arr[1]} >> ${arr[4]}
# 		#./topk.bin  29 33554431 8 2
# 		echo -n "Finihed processing for a k "
# 	done
# done
# printf "\n"
N=29
alpha=8
for (( k=2048; k<33554431; k=k*2 ))
do
	#for (( alpha=5;alpha<=16; alpha=alpha+1 ))
	#do
		./topk.bin $N $k $alpha /scratch/SIFT_SCORES_ID320578621.bin ${arr[2]} >> log.dat
		#./topk.bin  29 33554431 8 2
		echo -n "Finihed processing for a k "
	#done
done
printf "\n"

