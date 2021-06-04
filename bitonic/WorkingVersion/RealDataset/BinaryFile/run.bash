#!/bin/bash
alpha=5

N=30
#k=33554431
beta=2
declare -a arr=("UK_DrBitonic.csv" "kmer_V1r_DrBitonic.csv" "KNN_DrBitonic.csv" "UK_log" "kmer_V1r_log" "KNN_log" "U_K64" "U_K128" "U_K256" "U_K512")

N=29
alpha=8
for (( k=2048; k<33554431; k=k*2 ))
do
	for (( alpha=5;alpha<=16; alpha=alpha+1 ))
	do
		./topk.bin $N $k $alpha /scratch/SIFT_SCORES_ID320578621.bin ${arr[2]} >> log.dat
		#./topk.bin  29 33554431 8 2
		echo -n "Finihed processing for a k "
	done
done
printf "\n"

