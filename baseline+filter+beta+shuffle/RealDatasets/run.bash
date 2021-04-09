#!/bin/bash

N=30
#k=33554431
beta=2
#declare -a arr=("UK_DrRadix.csv" "kmer_V1r_DrRadix.csv" "KNN_DrRadix.csv" "UK_log" "kmer_V1r_log" "KNN_log" "U_K64" "U_K128" "U_K256" "U_K512")
declare -a arr=("Twitter_DrRadix.csv" "kmer_V1r_DrRadix.csv" "KNN_DrRadix.csv" "Twitter_log" "kmer_V1r_log" "KNN_log" "U_K64" "U_K128" "U_K256" "U_K512")
#UK: 105153953: N =26
#kmer_V1r : V=214005017: N= 27

#declare -a k=(1 2 4 8 16 31 64 128 256 512)
#for (( beta=2; beta<3; beta=beta+1 ))
for (( k=1; k<33554431; k=k*2 ))
do
	for (( N=26; N<=26; N=N+1 ))
	do
		./topk.bin $N $k 8 $beta /scratch/datasets/topkData/twitter/Twitter-COVID-dataset---Jan-2021/Scores_Twitter.dat ${arr[0]} >> ${arr[3]}
		#./topk.bin  29 33554431 8 2
		echo -n "Finihed processing for a k "
	done
done
printf "\n"

# for (( k=1; k<33554431; k=k*2 ))
# do
# 	for (( N=27; N<=27; N=N+1 ))
# 	do
# 		./topk.bin $N $k 8 $beta /scratch/datasets/topkData/graph_project_start/graph_reader/degreeKmer.dat ${arr[1]} >> ${arr[4]}
# 		#./topk.bin  29 33554431 8 2
# 		echo -n "Finihed processing for a k "
# 	done
# done
# printf "\n"

# for (( k=1; k<33554431; k=k*2 ))
# do
# 	for (( N=26; N<=26; N=N+1 ))
# 	do
# 		./topk.bin $N $k 8 $beta /scratch/datasets/topkData/graph_project_start/graph_reader/KNN.dat ${arr[2]} >> ${arr[5]}
# 		#./topk.bin  29 33554431 8 2
# 		echo -n "Finihed processing for a k "
# 	done
# done
# printf "\n"

