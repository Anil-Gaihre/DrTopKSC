#!/bin/bash
type=3
distributionType=0
testCount=5
startPower=22
stopPower=29 
N=30
#k=33554431
beta=2
declare -a arr=("optimum_FICO_Vector_SizeUniform" "optimum_FICO_Vector_Size_Normal" "baseline_filter" "baseline" "U_K16" "U_K31" "U_K64" "U_K128" "U_K256" "U_K512")

# for (( k=2097152; k<=33554431; k=k*2 ))
# do
# 	for (( N=30; N<=30; N=N+1 ))
# 	do
# 		./topk.bin $N $k 8 $beta >> ${arr[0]}
# 		#./topk.bin  29 33554431 8 2
# 		echo -n "Finihed processing for a k "
# 	done
# done
# printf "\n"

for (( k=2097152; k<=33554431; k=k*2 ))
do
	for (( N=30; N<=30; N=N+1 ))
	do
		./topk_diff_N.bin  $N $k 8 $beta >> ${arr[1]}
		
		#./topk.bin  29 33554431 8 2
		echo -n "Finihed processing for a k "
	done
done
printf "\n"


