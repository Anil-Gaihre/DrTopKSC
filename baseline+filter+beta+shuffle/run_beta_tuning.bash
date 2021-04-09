#!/bin/bash
type=3
distributionType=0
testCount=5
startPower=22
stopPower=29 
N=30
#k=33554431
beta=2
declare -a arr=("optimum_FICO" "baseline_filter_shuffle" "baseline_filter" "baseline" "U_K16" "U_K31" "U_K64" "U_K128" "U_K256" "U_K512")
#declare -a k=(1 2 4 8 16 31 64 128 256 512)
#for (( beta=2; beta<3; beta=beta+1 ))

for (( k=4194304; k<67108864; k=k*2 ))
do
	for (( beta=2; beta<5; beta=beta+1 ))
	do
		./topk.bin $N $k 8 $beta >> ${arr[0]}
		#./topk.bin  29 33554431 8 2
		echo -n "Finihed processing for a k "
	done
done
printf "\n"
