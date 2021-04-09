#!/bin/bash
type=3
distributionType=0
testCount=5
startPower=22
stopPower=29 
N=29
k=33554432
beta=2
declare -a arr=("DIFFLOGPARAMETER_VARY_N" "U_K2" "U_K4" "U_K8" "U_K16" "U_K32" "U_K64" "U_K128" "U_K256" "U_K512")
#declare -a k=(1 2 4 8 16 32 64 128 256 512)
#for (( beta=2; beta<5; beta=beta+1 ))
#do
	for (( N=27; N<=30; N=N+1 ))
	do
		./topk.bin $N $k 8 $beta >> ${arr[0]}
		#./topk.bin  29 33554432 8 2
		echo -n "Finihed processing for a k "
	done
#done
printf "\n"
