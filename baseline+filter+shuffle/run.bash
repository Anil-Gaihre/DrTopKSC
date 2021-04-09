#!/bin/bash
type=3
distributionType=0
testCount=5
startPower=22
stopPower=29
N=29
declare -a arr=("DIFFLOGPARAMETER" "U_K2" "U_K4" "U_K8" "U_K16" "U_K32" "U_K64" "U_K128" "U_K256" "U_K512")
#declare -a k=(1 2 4 8 16 32 64 128 256 512)
for (( beta=2; beta<5; beta=beta+1 ))
do
	for (( k=1; k<=33554432; k=k*2 ))
	do
		./topk.bin $N $k 8 $beta >> ${arr[0]}
		echo -n "Finihed processing for a k "
	done
done
printf "\n"
