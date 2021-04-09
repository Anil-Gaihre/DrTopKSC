#!/bin/bash
type=3
distributionType=0
testCount=5
startPower=22
stopPower=29
N=30
beta=2
declare -a arr=("DIFFLOGPARAMETER_Concave_alpha_figure" "U_K2" "U_K4" "U_K8" "U_K16" "U_K32" "U_K64" "U_K128" "U_K256" "U_K512")
#declare -a k=(1 2 4 8 16 32 64 128 256 512)
 for (( alpha=3; alpha<=16; alpha=alpha+1 ))
 do
	for (( k=8192; k<=8192; k=k*2 ))
	do
		./topk.bin $N $k 8 $beta $alpha >> ${arr[0]}
		echo -n "Finihed processing for a k "
	done
 done
printf "\n"
