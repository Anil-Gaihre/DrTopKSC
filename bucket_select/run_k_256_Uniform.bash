#!/bin/bash

COUNTER=0
bucket=256
p=30
k=8192
#k=4194304
alpha=11
beta=2
#set inside the code

k=1024
for (( p = 22 ; p < 31; p=p+1  )) ### Inner for loop ###
do
	for ((i = 0; i < 5; i++))
	do
		./topk.bin $p $k $bucket
	done 
	echo "Finished for a k"
done

