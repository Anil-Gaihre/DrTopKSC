#!/bin/bash

COUNTER=0
bucket=256
p=30
k=8192
#k=4194304
alpha=11
beta=2
#set inside the code

#   for (( k = 1 ; k <= 33554432; k=k*2  )) ### Inner for loop ###
#  do
#     for ((i = 0; i < 5; i++))
#          do
#                ./topk_Uint_uniform.bin $p $k $bucket
#          done 
#      echo "Finished for a k"
#  done

for (( k = 4194304 ; k <= 33554432; k=k*2  )) ### Inner for loop ###
do
	for ((i = 0; i < 5; i++))
	do
		./topk_Uint_Normal.bin $p $k $bucket
	done 
	echo "Finished for a k"
done

