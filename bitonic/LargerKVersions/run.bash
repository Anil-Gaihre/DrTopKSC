#!/bin/bash

COUNTER=0
digit=8
p=30
k=256
alpha=5

#for (( p = 22 ; p < 31; p=p+1  )) ### Inner for loop ###
#do
	for (( k = 4096; k < 16777216; k=k*2  ))
	do
		#     ./main $p $k $digit 0
		for (( alpha = 10; alpha <= 12; alpha++  )) 
		do
			#for ((i= 0; i <5; i++  )) 
			#do
			./topk.bin $p $k $alpha
			#done
		done
		echo""
		echo ""
		# echo -n "$i "
	done
#done

