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

for (( k = 1 ; k <= 33554432; k=k*2  )) ### Inner for loop ###
do
	# for ((i = 0; i < 5; i++))
	# do
		./topk.bin $p $k $bucket >> customized_bucket.dat
	# done 
	echo "Finished for a k"
done

# k=128
# for (( p = 22 ; p < 31; p=p+1  )) ### Inner for loop ###
# do
# 	for ((i = 0; i < 5; i++))
# 	do
# 		./topk.bin $p $k $bucket
# 	done 
# 	echo "Finished for a k"
# done

# k=256
# for (( p = 22 ; p < 31; p=p+1  )) ### Inner for loop ###
# do
# 	for ((i = 0; i < 5; i++))
# 	do
# 		./topk.bin $p $k $bucket
# 	done 
# 	echo "Finished for a k"
# done


# ##Bitonic test runs
# p=30
# for (( k = 1 ; k < 4096; k=k*2  )) ### Inner for loop ###
# do
# 	for ((alpha = 4; alpha < 13; alpha++))
# 	do
# 		#./exe exp_num_element k alpha 1=NormalBitonic/0=DrTopKBitonic
# 		/home/anil/topk_20/DrTopK/bitonic/topk_Uint_uniform.bin $p $k $alpha 0
# 		echo "Finished for a k"
# 		/home/anil/topk_20/DrTopK/bitonic/topk_Uint_Normal.bin $p $k $alpha 0
# 		echo "Finished for a k"
# 		/home/anil/topk_20/DrTopK/bitonic/topk_Uint_uniform.bin $p $k $alpha 1
# 		echo "Finished for a k"
# 		/home/anil/topk_20/DrTopK/bitonic/topk_Uint_Normal.bin $p $k $alpha 1
# 		echo "Finished for a k"
# 	done 

# done
#   for (( k = 1 ; k < 536870912; k=k*2  )) ### Inner for loop ###
#  do
#     for ((i = 0; i < 5; i++))
#          do
#                ./topk_float_Uniform.bin $p $k $bucket
#          done 
#      echo ""
#  done
#
#   for (( k = 1 ; k < 536870912; k=k*2  )) ### Inner for loop ###
#  do
#     for ((i = 0; i < 5; i++))
#          do
#                ./topk_float_Normal.bin $p $k $bucket
#          done 
#      echo ""
#  done


