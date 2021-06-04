#!/bin/bash

COUNTER=0
bucket=256
#p=30
k=8192
#k=4194304
alpha=11
beta=2
#set inside the code
p=29

for (( k = 2097152 ; k < 67108864; k=k*2  )) ### Inner for loop ###
do
      #  for ((alpha = 10; alpha < 14; alpha++))
      #  do
                #./exe exp_num_element k alpha 1=NormalBitonic/0=DrTopKBitonic
                ./topk.bin $p $k $alpha 1
     #           echo "Finished for a k"
    #            ./topk_Uint_Normal.bin $p $k $alpha 0
   #             echo "Finished for a k"
          
       # done
#              ./topk_Uint_uniform.bin $p $k $alpha 1
#                echo "Finished for a k"
 #               ./topk_Uint_Normal.bin $p $k $alpha 1
  #              echo "Finished for a k"

done

