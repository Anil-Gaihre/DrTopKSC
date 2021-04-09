#!/bin/bash

for (( k=1; k<=16777216; k=k*2 ))
do
	./topk_Increasing.bin 30 $k 8 2 1 >> BMWIncreasing.dat		
	echo -n "Finihed processing for a k "
done
