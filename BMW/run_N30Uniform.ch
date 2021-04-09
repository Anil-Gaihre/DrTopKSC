#!/bin/bash

for (( k=4194304; k<=16777216; k=k*2 ))
do
	./topk.bin 30 $k 8 2 1 >> HighUniform.dat		
	echo -n "Finihed processing for a k "
done

for (( k=4194304; k<=16777216; k=k*2 ))
do
	./topk.bin 30 $k 8 2 1 >> HighNormal.dat		
	echo -n "Finihed processing for a k "
done
