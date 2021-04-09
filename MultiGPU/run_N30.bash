#!/bin/bash
### Begin BSUB Options
#BSUB -P CSC289
#BSUB -J Supernodes_Pre2_6nodes
#BSUB -W 02:00
#BSUB -nnodes 6
#BSUB -alloc_flags "smt1"
### End BSUB Options and begin shell commands
module load cuda
N=30
k=128
beta=2
declare -a arr=("GPU_1_N30" "GPU_2_N30" "GPU_4_N30" "baseline" "U_K16" "U_K31" "U_K64" "U_K128" "U_K256" "U_K512")
#declare -a k=(1 2 4 8 16 31 64 128 256 512)
#for (( beta=2; beta<3; beta=beta+1 ))
for (( k=128; k<=128; k=k*2 ))
do
	# for (( NGPU=1; NGPU < 8; NGPU=NGPU*2 ))
	# do
		jsrun --nrs 1 --tasks_per_rs $NGPU --cpu_per_rs 42 --gpu_per_rs ${NGPU} --rs_per_host 1 ./topk.bin ${N} ${k} 8 ${beta} ${NGPU} >> ${arr[0]}
		echo -n "Finihed processing for a k "
	# done
done
