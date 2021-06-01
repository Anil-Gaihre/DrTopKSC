exe = topk.bin 
#exe = topk_normal.bin 
flags = -I. -O3
flags += -std=c++11
 
cucc= "$(shell which nvcc)"
cc= "$(shell which mpicxx)"
commflags=-lcudart -L"$(shell dirname $(cucc))"/../lib64
#cuflags= --compiler-options -Wall -Xptxas -v -Xcudafe -\# --resource-usage 
 cuflags= --compiler-options -Wall -v -Xcudafe -\# --resource-usage
objs	= $(patsubst %.cu,%.o,$(wildcard *.cu)) \
	$(patsubst %.cpp,%.o,$(wildcard *.cpp))

deps	= $(wildcard ./*.cuh) \
		  Makefile

%.o:%.cu $(deps)
	$(cucc) -c $(cuflags) $< -o $@ $(flags)

%.o:%.cpp $(deps)
	$(cc) -c  $< -o $@

$(exe):$(objs)
	$(cc) $(objs) $(commflags) -o $(exe)  $(flags)

test:$(exe)
	jsrun --nrs 1 --tasks_per_rs 4 --cpu_per_rs 42 --gpu_per_rs 4 --rs_per_host 1 ./topk.bin 30 128 8 2 4 1
#	jsrun --nrs 1 --tasks_per_rs 1 --cpu_per_rs 1 --gpu_per_rs 1 --rs_per_host 1 ./topk.bin 33 128 8 2 1
	# jsrun --nrs 1 --tasks_per_rs 1 --cpu_per_rs 1 --gpu_per_rs 1 --rs_per_host 1 ./topk.bin 32 128 8 2 1
	# jsrun --nrs 1 --tasks_per_rs 1 --cpu_per_rs 1 --gpu_per_rs 1 --rs_per_host 1 ./topk.bin 26 4 8 2 1
	# jsrun --nrs 1 --tasks_per_rs 1 --cpu_per_rs 1 --gpu_per_rs 1 --rs_per_host 1 ./topk.bin 29 16777216 8 2 1
 
test1:$(exe)
	# jsrun --nrs 1 --tasks_per_rs 2 --cpu_per_rs 42 --gpu_per_rs 2 --rs_per_host 1 ./topk.bin 29 4 8 2 2
	# jsrun --nrs 1 --tasks_per_rs 2 --cpu_per_rs 42 --gpu_per_rs 2 --rs_per_host 1 ./topk.bin 31 128 8 2 2
	jsrun --nrs 1 --tasks_per_rs 2 --cpu_per_rs 42 --gpu_per_rs 2 --rs_per_host 1 ./topk.bin 33 128 8 2 2

test2:$(exe)
	# jsrun --nrs 1 --tasks_per_rs 4 --cpu_per_rs 42 --gpu_per_rs 4 --rs_per_host 1 ./topk.bin 31 128 8 2 4
	# jsrun --nrs 1 --tasks_per_rs 4 --cpu_per_rs 42 --gpu_per_rs 4 --rs_per_host 1 ./topk.bin 34 128 8 2 4 #out of memory in main.cu at line 204
	jsrun --nrs 1 --tasks_per_rs 4 --cpu_per_rs 42 --gpu_per_rs 4 --rs_per_host 1 ./topk.bin 33 128 8 2 4
	# jsrun --nrs 1 --tasks_per_rs 4 --cpu_per_rs 42 --gpu_per_rs 4 --rs_per_host 1 ./topk.bin 29 16777216 8 2 4 


clean:
	rm -rf $(exe) $(objs) 
