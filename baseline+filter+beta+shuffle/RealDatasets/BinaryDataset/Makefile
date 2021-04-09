exe = topk.bin 
#exe = topk_normal.bin 

cc = "$(shell which nvcc)" 
#flags = -I. -fopenmp -march=athlon64 -O3
flags = -I. -O3
flags += -std=c++11
 
ifeq ($(debug), 1)
	flags+= -DDEBUG 
endif

objs = $(patsubst %.cu,%.o,$(wildcard *.cu))

deps = $(wildcard ./*.cuh) \
	   $(wildcard *.h) \
	   Makefile

%.o:%.cu $(deps)
	$(cc) -c $< -o $@ $(flags)

$(exe):$(objs)
	$(cc) $(objs) -o $(exe) $(flags)

test:$(exe)
	./$(exe) 30 16777216 8 2

test1:$(exe)
	./$(exe) 29 1 8 33554432

test2:$(exe)
	./$(exe) 30 1 8 33554432


clean:
	rm -rf $(exe) $(objs) 
