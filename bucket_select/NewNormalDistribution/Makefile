exe = topk.bin 
#exe = topk_Uint_uniform.bin 

#exe = topk_Uint_Normal.bin 
#exe = topk_float_Normal.bin 
#exe = topk_float_Uniform.bin 

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
	./$(exe) 30 1024 256

clean:
	rm -rf $(exe) $(objs) 
