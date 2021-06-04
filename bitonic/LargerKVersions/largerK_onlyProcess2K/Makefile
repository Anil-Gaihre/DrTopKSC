exe = topk.bin 
# exe = topk_Uint_uniform.bin 

# exe = topk_Uint_Normal.bin 
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
	#./$(exe) 20 1024 8 5 4
	./$(exe) 29 1024 12 1

clean:
	rm -rf $(exe) $(objs) 
