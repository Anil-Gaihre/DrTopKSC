# DrTopKSC
----
Software

g++ 7.5.0, CUDA V11.1.105, IBM Spectrum (MPI) for multi-GPU/multi-node settings

-----

Compilation flags: -O3, -std=c++11

-----
--
Hardware (tested)

Titan Xp, V100 (tested), Summit supercomputer for multi-GPU multi-node version (Used 4 GPUs per node while evaluation.)

------
--
Compile

Inside the folders type: (baseline+filter+beta+shuffle contains final radix select version)

make

------
--
Execute

Type: "./topk.bin" it will show you what is needed.

Tips: log2(|V|), k, Bits per digit in Radix = 8, beta=2   

Example run: make test
------

Additional information on different versions:

Top-k versions are classified as the folders name. Baseline, baseline+filter+shuffle, baseline+filter+beta,  baseline+filter+beta+shuffle are radix select versions with different optimizations. **baseline+filter+beta+shuffle** is the final radix select version. The running bash scripts are present in the respective folders.

The bitonic top-k is in home directory of bitonic folder. Other subdirectories contains different modified versions to accomodate large k in bitonic top-k. Real world datasets are in the respective versions testings.

