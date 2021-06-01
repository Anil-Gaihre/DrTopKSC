# DrTopKSC
--
Software (tested)
-----
g++ 7.5.0, CUDA V11.1.105
Compilation flag: -O3

-----
--
Hardware (tested)

Titan Xp, V100 (tested)

------
--
Compile

Inside the folders type: (baseline+filter+beta+shuffle contains final radis select version)
make

------
--
Execute

Type: "./topk.bin" it will show you what is needed.

Tips: log2(|V|), k, Bits per digit in Radix = 8, beta=2   

Example run: make test
------
