Recent Working directory : /home/anil/anil/topk/sample/TopK/test/OCT12 @alienware

The code includes the Max Sampling before the first Topk and the second Topk. The 3 operators (local sort, Merge and rebuild are merged into 1 kernel. The second kernel consists of Merge and rebuild and is called repeatedly untill the new number of elements reaches to K.

The implementation is done in the shared memory where the normal Number of elements per block is 8*BlockDim. This causes the limitation of usage of shared memory to higher value of K. Currently as we have to save the subrange ID information of the TopK elements . More shared memory is used. Does Bitonic sampling till 512.

But we can increase the value of K by only copying 2*K elements into the shared memory. And refilling it with other elements once the processing of the 2K elements is done in the same Kernel call.

With sampling the Bitonic Topk is a faster than the sigmod. for N=2^29, K=1024, alpha~11, the time is around 6 ms while the Sigmod Bitonic is taking around 100 ms.
