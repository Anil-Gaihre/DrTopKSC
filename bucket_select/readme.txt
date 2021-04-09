Recent Working directory : /home/anil/anil/topk/sample/TopK/bucket @alienware

The code includes both the implementation of the Inplace Bucket select and not inplace bucket select, into the same code and switches between them dynamically. The bucket seelect is similar to "Fast K-selection Algorithms for GPU" but is delivering faster performance. That could be because of the enhanced atomic operation in Titan X GPU.
