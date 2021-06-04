# DrTopK
----
**Software**

g++ 7.5.0, CUDA V11.1.105, IBM Spectrum (MPI) for multi-GPU/multi-node settings

-----

**Compilation flags: **-O3, -std=c++11

-----
--
**Hardware (tested)**

Titan Xp, V100 (tested), Summit supercomputer for multi-GPU multi-node version (Used 4 GPUs per node while evaluation.)

------
--
**Compile**

Inside the folders type:

make

Note: baseline+filter+beta+shuffle contains final radix top-k version

------
--
**Execute**

Type: "./topk.bin" it will show you what is needed.

Tips: log2(|V|), k, Bits per digit in Radix = 8, beta=2, Note: |V| and k are power of 2   

**Example test:** make test

------

**Additional information on different versions:**

Top-k versions are classified as the folders name. Baseline, baseline+filter+shuffle, baseline+filter+beta,  baseline+filter+beta+shuffle are radix select versions with different optimizations. **baseline+filter+beta+shuffle** is the final radix select version. The running bash scripts are present in the respective folders.

The bitonic top-k is in home directory of bitonic folder. Other subdirectories contains different modified versions to accomodate large k in bitonic top-k. Real world datasets are in the respective versions testings. In bitonic version the user enters alpha from the command line.

----
**Dataset generation:** We used std::uniform_int_distribution, std::normal_distribution a customized distribution (code to generate the customized distribution in included in the folder). The real world datasets (https://drive.google.com/drive/folders/1lc6iPOIOn0CIABboBxNIjZjLOsoxlVfM) are processed as mentioned in the paper and loaded to run the tests. Note, for binary dataset, we should use the code inside the folders "Binary dataets". The state of the art tools are also updated during data generation to run with the same datasets that Dr.Top-k uses.


**Estimated run time:**
It is relatively faster to test test the performance of Dr.Top-k assisted radix and Dr.top-k assisted bucket top-k. It will take approximately half an hour for a version (Dr.Top-k assisted radix (baseline+filter+beta+shuffle) and bucket_select for all k tests to run. 

For bitonic we have to switch between different versions because, they are designed in order to deliver the optimum performance for different values of k. Also we assign alpha to the bitonic through the commandline. The value of alpha is about 16 to 5 for different for increasing k from 2^0 to 2^24. (Total about 1-1.5 hours)

The real world datasets are read from an external file. It takes time to load the files given they are larger in size. It should take about 3-4 minutes per k (Total about 1.5 hour) to run for real world dataset.

**Scripts (names/links) to use for setup/evaluation:** Respective folders have required scripts “run.bash” in them. For instance, the Dr.Top-k assisted radix top-k (all optimizations included) version has the script run.bash at the location, /baseline+filter+beta+shuffle/run.bash

 
-----

**Others:** We also added a modified code (alphaManualTuning folder) where the user can manually enter the value of alpha and see how the performance vary with it.

-----
