#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <assert.h>
#include <cmath>
#include "wtime.h"
// #define _MONITOR_ 0
static void HandleError( cudaError_t err, const char *file, int line   ) {
	if (err != cudaSuccess) {
		printf( "\n%s in %s at line %d\n", \
				cudaGetErrorString( err   ), file, line );
		exit( EXIT_FAILURE   );
	}
}
#define H_ERR( err   ) \
	(HandleError( err, __FILE__, __LINE__   ))

using namespace std;
struct max_withIndex
{
	int index;
	int value;
};
	template<typename data_t, typename index_t>
__host__ __device__ data_t max(data_t a,data_t b,bool& firstselect)
{
	if (a>b)
	{
		firstselect=true;
		return a;
	}
	else
	{
		firstselect=false;
		return b;
	}
}
template<typename data_t, typename index_t>
void swap_ptr_index(index_t* &a, index_t* &b){
	index_t* temp = a;
	a = b;
	b = temp;

}
template<typename data_t, typename index_t>
void swap_ptr_data(data_t* &a, data_t* &b){
	data_t* temp = a;
	a = b;
	b = temp;

}

template<typename data_t, typename index_t>
__host__ __device__ __forceinline__ void Exchange(data_t& i, data_t& j) {
	data_t t;
	t = i;
	i= j;
	j = t;
}

template<typename data_t, typename index_t>
__host__ __device__ __forceinline__ void Exchange_Sub_index(index_t& m,index_t& n) {
	index_t p;
	p=m;
	m=n;
	n=p;
}

	template<typename data_t, typename index_t>
__global__ void CreateConcatenateRange(data_t* vec,data_t* ConcatenatedRange,index_t* SubrangeId,index_t ConcatenatedSize,index_t k,index_t Subrangesize)
{
	int thid = blockDim.x*blockIdx.x+threadIdx.x;
	int laneId= threadIdx.x & 0x1f;
	int myWarpID=thid >> 5;
	int NWarps=(blockDim.x*gridDim.x) >> 5;
	int NContributingSubrange=k; //for single maximum selection it will be k . For beta selection, the number is likely to be less than k
	while (myWarpID < NContributingSubrange)//
	{
		index_t mybegin_pos=(myWarpID*Subrangesize)+laneId;
		index_t myend_pos=(((myWarpID+1)*Subrangesize) < ConcatenatedSize) ? (myWarpID+1)*Subrangesize : ConcatenatedSize;
		index_t index=SubrangeId[myWarpID];
		index_t myvec_pos=index* Subrangesize+laneId;
		while (mybegin_pos<myend_pos)
		{
			if (mybegin_pos >= ConcatenatedSize)
			{
				printf("Error! Illegal memory access. \n");
			}
			ConcatenatedRange[mybegin_pos]=vec[myvec_pos];
			mybegin_pos+=32; 
			myvec_pos+=32;
		}
		myWarpID+=NWarps;
	}
	return;
}

	template<typename data_t, typename index_t>
__global__ void Combined_Sigmod_Merge(data_t* vec,index_t N,index_t top,index_t* SubrangeId,index_t NElementsperBlock,data_t* temp)
{
	index_t myBlockID=blockIdx.x;
	extern __shared__ data_t SMem[];
	index_t  SelectedNumber;//=NElementsperBlock;
	index_t factor= (NElementsperBlock >= top) ? NElementsperBlock >> 1: top >> 1 ;

	while(myBlockID*NElementsperBlock <N )
	{
		int count;
		SelectedNumber=NElementsperBlock;
		if (NElementsperBlock > blockDim.x)
		{
			count=0;
			index_t begin=threadIdx.x;//+blockDim.x;
			while(begin < NElementsperBlock)
			{
				SMem[begin]=vec[myBlockID*NElementsperBlock+begin];
				begin+=blockDim.x;
				count++;
			}
		}
		else//Copy all the elements from vec to SMem of size blockdim // if (threadIdx.x < top)//Loading only k elements in the shared memory. 1 block will work on k elements at 1 time
		{
			if (threadIdx.x<NElementsperBlock)
				SMem[threadIdx.x]=vec[myBlockID*blockDim.x+threadIdx.x];
		}

		__syncthreads(); //newly added OCT 10

		for (index_t len= 1 ;len< top; len= len<<1)//----------On completion of this loop--> Finishes local sorting of all the elements of k groups assigned to this block----------//
		{
			index_t dir=len << 1;
			for (index_t inc=len; inc > 0; inc=inc >> 1)//arranging the elements between the lower boundary and the upper boundary
			{
				index_t thid=threadIdx.x;
				index_t thid_end=factor;
				while(thid < thid_end)// For conditon when blockdim <k .1 thread in block will work for multiple times for 1 k sorting
				{
					index_t low=thid & (inc-1);
					index_t i= (thid <<1)-low;
					bool reverse=((dir & i)==0);
					bool swap=reverse^(SMem[i]<SMem[i+inc]);// ^ Xor--> A ^ B is only true when A,B are opposite boolean 
					// comparison++;
					if (swap)
					{      
						Exchange<data_t,index_t>(SMem[i],SMem[i+inc]);
					}
					thid+=blockDim.x;
				}
				__syncthreads();//synchronizes the threads within the block

			}

		}

		//----------------------------Starting Merge Operation------------------------------//

		index_t thid_shared=threadIdx.x;
		index_t thid_shared_end=factor;
		while(thid_shared<thid_shared_end)
		{
			index_t low= thid_shared & (top-1); //thid_shared might be used directly --> avoids mapping to the shared memory index like calculation of o
			index_t i= (thid_shared<<1)-low;
			bool firstselect=false;
			data_t temp_data=max<data_t,index_t>(SMem[i],SMem[i+top],firstselect);
			__syncthreads();
			SMem[thid_shared]=temp_data;
			//          SMem[thid_shared]=temp_data[threadIdx.x];
			thid_shared+=blockDim.x;
		}
		//----------------------------Finished Merge Operation------------------------------//

		SelectedNumber= SelectedNumber>>1;


		//   -----------------------------------------------------------------------------------------------------//       
		//----------------------------Start Rebuild Operation------------------------------//

		__syncthreads(); //newly added OCT 10

		index_t len=top>>1;        
		index_t dir=len << 1;
		for (index_t inc=len; inc > 0; inc=inc >> 1)//arranging the elements between the lower boundary and the upper boundary
		{
			index_t thid=threadIdx.x;
			index_t thid_end=SelectedNumber>>1;

			while(thid < thid_end)
			{
				index_t low=thid & (inc-1);
				index_t i= (thid <<1)-low;
				bool reverse=((dir & i)==0);
				bool swap=reverse^(SMem[i]<SMem[i+inc]);// ^ Xor--> A ^ B is only true when A,B are opposite boolean 
				if (swap)
				{
					Exchange<data_t,index_t>(SMem[i],SMem[i+inc]);       
				}
				thid+=blockDim.x;
			}
			__syncthreads();

		}

		//----------------------------Finished Rebuild Operation------------------------------//
		if (NElementsperBlock > blockDim.x)//NElementsperBlock is already halved by now
		{
			count=0;
			index_t begin=threadIdx.x;
			while(begin < SelectedNumber)//4 loops for 8*blockDim.x number of elements in the shared memory
			{
				temp[myBlockID*SelectedNumber+begin]=  SMem[begin];//New values are updated in temp
				begin+=blockDim.x;
				count++;
			}
		}
		else
		{ 
			if (threadIdx.x <NElementsperBlock)
				temp[myBlockID*blockDim.x+threadIdx.x]=SMem[threadIdx.x];
		}
		myBlockID+=gridDim.x;
		__syncthreads(); //newly added OCT 10

	}
	// printf("Returning abnormal way! :(\n");

}

	template<typename data_t, typename index_t>
__global__ void Combined_Sigmod_Merge_firstTopK(data_t* vec,index_t N,index_t top,index_t* SubrangeId,index_t NElementsperBlock,data_t* temp,index_t* temp_SubRange)
{
	index_t myBlockID=blockIdx.x;
	// extern __shared__ index_t Array[];
	extern __shared__ data_t Array[];

	index_t* SMemSubrange=(index_t*)&Array[0];
	data_t* SMem=&Array[NElementsperBlock];

	index_t  SelectedNumber;//=NElementsperBlock;
	index_t factor= (NElementsperBlock >= top) ? NElementsperBlock >> 1: top >> 1 ;

	while(myBlockID*NElementsperBlock <N )
	{
		int count;
		SelectedNumber=NElementsperBlock;
		if (NElementsperBlock > blockDim.x)
		{
			count=0;
			index_t begin=threadIdx.x;//+blockDim.x;
			while(begin < NElementsperBlock)
			{
				SMem[begin]=vec[myBlockID*NElementsperBlock+begin];
				SMemSubrange[begin]=SubrangeId[myBlockID*NElementsperBlock+begin];
				begin+=blockDim.x;
				count++;
			}
		}
		else//Copy all the elements from vec to SMem of size blockdim // if (threadIdx.x < top)//Loading only k elements in the shared memory. 1 block will work on k elements at 1 time
		{
			if (threadIdx.x<NElementsperBlock)
			{
				SMem[threadIdx.x]=vec[myBlockID*blockDim.x+threadIdx.x];
				SMemSubrange[threadIdx.x]=SubrangeId[myBlockID*blockDim.x+threadIdx.x];

			}
		}

		__syncthreads(); //newly added OCT 10

		for (index_t len= 1 ;len< top; len= len<<1)//----------On completion of this loop--> Finishes local sorting of all the elements of k groups assigned to this block----------//
		{
			index_t dir=len << 1;
			for (index_t inc=len; inc > 0; inc=inc >> 1)//arranging the elements between the lower boundary and the upper boundary
			{
				index_t thid=threadIdx.x;
				index_t thid_end=factor;
				while(thid < thid_end)// For conditon when blockdim <k .1 thread in block will work for multiple times for 1 k sorting
				{
					index_t low=thid & (inc-1);
					index_t i= (thid <<1)-low;
					bool reverse=((dir & i)==0);
					bool swap=reverse^(SMem[i]<SMem[i+inc]);// ^ Xor--> A ^ B is only true when A,B are opposite boolean 
					if (swap)
					{      
						Exchange<data_t,index_t>(SMem[i],SMem[i+inc]);
						Exchange_Sub_index<data_t,index_t>(SMemSubrange[i],SMemSubrange[i+inc]);
					}
					thid+=blockDim.x;
				}
				__syncthreads();//synchronizes the threads within the block

			}

		}

		//----------------------------Starting Merge Operation------------------------------//

		index_t thid_shared=threadIdx.x;
		index_t thid_shared_end=factor;
		while(thid_shared<thid_shared_end)
		{
			index_t low= thid_shared & (top-1); //thid_shared might be used directly --> avoids mapping to the shared memory index like calculation of o
			index_t i= (thid_shared<<1)-low;
			bool firstselect=false;
			data_t temp_data=max<data_t,index_t>(SMem[i],SMem[i+top],firstselect);
			data_t temp_SubRangedata;
			if (firstselect)
			{
				temp_SubRangedata=SMemSubrange[i];
			}
			else
			{
				temp_SubRangedata=SMemSubrange[i+top];
			}
			__syncthreads();
			SMem[thid_shared]=temp_data;
			SMemSubrange[thid_shared]=temp_SubRangedata;
			//          SMem[thid_shared]=temp_data[threadIdx.x];
			thid_shared+=blockDim.x;
		}
		//----------------------------Finished Merge Operation------------------------------//

		SelectedNumber= SelectedNumber>>1;


		//   -----------------------------------------------------------------------------------------------------//       
		//----------------------------Start Rebuild Operation------------------------------//

		__syncthreads(); //newly added OCT 10

		index_t len=top>>1;        
		index_t dir=len << 1;
		for (index_t inc=len; inc > 0; inc=inc >> 1)//arranging the elements between the lower boundary and the upper boundary
		{
			index_t thid=threadIdx.x;
			index_t thid_end=SelectedNumber>>1;

			while(thid < thid_end)
			{
				index_t low=thid & (inc-1);
				index_t i= (thid <<1)-low;
				bool reverse=((dir & i)==0);
				bool swap=reverse^(SMem[i]<SMem[i+inc]);// ^ Xor--> A ^ B is only true when A,B are opposite boolean 
				if (swap)
				{
					Exchange<data_t,index_t>(SMem[i],SMem[i+inc]);       
					Exchange_Sub_index<data_t,index_t>(SMemSubrange[i],SMemSubrange[i+inc]);


				}
				thid+=blockDim.x;
			}
			__syncthreads();

		}

		//----------------------------Finished Rebuild Operation------------------------------//
		if (NElementsperBlock > blockDim.x)//NElementsperBlock is already halved by now
		{
			count=0;
			index_t begin=threadIdx.x;
			while(begin < SelectedNumber)//4 loops for 8*blockDim.x number of elements in the shared memory
			{
				temp[myBlockID*SelectedNumber+begin]=  SMem[begin];//New values are updated in temp
				temp_SubRange[myBlockID*SelectedNumber+begin]= SMemSubrange[begin];
				begin+=blockDim.x;
				count++;
			}
		}
		else
		{ 
			if (threadIdx.x<NElementsperBlock)
			{
				temp[myBlockID*blockDim.x+threadIdx.x]=SMem[threadIdx.x];
				temp_SubRange[myBlockID*blockDim.x+threadIdx.x]=SMemSubrange[threadIdx.x];

			}
		}
		myBlockID+=gridDim.x;
		__syncthreads(); //newly added OCT 10

	}
	// printf("Returning abnormal way! :(\n");

}
	template<typename data_t, typename index_t>
__global__ void Combined_Sigmod_MergeRebuild_firstTopk(data_t* vec,index_t N,index_t top,index_t* SubrangeId,index_t NElementsperBlock,data_t* temp,index_t* temp_SubRange)
{
	index_t myBlockID=blockIdx.x;
	extern __shared__ data_t Array[];

	index_t* SMemSubrange=(index_t*)&Array[0];
	data_t* SMem=&Array[NElementsperBlock];

	index_t  SelectedNumber;//=NElementsperBlock;
	index_t factor= (NElementsperBlock >= top) ? NElementsperBlock >> 1: top >> 1 ;

	while(myBlockID*NElementsperBlock <N )
	{
		int count;
		SelectedNumber=NElementsperBlock;
		if (NElementsperBlock > blockDim.x)
		{
			count=0;
			index_t begin=threadIdx.x;//+blockDim.x;
			while(begin < NElementsperBlock)
			{
				SMem[begin]=vec[myBlockID*NElementsperBlock+begin];
				SMemSubrange[begin]=SubrangeId[myBlockID*NElementsperBlock+begin];
				begin+=blockDim.x;
				count++;
			}
		}
		else//Copy all the elements from vec to SMem of size blockdim // if (threadIdx.x < top)//Loading only k elements in the shared memory. 1 block will work on k elements at 1 time
		{
			if (threadIdx.x<NElementsperBlock)
			{
				SMem[threadIdx.x]=vec[myBlockID*blockDim.x+threadIdx.x];
				SMemSubrange[threadIdx.x]=SubrangeId[myBlockID*blockDim.x+threadIdx.x];

			}
		}

		//----------------------------Starting Merge Operation------------------------------//
		__syncthreads(); //newly added OCT 10

		index_t thid_shared=threadIdx.x;
		index_t thid_shared_end=factor;
		while(thid_shared<thid_shared_end)
		{
			index_t low= thid_shared & (top-1); //thid_shared might be used directly --> avoids mapping to the shared memory index like calculation of o
			index_t i= (thid_shared<<1)-low;
			bool firstselect=false;
			data_t temp_data=max<data_t,index_t>(SMem[i],SMem[i+top],firstselect);
			data_t temp_SubRangedata;
			if (firstselect)
			{
				temp_SubRangedata=SMemSubrange[i];
			}
			else
			{
				temp_SubRangedata=SMemSubrange[i+top];
			}
			__syncthreads();
			//  SMem[thid_shared]=temp_data[threadIdx.x];
			SMem[thid_shared]=temp_data;
			SMemSubrange[thid_shared]=temp_SubRangedata;
			thid_shared+=blockDim.x;
		}
		//----------------------------Finished Merge Operation------------------------------//

		//   -----------------------------------------------------------------------------------------------------//       
		//----------------------------Start Rebuild Operation------------------------------//

		__syncthreads(); //newly added OCT 10

		SelectedNumber= SelectedNumber>>1;
		index_t len=top>>1;        
		index_t dir=len << 1;
		for (index_t inc=len; inc > 0; inc=inc >> 1)//arranging the elements between the lower boundary and the upper boundary
		{
			index_t thid=threadIdx.x;
			index_t thid_end= SelectedNumber>>1;
			while(thid < thid_end)
			{
				index_t low=thid & (inc-1);
				index_t i= (thid <<1)-low;
				bool reverse=((dir & i)==0);
				bool swap=reverse^(SMem[i]<SMem[i+inc]);// ^ Xor--> A ^ B is only true when A,B are opposite boolean 
				if (swap)
				{
					Exchange<data_t,index_t>(SMem[i],SMem[i+inc]);       
					Exchange_Sub_index<data_t,index_t>(SMemSubrange[i],SMemSubrange[i+inc]);

				}
				thid+=blockDim.x;
			}
			__syncthreads();

		}
		//----------------------------Finished Rebuild Operation------------------------------//

		if (NElementsperBlock > blockDim.x)//NElementsperBlock is already halved by now
		{
			count=0;
			index_t begin=threadIdx.x;
			while(begin < SelectedNumber)//4 loops for 8*blockDim.x number of elements in the shared memory
			{
				temp[myBlockID*SelectedNumber+begin]=  SMem[begin];//New values are updated in temp
				temp_SubRange[myBlockID*SelectedNumber+begin]= SMemSubrange[begin];
				begin+=blockDim.x;
				count++;
			}
		}
		else
		{
			if (threadIdx.x<NElementsperBlock)
			{
				temp[myBlockID*blockDim.x+threadIdx.x]=SMem[threadIdx.x];
				temp_SubRange[myBlockID*blockDim.x+threadIdx.x]=SMemSubrange[threadIdx.x];

			}
		}
		myBlockID+=gridDim.x;
		__syncthreads(); //newly added OCT 10

	}

}
	template<typename data_t, typename index_t>
__global__ void Combined_Sigmod_MergeRebuild(data_t* vec,index_t N,index_t top,index_t* SubrangeId,index_t NElementsperBlock,data_t* temp)
{
	index_t myBlockID=blockIdx.x;
	extern __shared__ data_t SMem[];

	//    data_t* SMem= Array;
	//    data_t* temp_data= Array + NElementsperBlock;

	index_t  SelectedNumber;//=NElementsperBlock;
	index_t factor= (NElementsperBlock >= top) ? NElementsperBlock >> 1: top >> 1 ;

	while(myBlockID*NElementsperBlock <N )
	{
		int count;
		SelectedNumber=NElementsperBlock;
		if (NElementsperBlock > blockDim.x)
		{
			count=0;
			index_t begin=threadIdx.x;//+blockDim.x;
			while(begin < NElementsperBlock)
			{
				SMem[begin]=vec[myBlockID*NElementsperBlock+begin];
				begin+=blockDim.x;
				count++;
			}
		}
		else//Copy all the elements from vec to SMem of size blockdim // if (threadIdx.x < top)//Loading only k elements in the shared memory. 1 block will work on k elements at 1 time
		{
			if (threadIdx.x < NElementsperBlock)
				SMem[threadIdx.x]=vec[myBlockID*blockDim.x+threadIdx.x];
		}

		//----------------------------Starting Merge Operation------------------------------//

		__syncthreads(); //newly added OCT 10

		index_t thid_shared=threadIdx.x;
		index_t thid_shared_end=factor;
		while(thid_shared<thid_shared_end)
		{
			index_t low= thid_shared & (top-1); //thid_shared might be used directly --> avoids mapping to the shared memory index like calculation of o
			index_t i= (thid_shared<<1)-low;
			bool firstselect=false;
			data_t temp_data=max<data_t,index_t>(SMem[i],SMem[i+top],firstselect);
			__syncthreads();
			//  SMem[thid_shared]=temp_data[threadIdx.x];
			SMem[thid_shared]=temp_data;
			thid_shared+=blockDim.x;
		}
		__syncthreads(); //newly added OCT 10

		SelectedNumber= SelectedNumber>>1;
		index_t len=top>>1;        
		index_t dir=len << 1;
		for (index_t inc=len; inc > 0; inc=inc >> 1)//arranging the elements between the lower boundary and the upper boundary
		{
			index_t thid=threadIdx.x;
			index_t thid_end= SelectedNumber>>1;
			while(thid < thid_end)
			{
				index_t low=thid & (inc-1);
				index_t i= (thid <<1)-low;
				bool reverse=((dir & i)==0);//when the value of i>k, the reverse changes. i.e. reverse =false decreasing order
				bool swap=reverse^(SMem[i]<SMem[i+inc]);// ^ Xor--> A ^ B is only true when A,B are opposite boolean 
				if (swap)
				{
					Exchange<data_t,index_t>(SMem[i],SMem[i+inc]);       
				}
				thid+=blockDim.x;
			}
			__syncthreads();

		}
		//----------------------------Finished Rebuild Operation------------------------------//

		if (NElementsperBlock > blockDim.x)//NElementsperBlock is already halved by now
		{
			count=0;
			index_t begin=threadIdx.x;
			while(begin < SelectedNumber)//4 loops for 8*blockDim.x number of elements in the shared memory
			{
				temp[myBlockID*SelectedNumber+begin]=  SMem[begin];//New values are updated in temp
				begin+=blockDim.x;
				count++;
			}
		}
		else
		{ 
			if (threadIdx.x < NElementsperBlock)
				temp[myBlockID*blockDim.x+threadIdx.x]=SMem[threadIdx.x];
		}
		myBlockID+=gridDim.x;
		__syncthreads(); //newly added OCT 10

	}

}
	template<typename data_t, typename index_t>
void bitonic_firstTopk(data_t* vec1_d,index_t num_element,index_t k,index_t Subrangesize,index_t originalNum,index_t* SubrangeId_d,data_t* originalvec_d,data_t* ConcatenatedRange_d)
{

	cout<<"------Starting Bitonic TopK(First TopK)-------"<<endl;
	int BlockSize=128;
	int NBlocks=128;
	//  int NBlocks=num_element/(8*BlockSize);
	int Ssize;
	cout<<"Size of(data_t)"<<sizeof(data_t)<<endl;
	if (k>BlockSize)//k is less than block dimension
	{
		BlockSize=k;
		if (k>512) BlockSize=512;
	}
// Ssize=BlockSize<<3;
Ssize=BlockSize<<1;
if (num_element < Ssize)
{
	cout<<"NElements per block is less than NElements. Reducing Number of elemenrs per block to Number of Elements."<<endl;//possibility for optimizing workload distribution --> divide Ssize elemnts into many blocks
	Ssize=num_element;
}
cout<<"Shared Memory Size per block"<<(Ssize*sizeof(data_t)+Ssize*sizeof(index_t))/1024<<"KB"<<endl;;
if ((Ssize*sizeof(data_t)+Ssize*sizeof(index_t))/1024>=64)
{
	cout<<"Size of(data_t)"<<sizeof(data_t)<<endl;
	cout<<"SMem size exceeded"<<endl;
	exit(-1);
}
data_t* temp_d;
H_ERR(cudaMalloc((void**) &temp_d,sizeof(data_t)*num_element));
index_t* tempSubrangeId_d;
H_ERR(cudaMalloc((void**) &tempSubrangeId_d,sizeof(index_t)*num_element));
Combined_Sigmod_Merge_firstTopK<data_t,index_t> <<<NBlocks,BlockSize,((Ssize)*sizeof(data_t)+Ssize*sizeof(index_t))>>>(vec1_d,num_element,k,SubrangeId_d,Ssize,temp_d,tempSubrangeId_d);  
H_ERR(cudaDeviceSynchronize());
num_element=num_element>>1; 
swap_ptr_data<data_t,index_t> (vec1_d,temp_d);
swap_ptr_index<data_t,index_t> (SubrangeId_d,tempSubrangeId_d);
int loop=0;
while(num_element>k)
{
	cout<<endl<<"Inside loop:"<<loop<<endl;
	Combined_Sigmod_MergeRebuild_firstTopk<data_t,index_t> <<<NBlocks,BlockSize,((Ssize)*sizeof(data_t)+Ssize*sizeof(index_t))>>>(vec1_d,num_element,k,SubrangeId_d,Ssize,temp_d,tempSubrangeId_d);
	H_ERR(cudaDeviceSynchronize());   
	num_element =num_element >> 1;
	swap_ptr_data<data_t,index_t> (vec1_d,temp_d);  
	swap_ptr_index<data_t,index_t> (SubrangeId_d,tempSubrangeId_d);
	loop++;
}
assert(k==num_element);
index_t ConcatenatedSize=k*Subrangesize;
cout<<"Original number:"<<originalNum<<endl;
cout<<endl;
CreateConcatenateRange<data_t,index_t><<<512,512>>>(originalvec_d,ConcatenatedRange_d,SubrangeId_d,ConcatenatedSize,k,Subrangesize);
H_ERR(cudaDeviceSynchronize());
cout<<endl<<endl;
return;
} 

template<typename data_t, typename index_t>
void bitonic(data_t* vec1_d,index_t num_element,index_t k,data_t& TopKElement,index_t Subrangesize,data_t* originalvec,index_t originalNum,index_t* SubrangeId_d,data_t* originalvec_d,data_t* ConcatenatedRange_d)
{
cout<<"------Starting Second Bitonic TopK-------"<<endl;
int BlockSize=128;
//   int NBlocks=num_element/(8*BlockSize);
int NBlocks=128;
int Ssize;
cout<<"Size of(data_t)"<<sizeof(data_t)<<endl;
if (k>BlockSize)//k is less than block dimension
{
	BlockSize=k;
	if (k>512) BlockSize=512;
}
// Ssize=BlockSize<<3;
	Ssize=(k<<1);//Edited for larger k values/ From 3 to 1
	if (num_element < Ssize)
	{
		cout<<"NElements per block is less than NElements. Reducing Number of elemenrs per block to Number of Elements."<<endl;//possibility for optimizing workload distribution --> divide Ssize elemnts into many blocks
		Ssize=num_element;
	}
	cout<<"Shared Memory Size per block"<<Ssize*sizeof(data_t)/1024<<"KB"<<endl;;
	if (Ssize*sizeof(data_t)/1024>=64)
	{
		cout<<"Size of(data_t)"<<sizeof(data_t)<<endl;
		cout<<"SMem size exceeded"<<endl;
		exit(-1);
	}
	data_t* temp_d;
	H_ERR(cudaMalloc((void**) &temp_d,sizeof(data_t)*num_element));
	Combined_Sigmod_Merge<data_t,index_t> <<<NBlocks,BlockSize,(Ssize)*sizeof(data_t)>>>(vec1_d,num_element,k,SubrangeId_d,Ssize,temp_d);  
	H_ERR(cudaDeviceSynchronize());
	num_element=num_element>>1;
	swap_ptr_data<data_t,index_t> (vec1_d,temp_d);
	int loop=0;
	while(num_element>k)
	{
		cout<<endl<<"Inside loop:"<<loop<<endl;
		Combined_Sigmod_MergeRebuild<data_t,index_t> <<<NBlocks,BlockSize,(Ssize)*sizeof(data_t)>>>(vec1_d,num_element,k,SubrangeId_d,Ssize,temp_d);
		H_ERR(cudaDeviceSynchronize());    
		num_element =num_element >> 1;
		swap_ptr_data<data_t,index_t> (vec1_d,temp_d);  
		loop++;
	}
	assert(k==num_element);
#ifdef _MONITOR_
	cout<<"The top K elements"<<endl;
	cout<<"Normal sorting the original vector for testing"<<endl;
	sort(originalvec, originalvec + originalNum, std::greater<data_t>());
	cout<<"Normal sorting of original vector for testing finished!"<<endl;
	data_t* vec1=(data_t*)malloc(sizeof(data_t)*num_element);
	H_ERR(cudaMemcpy(vec1,vec1_d,sizeof(data_t)*num_element,cudaMemcpyDeviceToHost));
	for (index_t i=0;i<num_element;i++)
	{
		   cout<<i<<":"<<originalvec[k-i-1]<<" ";
		cout<<vec1[i]<<"   ";
		// assert(vec1[i]==originalvec[k-i-1]);
	}
#endif
	cout<<endl<<endl;
	return;
}
