// #include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
// #include <algorithm>
#include "wtime.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <assert.h>
#include <fstream> 
#include "PQ.h"
// #include <mpi.h>

const int blocksize=128;
const int gridsize=128;

#define INDEX(X) (X+(X>>5))
#include <cmath>
const int ShareMemory=512;
using namespace std;

static void HandleError( cudaError_t err, const char *file, int line   ) {
    if (err != cudaSuccess) {
        printf( "\n%s in %s at line %d\n", \
                cudaGetErrorString( err   ), file, line );
        exit( EXIT_FAILURE   );
    }
}
#define H_ERR( err   ) \
    (HandleError( err, __FILE__, __LINE__   ))


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

int power(int x,int n)
{
    int number=1;
    for (int i=0;i<n;i++)
        number*=x;
    return number;
}


__device__ __forceinline__ int ThreadLoad(int *ptr)
{
    int retval;           
    asm volatile ("ld.global.cg.s32 %0, [%1];" :    \
            "=r"(retval) :                        \
            "l" (ptr) );                          \
        return retval;  

}

template<typename data_t,typename index_t>
struct aux_device
{
    int gpuId; //gpu id
    data_t* vec_d;
};

// template<typename data_t,typename index_t>
// void Allocate_Initialize (struct aux_device* device_obj,index_t gpuId, index_t num_element, data_t* vec)
// {
// 	H_ERR(cudaMalloc((void**) &device_obj->vec_d,sizeof(data_t)*num_element));
// 	H_ERR(cudaMemcpy(device_obj->vec_d,vec,sizeof(data_t)*num_element,cudaMemcpyHostToDevice));

// }

    template<typename data_t, typename index_t>
__device__ __forceinline__ int ThreadLoad_ind(int *ptr)
{
    int retval;           
    asm volatile ("ld.global.cg.s32 %0, [%1];" :    \
            "=r"(retval) :                        \
            "l" (ptr) );                          \
        return retval;  

}
    template<typename data_t, typename index_t>
__global__ void FindFinalMinMax(data_t* maxArray_d,data_t* minArray_d,data_t* max_d,data_t* min_d)
{
    extern __shared__ data_t SMem[];
    data_t* Max_array=SMem;
    data_t* Min_array=SMem+blockDim.x;
    Max_array[threadIdx.x]=maxArray_d[threadIdx.x];
    Min_array[threadIdx.x]=minArray_d[threadIdx.x];
    //parallel reduction find the final Max and min
    for (int i=blockDim.x>>1;i>0;i=i>>1)
    {
        if (threadIdx.x <i )
        {
            Max_array[threadIdx.x]=(Max_array[threadIdx.x]>Max_array[threadIdx.x+i]) ? Max_array[threadIdx.x]: Max_array[threadIdx.x+i];
            Min_array[threadIdx.x]=(Min_array[threadIdx.x]<Min_array[threadIdx.x+i]) ? Min_array[threadIdx.x]:Min_array[threadIdx.x+i];
        }
        __syncthreads();
    }
    //~parallel reduction find the final Max and min
    if (threadIdx.x==0)
    {
        max_d[0]=Max_array[0];
        min_d[0]=Min_array[0];
    }
    return;
}


    template<typename data_t, typename index_t>
__global__ void FindRange(data_t* max, data_t* min, data_t* vec, index_t num_element)
{
    extern __shared__ data_t SMem[];
    data_t* Max_array=SMem;
    data_t* Min_array=SMem+blockDim.x;
    index_t thid=threadIdx.x+blockIdx.x*blockDim.x;
    data_t myMax=vec[thid];
    data_t myMin=vec[thid];
    while (thid<num_element)
    {
        data_t value=vec[thid];
        int is_larger=((value>myMax)<<1)+(value<myMin);       
        switch(is_larger)
        {
            case 2:
                myMax=value;
                break;
            case 1: 
                myMin=value;
                break;
            default:
                break;
        }
        thid+=blockDim.x*gridDim.x;
    }
    Max_array[threadIdx.x]=myMax;
    Min_array[threadIdx.x]=myMin; 
    __syncthreads();
    //parallel reduction find the final Max and min
    for (int i=blockDim.x>>1;i>0;i=i>>1)
    {
        if (threadIdx.x <i )
        {
            Max_array[threadIdx.x]=(Max_array[threadIdx.x]>Max_array[threadIdx.x+i]) ? Max_array[threadIdx.x]: Max_array[threadIdx.x+i];
            Min_array[threadIdx.x]=(Min_array[threadIdx.x]<Min_array[threadIdx.x+i]) ? Min_array[threadIdx.x]:Min_array[threadIdx.x+i];
        }
        __syncthreads();
    }
    //~parallel reduction find the final Max and min

    if (threadIdx.x==0)
    {
        max[blockIdx.x]=Max_array[0];
        min[blockIdx.x]=Min_array[0];
    }
    return;
}

    template<typename data_t,typename index_t>
void Max_Min(data_t& max,data_t& min,data_t* vec_d,index_t num_element)
{

    double tf_start=wtime();
    data_t *maxArray=(data_t*)malloc((sizeof(data_t))*blocksize);
    data_t *minArray=(data_t*)malloc((sizeof(data_t))*blocksize);

    data_t *maxArray_d;
    H_ERR(cudaMalloc((void**) &maxArray_d,sizeof(data_t)*blocksize));
    data_t *minArray_d;
    H_ERR(cudaMalloc((void**) &minArray_d,sizeof(data_t)*blocksize));
    if ((sizeof(data_t)*(blocksize<<1))/1024 > 32)
    {
        cout<<"Warning: 32 KB size limit of shared memory exceeded!!!"<<endl;
    }
    FindRange<data_t,index_t><<<gridsize,blocksize,sizeof(data_t)*(blocksize<<1)>>>(maxArray_d,minArray_d,vec_d,num_element);
    H_ERR(cudaDeviceSynchronize());
    if (gridsize>1024)
    {
        cout<<"BlockDim>1024: Cannot launch Min-Max kernel!"<<endl;
        return;
    }
    data_t* max_d;
    data_t* min_d;
    H_ERR(cudaMalloc((void**) &min_d,sizeof(data_t)));
    H_ERR(cudaMalloc((void**) &max_d,sizeof(data_t)));
    FindFinalMinMax<data_t,index_t><<<1,gridsize,(gridsize<<1)*sizeof(data_t)>>>(maxArray_d,minArray_d,max_d,min_d);//won't work when we make the number of blocks in FindRange kernel launch > 1024
    H_ERR(cudaDeviceSynchronize());
    H_ERR(cudaMemcpy(&max,max_d,sizeof(data_t),cudaMemcpyDeviceToHost));
    H_ERR(cudaMemcpy(&min,min_d,sizeof(data_t),cudaMemcpyDeviceToHost));

    double tf_Max_min=wtime()-tf_start;
    cout<<"Time for finding max min:"<<tf_Max_min*1000<<" ms"<<endl;
    free(maxArray);free(minArray);
    H_ERR(cudaFree(maxArray_d));H_ERR(cudaFree(minArray_d));


}

    template<typename data_t, typename index_t>
void CumulateCount_inplace(int* Count,index_t* CumCount,index_t num_bucket,int& Kdigit,index_t k,index_t num_element,index_t Belowcount,data_t& flag, int NBitsperDigit,int Currentdigit)
{
    index_t sum=Belowcount;// the cumulation starts from the elemens which are pushed away from the smallest bucket to the dummy smaller bucket
    index_t shiftleft=Currentdigit*NBitsperDigit;

    for (index_t i=0;i<num_bucket;i++)
    {
        sum+=Count[i];
        CumCount[i]=sum;  
        if (i!=0)
        {
            if ((CumCount[i] >= (num_element-k+1)) &&  (CumCount[i-1]< (num_element-k+1)))
            {
                Kdigit=i;
                flag=flag|((data_t)i<<shiftleft);

            }
        }
        else
        {
            if (CumCount[i] >= (num_element-k+1))
            {
                Kdigit=i;
                flag=flag|((data_t)i<<shiftleft);
            }
        }
    }
    return;
}


    template<typename data_t, typename index_t>
void CumulateCount_seq(int* Count,index_t* CumCount,index_t num_bucket,index_t& Kdigit,index_t k,index_t num_element,data_t& flag, int NBitsperDigit,int Currentdigit)
{
    index_t sum=0;//Belowcount;// the cumulation starts from the elemens which are pushed away from the smallest bucket to the dummy smaller bucket
    index_t shiftleft=Currentdigit*NBitsperDigit;
    //    data_t temp=0;
    for (index_t i=0;i<num_bucket;i++)
    {
        sum+=Count[i];
        CumCount[i]=sum;  
        if (i!=0)
        {
            if ((CumCount[i] >= (num_element-k+1)) &&  (CumCount[i-1]< (num_element-k+1)))
                //          if ((CumCount[i] >= (num_element-k)) &&  (CumCount[i-1]< (num_element-k)))
            {
                Kdigit=i;
                flag=flag|((data_t)i<<shiftleft);
            }
        }
        else
        {
            if (CumCount[i] >= (num_element-k+1))
                //      if (CumCount[i] >= (num_element-k)) 
            {
                Kdigit=i;
                flag=flag|((data_t)i<<shiftleft);
            }
        }
    }
    return;
}


    template<typename data_t, typename index_t>
__global__ void CumulateCount(int* Count,index_t* CumCount,index_t num_bucket,index_t& Kdigit,index_t k,index_t num_element)
{

    index_t sum=0;//Belowcount;// the cumulation starts from the elemens which are pushed away from the smallest bucket to the dummy smaller bucket
    for (index_t i=0;i<num_bucket;i++)
    {
        sum+=Count[i];
        CumCount[i]=sum;  
        if (i!=0)
        {
            if ((CumCount[i] >= (num_element-k)) &&  (CumCount[i-1]< (num_element-k)))
            {
                // if (i==0) Kdigit=i;
                Kdigit=i;
            }
        }
        else
        {
            if (CumCount[i] >= (num_element-k)) Kdigit=i;
        }
    }
    return;
}

    template<typename data_t, typename index_t>
int compare (const void * a, const void * b)
{
    return ( *(data_t*)a - *(data_t*)b   );//in ascending order
}



    template<typename data_t, typename index_t>
__global__ void AssignKBucketElements(data_t* vec,index_t N,int Kdigit,int* countfortemp_d,data_t* temp,data_t flag,int CurrentDigit,int NBitsperDigit,int shift)
{
    index_t thid=blockDim.x*blockIdx.x+threadIdx.x;
    while (thid<N)
    {
        data_t value=vec[thid];
        if (((value>>shift)<<shift)==flag)
        {
            int index=atomicAdd(countfortemp_d,1);
            temp[index]=value;
        }
        thid+=blockDim.x*gridDim.x;
    }
}

    template<typename data_t, typename index_t>
__global__ void AssignMaxMin(data_t* vec_d,index_t num_element,index_t* Count_d,data_t min,data_t max,int CurrentDigit,int NBitsperDigit,int KDigit,int* BelowCount,data_t flag)
{
    extern __shared__ index_t Below_Count[];
    index_t thid=threadIdx.x+blockIdx.x*blockDim.x;
    Below_Count[threadIdx.x] = 0;
    data_t test=0;
    test =~(test);
    int shleft=CurrentDigit*NBitsperDigit;
    test=test<<shleft;
    while (thid< num_element)
    {
        data_t value=vec_d[thid];
        if ((value >= min)&& (value<=max))
        {        
            if (value<flag)
            {
                vec_d[thid]=min-1; // set all bits to 0
                atomicAdd(&Below_Count[threadIdx.x],1);

            }
            else if (((flag|value)&test)!=flag)//if the value doesnot fall in the bucket
            {
                vec_d[thid]=max+1;
            }

        }
        thid+=blockDim.x*gridDim.x;
    }
    __syncthreads();

    //parallel reduction
    for (int i=blockDim.x>>1;i>0;i=i>>1)
    {
        if (threadIdx.x < i)
        {
            index_t temp=Below_Count[threadIdx.x]+Below_Count[threadIdx.x+i];
            Below_Count[threadIdx.x]= temp;
        }
        __syncthreads();
    }
    //~parallel reduction

    if (threadIdx.x==0)
        atomicAdd(BelowCount,Below_Count[0]);
}

    template<typename data_t, typename index_t>
__global__ void CalculateOccurence_inplace(data_t* vec,index_t num_element,int* Count,int NBitsperDigit,int CurrentDigit,int num_bucket,data_t flag, int shiftleft,int shiftRight)

{
    extern __shared__ int SMem_Count[];//The size is number of buckets
    for (int i = threadIdx.x; i < num_bucket; i += blockDim.x) 
        SMem_Count[i] = 0;
    __syncthreads();

    index_t mybegin=blockIdx.x*blockDim.x+threadIdx.x;

    data_t mask=num_bucket-1;
    while(mybegin < num_element)
    {
        data_t value=vec[mybegin];
        if (((value>>shiftleft)<<shiftleft)==flag)//The value falls within the bucket
        {
            data_t temp=value>>shiftRight;
            temp=temp&mask;
            atomicAdd(&SMem_Count[temp],1);
        }
        mybegin+=blockDim.x*gridDim.x;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < num_bucket; i += blockDim.x) 
    {
        atomicAdd(&Count[i],SMem_Count[i]);
    }
}

    template<typename data_t, typename index_t>
__global__ void CalculateOccurence_inplace_firsttopk(data_t* vec,index_t num_element,int* Count,int NBitsperDigit,int CurrentDigit,int num_bucket,data_t flag, int shiftleft,int shiftRight)

{
    extern __shared__ int SMem_Count[];//The size is number of buckets
    for (int i = threadIdx.x; i < num_bucket; i += blockDim.x) 
        SMem_Count[i] = 0;
    __syncthreads();

    index_t mybegin=blockIdx.x*blockDim.x+threadIdx.x;

    data_t mask=num_bucket-1;
    while(mybegin < num_element)
    {
        data_t value=vec[mybegin];
        if (((value>>shiftleft)<<shiftleft)==flag)//The value falls within the bucket
        {
            data_t temp=value>>shiftRight;
            temp=temp&mask;
            atomicAdd(&SMem_Count[temp],1);
        }
        mybegin+=blockDim.x*gridDim.x;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < num_bucket; i += blockDim.x) 
    {
        atomicAdd(&Count[i],SMem_Count[i]);
    }
}

    template<typename data_t,typename index_t>  
__global__ void FindTopKRanges(data_t* vec_d,index_t num_element,data_t flag,int shiftleft,index_t* SubrangeId_d,index_t* SelectedSubrangeId_d,int* CountSelectedSubrange_d)

{
    index_t mybegin=blockIdx.x*blockDim.x+threadIdx.x;

    while(mybegin < num_element)
    {
        data_t value=vec_d[mybegin];
        if (((value>>shiftleft)<<shiftleft) > flag)//Make sure all the subranges of the sampled elements which are greater than the Kth element are included in the subrange array
        {
            SelectedSubrangeId_d[atomicAdd(CountSelectedSubrange_d,1)]=SubrangeId_d[mybegin];
        }
        mybegin+=blockDim.x*gridDim.x;
    }
}

    template<typename data_t,typename index_t>  
__global__ void FindTopKRanges_beta_shLeft(data_t* vec_d,index_t num_element,data_t flag,int shiftleft,index_t* SubrangeId_d,index_t* SelectedSubrangeId_d,int* CountSelectedSubrange_d,data_t* ConcatenatedRange,index_t* CountLonelyElements,int beta)

{
    index_t mybegin=(blockIdx.x*blockDim.x+threadIdx.x)*beta;//*3 for Beta
    while(mybegin < num_element)
    {
        int count=0;

        data_t value=vec_d[mybegin];
        data_t value1,value2;
        if (((value>>shiftleft)<<shiftleft) >= flag)       
        {
            count++;


            value1=vec_d[mybegin+1];
            value2=vec_d[mybegin+2];

            count +=  
                (((value1>>shiftleft)<<shiftleft) >= flag) +
                (((value2>>shiftleft)<<shiftleft) >= flag);

            if (count==3)
            {
                //                //all Beta (3) elements are selected
                SelectedSubrangeId_d[atomicAdd(CountSelectedSubrange_d,1)]=SubrangeId_d[mybegin];
            }
            else
            {
                index_t curr_ptr = atomicAdd(CountLonelyElements,beta);
                //                
                ConcatenatedRange[curr_ptr]=value;
                ConcatenatedRange[curr_ptr + 1]=value1;
                ConcatenatedRange[curr_ptr + 2]=value2;
            }
        }

        mybegin+=blockDim.x*gridDim.x*beta;//*3 for beta
    }
}

    template<typename data_t,typename index_t>  
__global__ void FindTopKRanges_beta(data_t* vec_d,index_t num_element,data_t flag,int shiftleft,index_t* SubrangeId_d,index_t* SelectedSubrangeId_d,int* CountSelectedSubrange_d,data_t* ConcatenatedRange,int* CountLonelyElements,int beta)

{
    //Note: This kernel is desined for Beta=3, value,valu1,valu2 represents Max, Second Max and third max. Total of 3 Beta elements.
    index_t mybegin=(blockIdx.x*blockDim.x+threadIdx.x)*beta;//*3 for Beta
    switch(beta)
    {
        case 3:
            while(mybegin < num_element)
            {
                int count=0;

                data_t value=vec_d[mybegin];

                data_t value1,value2;
                if (value >= flag)       
                {
                    count++;


                    value1=vec_d[mybegin+1];
                    value2=vec_d[mybegin+2];

                    count +=  
                        (value1 >= flag) +
                        (value2 >= flag);

                    if (count==3)//For Beta=3 Need to modify for different values of Beta
                    {
                        //                //all Beta (3) elements are selected
                        SelectedSubrangeId_d[atomicAdd(CountSelectedSubrange_d,1)]=SubrangeId_d[mybegin];
                    }
                    else
                    {
                        index_t curr_ptr = atomicAdd(CountLonelyElements,beta);
                        ConcatenatedRange[curr_ptr]=value;
                        ConcatenatedRange[curr_ptr + 1]=value1;
                        ConcatenatedRange[curr_ptr + 2]=value2;
                    }
                }


                mybegin+=blockDim.x*gridDim.x*beta;//*3 for beta
            }
            break;


        case 4:
            while(mybegin < num_element)
            {
                int count=0;

                data_t value=vec_d[mybegin];

                data_t value1,value2,value3;
                if (value >= flag)       
                {
                    count++;


                    value1=vec_d[mybegin+1];
                    value2=vec_d[mybegin+2];
                    value3=vec_d[mybegin+3];

                    count +=  
                        (value1 >= flag) +
                        (value2 >= flag) +
                        (value3 >= flag);

                    if (count==4)//For Beta=3 Need to modify for different values of Beta
                    {
                        //                //all Beta (3) elements are selected
                        SelectedSubrangeId_d[atomicAdd(CountSelectedSubrange_d,1)]=SubrangeId_d[mybegin];
                    }
                    else
                    {
                        index_t curr_ptr = atomicAdd(CountLonelyElements,beta);
                        ConcatenatedRange[curr_ptr]=value;
                        ConcatenatedRange[curr_ptr + 1]=value1;
                        ConcatenatedRange[curr_ptr + 2]=value2;
                        ConcatenatedRange[curr_ptr + 3]=value2;
                    }
                }


                mybegin+=blockDim.x*gridDim.x*beta;//*3 for beta
            }
            break;
        default:
            //  case 2
            while(mybegin < num_element)
            {
                int count=0;

                data_t value=vec_d[mybegin];

                data_t value1,value2;
                if (value >= flag)       
                {
                    count++;


                    value1=vec_d[mybegin+1];
                    // value2=vec_d[mybegin+2];

                    count +=  
                        (value1 >= flag);
                    //  + (value2 >= flag);

                    if (count==2)//For Beta=3 Need to modify for different values of Beta
                    {
                        //                //all Beta (3) elements are selected
                        SelectedSubrangeId_d[atomicAdd(CountSelectedSubrange_d,1)]=SubrangeId_d[mybegin];
                    }
                    else
                    {
                        index_t curr_ptr = atomicAdd(CountLonelyElements,beta);
                        ConcatenatedRange[curr_ptr]=value;
                        ConcatenatedRange[curr_ptr + 1]=value1;
                        // ConcatenatedRange[curr_ptr + 2]=value2;
                    }
                }


                mybegin+=blockDim.x*gridDim.x*beta;//*3 for beta
            }
            break;
    }
}

    template<typename data_t,typename index_t>  
__global__ void FillWithKElementsID(data_t* vec_d,index_t num_element,data_t flag,int shiftleft,index_t* SubrangeId_d,index_t* SelectedSubrangeId_d,int* CountSelectedSubrange_d,index_t k)

{
    index_t mybegin=blockIdx.x*blockDim.x+threadIdx.x;
    while(mybegin < num_element)
    {
        data_t value=vec_d[mybegin];
        if (((value>>shiftleft)<<shiftleft) == flag)//fill out the remaining part of the array by the subranges ID of any of the kth elements
        {
            if (ThreadLoad_ind<data_t,index_t>(CountSelectedSubrange_d)==k) return;
            SelectedSubrangeId_d[atomicAdd(CountSelectedSubrange_d,(ThreadLoad_ind<data_t,index_t>(CountSelectedSubrange_d)<k))]=SubrangeId_d[mybegin];
            //SelectedSubrangeId_d[atomicAdd(CountSelectedSubrange_d,(ThreadLoad_ind<data_t,index_t>(CountSelectedSubrange_d)!=k))]=SubrangeId_d[mybegin];
        }
        mybegin+=blockDim.x*gridDim.x;
    }
}


    template<typename data_t, typename index_t>
__global__ void CalculateOccurence(data_t* vec,index_t num_element,int* Count,int NBitsperDigit,int CurrentDigit,int num_bucket,int shiftRight)
{
    extern __shared__ int SMem_Count[];//The size is number of buckets
    //    int num_bucket=1<<NBitsperDigit;
    for (int i = threadIdx.x; i < num_bucket; i += blockDim.x) 
        SMem_Count[i] = 0;
    __syncthreads();

    index_t mybegin=blockIdx.x*blockDim.x+threadIdx.x;

    //    int shiftRight=NBitsperDigit*CurrentDigit;
    data_t mask=num_bucket-1;
    while(mybegin < num_element)
    {
        data_t temp=vec[mybegin]>>shiftRight;
        temp=temp&mask;
        atomicAdd(&SMem_Count[temp],1);
        mybegin+=blockDim.x*gridDim.x;
    }
    __syncthreads();
    //Writing the local sum into the global memory
    for (int i = threadIdx.x; i < num_bucket; i += blockDim.x) 
    {
        //Directly add atomically into the global memory
        atomicAdd(&Count[i],SMem_Count[i]);
    }

}

    template<typename data_t, typename index_t>
__global__ void SelectKNumber_onlyselectsKthElement(data_t* vec,index_t num_element,data_t* TopKElement,int* check,data_t flag,int shift)
{
    index_t thid=threadIdx.x+blockIdx.x*blockDim.x;
    while(thid<num_element)
    {
        if (ThreadLoad(check)!=0) return; //return if any of the thread finds the  element
        data_t value=vec[thid];
        if ((value>>shift)<<shift==flag)
        {
            TopKElement[0]=value;
            atomicAdd(check,1);
            return;
        }        
        thid+=blockDim.x*gridDim.x;
    }
}

    template<typename data_t, typename index_t>
__global__ void SelectKNumber(data_t* vec,index_t num_element,data_t* TopKElement,int* countcheck,data_t flag,int shift,data_t* TopKElements,index_t k)
{
    index_t thid=threadIdx.x+blockIdx.x*blockDim.x;
    index_t KElement;
    while(thid<num_element)
    {
        //        if (ThreadLoad(check)==0) 
        //        {
        //            KElement=TopKElement[0];
        //        }
        data_t value=vec[thid];
        //        data_t TopK;
        data_t shiftedValue=(value>>shift)<<shift;
        //        if (shiftedValue > flag)
        //        {
        TopKElements[atomicAdd(countcheck,shiftedValue>flag)]=value;
        //            return;
        //        }
        if (shiftedValue==flag)
        {
            TopKElement[0]=value;
        }
        thid+=blockDim.x*gridDim.x;
    }
    __syncthreads();
    //Pad the remaining vacant places by the kth element
    index_t TopK=TopKElement[0];
    while (ThreadLoad_ind<data_t,index_t>(countcheck)<k) 
    {
        TopKElements[atomicAdd(countcheck,ThreadLoad_ind<data_t,index_t>(countcheck)<k)]=TopK;
    }
}


    template<typename data_t,typename index_t>
__global__ void sampleMax_old (data_t* A,data_t* SubRangeMax, index_t N,index_t NSubranges,index_t SubRangeSize,const index_t alpha,index_t* SubrangeId)
{	

    int thid = blockDim.x*blockIdx.x+threadIdx.x;
    int laneId= threadIdx.x & 0x1f;
    int myWarpID=thid >> 5;
    int NWarps=(blockDim.x*gridDim.x) >> 5;
    while (myWarpID < NSubranges)//WarpID is used as subrange ID
    {
        //        index_t mybegin_pos=myWarpID*SubRangeSize+laneId;//<<alpha+laneId;
        index_t mybegin_pos=(myWarpID<<alpha)+laneId;
        if (mybegin_pos >= N)
        {
            printf("Error! Illegal memory access in a thread. Return this thread\n");
            //return;
        }
        int Nthreadstowork=32;
        if (SubRangeSize<32)
        {
            Nthreadstowork=SubRangeSize;
        }
        index_t myend_pos=(((myWarpID+1)<<alpha) < (N-1)) ? ((myWarpID+1)<<alpha):N-1;
        //index_t myend_pos=(myWarpID+1)<<alpha;
        data_t Max=A[mybegin_pos];

        while(mybegin_pos < myend_pos)
        {
            if (mybegin_pos >= N)
            {
                printf("Error! Illegal memory access. at mybegin_pos:%d when warpId:%d \n",mybegin_pos,myWarpID);
            }
            Max=(Max < A[mybegin_pos]) ? A[mybegin_pos]:Max;
            //   printf("mybeginpos:%d ",mybegin_pos);
            mybegin_pos+=32;
        }
        data_t MaxFromOther;
        for (int j=Nthreadstowork >> 1;j >=1;j=j>>1)
        {
            MaxFromOther=__shfl_sync(0xffffffff, Max,laneId+j,32/*Nthreadstowork*//*32*/);
            if (laneId<j)
            {
                Max= (MaxFromOther > Max) ? MaxFromOther : Max ;
            }            
        }
        if(laneId==0)
        {
            SubRangeMax[myWarpID]=Max;
            SubrangeId[myWarpID]=myWarpID;
        }
        myWarpID+=NWarps;
    }
    return;
}





    template<typename data_t,typename index_t>
__global__ void sampleMax (data_t* A,data_t* SubRangeMax, index_t N,index_t NSubranges,index_t SubRangeSize,const index_t alpha,index_t* SubrangeId,int Nthreadstowork)
{	

    int thid = blockDim.x*blockIdx.x+threadIdx.x;
    int laneId= threadIdx.x & 0x1f;
    int myWarpID=thid >> 5;
    int NWarps=(blockDim.x*gridDim.x) >> 5;
    if (thid ==0 ) printf("Num_element: %d\n",N);
    while (myWarpID < NSubranges)//WarpID is used as subrange ID
    {
        index_t mybegin_pos=(myWarpID<<alpha)+laneId;
        index_t myend_pos=(((myWarpID+1)<<alpha) < (N)) ? ((myWarpID+1)<<alpha):N;
        data_t Max=0;//Assigning all the threads Max reads to 0 intially. Avoids the illegal memory access for Max=A[mybegin_pos]; condition

        while(mybegin_pos < myend_pos)
        {
            Max=(Max < A[mybegin_pos]) ? A[mybegin_pos]:Max;
            mybegin_pos+=32;
        }
        data_t MaxFromOther;
        for (int j=Nthreadstowork >> 1;j >=1;j=j>>1)
        {
            MaxFromOther=__shfl_sync(0xffffffff, Max,laneId+j,32/*Nthreadstowork*//*32*/);
            if (laneId<j)
            {
                Max= (MaxFromOther > Max) ? MaxFromOther : Max ;
            }            
        }
        if(laneId==0)
        {
            if (myWarpID < 0)
            {
                printf("Negative memory access!\n");
            }
            if (myWarpID > NSubranges)
            {
                printf("Nsubranges:%d myWarpID: %d\n",NSubranges,myWarpID);
            }
            SubRangeMax[myWarpID]=Max;
            SubrangeId[myWarpID]=myWarpID;
            // SubRangeMax[myWarpID]=0;
            // SubrangeId[0]=0;
        }
        myWarpID+=NWarps;
    }
    return;
}

    template<typename data_t,typename index_t>
__global__ void sampleMax_NoReduction_NoWarp (data_t* A,data_t* SubRangeMax, index_t N,index_t NSubranges,index_t SubRangeSize,const index_t alpha,index_t* SubrangeId,int NSharedMememoryElements)
{
    //every read for a thread is dumped into the shared memory without reduction
    //shared memory size= 32*BlockDim.x // So need to reduce the blockDim by half Shmem 64KB to 32KB
    //Subrange of Size 32 is considered
    extern __shared__ data_t ShMem[];
    int readIndex = ((blockDim.x*blockIdx.x)<<alpha) +threadIdx.x;//1 block reads 
    int laneId= threadIdx.x & 0x1f;
    data_t Max;
    //    int warpIDinBlock=threadIdx.x >> 5;
    int tid_block;
    int NWarpsInBlock=blockDim.x >> 5; //
    int pos=threadIdx.x+blockIdx.x*blockDim.x;
    while(readIndex < N)
    {
        tid_block=threadIdx.x;
        //        warpIDinBlock=threadIdx.x >> 5;

        int index=readIndex; 
        while (tid_block<NSharedMememoryElements)
            //        for (int i=0;i<SubRangeSize;i++)
        {
            //            Max=A[index];// the maximum value for every thread is 
            ShMem[INDEX(tid_block)]=A[index];
            //  warpIDinBlock+=NWarpsInBlock;
            index+=blockDim.x;
            tid_block+=blockDim.x;
        }    
        __syncthreads();   
        index_t start_sharedMem=threadIdx.x <<alpha; //thid*8
        index_t stop_sharedMem= ((threadIdx.x+1)<<alpha) < NSharedMememoryElements ? (threadIdx.x+1)<<alpha : NSharedMememoryElements;
        Max=ShMem[INDEX(start_sharedMem)];
        for (index_t i=start_sharedMem+1; i<stop_sharedMem;i++)
        {
            Max= (ShMem[INDEX(i)] < Max) ? Max :ShMem[INDEX(i)];
        }
        SubRangeMax[pos]=Max;
        SubrangeId[pos]=pos;
        readIndex+=((blockDim.x*gridDim.x) << alpha);// 1 block of threads work for 32*blockDim.x elements 
        pos+=blockDim.x*gridDim.x;
        __syncthreads();
    }    
    return;
}
    template<typename data_t,typename index_t>
__global__ void sampleMax_NoReduction_Beta2_multiGPU (data_t* A,data_t* SubRangeMax, index_t N,index_t NSubranges,index_t SubRangeSize,const index_t alpha,index_t* SubrangeId,int NSharedMememoryElements,int beta,int GPU_offset, int NSubrangePerGPU, int gpu_id, int vectorOffset)
{
    //every read for a thread is dumped into the shared memory without reduction
    //shared memory size= 32*BlockDim.x // So need to reduce the blockDim by half Shmem 64KB to 32KB
    //Subrange of Size 32 is considered
    extern __shared__ data_t ShMem[];
    int readIndex = ((blockDim.x*blockIdx.x)<<alpha) +threadIdx.x + vectorOffset;//1 block reads 
    int laneId= threadIdx.x & 0x1f;
    //    data_t Max;
    int warpIDinBlock=threadIdx.x >> 5;
    int NWarpsInBlock=blockDim.x >> 5; //
    int subrangeOffset = gpu_id*NSubrangePerGPU;
    int pos_offset = subrangeOffset*beta;
    int pos=(threadIdx.x+blockIdx.x*blockDim.x)*beta + pos_offset;//NEED to update

    int SubRangeID=threadIdx.x+blockIdx.x*blockDim.x + subrangeOffset;

    while(readIndex < (N + vectorOffset))
    {
        warpIDinBlock=threadIdx.x >> 5;
        int index=readIndex; 
        for (int i=0;i<SubRangeSize;i++)
        {
            //            Max=A[index];// the maximum value for every thread is
            //          if (index==125)
            //          {
            //              printf("Found %d at %d \n",A[index],index);
            //          }
            ShMem[INDEX((warpIDinBlock<<5)+laneId)]=A[index];
            warpIDinBlock+=NWarpsInBlock;
            index+=blockDim.x;
        }    
        __syncthreads();   
        index_t start_sharedMem=threadIdx.x <<alpha; //thid*8
        index_t stop_sharedMem= ((threadIdx.x+1)<<alpha) < NSharedMememoryElements ? (threadIdx.x+1)<<alpha : NSharedMememoryElements;

        data_t Max=ShMem[INDEX(start_sharedMem)];
        data_t Second=0;
        data_t Third=0;

        //       data_t Second=ShMem[INDEX(start_sharedMem+1)];
        //       data_t Third=ShMem[INDEX(start_sharedMem+2)];
        for (index_t i=start_sharedMem+1; i<stop_sharedMem;i++)
        {
            // data_t data;
            data_t data=ShMem[INDEX(i)];
            if (Max < data)
            {
                //    Third=Second;
                Second=Max;
                Max=data;
            }
            else if (Second<data)
            {
                //  Third=Second;
                Second=data;
            }
            //  else if (Third<data)
            //  {
            //      Third=data;
            //  }
            //Max= (ShMem[INDEX(i)] < Max) ? Max :ShMem[INDEX(i)];
        }
        SubRangeMax[pos]=Max;
        SubRangeMax[pos+1]=Second;
        //SubRangeMax[pos+2]=Third;

        SubrangeId[pos]=SubRangeID;
        SubrangeId[pos+1]=SubRangeID;
        // SubrangeId[pos+2]=SubRangeID;

        readIndex+=((blockDim.x*gridDim.x) << alpha);// 1 block of threads work for 32*blockDim.x elements 
        SubRangeID+=blockDim.x*gridDim.x;
        pos+=(blockDim.x*gridDim.x)*beta;
        __syncthreads();
    }    
    return;
}
    template<typename data_t,typename index_t>
__global__ void sampleMax_NoReduction_Beta2 (data_t* A,data_t* SubRangeMax, index_t N,index_t NSubranges,index_t SubRangeSize,const index_t alpha,index_t* SubrangeId,int NSharedMememoryElements,int beta)
{
    //every read for a thread is dumped into the shared memory without reduction
    //shared memory size= 32*BlockDim.x // So need to reduce the blockDim by half Shmem 64KB to 32KB
    //Subrange of Size 32 is considered
    extern __shared__ data_t ShMem[];
    int readIndex = ((blockDim.x*blockIdx.x)<<alpha) +threadIdx.x;//1 block reads 
    int laneId= threadIdx.x & 0x1f;
    //    data_t Max;
    int warpIDinBlock=threadIdx.x >> 5;
    int NWarpsInBlock=blockDim.x >> 5; //
    int pos=(threadIdx.x+blockIdx.x*blockDim.x)*beta;
    int SubRangeID=threadIdx.x+blockIdx.x*blockDim.x;

    while(readIndex < N)
    {
        warpIDinBlock=threadIdx.x >> 5;
        int index=readIndex; 
        for (int i=0;i<SubRangeSize;i++)
        {
            //            Max=A[index];// the maximum value for every thread is
            //          if (index==125)
            //          {
            //              printf("Found %d at %d \n",A[index],index);
            //          }
            ShMem[INDEX((warpIDinBlock<<5)+laneId)]=A[index];
            warpIDinBlock+=NWarpsInBlock;
            index+=blockDim.x;
        }    
        __syncthreads();   
        index_t start_sharedMem=threadIdx.x <<alpha; //thid*8
        index_t stop_sharedMem= ((threadIdx.x+1)<<alpha) < NSharedMememoryElements ? (threadIdx.x+1)<<alpha : NSharedMememoryElements;

        data_t Max=ShMem[INDEX(start_sharedMem)];
        data_t Second=0;
        data_t Third=0;

        //       data_t Second=ShMem[INDEX(start_sharedMem+1)];
        //       data_t Third=ShMem[INDEX(start_sharedMem+2)];
        for (index_t i=start_sharedMem+1; i<stop_sharedMem;i++)
        {
            // data_t data;
            data_t data=ShMem[INDEX(i)];
            if (Max < data)
            {
                //    Third=Second;
                Second=Max;
                Max=data;
            }
            else if (Second<data)
            {
                //  Third=Second;
                Second=data;
            }
            //  else if (Third<data)
            //  {
            //      Third=data;
            //  }
            //Max= (ShMem[INDEX(i)] < Max) ? Max :ShMem[INDEX(i)];
        }
        SubRangeMax[pos]=Max;
        SubRangeMax[pos+1]=Second;
        //SubRangeMax[pos+2]=Third;

        SubrangeId[pos]=SubRangeID;
        SubrangeId[pos+1]=SubRangeID;
        // SubrangeId[pos+2]=SubRangeID;

        readIndex+=((blockDim.x*gridDim.x) << alpha);// 1 block of threads work for 32*blockDim.x elements 
        SubRangeID+=blockDim.x*gridDim.x;
        pos+=(blockDim.x*gridDim.x)*beta;
        __syncthreads();
    }    
    return;
}
    template<typename data_t,typename index_t>
__global__ void sampleMax_NoReduction_Beta3 (data_t* A,data_t* SubRangeMax, index_t N,index_t NSubranges,index_t SubRangeSize,const index_t alpha,index_t* SubrangeId,int NSharedMememoryElements,int beta)
{
    //every read for a thread is dumped into the shared memory without reduction
    //shared memory size= 32*BlockDim.x // So need to reduce the blockDim by half Shmem 64KB to 32KB
    //Subrange of Size 32 is considered
    extern __shared__ data_t ShMem[];
    int readIndex = ((blockDim.x*blockIdx.x)<<alpha) +threadIdx.x;//1 block reads 
    int laneId= threadIdx.x & 0x1f;
    //    data_t Max;
    int warpIDinBlock=threadIdx.x >> 5;
    int NWarpsInBlock=blockDim.x >> 5; //
    int pos=(threadIdx.x+blockIdx.x*blockDim.x)*beta;
    int SubRangeID=threadIdx.x+blockIdx.x*blockDim.x;

    while(readIndex < N)
    {
        warpIDinBlock=threadIdx.x >> 5;
        int index=readIndex; 
        for (int i=0;i<SubRangeSize;i++)
        {
            //            Max=A[index];// the maximum value for every thread is
            //          if (index==125)
            //          {
            //              printf("Found %d at %d \n",A[index],index);
            //          }
            ShMem[INDEX((warpIDinBlock<<5)+laneId)]=A[index];
            warpIDinBlock+=NWarpsInBlock;
            index+=blockDim.x;
        }    
        __syncthreads();   
        index_t start_sharedMem=threadIdx.x <<alpha; //thid*8
        index_t stop_sharedMem= ((threadIdx.x+1)<<alpha) < NSharedMememoryElements ? (threadIdx.x+1)<<alpha : NSharedMememoryElements;

        data_t Max=ShMem[INDEX(start_sharedMem)];
        data_t Second=0;
        data_t Third=0;

        //       data_t Second=ShMem[INDEX(start_sharedMem+1)];
        //       data_t Third=ShMem[INDEX(start_sharedMem+2)];
        for (index_t i=start_sharedMem+1; i<stop_sharedMem;i++)
        {
            // data_t data;
            data_t data=ShMem[INDEX(i)];
            if (Max < data)
            {
                Third=Second;
                Second=Max;
                Max=data;
            }
            else if (Second<data)
            {
                Third=Second;
                Second=data;
            }
            else if (Third<data)
            {
                Third=data;
            }
            //Max= (ShMem[INDEX(i)] < Max) ? Max :ShMem[INDEX(i)];
        }
        SubRangeMax[pos]=Max;
        SubRangeMax[pos+1]=Second;
        SubRangeMax[pos+2]=Third;

        SubrangeId[pos]=SubRangeID;
        SubrangeId[pos+1]=SubRangeID;
        SubrangeId[pos+2]=SubRangeID;

        readIndex+=((blockDim.x*gridDim.x) << alpha);// 1 block of threads work for 32*blockDim.x elements 
        SubRangeID+=blockDim.x*gridDim.x;
        pos+=(blockDim.x*gridDim.x)*beta;
        __syncthreads();
    }    
    return;
}

    template<typename data_t,typename index_t>
__global__ void sampleMax_NoReduction_Beta4 (data_t* A,data_t* SubRangeMax, index_t N,index_t NSubranges,index_t SubRangeSize,const index_t alpha,index_t* SubrangeId,int NSharedMememoryElements,int beta)
{
    //every read for a thread is dumped into the shared memory without reduction
    //shared memory size= 32*BlockDim.x // So need to reduce the blockDim by half Shmem 64KB to 32KB
    //Subrange of Size 32 is considered
    extern __shared__ data_t ShMem[];
    int readIndex = ((blockDim.x*blockIdx.x)<<alpha) +threadIdx.x;//1 block reads 
    int laneId= threadIdx.x & 0x1f;
    //    data_t Max;
    int warpIDinBlock=threadIdx.x >> 5;
    int NWarpsInBlock=blockDim.x >> 5; //
    int pos=(threadIdx.x+blockIdx.x*blockDim.x)*beta;
    int SubRangeID=threadIdx.x+blockIdx.x*blockDim.x;

    while(readIndex < N)
    {
        warpIDinBlock=threadIdx.x >> 5;
        int index=readIndex; 
        for (int i=0;i<SubRangeSize;i++)
        {
            //            Max=A[index];// the maximum value for every thread is
            //          if (index==125)
            //          {
            //              printf("Found %d at %d \n",A[index],index);
            //          }
            ShMem[INDEX((warpIDinBlock<<5)+laneId)]=A[index];
            warpIDinBlock+=NWarpsInBlock;
            index+=blockDim.x;
        }    
        __syncthreads();   
        index_t start_sharedMem=threadIdx.x <<alpha; //thid*8
        index_t stop_sharedMem= ((threadIdx.x+1)<<alpha) < NSharedMememoryElements ? (threadIdx.x+1)<<alpha : NSharedMememoryElements;

        data_t Max=ShMem[INDEX(start_sharedMem)];
        data_t Second=0;
        data_t Third=0;
        data_t Fourth=0;

        //       data_t Second=ShMem[INDEX(start_sharedMem+1)];
        //       data_t Third=ShMem[INDEX(start_sharedMem+2)];
        for (index_t i=start_sharedMem+1; i<stop_sharedMem;i++)
        {
            // data_t data;
            data_t data=ShMem[INDEX(i)];
            if (Max < data)
            {
                Fourth= Third;
                Third=Second;
                Second=Max;
                Max=data;
            }
            else if (Second<data)
            {
                Fourth= Third;
                Third=Second;
                Second=data;
            }
            else if (Third<data)
            {
                Fourth= Third;
                Third=data;
            }
            else if (Fourth<data)
            {
                Fourth= Third;
            }
            //Max= (ShMem[INDEX(i)] < Max) ? Max :ShMem[INDEX(i)];
        }
        SubRangeMax[pos]=Max;
        SubRangeMax[pos+1]=Second;
        SubRangeMax[pos+2]=Third;

        SubrangeId[pos]=SubRangeID;
        SubrangeId[pos+1]=SubRangeID;
        SubrangeId[pos+2]=SubRangeID;

        readIndex+=((blockDim.x*gridDim.x) << alpha);// 1 block of threads work for 32*blockDim.x elements 
        SubRangeID+=blockDim.x*gridDim.x;
        pos+=(blockDim.x*gridDim.x)*beta;
        __syncthreads();
    }    
    return;
}
    template<typename data_t,typename index_t>
__global__ void sampleMax_NoReduction_Beta (data_t* A,data_t* SubRangeMax, index_t N,index_t NSubranges,index_t SubRangeSize,const index_t alpha,index_t* SubrangeId,int NSharedMememoryElements,int beta)
{
    //every read for a thread is dumped into the shared memory without reduction
    //shared memory size= 32*BlockDim.x // So need to reduce the blockDim by half Shmem 64KB to 32KB
    //Subrange of Size 32 is considered
    extern __shared__ data_t ShMem[];
    int readIndex = ((blockDim.x*blockIdx.x)<<alpha) +threadIdx.x;//1 block reads 
    int laneId= threadIdx.x & 0x1f;
    //    data_t Max;
    int warpIDinBlock=threadIdx.x >> 5;
    int NWarpsInBlock=blockDim.x >> 5; //
    int pos=(threadIdx.x+blockIdx.x*blockDim.x)*beta;
    int SubRangeID=threadIdx.x+blockIdx.x*blockDim.x;
    while(readIndex < N)
    {
        warpIDinBlock=threadIdx.x >> 5;
        int index=readIndex; 
        for (int i=0;i<SubRangeSize;i++)
        {
            //            Max=A[index];// the maximum value for every thread is
            //          if (index==125)
            //          {
            //              printf("Found %d at %d \n",A[index],index);
            //          }
            ShMem[INDEX((warpIDinBlock<<5)+laneId)]=A[index];
            warpIDinBlock+=NWarpsInBlock;
            index+=blockDim.x;
        }    
        __syncthreads();   
        index_t start_sharedMem=threadIdx.x <<alpha; //thid*8
        index_t stop_sharedMem= ((threadIdx.x+1)<<alpha) < NSharedMememoryElements ? (threadIdx.x+1)<<alpha : NSharedMememoryElements;
        data_t Max;data_t Second;data_t Third;data_t Fourth;
        switch(beta)
        {
            case 4:
                Max=ShMem[INDEX(start_sharedMem)];
                Second=0;
                Third=0;
                Fourth=0;

                //       data_t Second=ShMem[INDEX(start_sharedMem+1)];
                //       data_t Third=ShMem[INDEX(start_sharedMem+2)];
                for (index_t i=start_sharedMem+1; i<stop_sharedMem;i++)
                {
                    // data_t data;
                    data_t data=ShMem[INDEX(i)];
                    if (Max < data)
                    {
                        Fourth=Third;
                        Third=Second;
                        Second=Max;
                        Max=data;
                    }
                    else if (Second<data)
                    {
                        Fourth=Third;
                        Third=Second;
                        Second=data;
                    }
                    else if (Third<data)
                    {
                        Fourth=Third;
                        Third=data;
                    }
                    else if (Fourth<data)
                    {
                        Fourth=Third;
                    }
                    //Max= (ShMem[INDEX(i)] < Max) ? Max :ShMem[INDEX(i)];
                }
                SubRangeMax[pos]=Max;
                SubRangeMax[pos+1]=Second;
                SubRangeMax[pos+2]=Third;
                SubRangeMax[pos+2]=Fourth;

                SubrangeId[pos]=SubRangeID;
                SubrangeId[pos+1]=SubRangeID;
                SubrangeId[pos+2]=SubRangeID;
                SubrangeId[pos+3]=SubRangeID;
                break;

            case 3:
                Max=ShMem[INDEX(start_sharedMem)];
                Second=0;
                Third=0;

                //       data_t Second=ShMem[INDEX(start_sharedMem+1)];
                //       data_t Third=ShMem[INDEX(start_sharedMem+2)];
                for (index_t i=start_sharedMem+1; i<stop_sharedMem;i++)
                {
                    // data_t data;
                    data_t data=ShMem[INDEX(i)];
                    if (Max < data)
                    {
                        Third=Second;
                        Second=Max;
                        Max=data;
                    }
                    else if (Second<data)
                    {
                        Third=Second;
                        Second=data;
                    }
                    else if (Third<data)
                    {
                        Third=data;
                    }
                    //Max= (ShMem[INDEX(i)] < Max) ? Max :ShMem[INDEX(i)];
                }
                SubRangeMax[pos]=Max;
                SubRangeMax[pos+1]=Second;
                SubRangeMax[pos+2]=Third;

                SubrangeId[pos]=SubRangeID;
                SubrangeId[pos+1]=SubRangeID;
                SubrangeId[pos+2]=SubRangeID;
                break;

            default:
                // case 2
                Max=ShMem[INDEX(start_sharedMem)];
                Second=0;
                // data_t Third=0;

                //       data_t Second=ShMem[INDEX(start_sharedMem+1)];
                //       data_t Third=ShMem[INDEX(start_sharedMem+2)];
                for (index_t i=start_sharedMem+1; i<stop_sharedMem;i++)
                {
                    // data_t data;
                    data_t data=ShMem[INDEX(i)];
                    if (Max < data)
                    {
                        // Third=Second;
                        Second=Max;
                        Max=data;
                    }
                    else if (Second<data)
                    {
                        // Third=Second;
                        Second=data;
                    }
                    // else if (Third<data)
                    // {
                    // 	Third=data;
                    // }
                    //Max= (ShMem[INDEX(i)] < Max) ? Max :ShMem[INDEX(i)];
                }
                SubRangeMax[pos]=Max;
                SubRangeMax[pos+1]=Second;
                // SubRangeMax[pos+2]=Third;

                SubrangeId[pos]=SubRangeID;
                SubrangeId[pos+1]=SubRangeID;
                // SubrangeId[pos+2]=SubRangeID;

        }
        readIndex+=((blockDim.x*gridDim.x) << alpha);// 1 block of threads work for 32*blockDim.x elements 
        SubRangeID+=blockDim.x*gridDim.x;
        pos+=(blockDim.x*gridDim.x)*beta;
        __syncthreads();
    }    
    return;
}

    template<typename data_t, typename index_t>
__global__ void findTopk(data_t* vec_d,index_t num_element,index_t k,data_t TopKElement,data_t* TopKElements_d,int* topkOffset)
{
    index_t original_thid = blockDim.x*blockIdx.x+threadIdx.x;
    index_t thid = original_thid;
    while (thid < num_element)
    {
        data_t value = vec_d[thid];
        if (value > TopKElement)
        {
            TopKElements_d[atomicAdd(topkOffset,1)] = value;
        }
        thid += (blockDim.x*gridDim.x);
    }

}

    template<typename data_t, typename index_t>
__global__ void FillTopk(data_t* vec_d,index_t num_element,index_t k,data_t TopKElement,data_t* TopKElements_d,int* topkOffset)
{
    index_t original_thid = (blockDim.x*blockIdx.x+threadIdx.x) + topkOffset[0];
    index_t thid = original_thid;
    while (thid < k)
    {
        //fillup remaining space
        TopKElements_d[atomicAdd(topkOffset,1)] = TopKElement;
        thid += (blockDim.x*gridDim.x);
    }

}
    template<typename data_t,typename index_t>
__global__ void sampleMax_NoReduction_BU (data_t* A,data_t* SubRangeMax, index_t N,index_t NSubranges,index_t SubRangeSize,const index_t alpha,index_t* SubrangeId,int NSharedMememoryElements)
{
    //every read for a thread is dumped into the shared memory without reduction
    //shared memory size= 32*BlockDim.x // So need to reduce the blockDim by half Shmem 64KB to 32KB
    //Subrange of Size 32 is considered
    extern __shared__ data_t ShMem[];
    int readIndex = ((blockDim.x*blockIdx.x)<<alpha) +threadIdx.x;//1 block reads 
    int laneId= threadIdx.x & 0x1f;
    data_t Max;
    int warpIDinBlock=threadIdx.x >> 5;
    int NWarpsInBlock=blockDim.x >> 5; //
    int pos=threadIdx.x+blockIdx.x*blockDim.x;
    while(readIndex < N)
    {
        warpIDinBlock=threadIdx.x >> 5;
        int index=readIndex; 
        for (int i=0;i<SubRangeSize;i++)
        {
            //            Max=A[index];// the maximum value for every thread is 
            ShMem[INDEX((warpIDinBlock<<5)+laneId)]=A[index];
            warpIDinBlock+=NWarpsInBlock;
            index+=blockDim.x;
        }    
        __syncthreads();   
        index_t start_sharedMem=threadIdx.x <<alpha; //thid*8
        index_t stop_sharedMem= ((threadIdx.x+1)<<alpha) < NSharedMememoryElements ? (threadIdx.x+1)<<alpha : NSharedMememoryElements;
        Max=ShMem[INDEX(start_sharedMem)];
        for (index_t i=start_sharedMem+1; i<stop_sharedMem;i++)
        {
            Max= (ShMem[INDEX(i)] < Max) ? Max :ShMem[INDEX(i)];
        }
        SubRangeMax[pos]=Max;
        SubrangeId[pos]=pos;
        readIndex+=((blockDim.x*gridDim.x) << alpha);// 1 block of threads work for 32*blockDim.x elements 
        pos+=blockDim.x*gridDim.x;
        __syncthreads();
    }    
    return;
}



    template<typename data_t, typename index_t>
void radix_select_inplace(data_t* vec_d,index_t num_element,index_t k,index_t num_bucket,data_t& TopKElement,int NBitsperDigit,int CurrentDigit,data_t flag)
{
    double inplace_start=wtime();
    cout<<endl<<endl<<"Inside InPlace Radix Select(Phase II)"<<endl;
    index_t Kcount;

    int* Count=(int*)malloc(sizeof(int)*num_bucket);//new index_t[num_bucket];
    int* Count_d;
    H_ERR(cudaMalloc((void**) &Count_d,sizeof(int)*num_bucket));

    index_t* CumCount=(index_t*)malloc(sizeof(index_t)*num_bucket);//new index_t[num_bucket];

    index_t Belowcount=0;
    int Kdigit;
    cout<<"Number of elements"<<num_element<<endl;

    while(CurrentDigit>=0)
    {
        cout<<"Current Digit:"<<CurrentDigit<<endl<<endl;
        Kcount=0;
        H_ERR(cudaMemset(Count_d, 0, num_bucket * sizeof(int)));
        int shleft=(CurrentDigit+1)*(NBitsperDigit);
        int shright=CurrentDigit*NBitsperDigit;
        CalculateOccurence_inplace<data_t,index_t><<<128,128,num_bucket*sizeof(int)>>>(vec_d,num_element,Count_d,NBitsperDigit,CurrentDigit,num_bucket,flag,shleft,shright);
        H_ERR(cudaDeviceSynchronize());
        H_ERR(cudaMemcpy(Count,Count_d,sizeof(int)*num_bucket,cudaMemcpyDeviceToHost));
        memset(CumCount, 0, num_bucket * sizeof(index_t));
        CumulateCount_inplace<data_t,index_t>(Count,CumCount,num_bucket,Kdigit,k,num_element,Belowcount,flag,NBitsperDigit,CurrentDigit);
        if (Kdigit!=0)  Belowcount=CumCount[Kdigit-1];
        cout<<"KDigit"<<Kdigit<<endl;
        cout<<"BelowCount:"<<Belowcount<<endl;
        Kcount=Count[Kdigit];
        cout<<"Kcount:"<<Kcount<<endl;
        CurrentDigit--;
    } 

    if (CurrentDigit==-1)
    {
        TopKElement=flag;
    }
    else
    {
        data_t* TopKElement_d;
        H_ERR(cudaMalloc((void**) &TopKElement_d,sizeof(data_t)));
        int* check_d;
        H_ERR(cudaMalloc((void**) &check_d,sizeof(int)));
        H_ERR(cudaMemset(check_d,0,sizeof(int)));
        int* countcheck_d;
        H_ERR(cudaMalloc((void**) &countcheck_d,sizeof(int)));
        H_ERR(cudaMemset(countcheck_d,0,sizeof(int)));
        int shleft=(CurrentDigit+1)*NBitsperDigit;
        data_t* TopKElements_d;
        H_ERR(cudaMalloc((void**) &TopKElements_d,sizeof(data_t)*k));
        SelectKNumber<data_t,index_t><<<128,128>>>(vec_d,num_element,TopKElement_d,countcheck_d,flag,shleft,TopKElements_d,k);
        H_ERR(cudaDeviceSynchronize());
        H_ERR(cudaMemcpy(&TopKElement,TopKElement_d,sizeof(data_t),cudaMemcpyDeviceToHost));
        H_ERR(cudaFree(check_d));H_ERR(cudaFree(TopKElement_d));
    }
    cout<<"Total time of radix_select Inplace(Phase II):"<<(wtime()-inplace_start)*1000<<" ms"<<endl;
}



    template<typename data_t, typename index_t>
__global__ void CreateConcatenateRange(data_t* vec,data_t* ConcatenatedRange,index_t* SubrangeId,index_t ConcatenatedSize,index_t k,index_t Subrangesize)
{
    index_t thid = blockDim.x*blockIdx.x+threadIdx.x;
    index_t laneId= threadIdx.x & 0x1f;
    index_t myWarpID=thid >> 5;
    index_t NWarps=(blockDim.x*gridDim.x) >> 5;
    index_t NContributingSubrange=k; //for single maximum selection it will be k . For beta selection, the number is likely to be less than k
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
__global__ void CreateConcatenateRange_optimized(data_t* vec,data_t* ConcatenatedRange,index_t* SubrangeId,index_t ConcatenatedSize,index_t k,index_t Subrangesize,int* write_pos,data_t firsttopk)
{
    index_t thid = blockDim.x*blockIdx.x+threadIdx.x;
    index_t laneId= threadIdx.x & 0x1f;
    index_t myWarpID=thid >> 5;
    index_t NWarps=(blockDim.x*gridDim.x) >> 5;
    index_t NContributingSubrange=k; //for single maximum selection it will be k . For beta selection, the number is likely to be less than k
    while (myWarpID < NContributingSubrange)//
    {
        index_t mybegin_pos=(myWarpID*Subrangesize)+laneId;
        index_t myend_pos=(((myWarpID+1)*Subrangesize) < ConcatenatedSize) ? (myWarpID+1)*Subrangesize : ConcatenatedSize;
        index_t index=SubrangeId[myWarpID];
        index_t myvec_pos=index* Subrangesize+laneId;
        index_t l_id=laneId;
        while (l_id<Subrangesize)
        {
            if (mybegin_pos >= ConcatenatedSize)
            {
                printf("Error! Illegal memory access. \n");
            }
            data_t value=vec[myvec_pos];
            if (value >= firsttopk)
            {
                ConcatenatedRange[atomicAdd(write_pos,1)]=vec[myvec_pos];
            }
            mybegin_pos+=32; 
            myvec_pos+=32;
            l_id+=32;
        }
        myWarpID+=NWarps;
    }
    return;
}

    template<typename data_t, typename index_t>
__global__ void CreateConcatenateRange_Beta_optimized(data_t* vec,data_t* ConcatenatedRange,index_t* SubrangeId,index_t ConcatenatedSize,index_t NContributingSubrange,index_t Subrangesize,int* CountLonelyElements_d,int* write_pos,data_t firsttopk)
{
    index_t thid = blockDim.x*blockIdx.x+threadIdx.x;
    index_t laneId= threadIdx.x & 0x1f;
    index_t myWarpID=thid >> 5;
    index_t NWarps=(blockDim.x*gridDim.x) >> 5;
    //    index_t NContributingSubrange=k; //for single maximum selection it will be k . For beta selection, the number is likely to be less than k
    while (myWarpID < NContributingSubrange)//
    {
        index_t mybegin_pos=(myWarpID*Subrangesize)+laneId+CountLonelyElements_d[0];
        index_t myend_pos=(((myWarpID+1)*Subrangesize+CountLonelyElements_d[0]) < ConcatenatedSize) ? ((myWarpID+1)*Subrangesize+CountLonelyElements_d[0]) : ConcatenatedSize;
        index_t index=SubrangeId[myWarpID];
        index_t myvec_pos=index * Subrangesize+laneId;
        index_t l_id=laneId;
        while (l_id < Subrangesize)
        {
            data_t value=vec[myvec_pos];
            if (value >= firsttopk)
            {
                //  ConcatenatedRange[mybegin_pos]=vec[myvec_pos];
                ConcatenatedRange[atomicAdd(write_pos,1)]=vec[myvec_pos];
            }
            mybegin_pos+=32; 
            myvec_pos+=32;
            l_id+=32;
        }
        myWarpID+=NWarps;
    }
    return;
}

    template<typename data_t, typename index_t>
__global__ void CreateConcatenateRange_Beta(data_t* vec,data_t* ConcatenatedRange,index_t* SubrangeId,index_t ConcatenatedSize,index_t NContributingSubrange,index_t Subrangesize,int* CountLonelyElements_d)
{
    index_t thid = blockDim.x*blockIdx.x+threadIdx.x;
    index_t laneId= threadIdx.x & 0x1f;
    index_t myWarpID=thid >> 5;
    index_t NWarps=(blockDim.x*gridDim.x) >> 5;
    //    index_t NContributingSubrange=k; //for single maximum selection it will be k . For beta selection, the number is likely to be less than k
    while (myWarpID < NContributingSubrange)//
    {
        index_t mybegin_pos=(myWarpID*Subrangesize)+laneId+CountLonelyElements_d[0];
        index_t myend_pos=(((myWarpID+1)*Subrangesize+CountLonelyElements_d[0]) < ConcatenatedSize) ? ((myWarpID+1)*Subrangesize+CountLonelyElements_d[0]) : ConcatenatedSize;
        index_t index=SubrangeId[myWarpID];
        index_t myvec_pos=index* Subrangesize+laneId;
        while (mybegin_pos<myend_pos)
        {
            ConcatenatedRange[mybegin_pos]=vec[myvec_pos];
            mybegin_pos+=32; 
            myvec_pos+=32;
        }
        myWarpID+=NWarps;
    }
    return;
}

    template<typename data_t, typename index_t>
void radix_select_inplace_firsttopk(data_t* vec_d,index_t num_element,index_t k,index_t num_bucket,data_t& TopKElement,int NBitsperDigit,int CurrentDigit,data_t flag,index_t* SubrangeId_d,index_t* SelectedSubrangeId_d,int* CountSelectedSubrange_d)
{
    double inplace_start=wtime();
    cout<<endl<<endl<<"Inside InPlace Radix Select(Phase II)"<<endl;
    index_t Kcount;

    int* Count=(int*)malloc(sizeof(int)*num_bucket);//new index_t[num_bucket];
    int* Count_d;
    H_ERR(cudaMalloc((void**) &Count_d,sizeof(int)*num_bucket));

    index_t* CumCount=(index_t*)malloc(sizeof(index_t)*num_bucket);//new index_t[num_bucket];

    index_t Belowcount=0;
    int Kdigit;
    cout<<"Number of elements"<<num_element<<endl;
    int shleft,shright;    
    while(CurrentDigit>=0)
    {
        cout<<"Current Digit:"<<CurrentDigit<<endl<<endl;
        Kcount=0;
        H_ERR(cudaMemset(Count_d, 0, num_bucket * sizeof(int)));
        shleft=(CurrentDigit+1)*(NBitsperDigit);
        shright=CurrentDigit*NBitsperDigit;
        CalculateOccurence_inplace_firsttopk<data_t,index_t><<<128,128,num_bucket*sizeof(int)>>>(vec_d,num_element,Count_d,NBitsperDigit,CurrentDigit,num_bucket,flag,shleft,shright);
        H_ERR(cudaDeviceSynchronize());
        H_ERR(cudaMemcpy(Count,Count_d,sizeof(int)*num_bucket,cudaMemcpyDeviceToHost));
        memset(CumCount, 0, num_bucket * sizeof(index_t));
        // (int* Count,index_t* CumCount,index_t num_bucket,index_t& Kdigit,index_t k,index_t num_element,index_t Belowcount,data_t& flag, int NBitsperDigit,int Currentdigit)
        CumulateCount_inplace<data_t,index_t>(Count,CumCount,num_bucket,Kdigit,k,num_element,Belowcount,flag,NBitsperDigit,CurrentDigit);
        if (Kdigit!=0)  Belowcount=CumCount[Kdigit-1];
        cout<<"KDigit"<<Kdigit<<endl;

        cout<<"BelowCount:"<<Belowcount<<endl;

        Kcount=Count[Kdigit];
        cout<<"Kcount:"<<Kcount<<endl;
        //        if (CumCount[num_bucket-1]==1)
        //        {
        //            break;
        //        }
        CurrentDigit--;
        if (Kcount==1)
        {
            break;
        }

    }

    shleft=(CurrentDigit+1)*(NBitsperDigit);
    FindTopKRanges<data_t,index_t><<<128,128>>> (vec_d,num_element,flag,shleft,SubrangeId_d,SelectedSubrangeId_d,CountSelectedSubrange_d);
    H_ERR(cudaDeviceSynchronize());
    int count;
    H_ERR(cudaMemcpy(&count,CountSelectedSubrange_d,sizeof(int),cudaMemcpyDeviceToHost));
    cout<<"Number of subranges in first sweep:"<<count<<endl;
    FillWithKElementsID<data_t,index_t><<<128,128>>> (vec_d,num_element,flag,shleft,SubrangeId_d,SelectedSubrangeId_d,CountSelectedSubrange_d,k);
    H_ERR(cudaDeviceSynchronize());
    TopKElement=flag;
    cout<<"Total time of radix_select Inplace_firstTopk(Phase II):"<<(wtime()-inplace_start)*1000<<" ms"<<endl;
}


    template<typename data_t, typename index_t>
void radix_select_inplace_firsttopk_beta(data_t* vec_d,index_t num_element,index_t k,index_t num_bucket,data_t& TopKElement,int NBitsperDigit,int CurrentDigit,data_t flag,index_t* SubrangeId_d,index_t* SelectedSubrangeId_d,int* CountSelectedSubrange_d,data_t* ConcatenatedRange_d,int* CountLonelyElements_d,int beta)
{
    double inplace_start=wtime();
    cout<<endl<<endl<<"Inside InPlace Radix Select(Phase II)"<<endl;
    index_t Kcount;

    int* Count=(int*)malloc(sizeof(int)*num_bucket);//new index_t[num_bucket];
    int* Count_d;
    H_ERR(cudaMalloc((void**) &Count_d,sizeof(int)*num_bucket));

    index_t* CumCount=(index_t*)malloc(sizeof(index_t)*num_bucket);//new index_t[num_bucket];

    index_t Belowcount=0;
    int Kdigit;
    cout<<"Number of elements"<<num_element<<endl;
    int shleft,shright;    
    // while(CurrentDigit>=0)
    // while(CurrentDigit>=1)
    // while(CurrentDigit>=2)
    while(CurrentDigit>=3)
    {
        cout<<"Current Digit:"<<CurrentDigit<<endl<<endl;
        Kcount=0;
        H_ERR(cudaMemset(Count_d, 0, num_bucket * sizeof(index_t)));
        shleft=(CurrentDigit+1)*(NBitsperDigit);
        shright=CurrentDigit*NBitsperDigit;
        CalculateOccurence_inplace_firsttopk<data_t,index_t><<<128,128,num_bucket*sizeof(int)>>>(vec_d,num_element,Count_d,NBitsperDigit,CurrentDigit,num_bucket,flag,shleft,shright);
        H_ERR(cudaDeviceSynchronize());
        H_ERR(cudaMemcpy(Count,Count_d,sizeof(index_t)*num_bucket,cudaMemcpyDeviceToHost));
        memset(CumCount, 0, num_bucket * sizeof(index_t));
        CumulateCount_inplace<data_t,index_t>(Count,CumCount,num_bucket,Kdigit,k,num_element,Belowcount,flag,NBitsperDigit,CurrentDigit);
        if (Kdigit!=0)  Belowcount=CumCount[Kdigit-1];
        cout<<"KDigit"<<Kdigit<<endl;
        cout<<"BelowCount:"<<Belowcount<<endl;
        Kcount=Count[Kdigit];
        cout<<"Kcount:"<<Kcount<<endl;
        CurrentDigit--;
        //     if (Kcount==1)
        //     {
        //         break;
        //     }
    }
    shleft=(CurrentDigit+1)*(NBitsperDigit);

    FindTopKRanges_beta<data_t,index_t><<<128,128>>> (vec_d,num_element,flag,shleft,SubrangeId_d,SelectedSubrangeId_d,CountSelectedSubrange_d,ConcatenatedRange_d,CountLonelyElements_d,beta);
    H_ERR(cudaDeviceSynchronize());
    TopKElement=flag;
    //  int count;
    //  H_ERR(cudaMemcpy(&count,CountSelectedSubrange_d,sizeof(int),cudaMemcpyDeviceToHost));
    //  cout<<"Number of subranges in first sweep:"<<count<<endl;
    //  FillWithKElementsID<data_t,index_t><<<128,128>>> (vec_d,num_element,flag,shleft,SubrangeId_d,SelectedSubrangeId_d,CountSelectedSubrange_d,k);
    //  H_ERR(cudaDeviceSynchronize());
    cout<<"Total time of radix_select Inplace_firstTopk(Phase II):"<<(wtime()-inplace_start)*1000<<" ms"<<endl;
}






template<typename data_t, typename index_t>
void sample_radix_select(data_t* vec_d,index_t num_element,index_t k,index_t num_bucket,data_t& TopKElement,int NBitsperDigit,int CurrentDigit,index_t NSubranges,index_t SubRangesize,index_t alpha,double& timeforMaxsample,double& timeforFirstTopK,double& timeForsecondTopK,double& timeforConcatenation,
        data_t* Max_d,index_t* SubrangeId_d,int NSharedMemoryElements,int SizeOfSubWarp,int pow_size_Subwarp,index_t NSubWarps_InBlock,int NSubRangesPerBlock,int NElementsPerBlock_ReadFromGlobal,int
        TotalBlocksrequired,int SizeOfAllocation,
        int NThreadsPerBlock,int beta,bool defaultContribution,
        int NthreadstoworkInreduction,index_t* SelectedSubrangeId_d,
        int* CountLonelyElements_d,int* write_pos_d,
        data_t* ConcatenatedRange_d,int* CountSelectedSubrange_d,
        data_t* TopKElements_d,index_t NGpu, index_t myrank,
        data_t* TopKElementsRecv_d, double& timeForFinalTopK, double& overheadTime, data_t* vec,
    int& NSkipDocument)
{
    double sample_beg=wtime();
    // if (defaultContribution)
    {
        sampleMax<data_t,index_t><<<4096,512>>>(vec_d,Max_d,num_element,NSubranges,SubRangesize,alpha,SubrangeId_d,NthreadstoworkInreduction);
        // sampleMax<data_t,index_t><<<1,32>>>(vec_d,Max_d,num_element,NSubranges,SubRangesize,alpha,SubrangeId_d,NthreadstoworkInreduction);
    }
    // else
    // {
    //     if (beta ==2)
    //     {
    //         sampleMax_NoReduction_Beta2 <data_t,index_t><<<4096,NThreadsPerBlock,SizeOfAllocation*(sizeof(data_t))>>>(vec_d,Max_d,num_element,NSubranges,SubRangesize,alpha,SubrangeId_d,NSharedMemoryElements,beta);
    //     }
    //     else if (beta ==3)
    //     {
    //         //case beta =2
    //         sampleMax_NoReduction_Beta3<data_t,index_t><<<4096,NThreadsPerBlock,SizeOfAllocation*(sizeof(data_t))>>>(vec_d,Max_d,num_element,NSubranges,SubRangesize,alpha,SubrangeId_d,NSharedMemoryElements,beta);
    //     }
    //     else if (beta ==4)
    //     {
    //         //case beta =2
    //         sampleMax_NoReduction_Beta4<data_t,index_t><<<4096,NThreadsPerBlock,SizeOfAllocation*(sizeof(data_t))>>>(vec_d,Max_d,num_element,NSubranges,SubRangesize,alpha,SubrangeId_d,NSharedMemoryElements,beta);
    //     }

    // }
    H_ERR(cudaDeviceSynchronize());

    timeforMaxsample += (wtime()-sample_beg);
    cout<<"Process: "<<myrank<<" finished sampling!"<<endl;
    // return;
    data_t flag=0;
    data_t* Max= (data_t*) malloc (NSubranges*sizeof(data_t));
    H_ERR(cudaMemcpy(Max,Max_d,sizeof(data_t)*NSubranges,cudaMemcpyDeviceToHost));
    double startBMW=wtime();
    int threshold = 0;
    int docid=0;
    // int NSkipDocument = 0;
    int subIdx =0;
    PriorityQueue pq;

    while (docid< num_element)
    {
        bool skipblock =false;
        
        if ((docid % SubRangesize) ==0)
        {
            subIdx = docid/SubRangesize;
            if (Max[subIdx] < threshold)
            {
                skipblock=true;
                // cout<<"Skipped subrange: "<<subIdx<<" Max["<<subIdx<<"]: "<<Max[subIdx]<<" Threshold: "<<threshold<<endl;
            }
           
            // subIdx++;
        }

        if (skipblock)
        {
            docid += SubRangesize;
            NSkipDocument += SubRangesize;
        }
        else
        {
            //update the threshold
            if (vec[docid] > threshold)
            {
                //push vec[docid] into minimum heap tree
                pq.push(vec[docid]);
                if (pq.size() > k)
                {
                    //pop the minimum score
                    pq.pop();
                    // cout<<"Popping out the min value after full tree!"<<endl;
                    // cout<<"Size of k: "<<pq.size()<<endl;
                }
                //   update the threshold 
                threshold =  pq.top();
                // cout<<"New threshold: "<<threshold<<endl;
            }
            docid++;
        }
    }

    cout<<"TopkElement: "<<threshold<<endl;
    timeforFirstTopK += (wtime()-startBMW);
    cout<<"NSkipDocument: "<<NSkipDocument<<endl;
    
}


template<typename data_t, typename index_t>
void concatenate_radix_select(data_t* vec_d,index_t num_element,index_t k,index_t num_bucket,data_t& TopKElement,int NBitsperDigit,int CurrentDigit,index_t NSubranges,index_t SubRangesize,index_t alpha,double& timeforMaxsample,double& timeforFirstTopK,double& timeForsecondTopK,double& timeforConcatenation,
        data_t* Max_d,index_t* SubrangeId_d,int NSharedMemoryElements,int SizeOfSubWarp,int pow_size_Subwarp,index_t NSubWarps_InBlock,int NSubRangesPerBlock,int NElementsPerBlock_ReadFromGlobal,int
        TotalBlocksrequired,int SizeOfAllocation,
        int NThreadsPerBlock,int beta,bool defaultContribution,
        int NthreadstoworkInreduction,index_t* SelectedSubrangeId_d,
        int* CountLonelyElements_d,int* write_pos_d,
        data_t* ConcatenatedRange_d,int* CountSelectedSubrange_d,
        data_t* TopKElements_d,index_t NGpu, index_t myrank,data_t* TopKElementsRecv_d, double& timeForFinalTopK, double& overheadTime, index_t& ConcatenatedSize )
{

    int count,count_lonely;
    H_ERR(cudaMemcpy(&count,CountSelectedSubrange_d,sizeof(int),cudaMemcpyDeviceToHost));
    H_ERR(cudaMemcpy(&count_lonely,CountLonelyElements_d,sizeof(int),cudaMemcpyDeviceToHost));

    //Create Concatenate range
    //    index_t ConcatenatedSize=k*SubRangesize;//Maximum possible concatenated size
    //   index_t ConcatenatedSize;
    double StarttimeforConcatenation=wtime();
    if (defaultContribution)
    {
        ConcatenatedSize=k*SubRangesize;
        //   CreateConcatenateRange<data_t,index_t><<<512,512>>>(vec_d,ConcatenatedRange_d,SelectedSubrangeId_d,ConcatenatedSize,k,SubRangesize);
        H_ERR(cudaMemset(write_pos_d, 0, sizeof(int))); 
        CreateConcatenateRange_optimized<data_t,index_t><<<512,512>>>(vec_d,ConcatenatedRange_d,SelectedSubrangeId_d,ConcatenatedSize,k,SubRangesize,write_pos_d,TopKElement);
        //  CreateConcatenateRange<data_t,index_t><<<512,512>>>(vec_d,ConcatenatedRange_d,SelectedSubrangeId_d,ConcatenatedSize,count,SubRangesize,CountLonelyElements_d);
    }
    else
    {
        ConcatenatedSize=count*SubRangesize+count_lonely;
        H_ERR(cudaMemcpy(write_pos_d,&count_lonely,sizeof(int),cudaMemcpyHostToDevice));
        //       CreateConcatenateRange_Beta<data_t,index_t><<<512,512>>>(vec_d,ConcatenatedRange_d,SelectedSubrangeId_d,ConcatenatedSize,count,SubRangesize,CountLonelyElements_d);
        //   CreateConcatenateRange_Beta_optimized(data_t* vec,data_t* ConcatenatedRange,index_t* SubrangeId,index_t ConcatenatedSize,index_t NContributingSubrange,index_t Subrangesize,int* CountLonelyElements_d,int* write_pos,data_t firsttopk)
        CreateConcatenateRange_Beta_optimized<data_t,index_t><<<512,512>>>(vec_d,ConcatenatedRange_d,SelectedSubrangeId_d,ConcatenatedSize,count,SubRangesize,CountLonelyElements_d,write_pos_d,TopKElement);
        //      fContribute<<ConcatenatedSize<<endl;
    }
    H_ERR(cudaDeviceSynchronize());
    timeforConcatenation += (wtime()-StarttimeforConcatenation);
    H_ERR(cudaMemcpy(&ConcatenatedSize,write_pos_d,sizeof(int),cudaMemcpyDeviceToHost));
    cout<<"Process: "<<myrank<<" new Concatenated Size: "<<ConcatenatedSize<<endl;
    cout<<"Process: "<<myrank<<" top-k after concatenation: "<<TopKElement<<endl;
    //       fContribute<<ConcatenatedSize<<endl;
    //  fContribute.close();

    cout<<"Time for concatenation:"<<timeforConcatenation*1000<<" ms"<<endl;
    cout<<"The concatenation of the contributing subranges is done!"<<endl; 

    data_t flag=0;
    CurrentDigit=(sizeof(data_t)*8/NBitsperDigit)-1;   
    double startSecondTopK=wtime(); 
    radix_select_inplace<data_t,index_t>(ConcatenatedRange_d,ConcatenatedSize,k,num_bucket,TopKElement,NBitsperDigit,CurrentDigit,flag);
    H_ERR(cudaDeviceSynchronize());
    cout<<"Process: "<<myrank<<" top-k after immediate second top-k: "<<TopKElement<<endl;
    timeForsecondTopK += (wtime()-startSecondTopK);
}

template<typename data_t, typename index_t>
void find_topk(data_t* vec_d,index_t num_element,index_t k,index_t num_bucket,data_t& TopKElement,int NBitsperDigit,int CurrentDigit,index_t NSubranges,index_t SubRangesize,index_t alpha,double& timeforMaxsample,double& timeforFirstTopK,double& timeForsecondTopK,double& timeforConcatenation,
        data_t* Max_d,index_t* SubrangeId_d,int NSharedMemoryElements,int SizeOfSubWarp,int pow_size_Subwarp,index_t NSubWarps_InBlock,int NSubRangesPerBlock,int NElementsPerBlock_ReadFromGlobal,int
        TotalBlocksrequired,int SizeOfAllocation,
        int NThreadsPerBlock,int beta,bool defaultContribution,
        int NthreadstoworkInreduction,index_t* SelectedSubrangeId_d,
        int* CountLonelyElements_d,int* write_pos_d,
        data_t* ConcatenatedRange_d,int* CountSelectedSubrange_d,
        data_t* TopKElements_d,index_t NGpu, index_t myrank,data_t* TopKElementsRecv_d,
        double& timeForFinalTopK, double& overheadTime, index_t& ConcatenatedSize, int* topkOffset )
{       
    //finding top-k elements

    // cout<<"Myrank: "<<myrank<<" Top-k after second top-k: "<<TopKElement<<endl;
    findTopk<data_t,index_t><<<128,128>>>(ConcatenatedRange_d,ConcatenatedSize, k, TopKElement, TopKElements_d, topkOffset);
    H_ERR(cudaDeviceSynchronize());

}
