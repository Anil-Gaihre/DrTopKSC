#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include "wtime.h"
#include <cuda_runtime_api.h>
#include <assert.h>

using namespace std;

const int blocksize=128;
const int gridsize=128;
#define INDEX(X) (X+(X>>5))

static void HandleError( cudaError_t err, const char *file, int line    ) {
    if (err != cudaSuccess) {
        printf( "\n%s in %s at line %d\n", \
                cudaGetErrorString( err    ), file, line );
        exit( EXIT_FAILURE    );

    }
}
#define H_ERR( err    ) \
    (HandleError( err, __FILE__, __LINE__    ))

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



__device__ __forceinline__ int ThreadLoad(int *ptr)
{
    int retval;           
    asm volatile ("ld.global.cg.s32 %0, [%1];" :    \
            "=r"(retval) :                        \
            "l" (ptr) );                          \
        return retval;  

}

    template<typename data_t, typename index_t>
__device__ __forceinline__ index_t ThreadLoad_ind(index_t *ptr)
{
    index_t retval;           
    asm volatile ("ld.global.cg.s32 %0, [%1];" :    \
            "=r"(retval) :                        \
            "l" (ptr) );                          \
        return retval;  

}
 
    template<typename data_t, typename index_t>
void FindRange_seq(double& max, double&min, data_t* vec, index_t num_element)
{
    max=vec[0];
    min=vec[0];
    for (index_t i=0;i< num_element;i++)
    {
        int is_larger=((vec[i]>max)<<1)+(vec[i]<min);
        switch(is_larger)
        {
            case 2:
                max=vec[i];
                break;
            case 1: 
                min = vec[i];
                break;
            default:
                break;
        }
    }
    return;
}

    template<typename data_t, typename index_t>
__global__ void FindFinalMinMax(double* maxArray_d,double* minArray_d,double* max_d,double* min_d)
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
__global__ void FindRange(double* max, double* min, data_t* vec, index_t num_element)
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
  
    template<typename data_t, typename index_t>
__global__ void assignBuckets(data_t* vec,index_t num_element,index_t num_bucket,double slope,double min,double max,index_t* counter)
{
    extern __shared__ index_t SMem_Count[];
    for (int i = threadIdx.x; i < num_bucket; i += blockDim.x)
        SMem_Count[i] = 0;
    __syncthreads();

    index_t thid=threadIdx.x+blockIdx.x*blockDim.x;
    while(thid<num_element)
    {
        data_t value=vec[thid];
        if ((value>=(min))&&(value<=(max)))
        {
            int buck =(int)((value-min)*slope);
            atomicAdd(&SMem_Count[buck],1);
//            bucket[thid]=buck;
        }
  //      else
  //      {
  //          bucket[thid]=-1;//throwing the out of range element into the garbage bucket
  //      }
        thid+=blockDim.x*gridDim.x;
    }
    __syncthreads();
    for (int i=threadIdx.x;i<num_bucket;i+=blockDim.x)
    {
        atomicAdd(&counter[i],SMem_Count[i]);
    }

   return;
} 
   
    template<typename data_t, typename index_t>
index_t FindKBucket(index_t* counter,index_t num_bucket,index_t k,index_t& sum)
{
    //cout<<"Inside FindKBucket"<<endl;
    //    index_t sum=0;
    for (index_t i=0;i<num_bucket;i++)
    {
        sum+=counter[i];//adding the number of elements in the buckets 
        if (sum > k) 
        {
            sum-=counter[i];
            return i; // return the bucket containing the kth element (need to check this logic)

        }
    }
    std::cout<<"the number of elements in the buckets are less than k"<<"Number of Elements:"<<sum<<"  updated K value:"<<k<<std::endl;
    return 0;
}

   template<typename data_t, typename index_t>
__global__ void copyElements(data_t* vec,index_t num_element,index_t KBucket,data_t* newVec,index_t* countfornewvec,double max,double min,index_t KCount)
{
    index_t thid=blockDim.x*blockIdx.x+threadIdx.x;
    while (thid<num_element)
    {
        data_t value=vec[thid];
//        if (bucket[thid]==KBucket)
       if ((value>=(min))&&(value<(max)))
       //    if ((value>=min)&&(value<=max))
        {
            
            index_t index=atomicAdd(countfornewvec,1);
//            if (index > KCount)
  //          {
  //              printf("element exceeds the capacity of the vew vector\n");
  //          }
            newVec[index]=value;
        }
        thid+=blockDim.x*gridDim.x;
    }

}


    template<typename data_t, typename index_t>
data_t maximum(data_t a,data_t b)
{
    if (a>b) 
    {
        return a;

    }
    else
    {
        return b;
    }
}

    template<typename data_t, typename index_t>
data_t minimum(data_t a,data_t b)
{
    if (a<b)
    {
        return a;
    }
    else
    {
        return b;
    }
}

    template<typename data_t, typename index_t>
__global__ void GetKValue(data_t* vec,index_t num_element ,index_t KBucket,data_t* TopKElement,int* check,double max,double min)
{
    index_t thid=threadIdx.x+blockIdx.x*blockDim.x;
    while(thid<num_element)
    {
        if (ThreadLoad(check)!=0) return; //return if any of the thread finds the  element
        data_t value=vec[thid];
       //  int buck =(int)((value-min)*slope);
         //if (bucket[thid]==KBucket)
        if ((value>=min)&&(value<=max))
        {
            TopKElement[0]=value;
//            printf("topKN:%d \t",TopKElement[0]);
            atomicAdd(check,1);
            //        return;
        }
 //      if (bucket[thid]==KBucket)
 //      {
 //          TopKElement[0]=vec[thid];
 //          printf("topKN:%d \t",TopKElement[0]);
 //          atomicAdd(check,1);
 //          //        return;
 //      }
       thid+=blockDim.x*gridDim.x;
    }


}

    template<typename data_t,typename index_t>
void Max_Min(double& max,double& min,data_t* vec_d,index_t num_element)
{
    
         double tf_start=wtime();
        double *maxArray=(double*)malloc((sizeof(double))*blocksize);
        double *minArray=(double*)malloc((sizeof(double))*blocksize);

        double *maxArray_d;
        H_ERR(cudaMalloc((void**) &maxArray_d,sizeof(double)*blocksize));
        double *minArray_d;
        H_ERR(cudaMalloc((void**) &minArray_d,sizeof(double)*blocksize));
        if ((sizeof(double)*(blocksize<<1))/1024 > 32)
        {
            cout<<"Warning: 32 KB size limit of shared memory exceeded!!!"<<endl;
        }
        FindRange<data_t,index_t><<<gridsize,blocksize,sizeof(double)*(blocksize<<1)>>>(maxArray_d,minArray_d,vec_d,num_element);
        H_ERR(cudaDeviceSynchronize());
        if (gridsize>1024)
        {
            cout<<"BlockDim>1024: Cannot launch Min-Max kernel!"<<endl;
            return;
        }
        double* max_d;
        double* min_d;
        H_ERR(cudaMalloc((void**) &min_d,sizeof(double)));
        H_ERR(cudaMalloc((void**) &max_d,sizeof(double)));
        FindFinalMinMax<data_t,index_t><<<1,gridsize,(gridsize<<1)*sizeof(double)>>>(maxArray_d,minArray_d,max_d,min_d);//won't work when we make the number of blocks in FindRange kernel launch > 1024
        H_ERR(cudaDeviceSynchronize());
        H_ERR(cudaMemcpy(&max,max_d,sizeof(double),cudaMemcpyDeviceToHost));
        H_ERR(cudaMemcpy(&min,min_d,sizeof(double),cudaMemcpyDeviceToHost));

//       H_ERR(cudaMemcpy(maxArray,maxArray_d,sizeof(double)*blocksize,cudaMemcpyDeviceToHost));   
//       H_ERR(cudaMemcpy(minArray,minArray_d,sizeof(double)*blocksize,cudaMemcpyDeviceToHost));
//       min=minArray[0];
//       max=maxArray[0];
//       for (int i=1;i<gridsize;i++)
//       {
//           min=(min<minArray[i])? min: minArray[i];
//           max=(max>maxArray[i])? max: maxArray[i];
//       }
        double tf_Max_min=wtime()-tf_start;
        cout<<"Time for finding max min:"<<tf_Max_min*1000<<" ms"<<endl;
        free(maxArray);free(minArray);
        H_ERR(cudaFree(maxArray_d));H_ERR(cudaFree(minArray_d));
 

}

    template<typename data_t, typename index_t>
void bucket_select_PhaseII(data_t* vec_d,index_t num_element,index_t k,index_t num_bucket,data_t& TopKElement)
{
    cout<<"In Phase II: Inplace Bucket select"<<endl;
    double max=0;
    double min=0;
    Max_Min<data_t,index_t>(max,min,vec_d,num_element);
   if (max==min)
    {
        TopKElement=max;
        return;
    }
    double slope=((double)(num_bucket-1))/(max-min);//Number of bucket > 1 otherwise core dumped
    double* bucket_ranger=(double*)malloc((sizeof(double))*(num_bucket+1));//new double[num_bucket+1];

    for (index_t i=0;i<num_bucket;i++)
    {
        bucket_ranger[i]=i/slope+ min;
    }
    bucket_ranger[num_bucket]=max;

   index_t* counter=(index_t*)malloc((sizeof(index_t))*num_bucket);//new index_t[num_bucket];
    index_t* counter_d;
    H_ERR(cudaMalloc((void**) &counter_d,sizeof(index_t)*num_bucket));
    H_ERR(cudaMemset(counter_d, 0, num_bucket * sizeof(index_t)));
         double ta_start=wtime();
        assignBuckets<data_t,index_t><<<128,blocksize,num_bucket*sizeof(index_t)>>>(vec_d,num_element,num_bucket,slope,min,max,counter_d);
        H_ERR(cudaDeviceSynchronize());
        double ta=wtime()-ta_start;
        cout<<"Time for finding Bucket Count & assignement of bucket to the elements:"<<ta*1000<<" ms"<<endl;
//    assignBuckets<data_t,index_t><<<128,128,num_bucket*sizeof(index_t)>>>(vec_d,num_element,num_bucket,slope,min,max,bucket_d,counter_d);
//    H_ERR(cudaDeviceSynchronize());
    H_ERR(cudaMemcpy(counter,counter_d,sizeof(index_t)*num_bucket,cudaMemcpyDeviceToHost));
    index_t sum=0;
    index_t KBucket= FindKBucket<data_t,index_t>(counter,num_bucket,k,sum);//the bucket index where the kth element is present
    index_t KCount=counter[KBucket];//number of elements in the bucket consisting of the kth element
 //   data_t Lastmax=0;
 //   data_t Lastmin=0;
    int depth=0;

    while ((KCount >1) && (max-min)>0 && k>0)
    {
        cout<<"Loop:"<<depth<<endl;
        cout<<"KBucket:"<<KBucket<<endl;
        cout<<"KCount:"<<KCount<<endl;
        // for (int i=0;i<num_bucket;i++)
        // {
        //     cout<<counter[i]<<" ";
        // }
        // cout<<endl<<endl<<endl;
        
        max=bucket_ranger[KBucket+1];
        min=bucket_ranger[KBucket];
        slope=((double)(num_bucket-1))/(max-min);
        for (index_t i=0;i<num_bucket;i++)
        {
            bucket_ranger[i]=i/slope+min;
        }
        bucket_ranger[num_bucket]=max;

  //      if ((Lastmax==max)&&(Lastmin==min))
  //      {
  //          std::cout<<"Too few bucket number! Increase the bucket number!"<<std::endl;
  //          exit(-1);
  //      }
  //      Lastmax=max;
  //      Lastmin=min;
        if (KBucket!=0)
        {
            k=k-sum; //+KCount;
        }
        if ((max-min)>0)
        {
//            H_ERR(cudaMemset(bucket_d, -1, num_element * sizeof(int)));
            H_ERR(cudaMemset(counter_d, 0, num_bucket * sizeof(index_t)));
            depth++;
           ta_start=wtime();
             assignBuckets<data_t,index_t><<<128,blocksize,num_bucket*sizeof(index_t)>>>(vec_d,num_element,num_bucket,slope,min,max,counter_d);
            H_ERR(cudaDeviceSynchronize());
             ta=wtime()-ta_start;
        cout<<"Time for finding bucket count & assignement of bucket to the elements:"<<ta*1000<<" ms"<<endl;
//           assignBuckets<data_t,index_t><<<128,blocksize,num_bucket*sizeof(index_t)>>>(vec_d,num_element,num_bucket,slope,min,max,bucket_d,counter_d);
//            H_ERR(cudaDeviceSynchronize());
            H_ERR(cudaMemcpy(counter,counter_d,sizeof(index_t)*num_bucket,cudaMemcpyDeviceToHost));
            sum=0;

            KBucket=FindKBucket<data_t,index_t> (counter,num_bucket,k,sum);
            KCount=counter[KBucket];
        }
        else
        {
            TopKElement=max;
            return;
        }
    }

    data_t* TopKElement_d;
    H_ERR(cudaMalloc((void**) &TopKElement_d,sizeof(data_t)));
    int* check_d;
    //    int check;
    H_ERR(cudaMalloc((void**) &check_d,sizeof(int)));
    H_ERR(cudaMemset(check_d,0,sizeof(int)));
     slope=((double)(num_bucket-1))/(max-min);
     double FinalMin=KBucket/slope+min;
     double FinalMax;
     if (KBucket+1==num_bucket)
         FinalMax=max;
     else
         FinalMax=(KBucket+1)/slope+min;

    GetKValue<data_t,index_t><<<128,128>>>(vec_d,num_element,KBucket,TopKElement_d,check_d,FinalMax,FinalMin);
    H_ERR(cudaDeviceSynchronize());
    H_ERR(cudaMemcpy(&TopKElement,TopKElement_d,sizeof(data_t),cudaMemcpyDeviceToHost));
   std::cout<<"depth"<<depth<<std::endl;
    free(bucket_ranger);
    //free(bucket);
    free(counter);
    //H_ERR(cudaFree(bucket_d));
    H_ERR(cudaFree(counter_d));
    return;

}

    template<typename data_t, typename index_t>
void bucket_select(data_t* vec_d,index_t num_element,index_t k,index_t num_bucket,data_t& TopKElement)
{
    if (num_element <= 2097152)// if less than 2^21 go to phase II-> inplace
    {
        bucket_select_PhaseII<data_t,index_t>(vec_d,num_element,k,num_bucket,TopKElement);
        return;    
    }
    cout<<"In Phase I: Reduction of vector!"<<endl;
    int cut=0;
    int cutoff=2;
         double* bucket_ranger=(double*)malloc((sizeof(double))*(num_bucket+1));
         while(cut<cutoff)
    {
        cout<<"loop:"<<cut<<endl;
        double max=0;
        double min=0;
        Max_Min<data_t,index_t>(max,min,vec_d,num_element);
           //        FindRange<data_t,index_t>(max,min,vec,num_element);
       if (max==min)
        {
            TopKElement=max;
            return;
        }
        double slope=((double)(num_bucket-1))/(max-min);//Number of bucket > 1 otherwise core dumped
       index_t* counter=(index_t*)malloc((sizeof(index_t))*num_bucket);//new index_t[num_bucket];
        index_t* counter_d;
        H_ERR(cudaMalloc((void**) &counter_d,sizeof(index_t)*num_bucket));
        H_ERR(cudaMemset(counter_d, 0, num_bucket * sizeof(index_t)));
        double ta_start=wtime();
        assignBuckets<data_t,index_t><<<128,128,num_bucket*sizeof(index_t)>>>(vec_d,num_element,num_bucket,slope,min,max,counter_d);
        H_ERR(cudaDeviceSynchronize());
        double ta=wtime()-ta_start;
        cout<<"Time for finding bucket count & assignement of bucket to the elements:"<<ta*1000<<" ms"<<endl;

        for (index_t i=0;i<num_bucket;i++) 
        {
            bucket_ranger[i]=i/slope+ min;
        }
        bucket_ranger[num_bucket]=max;

        H_ERR(cudaMemcpy(counter,counter_d,sizeof(index_t)*num_bucket,cudaMemcpyDeviceToHost));
        index_t sum=0;
        index_t KBucket= FindKBucket<data_t,index_t>(counter,num_bucket,k,sum);//the bucket index where the kth element is present
        index_t KCount=counter[KBucket];//number of elements in the bucket consisting of the kth element
        if (KCount==1)//the bucket containing the Top k element has only 1 element --> Top K element
        {
            data_t* TopKElement_d;
            H_ERR(cudaMalloc((void**) &TopKElement_d,sizeof(data_t)));
            int* check_d;
            // int check;
            H_ERR(cudaMalloc((void**) &check_d,sizeof(int)));
            H_ERR(cudaMemset(check_d,0,sizeof(int)));
            GetKValue<data_t,index_t><<<128,128>>>(vec_d,num_element,KBucket,TopKElement_d,check_d,bucket_ranger[KBucket+1],bucket_ranger[KBucket]);
            H_ERR(cudaDeviceSynchronize());
            H_ERR(cudaFree(check_d));
            H_ERR(cudaMemcpy(&TopKElement,TopKElement_d,sizeof(data_t),cudaMemcpyDeviceToHost));
            cout<<"KCount=1 condition in Phase I !"<<endl;
            return;
        }
        else
        {
            if (KBucket!=0)
            {
                k=k-sum;//+KCount; Why KCount is used in the pseudocode?
            }
            data_t* newVec_d;
            H_ERR(cudaMalloc((void**) &newVec_d,sizeof(data_t)*KCount));
            index_t* countfornewvec=(index_t*)malloc(sizeof(index_t));
            index_t* countfornewvec_d;
            H_ERR(cudaMalloc((void**) & countfornewvec_d,sizeof(index_t)));
            H_ERR(cudaMemset(countfornewvec_d, 0, sizeof(index_t)));
            double tc_start=wtime();
            copyElements<data_t,index_t><<<128,128>>> (vec_d,num_element,KBucket,newVec_d,countfornewvec_d,bucket_ranger[KBucket+1],bucket_ranger[KBucket],KCount);
            H_ERR(cudaDeviceSynchronize());
            double tc=wtime()-tc_start;
            cout<<"Time for copying elements"<<tc*1000<<" ms"<<endl;
            num_element=KCount;
            free(countfornewvec);
            H_ERR(cudaFree(countfornewvec_d));
            if ((KCount <= 2097152)||(cut>cutoff))// call PhaseII when the number of element is below 2 million (approx)--> 2^21
            {
                cout<<endl<<endl;
                //free(bucket);
                free(counter);
                //H_ERR(cudaFree(bucket_d));
                H_ERR(cudaFree(counter_d));
                bucket_select_PhaseII<data_t,index_t>(newVec_d,num_element,k,num_bucket,TopKElement);
                H_ERR(cudaFree(newVec_d));
                return;
            }
            swap_ptr_data<data_t,index_t>(vec_d,newVec_d);
 //           swap_ptr_data<data_t,index_t>(vec,newVec);
        }
        cut++;
    }
}

    template<typename data_t,typename index_t>
__global__ void sampleMax (data_t* A,data_t* SubRangeMax, index_t N,index_t NSubranges,index_t SubRangeSize,const index_t alpha,index_t* SubrangeId,int Nthreadstowork)
{	

    int thid = blockDim.x*blockIdx.x+threadIdx.x;
    int laneId= threadIdx.x & 0x1f;
    int myWarpID=thid >> 5;
    int NWarps=(blockDim.x*gridDim.x) >> 5;
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
            SubRangeMax[myWarpID]=Max;
            SubrangeId[myWarpID]=myWarpID;
        }
        myWarpID+=NWarps;
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
              //  Third=Second;
                Second=Max;
                Max=data;
            }
            else if (Second<data)
            {
               // Third=Second;
                Second=data;
            }
        //   else if (Third<data)
        //   {
        //       Third=data;
        //   }
            //Max= (ShMem[INDEX(i)] < Max) ? Max :ShMem[INDEX(i)];
        }
        SubRangeMax[pos]=Max;
        SubRangeMax[pos+1]=Second;
      //  SubRangeMax[pos+2]=Third;

        SubrangeId[pos]=SubRangeID;
        SubrangeId[pos+1]=SubRangeID;
      //  SubrangeId[pos+2]=SubRangeID;

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
        data_t Max=ShMem[INDEX(start_sharedMem)];
        data_t Second=0;
//        data_t Third=0;
        
//       data_t Second=ShMem[INDEX(start_sharedMem+1)];
//       data_t Third=ShMem[INDEX(start_sharedMem+2)];
        for (index_t i=start_sharedMem+1; i<stop_sharedMem;i++)
        {
            // data_t data;
            data_t data=ShMem[INDEX(i)];
           if (Max < data)
            {
  //              Third=Second;
                Second=Max;
                Max=data;
            }
            else if (Second<data)
            {
    //            Third=Second;
                Second=data;
            }
     //      else if (Third<data)
     //      {
     //          Third=data;
     //      }
            //Max= (ShMem[INDEX(i)] < Max) ? Max :ShMem[INDEX(i)];
        }
        SubRangeMax[pos]=Max;
        SubRangeMax[pos+1]=Second;
      //  SubRangeMax[pos+2]=Third;

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

       template<typename data_t, typename index_t>
__global__ void CreateConcatenateRange_Beta_optimized(data_t* vec,data_t* ConcatenatedRange,index_t* SubrangeId,index_t ConcatenatedSize,index_t NContributingSubrange,index_t Subrangesize,index_t* CountLonelyElements_d,index_t* write_pos,data_t firsttopk)
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
__global__ void CreateConcatenateRange_optimized(data_t* vec,data_t* ConcatenatedRange,index_t* SubrangeId,index_t ConcatenatedSize,index_t k,index_t Subrangesize,index_t* write_pos,data_t firsttopk)
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
__global__ void CreateConcatenateRange_Beta(data_t* vec,data_t* ConcatenatedRange,index_t* SubrangeId,index_t ConcatenatedSize,index_t NContributingSubrange,index_t Subrangesize,index_t* CountLonelyElements_d)
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

     template<typename data_t,typename index_t>  
__global__ void FindTopKRanges_beta3(data_t* vec_d,index_t num_element,data_t flag,index_t* SubrangeId_d,index_t* SelectedSubrangeId_d,index_t* CountSelectedSubrange_d,data_t* ConcatenatedRange,index_t* CountLonelyElements,int beta)

{
    //Note: This kernel is desined for Beta=3, value,valu1,valu2 represents Max, Second Max and third max. Total of 3 Beta elements.
    index_t mybegin=(blockIdx.x*blockDim.x+threadIdx.x)*beta;//*3 for Beta
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
}
  template<typename data_t,typename index_t>  
__global__ void FindTopKRanges_beta(data_t* vec_d,index_t num_element,data_t flag,index_t* SubrangeId_d,index_t* SelectedSubrangeId_d,index_t* CountSelectedSubrange_d,data_t* ConcatenatedRange,index_t* CountLonelyElements,int beta)

{
    //Note: This kernel is desined for Beta=3, value,valu1,valu2 represents Max, Second Max and third max. Total of 3 Beta elements.
    index_t mybegin=(blockIdx.x*blockDim.x+threadIdx.x)*beta;//*3 for Beta
   while(mybegin < num_element)
    {
       int count=0;
//printf("beta%d\n",beta);
        data_t value=vec_d[mybegin];
        if (mybegin>=num_element)
        {
            printf("Eillegal psition %d\n",mybegin);
        }
        data_t value1,value2;
        if (value >= flag)       
        {
            count++;


            value1=vec_d[mybegin+1];
//            value2=vec_d[mybegin+2];
         if (mybegin+1>=num_element)
        {
            printf("Eillegal psition %d\n",mybegin+1);
        }
            
            count +=  
            (value1 >= flag) ;
            //+
  //          (value2 >= flag);

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
//                ConcatenatedRange[curr_ptr + 2]=value2;
           }
        }

        mybegin+=blockDim.x*gridDim.x*beta;//*3 for beta
    }
}


     template<typename data_t, typename index_t>
void bucket_select_inplace_betaFirstTopk(data_t* vec_d,index_t num_element,index_t k,index_t num_bucket,data_t& TopKElement,index_t* SubrangeId_d,index_t* SelectedSubrangeId_d,index_t* CountSelectedSubrange_d,data_t* ConcatenatedRange_d,index_t* CountLonelyElements_d,int beta)
{
    cout<<"In Phase II: Inplace Bucket select"<<endl;
    double max=0;
    double min=0;
    double inplace_start=wtime();
    Max_Min<data_t,index_t>(max,min,vec_d,num_element);
   if (max==min)
    {
        TopKElement=max;
      FindTopKRanges_beta<data_t,index_t><<<128,128>>> (vec_d,num_element,TopKElement,SubrangeId_d,SelectedSubrangeId_d,CountSelectedSubrange_d,ConcatenatedRange_d,CountLonelyElements_d,beta);
    H_ERR(cudaDeviceSynchronize());
       return;
    }
    double slope=((double)(num_bucket-1))/(max-min);//Number of bucket > 1 otherwise core dumped
    double* bucket_ranger=(double*)malloc((sizeof(double))*(num_bucket+1));//new double[num_bucket+1];

    for (index_t i=0;i<num_bucket;i++)
    {
        bucket_ranger[i]=i/slope+ min;
    }
    bucket_ranger[num_bucket]=max;

   index_t* counter=(index_t*)malloc((sizeof(index_t))*num_bucket);//new index_t[num_bucket];
    index_t* counter_d;
    H_ERR(cudaMalloc((void**) &counter_d,sizeof(index_t)*num_bucket));
    H_ERR(cudaMemset(counter_d, 0, num_bucket * sizeof(index_t)));
         double ta_start=wtime();
        assignBuckets<data_t,index_t><<<128,blocksize,num_bucket*sizeof(index_t)>>>(vec_d,num_element,num_bucket,slope,min,max,counter_d);
        H_ERR(cudaDeviceSynchronize());
        double ta=wtime()-ta_start;
        cout<<"Time for finding Bucket Count & assignement of bucket to the elements:"<<ta*1000<<" ms"<<endl;
    H_ERR(cudaMemcpy(counter,counter_d,sizeof(index_t)*num_bucket,cudaMemcpyDeviceToHost));
    index_t sum=0;
    index_t KBucket= FindKBucket<data_t,index_t>(counter,num_bucket,k,sum);//the bucket index where the kth element is present
    index_t KCount=counter[KBucket];//number of elements in the bucket consisting of the kth element
    int depth=0;

    // while ((KCount >1) && (max-min)>0 && k>0)
    {
        cout<<"Loop:"<<depth<<endl;
        cout<<"KBucket:"<<KBucket<<endl;
        cout<<"KCount:"<<KCount<<endl;
        for (int i=0;i<num_bucket;i++)
        {
            cout<<counter[i]<<" ";
        }
        cout<<endl<<endl<<endl;
        
        max=bucket_ranger[KBucket+1];
        min=bucket_ranger[KBucket];
        slope=((double)(num_bucket-1))/(max-min);
        for (index_t i=0;i<num_bucket;i++)
        {
            bucket_ranger[i]=i/slope+min;
        }
        bucket_ranger[num_bucket]=max;

       if (KBucket!=0)
        {
            k=k-sum; //+KCount;
        }
        if ((max-min)>0)
        {

//            H_ERR(cudaMemset(bucket_d, -1, num_element * sizeof(int)));
            H_ERR(cudaMemset(counter_d, 0, num_bucket * sizeof(index_t)));
            depth++;
           ta_start=wtime();
             assignBuckets<data_t,index_t><<<128,blocksize,num_bucket*sizeof(index_t)>>>(vec_d,num_element,num_bucket,slope,min,max,counter_d);
            H_ERR(cudaDeviceSynchronize());
             ta=wtime()-ta_start;
        cout<<"Time for finding bucket count & assignement of bucket to the elements:"<<ta*1000<<" ms"<<endl;
           H_ERR(cudaMemcpy(counter,counter_d,sizeof(index_t)*num_bucket,cudaMemcpyDeviceToHost));
            sum=0;

            KBucket=FindKBucket<data_t,index_t> (counter,num_bucket,k,sum);
            KCount=counter[KBucket];
        }
        else
        {
            TopKElement=max;
     FindTopKRanges_beta<data_t,index_t><<<128,128>>> (vec_d,num_element,TopKElement,SubrangeId_d,SelectedSubrangeId_d,CountSelectedSubrange_d,ConcatenatedRange_d,CountLonelyElements_d,beta);
    H_ERR(cudaDeviceSynchronize());
            return;
        }
    }

    data_t* TopKElement_d;
    H_ERR(cudaMalloc((void**) &TopKElement_d,sizeof(data_t)));
    int* check_d;
    //    int check;
    H_ERR(cudaMalloc((void**) &check_d,sizeof(int)));
    H_ERR(cudaMemset(check_d,0,sizeof(int)));
     slope=((double)(num_bucket-1))/(max-min);
     double FinalMin=KBucket/slope+min;
     double FinalMax;
     if (KBucket+1==num_bucket)
         FinalMax=max;
     else
         FinalMax=(KBucket+1)/slope+min;

    GetKValue<data_t,index_t><<<128,128>>>(vec_d,num_element,KBucket,TopKElement_d,check_d,FinalMax,FinalMin);
    H_ERR(cudaDeviceSynchronize());
    H_ERR(cudaMemcpy(&TopKElement,TopKElement_d,sizeof(data_t),cudaMemcpyDeviceToHost));
  printf("topkelement%d\n",TopKElement);
    FindTopKRanges_beta<data_t,index_t><<<128,128>>> (vec_d,num_element,TopKElement,SubrangeId_d,SelectedSubrangeId_d,CountSelectedSubrange_d,ConcatenatedRange_d,CountLonelyElements_d,beta);
    H_ERR(cudaDeviceSynchronize());
    //  int count;
    //  H_ERR(cudaMemcpy(&count,CountSelectedSubrange_d,sizeof(int),cudaMemcpyDeviceToHost));
    //  cout<<"Number of subranges in first sweep:"<<count<<endl;
    //  FillWithKElementsID<data_t,index_t><<<128,128>>> (vec_d,num_element,flag,shleft,SubrangeId_d,SelectedSubrangeId_d,CountSelectedSubrange_d,k);
    //  H_ERR(cudaDeviceSynchronize());
    cout<<"Total time of radix_select Inplace_firstTopk(Phase II):"<<(wtime()-inplace_start)*1000<<" ms"<<endl;
 std::cout<<"depth"<<depth<<std::endl;
    free(bucket_ranger);
    //free(bucket);
    free(counter);
    //H_ERR(cudaFree(bucket_d));
    H_ERR(cudaFree(counter_d));
    return;

}  
    
    
    template<typename data_t,typename index_t>  
__global__ void FindTopKRanges(data_t* vec_d,index_t num_element,data_t flag,index_t* SubrangeId_d,index_t* SelectedSubrangeId_d,index_t* CountSelectedSubrange_d)
{
    index_t mybegin=blockIdx.x*blockDim.x+threadIdx.x;
    while(mybegin < num_element)
    {
        data_t value=vec_d[mybegin];
        if (value > flag)//It makes sure all the subranges of the sampled elements which are greater than the Kth element are included in the subrange array
        {
            SelectedSubrangeId_d[atomicAdd(CountSelectedSubrange_d,1)]=SubrangeId_d[mybegin];
        }
        mybegin+=blockDim.x*gridDim.x;
    }
}

    template<typename data_t,typename index_t>  
__global__ void FillWithKElementsID(data_t* vec_d,index_t num_element,data_t flag,index_t* SubrangeId_d,index_t* SelectedSubrangeId_d,index_t* CountSelectedSubrange_d,index_t k)

{
    index_t mybegin=blockIdx.x*blockDim.x+threadIdx.x;
    while(mybegin < num_element)
    {
        data_t value=vec_d[mybegin];
        if (value == flag)//fill out the remaining part of the array by the subranges ID of any of the kth elements
        {
            if (ThreadLoad_ind<data_t,index_t>(CountSelectedSubrange_d)==k) return;
            SelectedSubrangeId_d[atomicAdd(CountSelectedSubrange_d,(ThreadLoad_ind<data_t,index_t>(CountSelectedSubrange_d)<k))]=SubrangeId_d[mybegin];
          
        }
        mybegin+=blockDim.x*gridDim.x;
    }
}

    template<typename data_t, typename index_t>
void bucket_select_inplaceFirstTopk(data_t* vec_d,index_t num_element,index_t k,index_t num_bucket,data_t& TopKElement,index_t* SubrangeId_d,index_t* SelectedSubrangeId_d,index_t* CountSelectedSubrange_d)
{
    cout<<"In Phase II: Inplace Bucket select"<<endl;
    double max=0;
    double min=0;
    double inplace_start=wtime();
    Max_Min<data_t,index_t>(max,min,vec_d,num_element);
   if (max==min)
    {
        TopKElement=max;
        cout<<"Max = min in the first topK inplace module"<<endl;
      FindTopKRanges<data_t,index_t><<<128,128>>> (vec_d,num_element,TopKElement,SubrangeId_d,SelectedSubrangeId_d,CountSelectedSubrange_d);
    H_ERR(cudaDeviceSynchronize());
    int count;
 FillWithKElementsID<data_t,index_t><<<128,128>>> (vec_d,num_element,TopKElement,SubrangeId_d,SelectedSubrangeId_d,CountSelectedSubrange_d,k);
    H_ERR(cudaDeviceSynchronize());
       return;
    }
    double slope=((double)(num_bucket-1))/(max-min);//Number of bucket > 1 otherwise core dumped
    double* bucket_ranger=(double*)malloc((sizeof(double))*(num_bucket+1));//new double[num_bucket+1];

    for (index_t i=0;i<num_bucket;i++)
    {
        bucket_ranger[i]=i/slope+ min;
    }
    bucket_ranger[num_bucket]=max;

   index_t* counter=(index_t*)malloc((sizeof(index_t))*num_bucket);//new index_t[num_bucket];
    index_t* counter_d;
    H_ERR(cudaMalloc((void**) &counter_d,sizeof(index_t)*num_bucket));
    H_ERR(cudaMemset(counter_d, 0, num_bucket * sizeof(index_t)));
         double ta_start=wtime();
        assignBuckets<data_t,index_t><<<128,blocksize,num_bucket*sizeof(index_t)>>>(vec_d,num_element,num_bucket,slope,min,max,counter_d);
        H_ERR(cudaDeviceSynchronize());
        double ta=wtime()-ta_start;
        cout<<"Time for finding Bucket Count & assignement of bucket to the elements:"<<ta*1000<<" ms"<<endl;
//    assignBuckets<data_t,index_t><<<128,128,num_bucket*sizeof(index_t)>>>(vec_d,num_element,num_bucket,slope,min,max,bucket_d,counter_d);
//    H_ERR(cudaDeviceSynchronize());
    H_ERR(cudaMemcpy(counter,counter_d,sizeof(index_t)*num_bucket,cudaMemcpyDeviceToHost));
    index_t sum=0;
    index_t KBucket= FindKBucket<data_t,index_t>(counter,num_bucket,k,sum);//the bucket index where the kth element is present
    index_t KCount=counter[KBucket];//number of elements in the bucket consisting of the kth element
 //   data_t Lastmax=0;
 //   data_t Lastmin=0;
    int depth=0;

    while ((KCount >1) && (max-min)>0 && k>0)
    {
        cout<<"Loop:"<<depth<<endl;
        cout<<"KBucket:"<<KBucket<<endl;
        cout<<"KCount:"<<KCount<<endl;
        for (int i=0;i<num_bucket;i++)
        {
            cout<<counter[i]<<" ";
        }
        cout<<endl<<endl<<endl;
        
        max=bucket_ranger[KBucket+1];
        min=bucket_ranger[KBucket];
        slope=((double)(num_bucket-1))/(max-min);
        for (index_t i=0;i<num_bucket;i++)
        {
            bucket_ranger[i]=i/slope+min;
        }
        bucket_ranger[num_bucket]=max;

  //      if ((Lastmax==max)&&(Lastmin==min))
  //      {
  //          std::cout<<"Too few bucket number! Increase the bucket number!"<<std::endl;
  //          exit(-1);
  //      }
  //      Lastmax=max;
  //      Lastmin=min;
        if (KBucket!=0)
        {
            k=k-sum; //+KCount;
        }
        if ((max-min)>0)
        {

//            H_ERR(cudaMemset(bucket_d, -1, num_element * sizeof(int)));
            H_ERR(cudaMemset(counter_d, 0, num_bucket * sizeof(index_t)));
            depth++;
           ta_start=wtime();
             assignBuckets<data_t,index_t><<<128,blocksize,num_bucket*sizeof(index_t)>>>(vec_d,num_element,num_bucket,slope,min,max,counter_d);
            H_ERR(cudaDeviceSynchronize());
             ta=wtime()-ta_start;
        cout<<"Time for finding bucket count & assignement of bucket to the elements:"<<ta*1000<<" ms"<<endl;
//           assignBuckets<data_t,index_t><<<128,blocksize,num_bucket*sizeof(index_t)>>>(vec_d,num_element,num_bucket,slope,min,max,bucket_d,counter_d);
//            H_ERR(cudaDeviceSynchronize());
            H_ERR(cudaMemcpy(counter,counter_d,sizeof(index_t)*num_bucket,cudaMemcpyDeviceToHost));
            sum=0;

            KBucket=FindKBucket<data_t,index_t> (counter,num_bucket,k,sum);
            KCount=counter[KBucket];
        }
        else
        {
            TopKElement=max;
     FindTopKRanges<data_t,index_t><<<128,128>>> (vec_d,num_element,TopKElement,SubrangeId_d,SelectedSubrangeId_d,CountSelectedSubrange_d);
    H_ERR(cudaDeviceSynchronize());
    int count;
 FillWithKElementsID<data_t,index_t><<<128,128>>> (vec_d,num_element,TopKElement,SubrangeId_d,SelectedSubrangeId_d,CountSelectedSubrange_d,k);
    H_ERR(cudaDeviceSynchronize());
            return;
        }
    }

    data_t* TopKElement_d;
    H_ERR(cudaMalloc((void**) &TopKElement_d,sizeof(data_t)));
    int* check_d;
    //    int check;
    H_ERR(cudaMalloc((void**) &check_d,sizeof(int)));
    H_ERR(cudaMemset(check_d,0,sizeof(int)));
     slope=((double)(num_bucket-1))/(max-min);
     double FinalMin=KBucket/slope+min;
     double FinalMax;
     if (KBucket+1==num_bucket)
         FinalMax=max;
     else
         FinalMax=(KBucket+1)/slope+min;

    GetKValue<data_t,index_t><<<128,128>>>(vec_d,num_element,KBucket,TopKElement_d,check_d,FinalMax,FinalMin);
    H_ERR(cudaDeviceSynchronize());
    H_ERR(cudaMemcpy(&TopKElement,TopKElement_d,sizeof(data_t),cudaMemcpyDeviceToHost));
   std::cout<<"depth"<<depth<<std::endl;
  //Getting the subrange info
     FindTopKRanges<data_t,index_t><<<128,128>>> (vec_d,num_element,TopKElement,SubrangeId_d,SelectedSubrangeId_d,CountSelectedSubrange_d);
    H_ERR(cudaDeviceSynchronize());
    int count;
    H_ERR(cudaMemcpy(&count,CountSelectedSubrange_d,sizeof(int),cudaMemcpyDeviceToHost));
    cout<<"Number of subranges in first sweep:"<<count<<endl;
    FillWithKElementsID<data_t,index_t><<<128,128>>> (vec_d,num_element,TopKElement,SubrangeId_d,SelectedSubrangeId_d,CountSelectedSubrange_d,k);
    H_ERR(cudaDeviceSynchronize());
    cout<<"Total time of radix_select Inplace_firstTopk(Phase II):"<<(wtime()-inplace_start)*1000<<" ms"<<endl;

   //~Getting the subrange info 
    free(bucket_ranger);
    //free(bucket);
    free(counter);
    //H_ERR(cudaFree(bucket_d));
    H_ERR(cudaFree(counter_d));
    return;

}

 
template<typename data_t, typename index_t>
void sample_bucket_select(data_t* vec_d,index_t num_element,index_t k,index_t num_bucket,data_t& TopKElement,index_t NSubranges,index_t SubRangesize,index_t alpha,double& timeforMaxsample,double& timeforFirstTopK,double& timeForsecondTopK,double& timeforConcatenation,data_t* Max_d,index_t* SubrangeId_d,int beta,bool defaultContribution,int NthreadstoworkInreduction,int NThreadsPerBlock,int SizeOfAllocation,int NSharedMemoryElements,index_t* SelectedSubrangeId_d,index_t* CountSelectedSubrange_d,index_t* CountLonelyElements_d,index_t* write_pos_d,data_t* ConcatenatedRange_d)
{
    double sample_beg=wtime();
   if (defaultContribution)
    {
         sampleMax<data_t,index_t><<<4096,512>>>(vec_d,Max_d,num_element,NSubranges,SubRangesize,alpha,SubrangeId_d,NthreadstoworkInreduction);
     // sampleMax<data_t,index_t><<<4096,512>>>(vec_d,Max_d,num_element,NSubranges,SubRangesize,alpha,SubrangeId_d,Nthreadstowork,SizeOfSubWarp,pow_size_Subwarp);
    }
   else
   {
        sampleMax_NoReduction_Beta <data_t,index_t><<<4096,NThreadsPerBlock,SizeOfAllocation*(sizeof(data_t))>>>(vec_d,Max_d,num_element,NSubranges,SubRangesize,alpha,SubrangeId_d,NSharedMemoryElements,beta);

   }
   H_ERR(cudaDeviceSynchronize());
    //For testing First TopK result
    //The following allocations alone take 0.5 ms
//   index_t* SelectedSubrangeId_d;
//   H_ERR(cudaMalloc((void**) &SelectedSubrangeId_d,sizeof(index_t)*(NSubranges-k)*beta));//updated *3 for beta
//   index_t* CountSelectedSubrange_d;
//   index_t* CountLonelyElements_d;
//   H_ERR(cudaMalloc((void**) &CountSelectedSubrange_d,sizeof(index_t)));
//   H_ERR(cudaMalloc((void**) &CountLonelyElements_d,sizeof(index_t)));
//   H_ERR(cudaMemset(CountSelectedSubrange_d, 0,  sizeof(index_t)));
//   H_ERR(cudaMemset(CountLonelyElements_d, 0,  sizeof(index_t)));
//     index_t* write_pos_d;
 //       H_ERR(cudaMalloc((void**) &ConcatenatedRange_d,sizeof(data_t)*k*SubRangesize));
  //      H_ERR(cudaMalloc((void**) &write_pos_d,sizeof(index_t)));
  //data_t* ConcatenatedRange_d;
  //  H_ERR(cudaMalloc((void**) &ConcatenatedRange_d,sizeof(data_t)*k*SubRangesize));
   
 timeforMaxsample=wtime()-sample_beg;
    double startFirstTopK=wtime();
    data_t flag=0;
   if (defaultContribution)
    {
         bucket_select_inplaceFirstTopk<data_t,index_t>(Max_d,NSubranges,/*k*/NSubranges-k,num_bucket,TopKElement,SubrangeId_d,SelectedSubrangeId_d,CountSelectedSubrange_d);

    }
    else
    {
        bucket_select_inplace_betaFirstTopk<data_t,index_t>(Max_d,NSubranges*beta,/*k*/NSubranges*beta-k,num_bucket,TopKElement,SubrangeId_d,SelectedSubrangeId_d,CountSelectedSubrange_d,ConcatenatedRange_d,CountLonelyElements_d,beta);
    }    // 
    index_t count,count_lonely;
    H_ERR(cudaMemcpy(&count,CountSelectedSubrange_d,sizeof(index_t),cudaMemcpyDeviceToHost));
    H_ERR(cudaMemcpy(&count_lonely,CountLonelyElements_d,sizeof(index_t),cudaMemcpyDeviceToHost));
   timeforFirstTopK=wtime()-startFirstTopK;
    //    double t_start_concatenate=wtime();
    timeforConcatenation=wtime();
    //Create Concatenate range
    //    index_t ConcatenatedSize=k*SubRangesize;//Maximum possible concatenated size
    index_t ConcatenatedSize;
    if (defaultContribution)
    {
        ConcatenatedSize=k*SubRangesize;
       H_ERR(cudaMemset(write_pos_d, 0, sizeof(index_t)));  CreateConcatenateRange_optimized<data_t,index_t><<<512,512>>>(vec_d,ConcatenatedRange_d,SelectedSubrangeId_d,ConcatenatedSize,k,SubRangesize,write_pos_d,TopKElement);
//   CreateConcatenateRange<data_t,index_t><<<512,512>>>(vec_d,ConcatenatedRange_d,SelectedSubrangeId_d,ConcatenatedSize,k,SubRangesize);
//  CreateConcatenateRange<data_t,index_t><<<512,512>>>(vec_d,ConcatenatedRange_d,SelectedSubrangeId_d,ConcatenatedSize,count,SubRangesize,CountLonelyElements_d);
    }
    else
    {
        ConcatenatedSize=count*SubRangesize+count_lonely;
       H_ERR(cudaMemcpy(write_pos_d,&count_lonely,sizeof(index_t),cudaMemcpyHostToDevice));
//       CreateConcatenateRange_Beta<data_t,index_t><<<512,512>>>(vec_d,ConcatenatedRange_d,SelectedSubrangeId_d,ConcatenatedSize,count,SubRangesize,CountLonelyElements_d);
      CreateConcatenateRange_Beta_optimized<data_t,index_t><<<512,512>>>(vec_d,ConcatenatedRange_d,SelectedSubrangeId_d,ConcatenatedSize,count,SubRangesize,CountLonelyElements_d,write_pos_d,TopKElement);
 //   CreateConcatenateRange_Beta<data_t,index_t><<<512,512>>>(vec_d,ConcatenatedRange_d,SelectedSubrangeId_d,ConcatenatedSize,count,SubRangesize,CountLonelyElements_d);
    }
    H_ERR(cudaDeviceSynchronize());
         H_ERR(cudaMemcpy(&ConcatenatedSize,write_pos_d,sizeof(index_t),cudaMemcpyDeviceToHost));
      cout<<"New Concatenated Size: "<<ConcatenatedSize<<endl;
 timeforConcatenation=wtime()-timeforConcatenation;
    cout<<"Time for concatenation:"<<timeforConcatenation*1000<<" ms"<<endl;
    cout<<"The concatenation of the contributing subranges is done!"<<endl; 
    double startSecondTopK=wtime();
    //flag=0;
    //CurrentDigit=(sizeof(data_t)*8/NBitsperDigit)-1;    
    bucket_select<data_t,index_t>(ConcatenatedRange_d,ConcatenatedSize,/*k*/ConcatenatedSize-k,num_bucket,TopKElement);
    timeForsecondTopK=wtime()-startSecondTopK;
    //           radix_select_inplace<data_t,index_t>(ConcatenatedRange_d,ConcatenatedSize,k,num_bucket,TopKElement,NBits,CurrentDigit);   
}
