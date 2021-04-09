#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <ctime>
#include <assert.h>
#include <math.h>
#include "bitonic_com.cuh"
#include "wtime.h"
#include <fstream>
#include <random>
using namespace std;

typedef unsigned int data_t;

//typedef unsigned int data_t;
typedef int index_t;
int compare(const void *p1, const void *p2)
{
    const struct max_withIndex *elem1 = (const struct max_withIndex *)p1;
    const struct max_withIndex *elem2 = (const struct max_withIndex *)p2;
    if (elem1->value > elem2->value)//For descending order
    {
        return -1;
    }
    else if (elem1->value < elem2->value)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

    template<typename data_t,typename index_t>
index_t power(index_t x,index_t n)
{
    index_t number=1;
    for (index_t i=0; i<n ;i++)
    {
        number*=x;
    }
    return number;
}

    template<typename data_t, typename index_t>
__global__ void alpha_range_max_sample(
        data_t* vec, 
        data_t* sampled_top, 
        index_t num_element, 
        index_t num_subrange, 
        index_t alpha,
        index_t* SubrangeId)
{
    index_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    index_t warp_id = tid >> 5;
    const index_t lane_id = threadIdx.x & 31;
    const index_t warp_count = (blockDim.x * gridDim.x) >> 5;
    //    const index_t subrange_size = 1<<alpha;

    //schedule one warp to work on one subrange
    while(warp_id < num_subrange - 1)
    {
        index_t my_beg = (warp_id << alpha) + lane_id;
        index_t my_end = (warp_id+1) << alpha;
        assert(my_end < num_element);

        data_t my_max = vec[my_beg];

        while(my_beg < my_end)
        {
            my_max = (my_max < vec[my_beg] ? vec[my_beg]:my_max);
            my_beg += 32;
        }

        //max across the warp
        for (int i=16; i>0; i>>=1)
        {
            data_t new_max =  __shfl_down_sync(0xffffffff,my_max, i);
            my_max = (my_max < new_max ? new_max:my_max);
        }

        if(!lane_id)
        {   
            sampled_top[warp_id] = my_max;
            SubrangeId[warp_id]=my_max;
            //            printf("%d\n",my_max);
        }

        warp_id += warp_count;
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
bool IsPowerof2(index_t x)
{
    return (x != 0) && ((x & (x - 1)) == 0);
}

int main(int argc,char**argv)
{
    cout<<"./exe exp_num_element k alpha 1=NormalBitonic/0=DrTopKBitonic"<<endl;
    if (argc != 5) {cout<<"wrong input"<<endl;exit(-1);}
    index_t p=atol(argv[1]);
    index_t k= atol(argv[2]);
    if (!IsPowerof2<data_t,index_t>(k))
    {
        cout<<"k should be power of 2!"<<endl;
        exit(-1);
    }
    index_t alpha= atol(argv[3]);
    index_t NormalBitonic= atol(argv[4]);
    cout<<"NormalBitonic: "<<NormalBitonic<<endl;

    index_t base=2;
    index_t N=power<data_t,index_t>(base,p);
    cudaSetDevice(0);
    data_t* A=(data_t*)malloc((sizeof(data_t)*N));
    data_t *A_d;
    H_ERR(cudaMalloc((void**) &A_d,sizeof(data_t)*N));
    int count=0;
    cout<<"subrange:"<<count<<endl;
    index_t SubRangesize=pow(2,alpha);
    index_t NSubranges=N/SubRangesize;
    if (NSubranges<k)
    {
        cout<<"Small number of subranges!. Decrease the value of alpha!"<<endl;
        // exit(-1);
    }
    std::fstream statusLog;
    //     statusLog.open("testSMemDiffK_alpha_N.csv",std::fstream::out | std::fstream::app);
    //     statusLog<<endl<<endl<<"Started Top K with N_"<<N<<"k_"<<k<<"alpha_"<<alpha<<endl;
    int c=0;
    data_t* Subrange=(data_t*)malloc((sizeof(data_t)*NSubranges));
    max_withIndex* max_Obj=new max_withIndex[NSubranges];
    float a=1500.0;
//   for (index_t i=0;i<N;i++)
//   {
///        A[i]=rand()%(2147483648);//2^31 -1 
//       A[i]=(float)rand()/(float)(RAND_MAX)*a;//2^31 -1 
//       if (A[i]==2147481384) c++;
//   }
    std::random_device rd;
    std::mt19937 gen(rd());

    float value; // For Normal distribution
    float minvalue=100000000;

//   int value;
//   int minvalue=100000000;



 //   int dis=atoi(argv[4]);
            std::uniform_int_distribution <unsigned int> d(0, 4294967295);
//   std::normal_distribution<float> d(100000000, 10);//Mean =100 mill , sd=100
  for (index_t i=0;i<N;i++)
    {
 //     value=d(gen);
 //      if (minvalue > value)
 //      {
 //          minvalue=value;
 //      }
 //      if (value > 4294967295)
 //      {
 //          cout<<"Overflow of unsigned int detected"<<endl;
 //          return -1;
 //      }
        A[i]=d(gen);
       // vec1[i]=vec[i];
    }
    cout<<endl;
    if (minvalue < 0)
    {
        cout<<"-ve value detected:"<<minvalue<<endl;
        return -1;
    }

    cout<<endl;
    cout<<"SubRangeSize:"<<SubRangesize<<endl;
    cout<<"Number of Subranges:"<<NSubranges<<endl;
    data_t* Max_d;
    H_ERR(cudaMalloc((void**) &Max_d,sizeof(data_t)*NSubranges));
    H_ERR(cudaMemcpy(A_d,A,sizeof(data_t)*N,cudaMemcpyHostToDevice));
    index_t* SubrangeId_d;
    H_ERR(cudaMalloc((void**) &SubrangeId_d,sizeof(index_t)*NSubranges));
    data_t TopKElement;
    data_t* ConcatenatedRange_d;

    H_ERR(cudaMalloc((void**) &ConcatenatedRange_d,sizeof(data_t)*k*SubRangesize));
    // H_ERR(cudaMalloc((void**) &ConcatenatedRange_d,sizeof(data_t)*N));
    double time_beg=wtime();
    double t2=0;double t3=0;double t4=0;double totalTime=0;double timeforFirstTopk=0;double timeforMaxsample=0;double timeforSecondTopk=0;double timeforNormalBitonicsort=0;
    if ((NSubranges>k) && (NormalBitonic==0))
    {
        int Nthreadstowork=32;
        if (SubRangesize<32)
        {
            Nthreadstowork=SubRangesize;
        }
        sampleMax<data_t,index_t><<<128,128>>>(A_d,Max_d,N,NSubranges,SubRangesize,alpha,SubrangeId_d,Nthreadstowork);
        H_ERR(cudaDeviceSynchronize());
        t2=wtime();
        timeforMaxsample=t2-time_beg;
        //     cout<<"timeforMaxsample:"<<timeforMaxsample<<endl;
        //     cout<<"Max for every subranges"<<endl;
        cout<<"Starting first bitonic!"<<endl;
        bitonic_firstTopk<data_t,index_t>(Max_d,NSubranges,k,SubRangesize,NSubranges,SubrangeId_d,A_d,ConcatenatedRange_d);
        t3=wtime();
        cout<<"Finished first topk and concatenation"<<endl;
        timeforFirstTopk=t3-t2;
        bitonic<data_t,index_t>(ConcatenatedRange_d,k*SubRangesize,k,TopKElement,SubRangesize,A,N,SubrangeId_d,A_d,ConcatenatedRange_d);
    }
    else
    {
        double NormalBitonicstart=wtime();
        bitonic<data_t,index_t>(A_d,N,k,TopKElement,SubRangesize,A,N,SubrangeId_d,A_d,ConcatenatedRange_d);
        timeforNormalBitonicsort=wtime()-NormalBitonicstart;
    }
    cout <<endl;
    t4=wtime();

    timeforSecondTopk=t4-t3;

    cudaFree(A_d);
    cudaFree(Max_d);
    free(A);
    totalTime=t4-time_beg;
    cout<<"timeforMaxsample:"<<timeforMaxsample*1000<<" ms"<<endl;
    cout<<"timeforFirstTopk:"<<timeforFirstTopk*1000<<" ms"<<endl;
    cout<<"timeforSecondTopk:"<<timeforSecondTopk*1000<<" ms"<<endl;
    cout<<"totalTime:"<<totalTime*1000<<" ms"<<endl;
    cout<<"timeforNormalBitonicsort"<<timeforNormalBitonicsort*1000<<" ms"<<endl;
    std::fstream timeLog;
   // timeLog.open("timeBitonicsampleOCT11.csv",std::fstream::out | std::fstream::app);
    // timeLog.open("Uniform2^30_Bitonic_DrTopkBitonic.csv",std::fstream::out | std::fstream::app);
    timeLog.open("testV100.csv",std::fstream::out | std::fstream::app);
    // timeLog.open("Normal2^30_Bitonic_DrTopkBitonic.csv",std::fstream::out | std::fstream::app);
    timeLog<<p<<";"<<k<<";"<<alpha<<";"<<timeforMaxsample*1000<<";"<<timeforFirstTopk*1000<<";"<<timeforSecondTopk*1000<<";"<<timeforNormalBitonicsort*1000<<";"<<totalTime*1000<<endl;
    //    timeLog<<"N_"<<N<<"k_"<<k<<"alpha_"<<alpha<<";"<<timeforNormalBitonicsort*1000<<endl;
    timeLog.close();

    //    statusLog<<"Successfully Finished Top K with N_"<<N<<"k_"<<k<<"alpha_"<<alpha<<endl;
    //    statusLog.close();

    return 0;
}
