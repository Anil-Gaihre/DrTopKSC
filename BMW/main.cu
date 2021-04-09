//GPU 0 is the default GPU. Only max sampling is done in multiple GPU in this version
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <algorithm>
#include "radixselect.cuh"
//#include "radixselectNormalInplaceWorking.cuh"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <random>
#include <cstdio>
// #include <mpi.h>
// #define Enabletest 1

using namespace std;
#define enable_overheads 1
typedef unsigned int data_t;
// typedef int index_t;
typedef int64_t index_t;


int compare (const void * a, const void * b)
{
    return ( *(int*)a - *(int*)b  );//in ascending order
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


void SynchronizeAllDevices(cudaStream_t* stream,int N_gpu)
{
    for (int i = 0; i < N_gpu; i++)
    {
        //ensures every GPU finishes their work
        H_ERR(cudaSetDevice(i));
        H_ERR(cudaStreamSynchronize(stream[i]));
        //	H_ERR(cudaDeviceSynchronize());
    }
}

void getminmax(data_t* arr,index_t n,data_t& max,data_t& min)
{

    for (index_t i=1;i<n;i++)
    {
        if (arr[i]> max)
        {
            max=arr[i];
        }
        if (arr[i]<min)
        {
            min=arr[i];
        }
    }
    return;
}



    template<typename data_t,typename index_t>
bool IsPowerof2(index_t x)
{
    return (x != 0) && ((x & (x - 1)) == 0);
}

int main(int argc,char** argv)
{
    char message[20];
    int myrank, tag=99;
    int size=0;
    myrank=0;
    // MPI_Status status;
    // MPI_Init(&argc, &argv);
    // /* Determine unique id of the calling process of all processes participating
    //  *        in this MPI program. This id is usually called MPI rank. */
    // MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    printf("setting up GPU: %d\n",myrank);
    H_ERR(cudaSetDevice(myrank));
    // MPI_Comm_size( MPI_COMM_WORLD, &size);


    cout<<"./exe num_element k NBitsPerDigit beta NGPU"<<endl;
    cout<<"Size of unsigned int"<<sizeof(unsigned int)<<endl;
    if (argc != 6) {cout<<"wrong input"<<endl;exit(-1);}
    index_t num_pow = atol(argv[1]);
    index_t base=2;
    index_t num_element = power<data_t,index_t>(base,num_pow);
    index_t total_num_element = num_element;

    cout<<"num_element: "<<num_element<<endl;
    index_t k= atol(argv[2]);
    index_t NBits=atol(argv[3]);//atol(argv[3]);
    // int NBitsperDigit = 
    int sd[]={10,100000,1000000,100,100000000};
    int beta=atoi(argv[4]);//SampleBeta function is designed for Beta=3 only. So we need to update the SampleBetafunction in radix select if we want to change Beta value
    int NGPU=atoi(argv[5]);

    int num_process = NGPU;
    index_t new_num_pow = num_pow - log(NGPU)/log(2);
    index_t final_num_pow = new_num_pow;
    int NIterationPerGPU=1;
    index_t numerator;
    if (new_num_pow > 30) 
    {
        //numerator = power<data_t,index_t>(base,new_num_pow-30);
        final_num_pow = 30;
        //final_num_pow = new_num_pow - log(numerator)/log(2);
        NIterationPerGPU =  power<data_t,index_t>(base, new_num_pow -30);
        //NIterationPerGPU =  numerator/NGPU;
    }
    cout<<"NIterationPerGPU: "<<NIterationPerGPU<<endl;
    // if (size != num_process) {cout<<"Err: Process not matching!"<<endl;exit(-1);}
    // H_ERR(cudaSetDevice(0)); //GPU 0 is the default GPU. Only max sampling is done in multiple GPU in this version

    //    H_ERR(cudaSetDevice(1));


    data_t* vec= (data_t*)malloc(sizeof(data_t)*num_element);//new data_t[num_element];
    data_t* vec1= (data_t*)malloc(sizeof(data_t)*num_element);//new data_t[num_element];

    std::random_device rd;
    std::mt19937 gen(rd());

    unsigned int value;
    int over;
    int minvalue=2147483643;
    bool test=false;

    // index_t alpha=0.5*((num_pow-(NGPU-1))-log(k)/log(2)+3);//modified for multiGPU
    // index_t alpha=0.5*(new_num_pow-log(k)/log(2)+3);//modified for multiGPU
    index_t alpha=0.5*(final_num_pow-log(k)/log(2)+3);//modified for multiGPU

    cout<<"Calculated alpha: "<<alpha<<endl;
    cout<<"final_num_pow: "<<final_num_pow<<endl;

    bool defaultContribution=true;

    if (alpha <=5)  defaultContribution=false;


    index_t SubRangesize=pow(2,alpha);
    cout<<"SubRangesize: "<<SubRangesize<<endl;
   
    // std::normal_distribution<float> d(100000000, 10);//Normal
    for (index_t i=0;i<num_element;i++)
    { 
        value=rand()%2147483648;//2^31 -1//Uniform        
        // value=i;//2^31 -1//Increasing 

     
        // value=d(gen);

        if (minvalue > value)
        {
            minvalue=value;
        }
        // if (value > 2147483650) test=true;
        if (value > 4294967295)
        {
            cout<<"Overflow of unsigned int detected"<<endl;
            return -1;
        }
        vec[i]=value;
        vec1[i]=vec[i];
    }

    if (minvalue < 0)
    {
        cout<<"-ve value detected:"<<minvalue<<endl;
        return -1;
    }
    cout<<"Minimum value:"<<minvalue<<endl;
    if (test) cout<<"Data generated Ok"<<endl;
    else
        cout<<"Data generated not Ok"<<endl;

    num_element = power<data_t,index_t>(base,final_num_pow);//num_element/num_process;
    cout<<"final_num_pow: "<<final_num_pow<<endl;
    int iteration =0;
    index_t vec_offset = iteration * NGPU*num_element + myrank * num_element;
    data_t* vec_d;
    H_ERR(cudaMalloc((void**) &vec_d,sizeof(data_t)*num_element));
    H_ERR(cudaMemcpy(vec_d,&vec[vec_offset],sizeof(data_t)*num_element,cudaMemcpyHostToDevice));

    index_t num_bucket=1<<NBits;
    int CurrentDigit=(sizeof(data_t)*8/NBits)-1;
    index_t NSubranges=num_element/SubRangesize;
    cout<<"Final number of element: "<<num_element<<endl;
    int NthreadstoworkInreduction=32;
    if (SubRangesize<32)
    {
        NthreadstoworkInreduction=SubRangesize;
    }

    cout<<"Number of Subranges:"<<NSubranges<<endl;
    if (NSubranges<k)
    {
        cout<<"Small number of subranges!. Decrease the value of alpha!"<<endl;
        //        exit(-1);
    }

    if ((!IsPowerof2<data_t,index_t>(NBits)) || (NBits > sizeof(data_t)*8)) 
    {
        cout<<"Enter correct number of bits per digit"<<endl;
        return -1;
    }


    cout<<"Starting TopK with Npow:"<<num_pow<<" K:"<<k<<" alpha:"<<alpha<<"DistributionU(0,2^31-1)"<<endl;
    // std::fstream statusLog;
    //  timeLog.open("timeRadixSampleOCT11_N_K_alphaVaried.csv",std::fstream::out | std::fstream::app);
    cout<<vec[0];
    cout<<endl;
    data_t* TopArray=new data_t[k];
    data_t TopKElement=0;
    // // raelse dix_select_inplace<data_t,index_t>(vec_d,num_element,k,num_bucket,TopKElement,NBits,CurrentDigit,0);
    double timeforMaxsample=0;double timeforFirstTopk=0;double timeforSecondTopk=0;double timeforNormalRadixSelect=0;double timeforConcatenation=0;

    data_t* Max_d;
    H_ERR(cudaMalloc((void**) &Max_d,sizeof(data_t)*NSubranges*beta));// updated for Beta

    index_t* SubrangeId_d;
    H_ERR(cudaMalloc((void**) &SubrangeId_d,sizeof(index_t)*NSubranges*beta));//updated for beta

    int NThreadsPerBlock=256;//only shared memory
    //   int NThreadsPerBlock=1024;//Shared memory with subwarp
    int SizeOfSubWarp=8;
    int pow_size_Subwarp=3;
    //    int NSharedMemoryElements=NThreadsPerBlock<<alpha;//only shared Memory
    int NSharedMemoryElements=NThreadsPerBlock<<5;//3 is giving best result in different values of SubWarp size //Each thread responsible for 32 elements and contribute to 8 Subranges from a group of 4 elements
    int SizeOfAllocation=NSharedMemoryElements+(NSharedMemoryElements >> 5);
    //sampleMax_multirange<data_t,index_t><<<4096,512>>>(A_d,Max_d,N,NSubranges,SubRangesize,alpha,SubrangeId_d,Nthreadstowork, NSubrangesperWarp, SubWarpSize,NThreadsPerSubRange);
    int NumberOfSpace_WithPadding=NSharedMemoryElements+(NSharedMemoryElements >>5);
    int NSubRangesPerBlock=NSharedMemoryElements/SizeOfSubWarp;//can be in CPU
    int NSubWarps_InBlock=NThreadsPerBlock >> pow_size_Subwarp;// Can be in CPU
    //Note NTotalVirtualSubWarpsInBlock=NSubrangesDealtBy1Block as 1 subwarp is responsible for 1 Subrange
    int NElementsPerBlock_ReadFromGlobal=NSubRangesPerBlock*SubRangesize;//1 Subwarp works for 1 subrange --> Can be in CPU
    int TotalBlocksrequired=num_element/NElementsPerBlock_ReadFromGlobal;
    // if (TotalBlocksrequired<1)
    // {
    //     cout<<"reduce blockDim or sizeofSubrange(alpha), for the kernel to work"<<endl;
    //     exit(-1);
    // } 
    cout<<"Size of shared memory per block:"<<SizeOfAllocation*sizeof(data_t)/1024.0 <<"KB"<<endl;

    //   statusLog.open("Status_alpha_0_3_4_5_TotalSOK_Radix.csv",std::fstream::out | std::fstream::app);
    // statusLog.open("StatusFile.csv",std::fstream::out | std::fstream::app);
    // statusLog<<endl<<endl<<"Started Radix select with:2^"<<num_pow<<" elements "<<k<<" as Kth element and "<<alpha<<"as alpha!."<<"Distribution:U(0,2^31-1)"<<endl;

    index_t* SelectedSubrangeId_d;
    // H_ERR(cudaMalloc((void**) &SelectedSubrangeId_d,sizeof(index_t)*k*beta));//updated *3 for beta
    // H_ERR(cudaMalloc((void**) &SelectedSubrangeId_d,sizeof(index_t)*k*beta));//updated *3 for beta
    H_ERR(cudaMalloc((void**) &SelectedSubrangeId_d,sizeof(index_t)*num_element));//When digit skip is enabled in first topk
    int* CountSelectedSubrange_d;
    int* CountLonelyElements_d;
    H_ERR(cudaMalloc((void**) &CountSelectedSubrange_d,sizeof(int)));
    H_ERR(cudaMalloc((void**) &CountLonelyElements_d,sizeof(int)));
    H_ERR(cudaMemset(CountSelectedSubrange_d, 0,  sizeof(int)));
    H_ERR(cudaMemset(CountLonelyElements_d, 0,  sizeof(int)));

    data_t* ConcatenatedRange_d;
    int* write_pos_d;
    H_ERR(cudaMalloc((void**) &ConcatenatedRange_d,sizeof(data_t)*k*SubRangesize));
    // H_ERR(cudaMalloc((void**) &ConcatenatedRange_d,sizeof(data_t)*num_element));// for skipping digits in first top-k
    H_ERR(cudaMalloc((void**) &write_pos_d,sizeof(int)));
    data_t* TopKElements=	(data_t*) malloc ((k*NIterationPerGPU)*sizeof(data_t));
    data_t* TopKElements_d;
    H_ERR(cudaMalloc((void**) &TopKElements_d,sizeof(data_t)*k*NIterationPerGPU));// for skipping digits in first top-k
    data_t* TopKElementsRecv_d;
    data_t* TopKElementsRecv=(data_t*) malloc ((k*NGPU*NIterationPerGPU)*sizeof(data_t));
    H_ERR(cudaMalloc((void**) &TopKElementsRecv_d,sizeof(data_t)*k*NGPU*NIterationPerGPU));// for skipping digits in first top-k
    // if (myrank ==0 ) TopKElements_d = TopKElementsRecv_d;//Rank 0 will directly put its topk into the corresponding position in receive buffer

    double timeForFinalTopK =0;
    double overheadTime = 0;
    double SampleFirstTopk =0;
    double KthCommunicateTime=0;
    double FindAndFillTime =0;
    double ConcatenateAndSecondTopkTime=0;
    double startTime =0;
    double TopKCommunication=0;
    double StartTopKCommunication=0;
    double TimeForTopKCommunication=0;
    double  ReloadOverhead=0;
    index_t ConcatenatedSize =0;
    double OffsetGatherTime =0;
    int* topkOffset_d;
    int* topkOffset =0;
    H_ERR(cudaMalloc((void**) &topkOffset_d,sizeof(int)));
    H_ERR(cudaMemset(topkOffset_d, 0, sizeof(int)));
    index_t* displs= (index_t*) malloc (num_process*sizeof(index_t));
    index_t* rcount= (index_t*) malloc (num_process*sizeof(index_t));
    for (int i=0;i<num_process;i++)
    {
        displs[i] = i;
        rcount[i] = 1;
    }
    index_t* IndivProcessKElements = (index_t*)malloc((sizeof(index_t))*num_process);
    double start=wtime();
    // if (alpha==0)
    // {
    //     timeforNormalRadixSelect=wtime();
    //     radix_select<data_t,index_t>(vec_d,num_element,k,num_bucket,TopKElement,NBits,CurrentDigit);
    //     timeforNormalRadixSelect=wtime()-timeforNormalRadixSelect;
    // }
    // else// if(NSubranges > k)
    // {		
    index_t GPUID = myrank;		
    // H_ERR(cudaSetDevice(GPUID));
    // multiGPUSample( NGPU, GPUID,  Max_d/*unified*/,&vec_d[num_element*myrank], num_element, NSubranges, SubRangesize, alpha,SubrangeId_d, NthreadstoworkInreduction,  NSharedMemoryElements, beta,defaultContribution,NThreadsPerBlock,SizeOfAllocation,timeforMaxsample);
    // GPUID = 1;
    // H_ERR(cudaSetDevice(GPUID));
    // multiGPUSample( NGPU, GPUID,  Max_d/*unified*/,vec_dG1, num_element, NSubranges, SubRangesize, alpha,SubrangeId_d, NthreadstoworkInreduction,  NSharedMemoryElements, beta,defaultContribution,NThreadsPerBlock,SizeOfAllocation,timeforMaxsample);

    // SynchronizeAllDevices(stream,NGPU);
    // H_ERR(cudaSetDevice(0));//All other operations are in GPU 0

    int offset = 0;  
    int NSkipDocument =0;             
    // while (iteration < NIterationPerGPU)
    // {
    cout<<"Starting iteration: "<<iteration<<endl;
    startTime= wtime();
    CurrentDigit=(sizeof(data_t)*8/NBits)-1;
    sample_radix_select<data_t,index_t>(vec_d, num_element,k,num_bucket,TopKElement,NBits,CurrentDigit,
            NSubranges,SubRangesize,alpha,timeforMaxsample,timeforFirstTopk,
            timeforSecondTopk,timeforConcatenation,
            Max_d,SubrangeId_d,NSharedMemoryElements,SizeOfSubWarp,
            pow_size_Subwarp,NSubWarps_InBlock,NSubRangesPerBlock,
            NElementsPerBlock_ReadFromGlobal,TotalBlocksrequired,SizeOfAllocation,
            NThreadsPerBlock,beta,defaultContribution,NthreadstoworkInreduction,
            SelectedSubrangeId_d,CountLonelyElements_d,write_pos_d,ConcatenatedRange_d,
            CountSelectedSubrange_d, &TopKElements_d[k*iteration],
            NGPU, myrank, TopKElementsRecv_d,  timeForFinalTopK, overheadTime,vec,NSkipDocument);
    // cout<<"myrank: "<<myrank<<" TopKElement after first topk: "<<TopKElement<<endl;
    SampleFirstTopk += wtime() - startTime;
    data_t recvTopK;

#ifdef Enabletest
    if (myrank==0)
    {
        sort(vec1, vec1 + total_num_element);

        cout<<endl;

        if (vec1[total_num_element-k]==TopKElement) 
        {
            cout<<"Success!"<<endl;
        }
        else
        {
            cout<<"Not Success!"<<endl;
        }
        cout<<"Required value"<<vec1[total_num_element-k]<<endl;
        assert(vec1[total_num_element-k]==TopKElement);
    }
#endif
    if (myrank==0)
    {
        std::fstream timeLog;
        timeLog.open("BMW_LogUniform.csv",std::fstream::out | std::fstream::app);
        // timeLog.open("BMW_LogNormal.csv",std::fstream::out | std::fstream::app);
        // timeLog.open("BMW_LogIncreasing.csv",std::fstream::out | std::fstream::app);

        timeLog<<" "<<num_element<<";"<<k<<";"<<alpha<<";"<<beta<<";"<<NSkipDocument<<endl;
        // cout<<num_element<<";"<<k<<";"<<alpha<<";"<<beta<<";"<<NSkipDocument<<endl;
        timeLog.close();
    }
    // MPI_Finalize();

    return 0;


}
