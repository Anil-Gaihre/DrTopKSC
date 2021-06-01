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
#include <mpi.h>

// #include "/scratch/openmpi-3.0.6/ompi/include/mpi.h"
// #include <iostream> 
//#include <random>
//  #define Enabletest 1
// #define CommunicateFirstAndSecondK 1
// #define EnableOffsetGather 1
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
    MPI_Status status;
    MPI_Init(&argc, &argv);
    /* Determine unique id of the calling process of all processes participating
     *        in this MPI program. This id is usually called MPI rank. */
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    printf("setting up GPU: %d\n",myrank);
    // H_ERR(cudaSetDevice(myrank));
    int NGPUPerNode=atoi(argv[5]);
    H_ERR(cudaSetDevice(myrank%NGPUPerNode));
    MPI_Comm_size( MPI_COMM_WORLD, &size);


    cout<<"./exe num_element k NBitsPerDigit beta NGPUPerNode num_Node"<<endl;
    cout<<"Size of unsigned int"<<sizeof(unsigned int)<<endl;
    if (argc != 7) {cout<<"wrong input"<<endl;exit(-1);}
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
    
    int num_Node = atoi(argv[6]);
    

    // int num_process = NGPU;
    int num_process = size;
    // int NGPU = NGPUPerNode * num_process;
    int NGPU = NGPUPerNode * num_Node;
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
    if (size != num_process) {cout<<"Err: Process not matching!"<<endl;exit(-1);}
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

    //   for (int dis=3;dis<4;dis++)
    {
        //         std::uniform_int_distribution <unsigned int> d(0, 2147483643);
        int minvalue=2147483643;

        // std::normal_distribution<float> d(100000000, 10);//Mean =100 mill , sd=100
        // std::normal_distribution<float> d(100000000, 10000000);//Mean =100 mill , sd=100
        // std::uniform_int_distribution <unsigned int> d(0, 4294967295);

        //    for (int dis=3;dis<4;dis++)
        //        {
        for (index_t i=0;i<num_element;i++)
        { 
            value=rand()%2147483648;//2^31 -1
            // value=i;//2^31 -1
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

        
        //   sort(vec, vec + num_element);

        //    for (int Kiteration=atol(argv[2]);Kiteration<536870912;Kiteration=Kiteration*2)
        {
            //        k=Kiteration;
            // index_t alpha=atol(argv[4]);
            // int beta=3;//SampleBeta function is designed for Beta=3 only. So we need to update the SampleBetafunction in radix select if we want to change Beta value
            // num_element = num_element/num_process;
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

            MPI_Barrier(MPI_COMM_WORLD);    

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
                          
                while (iteration < NIterationPerGPU)
                {
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
                        NGPU, myrank, TopKElementsRecv_d,  timeForFinalTopK, overheadTime );
                // cout<<"myrank: "<<myrank<<" TopKElement after first topk: "<<TopKElement<<endl;
                SampleFirstTopk += wtime() - startTime;
                data_t recvTopK;
                startTime = wtime();
                #ifdef CommunicateFirstAndSecondK
                MPI_Allreduce(&TopKElement, &recvTopK, 1,
                        MPI_UNSIGNED, MPI_MAX, MPI_COMM_WORLD);
                TopKElement = recvTopK;
                #endif
                KthCommunicateTime += (wtime() - startTime);
                startTime = wtime();
                // cout<<"myrank: "<<myrank<<" TopKElement after communication: "<<TopKElement<<endl;
                concatenate_radix_select<data_t,index_t>(vec_d, num_element,k,num_bucket,TopKElement,NBits,CurrentDigit,
                        NSubranges,SubRangesize,alpha,timeforMaxsample,timeforFirstTopk,
                        timeforSecondTopk,timeforConcatenation,
                        Max_d,SubrangeId_d,NSharedMemoryElements,SizeOfSubWarp,
                        pow_size_Subwarp,NSubWarps_InBlock,NSubRangesPerBlock,
                        NElementsPerBlock_ReadFromGlobal,TotalBlocksrequired,SizeOfAllocation,
                        NThreadsPerBlock,beta,defaultContribution,NthreadstoworkInreduction,
                        SelectedSubrangeId_d,CountLonelyElements_d,write_pos_d,ConcatenatedRange_d,
                        CountSelectedSubrange_d,&TopKElements_d[k*iteration],
                        NGPU, myrank, TopKElementsRecv_d,  timeForFinalTopK, overheadTime, ConcatenatedSize );
                ConcatenateAndSecondTopkTime += (wtime()-startTime);
                //  cout<<"myrank: "<<myrank<<" TopKElement after second topk: "<<TopKElement<<endl;

                // cout<<"myrank: "<<myrank<<" TopKElement before communication/After second topk: "<<TopKElement<<endl;                              
                // data_t recvTopK;
                startTime = wtime();
                #ifdef CommunicateFirstAndSecondK
                MPI_Allreduce(&TopKElement, &recvTopK, 1,
                        MPI_UNSIGNED, MPI_MAX, MPI_COMM_WORLD);
                TopKElement = recvTopK;
                #endif
                KthCommunicateTime += (wtime() - startTime);
                // cout<<"myrank: "<<myrank<<" TopKElement after communication: "<<TopKElement<<endl;
                startTime = wtime();
                find_topk<data_t,index_t>(vec_d, num_element,k,num_bucket,TopKElement,NBits,CurrentDigit,
                        NSubranges,SubRangesize,alpha,timeforMaxsample,timeforFirstTopk,
                        timeforSecondTopk,timeforConcatenation,
                        Max_d,SubrangeId_d,NSharedMemoryElements,SizeOfSubWarp,
                        pow_size_Subwarp,NSubWarps_InBlock,NSubRangesPerBlock,
                        NElementsPerBlock_ReadFromGlobal,TotalBlocksrequired,SizeOfAllocation,
                        NThreadsPerBlock,beta,defaultContribution,NthreadstoworkInreduction,
                        SelectedSubrangeId_d,CountLonelyElements_d,write_pos_d,ConcatenatedRange_d,
                        CountSelectedSubrange_d,&TopKElements_d[k*iteration],
                        NGPU, myrank, TopKElementsRecv_d,  timeForFinalTopK, overheadTime, ConcatenatedSize,topkOffset_d );


                FillTopk<data_t,index_t><<<128,128>>>(ConcatenatedRange_d,ConcatenatedSize, k, TopKElement, TopKElements_d, topkOffset_d);
                H_ERR(cudaDeviceSynchronize());
                FindAndFillTime += (wtime() - startTime);
                #ifdef EnableOffsetGather
                // startTime = wtime();
                // H_ERR(cudaMemcpy(&topkOffset,topkOffset_d,sizeof(data_t),cudaMemcpyDeviceToHost));
                // //Gather offset information in root (process 0)
                // MPI_Gatherv(&topkOffset,1,MPI_INT,IndivProcessKElements, rcount,displs,MPI_INT,0,MPI_COMM_WORLD);
                // OffsetGatherTime += (wtime()-startTime);
                #endif
                double StartReloadOverhead = wtime();  
                iteration++; 
                if (iteration < NIterationPerGPU)
                {
                    H_ERR(cudaMemset(topkOffset_d, 0, sizeof(int)));
                    H_ERR(cudaMemset(CountSelectedSubrange_d, 0,  sizeof(int)));
                    H_ERR(cudaMemset(CountLonelyElements_d, 0,  sizeof(int)));
                    vec_offset = iteration * NGPU*num_element + myrank * num_element;
                    //load vec_d with new elements
                    H_ERR(cudaMemcpy(vec_d,&vec[vec_offset],sizeof(data_t)*num_element,cudaMemcpyHostToDevice));
                    H_ERR(cudaDeviceSynchronize());
                }
                 ReloadOverhead += (wtime()-StartReloadOverhead);
            } 
            //communicate k among the processes
#ifdef enable_overheads
            StartTopKCommunication = wtime();
            if (myrank == 0)
            {
                MPI_Status* status = (MPI_Status*)malloc((sizeof(MPI_Status))*NGPU);
                MPI_Request* request = (MPI_Request*)malloc((sizeof(MPI_Request))*NGPU);
                int receivedTopKs = 0;
                // Receive offset information
                // ~Receive offset information      
                int* receiveOffset = (int*)malloc((sizeof(int))*NGPU);       

                for(int i=0; i < NGPU; i++)
                {
                    #ifndef EnableOffsetGather
                    // IndivProcessKElements[i+iteration*NGPU] =k;
                    #endif
                    //int receiveOffset =   i * k * NIterationPerGPU;
                     receiveOffset[i] =   i * k * NIterationPerGPU;
                    if (i!=myrank)
                    {
                        //i.e. i!=0 ==> Recieve from other GPUS
                        receivedTopKs=1;
                        cout<<"Process: "<<myrank<< "receiving from process: "<<i<<endl;
                        MPI_Irecv(&TopKElementsRecv[receiveOffset[i]], k*NIterationPerGPU, MPI_UNSIGNED, i, i, MPI_COMM_WORLD, &request[i]);
                      
                    }
                    else
                    {
                        // i = 0
                        H_ERR(cudaMemcpy(&TopKElementsRecv_d[receiveOffset[i]],TopKElements_d,sizeof(data_t)*k*NIterationPerGPU,cudaMemcpyDeviceToDevice));
 
                    } 
                    // offset += IndivProcessKElements[i+iteration*NGPU];
                    // offset += IndivProcessKElements[i+iteration*NGPU];
                }
                // MPI_Wait();
                for(int i=1;i<NGPU;i++)
                {
                    MPI_Wait(&request[i], &status[i]);
                    cout<<"Process: "<<myrank<< "received topk from process: "<<i<<endl;
                }
                // if (receivedTopKs==1) H_ERR(cudaMemcpy(&TopKElementsRecv_d[NIterationPerGPU*k],&TopKElementsRecv[NIterationPerGPU*k],sizeof(data_t)*k*NIterationPerGPU,cudaMemcpyHostToDevice));
                if (receivedTopKs==1) H_ERR(cudaMemcpy(&TopKElementsRecv_d[NIterationPerGPU*k],&TopKElementsRecv[NIterationPerGPU*k],sizeof(data_t)*k*NIterationPerGPU*(NGPU-1),cudaMemcpyHostToDevice));
             }
            else
            {                
                // startTime = wtime();
                H_ERR(cudaMemcpy(TopKElements,TopKElements_d,sizeof(data_t)*k*NIterationPerGPU,cudaMemcpyDeviceToHost));
                cout<<"Process: "<<myrank<< "sending its topk"<<endl;

                // MPI_Send(TopKElements, k*NIterationPerGPU, MPI_UNSIGNED, 0, myrank, MPI_COMM_WORLD);

                MPI_Send(TopKElements, k*NIterationPerGPU, MPI_UNSIGNED, 0, myrank, MPI_COMM_WORLD);
                cout<<"Process: "<<myrank<< "sent its topk"<<endl;
                
            }
            
            TopKCommunication += (wtime() - StartTopKCommunication);
           
            // double StartReloadOverhead = wtime();  
            // iteration++;
            // if (iteration < NIterationPerGPU)
            // {
            //     H_ERR(cudaMemset(topkOffset_d, 0, sizeof(int)));
            //     H_ERR(cudaMemset(CountSelectedSubrange_d, 0,  sizeof(int)));
            //     H_ERR(cudaMemset(CountLonelyElements_d, 0,  sizeof(int)));
            //     vec_offset = iteration * NGPU*num_element + myrank * num_element;
            //     //load vec_d with new elements
            //     H_ERR(cudaMemcpy(vec_d,&vec[vec_offset],sizeof(data_t)*num_element,cudaMemcpyHostToDevice));
            //     H_ERR(cudaDeviceSynchronize());
            // }
            //  ReloadOverhead += (wtime()-StartReloadOverhead);
        // }

        //Perform final radix select in Process 0
        if (myrank ==0)
        {
             //Final normal top-k on the elements
             double startFinalTopK=wtime();
             data_t flag=0;
             CurrentDigit=(sizeof(data_t)*8/NBits)-1;  
             // cout<<"kth element in rank 3: "<< TopKElementsRecv[3*k]<<endl;
             int final_vec_size_radix_select = NIterationPerGPU*k*NGPU;
             cout<<"Final number of elements after reduction to Process 0: "<<final_vec_size_radix_select<<endl;
            radix_select_inplace<data_t,index_t>(TopKElementsRecv_d,final_vec_size_radix_select,k,num_bucket,TopKElement,NBits,CurrentDigit,flag);    
            // radix_select_inplace<data_t,index_t>(TopKElementsRecv_d,k*NGPU,k,num_bucket,TopKElement,NBits,CurrentDigit,flag);    
            timeForFinalTopK += (wtime()-startFinalTopK);
            //~Final normal top-k on the elements
        }
            //~communicate k among the processes
#endif
        
            double totalTime = wtime()-start;
            if (myrank==0) cout<<"The kth element from top is:"<<TopKElement<<endl;

            cout<<"Process: "<<myrank<<" Sampling Time: "<<timeforMaxsample*1000<<" ms"<<endl;
            cout<<"Process: "<<myrank<<" Time for First TopK: "<<timeforFirstTopk*1000<<" ms"<<endl;
            cout<<"Process: "<<myrank<<" Time for sampling and First TopK: "<<SampleFirstTopk*1000<<" ms"<<endl<<endl;
            cout<<"Process: "<<myrank<<" Time for Concatenation:"<<timeforConcatenation*1000<<" ms"<<endl;
            cout<<"Process: "<<myrank<<" Time for Second TopK: "<<timeforSecondTopk*1000<<" ms"<<endl;
            cout<<"Process: "<<myrank<<" Time for Concatenation and Second TopK: "<<ConcatenateAndSecondTopkTime*1000<<" ms"<<endl<<endl;

            // cout<<"Process: "<<myrank<<" Time for communicating FirstTopkAndSecondTopk: "<<KthCommunicateTime*1000<<" ms"<<endl<<endl;
            cout<<"Process: "<<myrank<<" Time for ReloadOverhead: "<<ReloadOverhead*1000<<" ms"<<endl;
            cout<<"Process: "<<myrank<<" Time for Find and Fill: "<<FindAndFillTime*1000<<" ms"<<endl<<endl;
            cout<<"Process: "<<myrank<<" Time for Offset gather: "<<OffsetGatherTime*1000<<" ms"<<endl<<endl;
            cout<<"Process: "<<myrank<<" Time for Communication (in P0): "<<TopKCommunication*1000<<" ms"<<endl;
            cout<<"Process: "<<myrank<<" Time for Final TopK(in P0): "<<timeForFinalTopK*1000<<" ms"<<endl;
           

            // cout<<"Process: "<<myrank<<" Time for Normal Radix Select:"<<timeforNormalRadixSelect*1000<<" ms"<<endl;
            cout<<"Process: "<<myrank<<" Total Time:"<<totalTime*1000<<" ms"<<endl;
            MPI_Barrier(MPI_COMM_WORLD); 
            double globalTotalTime =0;
            MPI_Reduce(&totalTime, &globalTotalTime, 1, MPI_DOUBLE ,
                    MPI_MAX, 0, MPI_COMM_WORLD );
            if (myrank ==0 ) cout<<"The total global time for top-k computation: "<<globalTotalTime*1000<<" ms"<<endl;

            
            MPI_Finalize();


            // statusLog<<"Successfully Finished Radix select with:2^"<<num_pow<<" elements "<<k<<" as Kth element and "<<alpha<<"as alpha!."<<endl;
            // statusLog.close();


            // MPI_Barrier(MPI_COMM_WORLD); 

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
            //  timeLog.open("N_29UniformDistributedAutoTuneAdaptive22Feb_TitanSorted_Unsorted.csv",std::fstream::out | std::fstream::app);
            //         timeLog.open("N_29TestingFor2^32.csv",std::fstream::out | std::fstream::app);
            // timeLog.open("U_VectorSize_VaryingK2^29.csv",std::fstream::out | std::fstream::app);
            timeLog.open("MultGPU.csv",std::fstream::out | std::fstream::app);
            // timeLog.open("FirstTopK3DigitsSkipped_Uniform.csv",std::fstream::out | std::fstream::app);
            // timeLog.open("U_beta_tuning.csv",std::fstream::out | std::fstream::app);
            if  (defaultContribution)
            {
                timeLog<<"D"<<";";
            }
            else
            {
                timeLog<<"B"<<";";
            }

            timeLog<<" "<<num_pow<<";"<<k<<";"<<alpha<<";"<<beta<<";"<<timeforMaxsample*1000<<";"<<timeforFirstTopk*1000<<";"<<timeforConcatenation*1000<<";"<<timeforSecondTopk*1000<<";"<<timeforNormalRadixSelect*1000<<";"<<totalTime*1000<<endl;
            timeLog.close();
        }
            //         H_ERR(cudaFree(vec_d));H_ERR(cudaFree(Max_d));H_ERR(cudaFree(SubrangeId_d));
        }
    }
    //    free(vec);free(vec1);

    return 0;


    }
