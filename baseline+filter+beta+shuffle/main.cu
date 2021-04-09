#include <iostream>
#include <stdlib.h>
#include <algorithm>
#include "radixselect.cuh"
//#include "radixselectNormalInplaceWorking.cuh"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <random>
//#include <random>
// #define Enabletest 1
using namespace std;

typedef unsigned int data_t;
typedef int index_t;
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
	cout<<"./exe num_element k NBitsPerDigit beta"<<endl;
	cout<<"Size of unsigned int"<<sizeof(unsigned int)<<endl;
	if (argc != 5) {cout<<"wrong input"<<endl;exit(-1);}
	index_t num_pow = atol(argv[1]);
	index_t base=2;
	index_t num_element = power<data_t,index_t>(base,num_pow);
	cout<<"num_element: "<<num_element<<endl;
	index_t k= atol(argv[2]);
	index_t NBits=atol(argv[3]);//atol(argv[3]);
	int sd[]={10,100000,1000000,100,100000000};
	int beta=atoi(argv[4]);//SampleBeta function is designed for Beta=3 only. So we need to update the SampleBetafunction in radix select if we want to change Beta value

	//    H_ERR(cudaSetDevice(1));

	data_t* vec= (data_t*)malloc(sizeof(data_t)*num_element);//new data_t[num_element];
	data_t* vec1= (data_t*)malloc(sizeof(data_t)*num_element);//new data_t[num_element];

	std::random_device rd;
	std::mt19937 gen(rd());

	unsigned int value;
	int over;
	int minvalue=2147483643;
	bool test=false;

	index_t alpha=0.5*(num_pow-log(k)/log(2)+3);

cout<<"Calculated alpha: "<<alpha<<endl;

	bool defaultContribution=true;

	if (alpha <=5)  defaultContribution=false;


	index_t SubRangesize=pow(2,alpha);

	//   for (int dis=3;dis<4;dis++)
	{
		//         std::uniform_int_distribution <unsigned int> d(0, 2147483643);
		int minvalue=2147483643;
		
		// std::normal_distribution<float> d(100000000, 10);//Mean =100 mill , sd=100
		std::normal_distribution<float> d(100000000, 10000000);//Mean =100 mill , sd=100
		// std::uniform_int_distribution <unsigned int> d(0, 4294967295);

		//    for (int dis=3;dis<4;dis++)
		//        {
			for (index_t i=0;i<num_element;i++)
		{
			//        vec[i]=rand()%2147483648;//2^31 -1
			value=d(gen);
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

		//   sort(vec, vec + num_element);

		//    for (int Kiteration=atol(argv[2]);Kiteration<536870912;Kiteration=Kiteration*2)
		{
			//        k=Kiteration;
			// index_t alpha=atol(argv[4]);
			// int beta=3;//SampleBeta function is designed for Beta=3 only. So we need to update the SampleBetafunction in radix select if we want to change Beta value
			index_t num_bucket=1<<NBits;
			int CurrentDigit=(sizeof(data_t)*8/NBits)-1;
			index_t NSubranges=num_element/SubRangesize;
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

			//          data_t* vec= (data_t*)malloc(sizeof(data_t)*num_element);//new data_t[num_element];
			//          data_t* vec1= (data_t*)malloc(sizeof(data_t)*num_element);//new data_t[num_element];

			//          std::random_device rd;
			//          std::mt19937 gen(rd());

			//          float value;
			//          float minvalue=10000000;

			//          for (index_t i=0;i<num_element;i++)
			//          {
			//              std::normal_distribution<float> d(10000000, sd[d]);//Mean =100 mill , sd=100
			//              //        vec[i]=rand()%2147483648;//2^31 -1
			//              value=d(gen);
			//              if (minvalue > value)
			//              {
			//                  minvalue=value;
			//              }
			//              if (value > 4294967295)
			//              {
			//                  cout<<"Overflow of unsigned int detected"<<endl;
			//              }
			//              vec[i]=value;
			//              vec1[i]=vec[i];
			//         }
			//          cout<<endl;
			//          if (minvalue < 0)
			//          {
			//              cout<<"-ve value detected"<<endl;
			//          }
			cout<<"Starting TopK with Npow:"<<num_pow<<" K:"<<k<<" alpha:"<<alpha<<"DistributionU(0,2^31-1)"<<endl;
			std::fstream statusLog;
			//  timeLog.open("timeRadixSampleOCT11_N_K_alphaVaried.csv",std::fstream::out | std::fstream::app);
			cout<<vec[0];
			cout<<endl;
			data_t* TopArray=new data_t[k];
			data_t TopKElement=0;
			data_t* vec_d;
			H_ERR(cudaMalloc((void**) &vec_d,sizeof(data_t)*num_element));
			H_ERR(cudaMemcpy(vec_d,vec,sizeof(data_t)*num_element,cudaMemcpyHostToDevice));
			// raelse dix_select_inplace<data_t,index_t>(vec_d,num_element,k,num_bucket,TopKElement,NBits,CurrentDigit,0);
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
			if (TotalBlocksrequired<1)
			{
				cout<<"reduce blockDim or sizeofSubrange(alpha), for the kernel to work"<<endl;
				exit(-1);
			}
			cout<<"Size of shared memory per block:"<<SizeOfAllocation*sizeof(data_t)/1024.0 <<"KB"<<endl;

			//   statusLog.open("Status_alpha_0_3_4_5_TotalSOK_Radix.csv",std::fstream::out | std::fstream::app);
			statusLog.open("StatusFile.csv",std::fstream::out | std::fstream::app);
			statusLog<<endl<<endl<<"Started Radix select with:2^"<<num_pow<<" elements "<<k<<" as Kth element and "<<alpha<<"as alpha!."<<"Distribution:U(0,2^31-1)"<<endl;

			index_t* SelectedSubrangeId_d;
			// H_ERR(cudaMalloc((void**) &SelectedSubrangeId_d,sizeof(index_t)*k*beta));//updated *3 for beta
			// H_ERR(cudaMalloc((void**) &SelectedSubrangeId_d,sizeof(index_t)*k*beta));//updated *3 for beta
			H_ERR(cudaMalloc((void**) &SelectedSubrangeId_d,sizeof(index_t)*num_element));//When digit skip is enabled in first topk
			index_t* CountSelectedSubrange_d;
			index_t* CountLonelyElements_d;
			H_ERR(cudaMalloc((void**) &CountSelectedSubrange_d,sizeof(index_t)));
			H_ERR(cudaMalloc((void**) &CountLonelyElements_d,sizeof(index_t)));
			H_ERR(cudaMemset(CountSelectedSubrange_d, 0,  sizeof(index_t)));
			H_ERR(cudaMemset(CountLonelyElements_d, 0,  sizeof(index_t)));

			data_t* ConcatenatedRange_d;
			index_t* write_pos_d;
			// H_ERR(cudaMalloc((void**) &ConcatenatedRange_d,sizeof(data_t)*k*SubRangesize));
			H_ERR(cudaMalloc((void**) &ConcatenatedRange_d,sizeof(data_t)*num_element));// for skipping digits in first top-k
			H_ERR(cudaMalloc((void**) &write_pos_d,sizeof(index_t)));

			double start=wtime();
			if (alpha==0)
			{
				timeforNormalRadixSelect=wtime();
				radix_select<data_t,index_t>(vec_d,num_element,k,num_bucket,TopKElement,NBits,CurrentDigit);
				timeforNormalRadixSelect=wtime()-timeforNormalRadixSelect;
			}
			else// if(NSubranges > k)
			{
				sample_radix_select<data_t,index_t>(vec_d,num_element,k,num_bucket,TopKElement,NBits,CurrentDigit,NSubranges,SubRangesize,alpha,timeforMaxsample,timeforFirstTopk,timeforSecondTopk,timeforConcatenation,Max_d,SubrangeId_d,NSharedMemoryElements,SizeOfSubWarp,pow_size_Subwarp,NSubWarps_InBlock,NSubRangesPerBlock,NElementsPerBlock_ReadFromGlobal,TotalBlocksrequired,SizeOfAllocation,NThreadsPerBlock,beta,defaultContribution,NthreadstoworkInreduction,SelectedSubrangeId_d,CountLonelyElements_d,write_pos_d,ConcatenatedRange_d,CountSelectedSubrange_d);
			}
			//  else
			//   {
			//       timeforNormalRadixSelect=wtime();
			//       radix_select<data_t,index_t>(vec_d,num_element,k,num_bucket,TopKElement,NBits,CurrentDigit);
			//       timeforNormalRadixSelect=wtime()-timeforNormalRadixSelect;
			//   }
			double totalTime=wtime()-start;
			cout<<"The kth element from top is:"<<TopKElement<<endl;

			statusLog<<"Successfully Finished Radix select with:2^"<<num_pow<<" elements "<<k<<" as Kth element and "<<alpha<<"as alpha!."<<endl;
			statusLog.close();

			cout<<"Sampling Time:"<<timeforMaxsample*1000<<" ms"<<endl;
			cout<<"Time for First TopK:"<<timeforFirstTopk*1000<<" ms"<<endl;
			cout<<"Time for Concatenation:"<<timeforConcatenation*1000<<" ms"<<endl;
			cout<<"Time for Second TopK:"<<timeforSecondTopk*1000<<" ms"<<endl;
			cout<<"Time for Normal Radix Select:"<<timeforNormalRadixSelect*1000<<" ms"<<endl;
			cout<<"Total Time:"<<totalTime*1000<<" ms"<<endl;
			#ifdef Enabletest
			sort(vec1, vec1 + num_element);

			cout<<endl;

			if (vec1[num_element-k]==TopKElement) 
			{
				cout<<"Success!"<<endl;
			}
			else
			{
				cout<<"Not Success!"<<endl;
			}
			cout<<"Required value"<<vec1[num_element-k]<<endl;
			assert(vec1[num_element-k]==TopKElement);
#endif
			std::fstream timeLog;
			//  timeLog.open("N_29UniformDistributedAutoTuneAdaptive22Feb_TitanSorted_Unsorted.csv",std::fstream::out | std::fstream::app);
			//         timeLog.open("N_29TestingFor2^32.csv",std::fstream::out | std::fstream::app);
			// timeLog.open("U_VectorSize_VaryingK2^29.csv",std::fstream::out | std::fstream::app);
			timeLog.open("FirstTopK3DigitsSkipped_NORMAL_ALL_K.csv",std::fstream::out | std::fstream::app);
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
			//         H_ERR(cudaFree(vec_d));H_ERR(cudaFree(Max_d));H_ERR(cudaFree(SubrangeId_d));
		}
	}
	//    free(vec);free(vec1);

	return 0;


	}
