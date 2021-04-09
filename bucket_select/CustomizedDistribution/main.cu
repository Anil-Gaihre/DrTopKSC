#include <iostream>
#include <stdlib.h>
#include <algorithm>
#include "bucketselect_combined.cuh"
//#include "bucketselect.cuh"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <random>
// #define Enabletest 1

using namespace std;

typedef unsigned int data_t;

//typedef unsigned int data_t;
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
	//data_t max=arr[0];

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

int main(int argc,char** argv)
{
	cout<<"./exe num_element k NBucket"<<endl;
	if (argc != 4) {cout<<"wrong input"<<endl;exit(-1);}
	index_t num_pow = atol(argv[1]);
	index_t base=2;
	index_t num_element = power<data_t,index_t>(base,num_pow);
	index_t k= atol(argv[2]);
	index_t num_bucket=atol(argv[3]);//atol(argv[3]);
	//    index_t num_bucket=1<<NBits;

	index_t alpha=0.5*(num_pow-log(k)/log(2)+3);
	if (alpha<5) alpha++;
	bool defaultContribution=true;
	int beta=2;

	index_t SubRangesize=pow(2,alpha);
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

	if (alpha <=5) 
	{
		defaultContribution=false;
		beta=2;//SampleBeta function is designed for Beta=3 only. So we need to update the SampleBetafunction in radix select if we want to change Beta value
	}

	data_t* Max_d;
	H_ERR(cudaMalloc((void**) &Max_d,sizeof(data_t)*NSubranges*beta));// updated for Beta
	index_t* SubrangeId_d;
	H_ERR(cudaMalloc((void**) &SubrangeId_d,sizeof(index_t)*NSubranges*beta));//updated for beta


	data_t* vec= new data_t[num_element];
	data_t* vec1= new data_t[num_element];

	std::random_device rd;
	std::mt19937 gen(rd());
	float minvalue=100000000;
	unsigned int value;

	// std::uniform_int_distribution <unsigned int> d(0, 4294957295);//50%
	std::uniform_int_distribution <unsigned int> d_lower(0, 4294967295);//50%
	index_t num_element1 = 0.5*num_element;
	std::uniform_int_distribution <unsigned int> d_upper(4294957295, 4294967295);//Should be 50%
	// std::normal_distribution<float> d(100000000, 10);//Mean =100 mill , sd=100
	//     std::uniform_real_distribution<> d(0.0, 4294967295.0);//Generates random uniformly distributed floats within the given range
	for (index_t i=0;i<num_element1;i++)
	{
		//        vec[i]=rand()%2147483648;//2^31 -1
		value=d_lower(gen);//2^31 -1
		if (minvalue > value)
		{
			minvalue=value;
		}
		vec[i]=value;
		vec1[i]=vec[i];
	}
	for (index_t i=num_element1;i<num_element;i++)
	{
		//        vec[i]=rand()%2147483648;//2^31 -1
		value=d_upper(gen);//2^31 -1
		if (minvalue > value)
		{
			minvalue=value;
		}
		vec[i]=value;
		vec1[i]=vec[i];
	}
	if (minvalue < 0)
	{
		cout<<"-ve value detected:"<<minvalue<<endl;
		return -1;
	}


	cout<<vec[0];
	cout<<endl;
	data_t* TopArray=new data_t[k];
	data_t TopKElement=0;
	//    data_t NNewTopElements;
	data_t* vec_d;
	H_ERR(cudaMalloc((void**) &vec_d,sizeof(data_t)*num_element));
	H_ERR(cudaMemcpy(vec_d,vec,sizeof(data_t)*num_element,cudaMemcpyHostToDevice));
	double timeforMaxsample=0;double timeforFirstTopk=0;double timeforSecondTopk=0;double timeforNormalRadixSelect=0;double timeforConcatenation=0;

	int NThreadsPerBlock=256;//only shared memory
	int NSharedMemoryElements=NThreadsPerBlock<<5;//3 is giving best result in different values of SubWarp size //Each thread responsible for 32 elements and contribute to 8 Subranges from a group of 4 elements
	int SizeOfAllocation=NSharedMemoryElements+(NSharedMemoryElements >> 5);

	H_ERR(cudaDeviceSynchronize());
	index_t* SelectedSubrangeId_d;
	H_ERR(cudaMalloc((void**) &SelectedSubrangeId_d,sizeof(index_t)*(NSubranges-k)*beta));//updated *3 for beta
	index_t* CountSelectedSubrange_d;
	index_t* CountLonelyElements_d;
	H_ERR(cudaMalloc((void**) &CountSelectedSubrange_d,sizeof(index_t)));
	H_ERR(cudaMalloc((void**) &CountLonelyElements_d,sizeof(index_t)));
	H_ERR(cudaMemset(CountSelectedSubrange_d, 0,  sizeof(index_t)));
	H_ERR(cudaMemset(CountLonelyElements_d, 0,  sizeof(index_t)));
	index_t* write_pos_d;
	//        H_ERR(cudaMalloc((void**) &ConcatenatedRange_d,sizeof(data_t)*k*SubRangesize));
	H_ERR(cudaMalloc((void**) &write_pos_d,sizeof(index_t)));
	data_t* ConcatenatedRange_d;
	H_ERR(cudaMalloc((void**) &ConcatenatedRange_d,sizeof(data_t)*k*SubRangesize));

	double start=wtime();

	// sample_bucket_select<data_t,index_t>(vec_d,num_element,/*num_element-k*/k,num_bucket,TopKElement,NSubranges,SubRangesize,alpha,timeforMaxsample,timeforFirstTopk,timeforSecondTopk,timeforConcatenation,Max_d,SubrangeId_d,beta,defaultContribution,NthreadstoworkInreduction,NThreadsPerBlock,SizeOfAllocation,NSharedMemoryElements,  SelectedSubrangeId_d, CountSelectedSubrange_d, CountLonelyElements_d, write_pos_d, ConcatenatedRange_d);

		// bucket_select<data_t,index_t>(vec_d,num_element,num_element-k,num_bucket,TopArray,TopKElement);
		bucket_select<data_t,index_t>(vec_d,num_element,num_element-k,num_bucket,TopKElement);
	double totalTime=wtime()-start;
	cout<<"Time for selecting the top k element is:"<<totalTime*1000<<" ms"<<endl;

	//     bucket_select_PhaseII<data_t,index_t>(vec_d,num_element,k,num_bucket,TopKElement,vec);
	cout<<"The kth element from top is:"<<TopKElement<<endl;
	cout<<endl;
#ifdef Enabletest
	sort(vec1, vec1 + num_element);

	cout<<endl;

	cout<<"kth element"<<vec1[num_element-k]<<endl;
	cout<<"k-1 th element"<<vec1[num_element-k+1]<<endl;
	cout<<"k+1 th element"<<vec1[num_element-k-1]<<endl;
	if (vec1[num_element-k]==TopKElement) 
	{
		cout<<"Success!"<<endl;
	}
	else
	{
		cout<<"Not Success!"<<endl;
	}
	assert(vec1[num_element-k]==TopKElement);
#endif
	std::fstream timeLog;
	// timeLog.open("Uniform_Unsigned__N_30_SOKBucket.csv",std::fstream::out | std::fstream::app);
    //timeLog.open("Normal_float_N_29_SOKBucket.csv",std::fstream::out | std::fstream::app);
    // timeLog.open("Normal_UINT_N_30_SOKBucket.csv",std::fstream::out | std::fstream::app);
	timeLog.open("Test.csv",std::fstream::out | std::fstream::app);
	//  timeLog.open("Uniform_UINT_N_29_SOKBucket.csv",std::fstream::out | std::fstream::app);
	if  (defaultContribution)
	{
		timeLog<<"D"<<";";
	}
	else
	{
		timeLog<<"B"<<";";
	}

	timeLog<<num_pow<<";"<<k<<";"<<alpha<<";"<<timeforMaxsample*1000<<";"<<timeforFirstTopk*1000<<";"<<timeforConcatenation*1000<<";"<<timeforSecondTopk*1000<<";"<<timeforNormalRadixSelect*1000<<";"<<totalTime*1000<<endl;
	//          timeLog<<num_pow<<"_N_"<<num_element<<"k_"<<k<<"num_bucket_"<<num_bucket<<";"<<totalTime*1000<<endl;
	timeLog.close();
	return 0;
}
