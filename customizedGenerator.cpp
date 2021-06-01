//**The code generates the customized distribution mentioned in the paper and writes it into the output directory mentioned in the code. The top-k tools read the data and perform read**/
#include <iostream>
#include <fstream>
using namespace std;


int main()
{
	unsigned int max= 4294967295;
//	unsigned int max= 2147483647;

	unsigned int min= 0;
	//  unsigned int NBucket = 256;
	unsigned int NBucket = 256;
	unsigned int num_element = 1073741824;
	unsigned int diff;
	//     int NElement = 2147483647;
	unsigned int* data = new unsigned int[num_element];
	int finish = 0;
	int i=0;
	int j=0;
	int pos=0;
	unsigned int offset =0;
	ofstream myfile;
	myfile.open ("/scratch/datasets/topkData/customized.dat");
	while (finish == 0)
	{
		diff = (max-min)/NBucket;
		if (min >= max) break;
		cout<<"min: "<<min<<endl;
		cout<<"diff: "<<diff<<endl;
		for (i=0;i<NBucket; i++)
		{
			pos = i+j*NBucket;
			if (pos > 0)
			{
				if (data[pos-2] > data[pos-1])
				{
					finish=1;break;
				}
			}
			if (pos >= num_element) {finish=1;break;}
			//   data[pos] = (i*diff+(i+1)*diff)/2;
			data[pos] = offset + (pos*diff+(pos+1)*diff)/2;
			//   cout<<pos<<":"<<data[pos]<<" ";
			myfile<<data[pos]<<" ";
		}
		min =data[pos];
		offset = min;
		j++;
	}

	for (int i=pos; i<num_element;i++)
	{
		data[i] = (unsigned int)4294967295;
		//data[i] = (unsigned int)(rand()%2147483646)+2147483000;
			//   cout<<i<<":"<<data[i]<<" ";
		myfile<<data[pos]<<" ";
	}
	myfile.close();
	return 0;
}

