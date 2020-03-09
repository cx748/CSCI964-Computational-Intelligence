#include<iostream>
#include<iomanip>
#include<fstream>
#include<cstdlib>
#include<cstdio>
#include<cmath>
#include<ctime>

using namespace std;
const int MAXN = 50;       // Max neurons in any layer
const int MAXPATS = 5000;  // Max training patterns

const long NumIts=1000;
const int NumIn=4;
const int NumHN=3;
const int NumOut=3;
const float r=0.9;
const float ObjErr=0.0001;


extern double Weight1[NumIn][NumHN]={0.2};
extern double Weight2[NumHN][NumOut]={0.2};//初始化权重
extern double theta_H[NumHN]={0.1};
extern double theta_O[NumOut]={0.1};

void train(float **x,float **d,int NumIn,int NumOut,int NumPats)
{ //x[][]是输入数据 ， d[][]是目标输出
    float *h1 = new float[NumHN]; // O/Ps of hidden layer
    float *y  = new float[NumOut]; // O/P of Net
    float *ad1= new float[NumHN]; // HN1 back prop errors
    float *ad2= new float[NumOut]; // O/P back prop errors
    float PatErr,MinErr,AveErr,MaxErr;  // Pattern errors
    int p,i,j;     // for loops indexes
    long ItCnt=0;  // Iteration counter
    long NumErr=0; // Error counter (added for spiral problem)

    for(;;)
    {
      // Main learning loop
        MinErr=3.4e38; AveErr=0; MaxErr=-3.4e38; NumErr=0;
        for(p=0;p<NumPats;p++)
        {
            for(i=0;i<NumHN;i++)
            { // Cal O/P of hidden layer 1
                float in=0;
                for(j=0;j<NumIn;j++){
                in+=Weight1[j][i]*x[p][j];
                }
                in-=theta_H[i];
                h1[i]=(float)(1.0/(1.0+exp(double(-in))));// Sigmoid fn
            }
            for(i=0;i<NumOut;i++)
            { // Cal O/P of output layer
                float in=0;
                for(j=0;j<NumHN;j++){
                in+=Weight2[j][i]*h1[j];
                }
                in-=theta_O[i];
                y[i]=(float)(1.0/(1.0+exp(double(-in))));// Sigmoid fn
            }
 
            PatErr=0.0;
            for(i=0;i<NumOut;i++)
            {
                float err=y[i]-d[p][i]; // actual-desired O/P
                if(err>0)PatErr+=err; else PatErr-=err;
                NumErr += ((y[i]<0.5&&d[p][i]>=0.5)||(y[i]>=0.5&&d[p][i]<0.5));//added for binary classification problem
            }
            if(PatErr<MinErr)MinErr=PatErr;
            if(PatErr>MaxErr)MaxErr=PatErr;
            AveErr+=PatErr;
        }

        //计算输出层的BP
        for(i=0;i<NumOut;i++)
        { 
            ad2[i]=(d[p][i]-y[i])*y[i]*(1.0-y[i]);
            for(j=0;j<NumHN;j++)
            {
                Weight2[j][i]+= r*h1[j]*ad2[i];
                theta_O[j]-=  r*ad2[i];
            }
        }
        //计算隐藏层的BP
        for(i=0;i<NumHN;i++)
        {
            float err=0.0;
            for(j=0;j<NumOut;j++)
            err+=ad2[j]*Weight2[i][j];
            ad1[i]=err*h1[i]*(1.0-h1[i]);
            for(j=0;j<NumIn;j++)
            {
                Weight1[j][i]+=r*x[p][j]*ad1[i];
                theta_O[j] -=  r*ad1[i];
            }
        }
    ItCnt++;
    AveErr/=NumPats;
    float PcntErr = NumErr/float(NumPats) * 100.0;
    if((AveErr<=ObjErr)||(ItCnt==NumIts)) break;
    }
}