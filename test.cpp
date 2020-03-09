#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <string>
#include <fstream>
#include <iomanip>
using namespace std; 

#define innode 4//输入层结点数
#define hidenode 10//隐层结点数
#define outnode 3//输出层结点数
#define trainsample 75//训练样本数
#define testsample 75//测试样本数

double trainData[trainsample][innode];//输入样本
double outData[trainsample][outnode];//输出样本

double testData[testsample][innode];//测试样本

double w[innode][hidenode];//输入层到隐层的权值
double w1[hidenode][outnode];//隐层到输出层的权值
double b1[hidenode];//隐层阈值
double b2[outnode];//输出层阈值

double e=0.0;//误差计算
double error=1.0;//允许的最大误差

double rate_w=0.9;//输入层到隐层的学习率
double rate_w1=0.9;//隐层到输出层的学习率
double rate_b1=0.9;//隐层阈值学习率
double rate_b2=0.9;//输出层阈值学习率

double result[outnode];//bp输出

//初始化函数
void init(double w[], int n);
//Bp训练函数
void train(double trainData[trainsample][innode], double label[trainsample][outnode]);
//Bp识别 
double *recognize(double *p);
//从文件夹读取数据
void readData(std::string filename, double data[][innode], int x);
//数据归一化处理
void changeData(double data[][innode], int x);

int main()
{
    int i,j;
    int trainNum=0;//样本训练次数
    double *r; //测试结果
    int count=0;//正确测试结果数
    double maxRate = 1.0;//输出结果中的最大概率
    //对权值和阈值进行初始化
    init((double*)w, innode*hidenode);       //double *表示指向double型的指针，w为输入层到隐层的权值数组，innode*hidenode=4*10=40
    init((double*)w1, hidenode*outnode);
    init(b1, hidenode);    //b1为隐层阈值，hidenode为隐层结点数
    init(b2, outnode);	  //b2为输出层阈值，outnode为输出层结点数

    //读取训练数据
    readData("./Iris-train2.txt", trainData, trainsample);
    //对训练数据进行归一化处理
    changeData(trainData, trainsample);

  for(i=0; i<trainsample; i++)    //trainsample为测试样本数
    {
        printf("%d: ",i+1);
        for(j=0; j<innode; j++)
            printf("%5.2lf",trainData[i][j]);
        printf("\n");
    }
  
    //准备输出样本——3类花，每类花有25个样本
    for(i=0; i<trainsample; i++)
    {
        if(i<25)
        {
            outData[i][0] = 1.0;
            outData[i][1] = 0.000001;
            outData[i][2] = 0.000001;
        }
        else if(i<50)
        {
            outData[i][0] = 0.000001;
            outData[i][1] = 1.0;
            outData[i][2] = 0.000001;
        }
        else
        {
            outData[i][0] = 0.000001;
            outData[i][1] = 0.000001;
            outData[i][2] = 1.0;
        }
    }

    printf("开始训练\n");
    while(trainNum < 10000)    //trainNum为样本训练次数
    {
        e = 0.0;                 //e为误差
        trainNum++;
        train(trainData, outData);      //BP训练
        printf("训练第%d次， error=%8.4lf\n", trainNum, error);
    }
    printf("训练完成\n\n");

    //读入测试数据
    readData("./Iris-test.txt", testData, testsample);
    //归一化测试数据
    changeData(testData, testsample);
    for(i=0; i<testsample; i++)
    {
        r = recognize(testData[i]);
        for(j=0; j<outnode; j++)
            printf("\t%7.4lf\t",r[j]);
        printf("\n");
        //判断检测结果是否正确
        if(i<25 && r[0]>r[1] && r[0]>r[2])
            count++;
        if(i>=25 && i<50 && r[1]>r[0] && r[1]>r[2])
            count++;
        if(i>=50 && r[2]>r[0] && r[2]>r[1])
            count++;
    }

    printf("\n\n共有%d个检测样本， 正确检测出%d个， 准确率: %7.4lf\n\n",testsample, count, (double)count/testsample);
    system("pause");
    system("pause");
    return 0;
}

//初始化函数（0到1之间的数）
void init(double w[], int n)
{
    int i;
    srand((unsigned int)time(NULL));    //这个是种子函数srand（）为rand函数提供不同的种子，每次运行程序产生不同的随机数，不然rand函数每次运行程序产生的随机数都是一样的
    for(i=0; i<n; i++)
    {
        w[i] = 2.0*((double)rand()/RAND_MAX)-1;     //RAND_MAX 是 <stdlib.h> 中伪随机数生成函数 rand 所能返回的最大数值。
																				 //这意味着，任何一次对 rand 的调用，都将得到一个 0~RAND_MAX 之间的伪随机数。RAND_MAX=0x7fff
    }
}

//BP训练函数
void train(double trainData[trainsample][innode], double label[trainsample][outnode])
{
    double x[innode];//输入层的输入值
    double yd[outnode];//期望的输出值

    double o1[hidenode];//隐层结点激活值
    double o2[hidenode];//输出层结点激活值
    double x1[hidenode];//隐层向输出层的输入
    double x2[outnode];//输出结点的输出
    double qq[outnode];//期望的输出与实际输出的偏差

    double pp[hidenode];//隐含结点校正误差


    int issamp;
    int i,j,k;
    for(issamp=0; issamp<trainsample; issamp++)
    {
        for(i=0; i<innode; i++)
            x[i] = trainData[issamp][i];

        for(i=0; i<outnode; i++)
            yd[i] = label[issamp][i];

        //计算隐层各结点的激活值和隐层的输出值
        for(i=0; i<hidenode; i++)
        {
            o1[i] = 0.0;
            for(j=0; j<innode; j++)
                o1[i] = o1[i]+w[j][i]*x[j];              //w[][]为输入层到隐含层的网络权重，o1是隐层结点激活值
            x1[i] = 1.0/(1.0+exp(-o1[i]-b1[i]));  //x1是隐含层的输出
        }

        //计算输出层各结点的激活值和输出值
        for(i=0; i<outnode; i++)
        {
            o2[i] = 0.0;
            for(j=0; j<hidenode; j++)
                o2[i] = o2[i]+w1[j][i]*x1[j];                //w1[][]为隐层到输出层的权值，o2为输出层结点激活值
            x2[i] = 1.0/(1.0+exp(-o2[i]-b2[i]));        //x2为输出结点的输出
        }

        //得到了x2输出后接下来就要进行反向传播了

        //计算实际输出与期望输出的偏差，反向调节隐层到输出层的路径上的权值
        for(i=0; i<outnode; i++)
        {
            qq[i] = (yd[i]-x2[i]) * x2[i] * (1-x2[i]);      //输出节点j的偏差
            for(j=0; j<hidenode; j++)
                w1[j][i] = w1[j][i]+rate_w1*qq[i]*x1[j];   //隐层到输出层的路径上的权值
        }

        //继续反向传播调整输出层到隐层的各路径上的权值
        for(i=0; i<hidenode; i++)
        {
            pp[i] = 0.0;
            for(j=0; j<outnode; j++)
                pp[i] = pp[i]+qq[j]*w1[i][j];     
            pp[i] = pp[i]*x1[i]*(1.0-x1[i]);      //隐含层节点i的偏差

            for(k=0; k<innode; k++)
                w[k][i] = w[k][i] + rate_w*pp[i]*x[k];  //输出层到隐层的各路径上的权值
        }

        //调整允许的最大误差
        for(k=0; k<outnode; k++)  
        {  
            e+=fabs(yd[k]-x2[k])*fabs(yd[k]-x2[k]); //计算均方差  
        }  
        error=e/2.0; 

        //调整输出层各结点的阈值
        for(k=0; k<outnode; k++)  
            b2[k]=b2[k]+rate_b2*qq[k];

        //调整隐层各结点的阈值
        for(j=0; j<hidenode; j++)  
            b1[j]=b1[j]+rate_b1*pp[j];
    }
}

//Bp识别
double *recognize(double *p)
{
    double x[innode];//输入层的个输入值
    double o1[hidenode];//隐层结点激活值
    double o2[hidenode];//输出层结点激活值
    double x1[hidenode];//隐层向输出层的输入
    double x2[outnode];//输出结点的输出

    int i,j,k;

    for(i=0;i<innode;i++)  
        x[i]=p[i];  

    for(j=0;j<hidenode;j++)  
    {  
        o1[j]=0.0;  
        for(i=0;i<innode;i++)  
            o1[j]=o1[j]+w[i][j]*x[i]; //隐含层各单元激活值  
        x1[j]=1.0/(1.0+exp(-o1[j]-b1[j])); //隐含层各单元输出  
    }  

    for(k=0;k<outnode;k++)  
    {  
        o2[k]=0.0;  
        for(j=0;j<hidenode;j++)  
            o2[k]=o2[k]+w1[j][k]*x1[j]; 
        x2[k]=1.0/(1.0+exp(-o2[k]-b2[k]));  
    }  

    for(k=0;k<outnode;k++)  
    {  
        result[k]=x2[k];  
    }  
    return result;
}

//从文件夹读取数据
void readData(std::string filename, double data[][innode], int x)
{
    ifstream inData(filename, std::ios::in);
    int i,j;
    double dataLabel;
    for(i=0; i<x; i++)
    {
        for(j=0; j<innode; j++)
        {
            inData >>data[i][j];
        }
        inData >>dataLabel;
    }
    inData.close();
}

void changeData(double data[][innode], int x)
{
    double minNum,maxNum;
    int i,j;
    minNum = data[0][0];
    maxNum = data[0][0];
    for(i=0; i<x; i++)
    {
        for(j=0; j<innode; j++)
        {
            if(minNum > data[i][j])
                minNum = data[i][j];
            if(maxNum < data[i][j])
                maxNum = data[i][j];
        }
    }
    for(i=0; i<x; i++)
    {
        for(j=0; j<innode; j++)
            data[i][j] = (data[i][j]-minNum)/(maxNum-minNum);
    }
}