#include <stdio.h>  
#include <stdlib.h>
#include <cmath>

int w11=3, w12=2, w21=1, w22=4;
int v11=3, v12=5, v21=2, v22=1;
int x1=1,x2=2;

int main()
{
    int w11=3, w12=2, w21=1, w22=4;
    int v11=3, v12=5, v21=2, v22=1;
    int x1=1,x2=2;
    int u,v;
    u = w11 * x1 + w21 * x2 + 1;
    v = w12 * x1 + w22 * x2 + 1;
    float h1,h2,o1,o2;
    h1 = 1/(1+exp(-u));
    h2 = 1/(1+exp(-v));
    printf("h1=%f, h2=%f\n", h1, h2);
    float Net_o1, Net_o2; o1, o2;
    Net_o1 = v11 * h1 + v21 * h2 + 1;
    Net_o2 = v12 * h1 + v22 * h2 + 1;
    o1 = 1/(1+exp(-Net_o1));
    o2 = 1/(1+exp(-Net_o2));
    printf("o1=%f, o2=%f\n", o1, o2);
}

