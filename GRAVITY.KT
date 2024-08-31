#include <chrono> // 包含chrono库
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<iostream>
#include<fstream>
#include<string>
#include <vector>
#include <sstream>
#include <math.h>
#include <algorithm>
#include <cmath>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <cuda_runtime.h>
using namespace std::chrono; // 命名空间
using namespace std;
#define N jing*wei
#define jing 241
#define wei 273
__device__ __host__ double ff(double L, double B, double L1, double B2, double g, int n)
{
    double PI = 3.14159265358979323846;//定义Π
    double R = 6381372;
    double r = R + 160;
    double   cosw, sinw, cotw, Sr, Srr, Sw, Sww, Srw, Cr, C1w, C1ww, C2w, C2ww, C3w, C3rw, C4w, C4ww, C5w, C5rw;
    double cosa, sina, cota, Sxx, Syy, Szz, Sxy, Sxz, Syz, l;
    cosw = sin(B * PI / 180) * sin(B2 * PI / 180) + cos(B * PI / 180) * cos(B2 * PI / 180) * cos((L - L1) * PI / 180);
    sinw = sqrt(1 - pow(cosw, 2));
    l = sqrt(pow(r, 2) + pow(R, 2) - 2 * R * r * cosw);
    Sr = -2 * R * (r - R * cosw) / pow(l, 3) - R / pow(r, 2) - 3 * R / (l * r) + 6 * R * l / pow(r, 3) + pow(R, 2) * cosw * (13 + 6 * log((r - R * cosw + l) / (2 * r))) / pow(r, 3);
    Srr = R / pow(l, 3) + 6 * R * pow((r - R * cosw), 2) / pow(l, 5) + 2 * R / pow(r, 3) + 9 * R / (l * r * r) - 18 * l * R / pow(r, 4) - 45 * R * R * cosw / pow(r, 4) - 3 * R * R * cosw / (l * l * l * r) - 18 * R * R * cosw * log((r - R * cosw + l) / (2 * r)) / pow(r, 4);
    Sw = -2 * pow(R, 2) * r * sinw / pow(l, 3) - 3 * pow(R, 2) * sinw / (l * r) + pow(R, 2) * sinw * (5 + 3 * log((r - R * cosw + l) / (2 * r))) / pow(r, 2) - 3 * pow(R, 3) * cosw * sinw / (pow(r, 2) * (r - R * cosw + l)) - 3 * pow(R, 3) * cosw * sinw / (r * l * (r - R * cosw + l));
    Sww = 6 * pow(R, 3) * pow(r, 2) * pow(sinw, 2) / pow(l, 5) + (3 * pow(R, 3) * pow(sinw, 2) - 2 * pow(R, 2) * r * cosw) / pow(l, 3) - 3 * pow(R, 2) * cosw / (l * r) + pow(R, 2) * cosw * (5 + 3 * log((r - R * cosw + l) / (2 * r))) / pow(r, 2) + 3 * pow(R, 3) * (l + r) * (2 * pow(sinw, 2) - pow(cosw, 2)) / (pow(r, 2) * l * (r - R * cosw + l)) + 3 * pow(R, 4) * cosw * sinw * sinw * pow((l + r), 2) / (r * r * l * l * (r - R * cosw + l) * (r - R * cosw + l)) + 3 * pow(R, 4) * cosw * sinw * sinw / (pow(l, 3) * (r - R * cosw + l));
    Srw = pow(R, 2) * sinw / pow(l, 3) + (6 * R * R * r * sinw * (r - R * cosw)) / pow(l, 5) + 6 * R * R * sinw / (l * r * r) - 13 * R * R * sinw / pow(r, 3) - 6 * R * R * sinw * log((r - R * cosw + l) / (2 * r)) / pow(r, 3) + 6 * R * R * R * sinw * cosw / ((r - R * cosw + l) * pow(r, 3)) + 6 * R * R * R * sinw * cosw / ((r - R * cosw + l) * pow(r, 2) * l);
    Cr = 1 / r;
    sina = sin((L1- L) * PI / 180) * cos(B2 * PI / 180) / sinw;
    cosa = (cos(B * PI / 180) * sin(B2 * PI / 180) - sin(B * PI / 180) * cos(B2 * PI / 180) * cos((L1 - L) * PI / 180)) / sinw;
    cota = cosa / sina;
    cotw = cosw / sinw;
    C1w = cotw * pow(cosa, 2) / pow(r, 2);
    C1ww = sina * sina / (r * r);
    C2w = cotw * cosa * sina / pow(r, 2);
    C2ww = -cosa * sina / pow(r, 2);
    C3w = -sina / pow(r, 2);
    C3rw = sina / r;
    C4w = cotw * pow(sina, 2) / pow(r, 2);
    C4ww = cosa * cosa / (r * r);
    C5w = cosa / pow(r, 2);
    C5rw = -cosa / r;
    Sxx = C1w * Sw + Cr * Sr + C1ww * Sww;
    Syy = C4w * Sw + Cr * Sr + C4ww * Sww;
    Szz = Srr;
    Sxy = C2w * Sw + C2ww * Sww;
    Sxz = C3w * Sw + C3rw * Srw;
    Syz = C5rw * Srw + C5w * Sw;
    if (n == 1)
    {
        return g * R * Sxx * cos(B2 * PI / 180) / (4 * PI) * PI * PI / (180 * 180);
    }
    else if (n == 2)
    {
        return g * R * Syy * cos(B2 * PI / 180) / (4 * PI) * PI * PI / (180 * 180);
    }
    else if (n == 3)
    {
        return g * R * Szz * cos(B2 * PI / 180) / (4 * PI) * PI * PI / (180 * 180);
    }
    else if (n == 4)
    {
        return g * R * Sxy * cos(B2 * PI / 180) / (4 * PI) * PI * PI / (180 * 180);
    }
    else if (n == 5)
    {
        return g * R * Sxz * cos(B2 * PI / 180) / (4 * PI) * PI * PI / (180 * 180);
    }
    else if (n == 6)
    {
        return g * R * Syz * cos(B2 * PI / 180) / (4 * PI) * PI * PI / (180 * 180);
    }
}
__global__ void addArrays(double* L1, double* B2, double* g, double* zong1, double* zong2, double* zong3, double* zongxy, double* zongxz, double* zongyz) 
{
    double gauss_30coef[30][2] = { {  0.99689348407465, 0.00796819249616665    },
{ 0.983668123279747, 0.0184664683110908     }  ,
{ 0.960021864968308, 0.0287847078833233     }  ,
{ 0.926200047429274, 0.038799192569627      }  ,
{ 0.882560535792053, 0.048402672830594      }  ,
{ 0.829565762382768, 0.0574931562176191     }  ,
{ 0.767777432104826, 0.0659742298821805     }  ,
{ 0.697850494793316, 0.0737559747377051     }  ,
{ 0.620526182989243, 0.0807558952294202     }  ,
{  0.53662414814202, 0.086899787201083      }  ,
{  0.447033769538089, 0.0921225222377861      },
{  0.352704725530878, 0.0963687371746442      },
{  0.25463692616789, 0.0995934205867953      } ,
{  0.153869913608584, 0.101762389748406      } ,
{  0.0514718425553177, 0.102852652893559     } ,
{  -0.0514718425553177, 0.102852652893559   }  ,
{  -0.153869913608584, 0.101762389748406    }  ,
{  -0.25463692616789, 0.0995934205867953    }  ,
{  -0.352704725530878, 0.0963687371746442   }  ,
{  -0.447033769538089, 0.0921225222377861   }  ,
{  -0.53662414814202, 0.086899787201083     }  ,
{  -0.620526182989243, 0.0807558952294202   }  ,
{  -0.697850494793316, 0.0737559747377051   }  ,
{  -0.767777432104826, 0.0659742298821805   }  ,
{  -0.829565762382768, 0.0574931562176191   }  ,
{  -0.882560535792053, 0.048402672830594    }  ,
{  -0.926200047429274, 0.038799192569627    }  ,
{  -0.960021864968308, 0.0287847078833233   }  ,
{  -0.983668123279747, 0.0184664683110908   }  ,
{  -0.99689348407465, 0.00796819249616665   } };
    double xiaobo_23[6][2] = {
{0.083333333,	0.1875},
{0.25,	0.125         },
{0.416666667,	0.1875},
{0.583333333,	0.1875},
{0.75,	0.125         },
{0.916666667,	0.1875},

    };

    double xiaobo_54[64][2] = {
        {0.0078125,0.016927083},
        {0.0234375,0.014322917},
        {0.0390625,0.014322917},
        {0.0546875,0.016927083},
        {0.0703125,0.016927083},
        {0.0859375,0.014322917},
        {0.1015625,0.014322917},
        {0.1171875,0.016927083},
        {0.1328125,0.016927083},
        {0.1484375,0.014322917},
        {0.1640625,0.014322917},
        {0.1796875,0.016927083},
        {0.1953125,0.016927083},
        {0.2109375,0.014322917},
        {0.2265625,0.014322917},
        {0.2421875,0.016927083},
        {0.2578125,0.016927083},
        {0.2734375,0.014322917},
        {0.2890625,0.014322917},
        {0.3046875,0.016927083},
        {0.3203125,0.016927083},
        {0.3359375,0.014322917},
        {0.3515625,0.014322917},
        {0.3671875,0.016927083},
        {0.3828125,0.016927083},
        {0.3984375,0.014322917},
        {0.4140625,0.014322917},
        {0.4296875,0.016927083},
        {0.4453125,0.016927083},
        {0.4609375,0.014322917},
        {0.4765625,0.014322917},
        {0.4921875,0.016927083},
        {0.5078125,0.016927083},
        {0.5234375,0.014322917},
        {0.5390625,0.014322917},
        {0.5546875,0.016927083},
        {0.5703125,0.016927083},
        {0.5859375,0.014322917},
        {0.6015625,0.014322917},
        {0.6171875,0.016927083},
        {0.6328125,0.016927083},
        {0.6484375,0.014322917},
        {0.6640625,0.014322917},
        {0.6796875,0.016927083},
        {0.6953125,0.016927083},
        {0.7109375,0.014322917},
        {0.7265625,0.014322917},
        {0.7421875,0.016927083},
        {0.7578125,0.016927083},
        {0.7734375,0.014322917},
        {0.7890625,0.014322917},
        {0.8046875,0.016927083},
        {0.8203125,0.016927083},
        {0.8359375,0.014322917},
        {0.8515625,0.014322917},
        {0.8671875,0.016927083},
        {0.8828125,0.016927083},
        {0.8984375,0.014322917},
        {0.9140625,0.014322917},
        {0.9296875,0.016927083},
        {0.9453125,0.016927083},
        {0.9609375,0.014322917},
        {0.9765625,0.014322917},
        {0.9921875,0.016927083},
    };

    double PI = 3.14159265358979323846;//定义Π
    double R = 6381372;
    double r;
    //r = R+160 ;
    double L, B, dm, l;
    //float x[4270], y[4270];
    double   cosw, sinw, cotw, Sr, Srr, Sw, Sww, Srw, Cr, C1w, C1ww, C2w, C2ww, C3w, C3rw, C4w, C4ww, C5w, C5rw;
    double cosa, sina, cota, Sxx, Syy, Szz, Sxy, Sxz, Syz;
    double dL11 = 0.0166666666667, dB22 = 0.0146666666666667;
    double x1, x2, y1, y2;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < 1) {
        L = L1[tid] , B = B2[tid] ;
        double xx = 0, yy = 0, zz = 0, xy = 0, xz = 0, yz = 0;
        for (int a = 0; a < 2; a++)
        {
            x2 = L1[a] + dL11 / 2;
            x1 = L1[a] - dL11 / 2;
            y2 = B2[a] + dB22 / 2;
            y1 = B2[a] - dB22 / 2;
            double s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0, aaaa = 1 ; // 积分结果
            if (a=tid)//(L1[tid] - L1[a] < 0.0167*2) && (L1[a] - L1[tid] < 0.0167*2) && (B2[tid] - B2[a] < 0.0167*2) && (B2[a] - B2[tid] < 0.0167*2))// || L1[tid]-L1[a] > 2 || L1[tid] - L1[a] < -2 || B2[tid] - B2[a] > 2 || B2[tid] - B2[a] < -2)
            {
               // //double gy1 = 0, gy2 = 0, gx1 = 0, gx2 = 0, gg;
               // //double gx2y2, gx2y1, gx1y2, gx1y1;
               // //double g1, g2, g11, g12, g22;
               // //gg = g[a];
               // //gx1 = g[a - 1], gx2 = g[a + 1], gy1 = g[a + jing], gy2 = g[a - jing];
               // //gx2y2 = g[a + 1 - jing], gx2y1 = g[a + 1 + jing], gx1y2 = g[a - 1 - jing], gx1y1 = g[a - 1 + jing];
               // //g1 = (gx2 - gx1) / (2 * 1600);
               // //g2 = (gy2 - gy1) / (2 * 1600);
               // //s5 += -g1;
               // //s6 += -g2;
               // for (int i = 0; i < 30; i++)
               // {
               //     for (int j = 0; j < 30; j++)
               //     {
               //         double x = ((x2 - x1) / 2) * gauss_30coef[i][0] + ((x2 + x1) / 2);
               //         double y = ((y2 - y1) / 2) * gauss_30coef[j][0] + ((y2 + y1) / 2);
               //         //double x = (x2 - x1) * xiaobo_54[i][0] + x1;
               //         //double y = (y2 - y1) * xiaobo_54[j][0] + y1;
               //         double w = gauss_30coef[i][1] * gauss_30coef[j][1];
               //         //s1 += w * ff(L, B, x, y, g[a], 1) * (x2 - x1) * (y2 - y1);
               //         //s2 += w * ff(L, B, x, y, g[a], 2) * (x2 - x1) * (y2 - y1);
               //         s3 += w * ff(L, B, x, y, g[a], 3) * (x2 - x1) * (y2 - y1)/4;
               //         //s4 += w * ff(L, B, x, y, g[a], 4) * (x2 - x1) * (y2 - y1);
               //         //s5 += w * ff(L, B, x, y, g[b], 5) * (x2 - x1) * (y2 - y1) ;
               //         //s6 += w * ff(L, B, x, y, g[b], 6) * (x2 - x1) * (y2 - y1) ;                            
               //     }//* g[b]  
               // }
            }
            else if((L1[tid] - L1[a] < 0.0167 * aaaa) && (L1[a] - L1[tid] < 0.0167 * aaaa) && (B2[tid] - B2[a] < 0.0147 * aaaa) && (B2[a] - B2[tid] < 0.0147 * aaaa))
            {
                for (int i = 0; i < 1; i++)
                {
                    for (int j = 0; j < 1; j++)
                    {
                        //double x = ((x2 - x1) / 2) * gauss_30coef[i][0] + ((x2 + x1) / 2);
                        //double y = ((y2 - y1) / 2) * gauss_30coef[j][0] + ((y2 + y1) / 2);
                        double x = (x2 - x1) * xiaobo_23[i][0] + x1;
                        double y = (y2 - y1) * xiaobo_23[j][0] + y1;
                        double w = xiaobo_23[i][1] * xiaobo_23[j][1];
                        s1 += w * ff(L, B, x, y, g[a], 1) * (x2 - x1) * (y2 - y1);
                        s2 += w * ff(L, B, x, y, g[a], 2) * (x2 - x1) * (y2 - y1);
                        s3 += w * ff(L, B, x, y, g[a], 3) * (x2 - x1) * (y2 - y1);
                        s4 += w * ff(L, B, x, y, g[a], 4) * (x2 - x1) * (y2 - y1);
                        //s5 += w * ff(L, B, x, y, g[b], 5) * (x2 - x1) * (y2 - y1) ;
                        //s6 += w * ff(L, B, x, y, g[b], 6) * (x2 - x1) * (y2 - y1) ;                            
                    }//* g[b]  
                }

                //for (int i = 0; i < 6; i++) {
                //    for (int j = 0; j < 6; j++) {
                //        double x = (x2 - x1) * xiaobo_23[i][0] + x1;
                //        double y = (y2 - y1) * xiaobo_23[j][0] + y1;
                //        double w = xiaobo_23[i][1] * xiaobo_23[j][1];
                //        //s1 += w * ff(L, B, x, y, g[a], 1) * (x2 - x1) * (y2 - y1); // 计算积分值
                //        //s2 += w * ff(L, B, x, y, g[a], 2) * (x2 - x1) * (y2 - y1);
                //        s3 += w * ff(L, B, x, y, g[a], 3) * (x2 - x1) * (y2 - y1);
                //        //s4 += w * ff(L, B, x, y, g[a], 4) * (x2 - x1) * (y2 - y1);
                //        //s5 += w * ff(L, B, x, y, g[a], 5) * (x2 - x1) * (y2 - y1);
                //        //s6 += w * ff(L, B, x, y, g[a], 6) * (x2 - x1) * (y2 - y1);
                //    }
                //}
            }

            xx = xx + s1;
            yy = yy + s2;
            zz = zz + s3;
            xy = xy + s4;
            xz = xz + s5;
            yz = yz + s6;
        }
        double s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0; // 积分结果
        {
            int b = tid;
            x2 = L1[b] + dL11 / 2;
            x1 = L1[b] - dL11 / 2;
            y2 = B2[b] + dB22 / 2;
            y1 = B2[b] - dB22 / 2;
            double s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0; // 积分结果
            if (b == tid)
            {
                double gy1 = 0, gy2 = 0, gx1 = 0, gx2 = 0, gg;
                double gx2y2, gx2y1, gx1y2, gx1y1;
                double g1, g2, g11, g12, g22;
                gg = g[b];
                gx1 = g[b - 1], gx2 = g[b + 1], gy1 = g[b + jing], gy2 = g[b - jing];
                gx2y2 = g[b + 1 - jing], gx2y1 = g[b + 1 + jing], gx1y2 = g[b - 1 - jing], gx1y1 = g[b - 1 + jing];
                g1 = (gx2 - gx1) / (2 * 1600);
                g2 = (gy2 - gy1) / (2 * 1600);
                s5 += -g1;
                s6 += -g2;
                for (int i = 0; i < 1; i++)
                {
                      for (int j = 0; j < 1; j++)
                      {
                          //double x = ((x2 - x1) / 2) * gauss_30coef[i][0] + ((x2 + x1) / 2);
                          //double y = ((y2 - y1) / 2) * gauss_30coef[j][0] + ((y2 + y1) / 2);
                          double x = (x2 - x1) * xiaobo_54[i][0] + x1;
                          double y = (y2 - y1) * xiaobo_54[j][0] + y1;
                          double w = xiaobo_54[i][1] * xiaobo_54[j][1];
                          s1 += w * ff(L, B, x, y, g[b], 1) * (x2 - x1) * (y2 - y1) ;
                          s2 += w * ff(L, B, x, y, g[b], 2) * (x2 - x1) * (y2 - y1) ;
                          s3 += w * ff(L, B, x, y, g[b], 3) * (x2 - x1) * (y2 - y1) ;
                          s4 += w * ff(L, B, x, y, g[b], 4) * (x2 - x1) * (y2 - y1) ;
                          //s5 += w * ff(L, B, x, y, g[b], 5) * (x2 - x1) * (y2 - y1) ;
                          //s6 += w * ff(L, B, x, y, g[b], 6) * (x2 - x1) * (y2 - y1) ;                            
                      }//* g[b]  
                }
            }
            xx = xx + s1;
            yy = yy + s2;
            zz = zz + s3;
            xy = xy + s4;
            xz = xz + s5;
            yz = yz + s6;
        }




        zong1[tid] = xx;
        zong2[tid] = yy;
        zong3[tid] = zz;
        zongxy[tid] = xy;
        zongxz[tid] = xz;
        zongyz[tid] = yz;
    }
}

int main() 
{
    auto start_time = high_resolution_clock::now(); // 获取开始时间点

    
    double* L1 = new double[N], * B2 = new double[N], * g = new double[N],
        * zong1 = new double[N], * zong2 = new double[N], * zong3 = new double[N],
        * zongxy = new double[N], * zongxz = new double[N], * zongyz = new double[N],


    double* dev_L1, * dev_B2, * dev_g, * dev_zong1, * dev_zong2, * dev_zong3, * dev_zongxy, * dev_zongxz, * dev_zongyz, * dev_L11, * dev_B22 ;

    fstream  f("4x4.txt");//创建一个fstream文件流对象

    vector<string> words; //创建一个vector<string>对象
    string  line; //保存读入的每一行
    while (getline(f, line))//会自动把\n换行符去掉 
    {
        words.push_back(line);
    }
    for (int i = 0; i < N; i++) { // 遍历字符串数组
        string str = words[i];
        stringstream ss(str); // 使用stringstream将字符串转为流
        float num1, num2, num3;
        ss >> num1 >> num2 >> num3; // 从流中依次读取三个浮点型数字
        L1[i] = num1; // 将读取的数字存入对应数组中
        B2[i] = num2;
        g[i] = num3 * 10000;
    }
    for (int i = 0; i < wei; i++)
    {
        for (int j = 0; j < jing; j++)
        {
            L1[jing * i + j] = 0.0166666666666666666667 * (j + 1);
        }
    }
    for (int i = 0; i < jing; i++)
    {
        for (int j = 0; j < wei; j++)
        {
            B2[jing * j + i] = 0.0146666666666667 - 0.0146666666666667 * (j + 1);
        }
    }

    //for (int i = 0; i < N1; i++)
    //{
    //    cout << L11[i]<<endl;
    //}
    //cout <<  endl;

    //for (int i = 0; i < N; i++)
    //{
    //    cout << L1[i] << endl;
    //}
    //cout << L11[N-1]<<"        " << L1[N1-1] << "        " << N  << "        " << N1  << endl;

    //for (int i = 0; i < words.size(); i++)
        //cout << "第" << i + 1 << "行：" << L1[i] << "     " << B2[i] << "     " << g[i] << endl;
    // Allocate memory on device在设备上分配内存
    cudaMalloc((void**)&dev_L1, N * sizeof(double));
    cudaMalloc((void**)&dev_B2, N * sizeof(double));
    cudaMalloc((void**)&dev_g, N * sizeof(double));
    cudaMalloc((void**)&dev_zong1, N * sizeof(double));
    cudaMalloc((void**)&dev_zong2, N * sizeof(double));
    cudaMalloc((void**)&dev_zong3, N * sizeof(double));
    cudaMalloc((void**)&dev_zongxy, N * sizeof(double));
    cudaMalloc((void**)&dev_zongxz, N * sizeof(double));
    cudaMalloc((void**)&dev_zongyz, N * sizeof(double));
    // Copy input data to device将输入数据复制到设备
    cudaMemcpy(dev_L1, L1, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_B2, B2, N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_g, g, N * sizeof(double), cudaMemcpyHostToDevice);
    // Launch kernel启动内核
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    addArrays << <blocksPerGrid, threadsPerBlock >> > (dev_L1, dev_B2, dev_g, dev_zong1, dev_zong2, dev_zong3, dev_zongxy, dev_zongxz, dev_zongyz);
    // Copy result back to host将结果复制回主机blocksPerGrid, threadsPerBlock
    cudaMemcpy(zong1, dev_zong1, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(zong2, dev_zong2, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(zong3, dev_zong3, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(zongxy, dev_zongxy, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(zongxz, dev_zongxz, N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(zongyz, dev_zongyz, N * sizeof(double), cudaMemcpyDeviceToHost);
    ofstream file("Txx.txt");
    ofstream fout1("Tyy.txt");
    ofstream file3("Tzz.txt");
    ofstream file4("Tsum.txt");
    ofstream file5("gg.txt");
    ofstream file6("Txy.txt");
    ofstream file7("Txz.txt");
    ofstream file8("Tyz.txt");

    for (int i = 0; i < N; i++)
    {
        file << zong1[i] << endl;
        fout1 << zong2[i] << endl;
        file3 << zong3[i] << endl;
        file4 << zong1[i] + zong2[i] + zong3[i] << endl;
        file6 << zongxy[i] << endl;
        file7 << zongxz[i] << endl;
        file8 << zongyz[i] << endl;
    }
    for (int i = 0; i < N; i++)
    {
        file5 << g[i] * 0.000000001 << endl;
    }
    file.close();
    fout1.close();
    file3.close();
    file4.close();
    file5.close();
    file6.close();
    file7.close();
    file8.close();
    // Free memory on device释放设备上的内存
    cudaFree(dev_L1);
    cudaFree(dev_B2);
    cudaFree(dev_g);
    cudaFree(dev_zong1);
    cudaFree(dev_zong2);
    cudaFree(dev_zong3);
    cudaFree(dev_zongxy);
    cudaFree(dev_zongxz);
    cudaFree(dev_zongyz);
    auto end_time = high_resolution_clock::now(); // 获取结束时间点
    auto duration = duration_cast<microseconds>(end_time - start_time); // 计算持续时间
    int sec;
    sec = duration.count() * 1e-6;
    cout << "计算花费了 " << sec << " 秒" << endl;
    return 0;
}
