#pragma once

#include <cstdio>
#include <iostream>
#include <queue>
// #include <execinfo.h>
// #include <csignal>
#include <opencv2/opencv.hpp>
// #include "/home/shinan/catkin_ws/src/VINS-Fusion/vins_estimator/src/depth_process/persistence1d.hpp"
// #include<NumCpp.hpp>
#include<math.h>
#include<numeric>

using namespace std;
// using namespace p1d;

// static bool cmp(const pair<int,int>& a,const pair<int,int>& b){
//     return a.second>b.second;
// }

class Depth_process
{
public:
    Depth_process(/* args */);
    ~Depth_process();
    //通过对深度分类得到不同的图像块。
    //调用位置现在考虑有两个位置，一个是练习尝试时对整张图进行分割，之后对框出部分分割
    //得到的应该是多个图像块的质心
    int k_means(cv::Mat srcImage);
    cv::Mat dep_k_means(cv::Mat srcImage);
    
    cv::Mat dep_contours(cv::Mat srcImage);
    cv::Mat dep_afkmc2(cv::Mat srcImage);
    cv::Mat afkmc2(cv::Mat X , int k, int m = 200);
    float distance(int i,int j,std::vector<std::vector<float>> &d_store,std::vector<float> &X,std::vector<float> &center);
    int  randomHit(const vector<float>&oddsList,float& sum,const vector<float> &rank);
    cv::Mat ratiopro(cv::Mat srcImage );
    cv::Mat dep_region_grow(cv::Mat srcImage);
    cv::Mat Region_Growing(cv::Mat input, cv::Mat output, cv::Point fristseed, int value);
};
