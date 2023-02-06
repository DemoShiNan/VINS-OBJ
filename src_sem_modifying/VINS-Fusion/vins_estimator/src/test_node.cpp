#include <stdio.h>
#include <string>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include "ros/ros.h"
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include "depth_process/depth_process.h"
#include "boost/function.hpp"
#include <chrono>

using namespace std;
using namespace cv;
using namespace sensor_msgs;
using namespace chrono;

queue<sensor_msgs::ImageConstPtr> depth_buf;
std::mutex d_buf;

Depth_process Dp;

void depth_callback(const sensor_msgs::ImageConstPtr &depth_msg)
{
    d_buf.lock();
    depth_buf.push(depth_msg);
    d_buf.unlock();
}

cv::Mat getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg)
{
    cv_bridge::CvImageConstPtr ptr;
    if (img_msg->encoding == "16UC1")
    {
        sensor_msgs::Image img;
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "16UC1";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::TYPE_16UC1);
    }
    else{
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::TYPE_16UC1);
    }
    cv::Mat img = ptr->image.clone();
    return img;
}


void sync_process()
{
    while(1)
    {
        d_buf.lock();
        cv::Mat depth_img;
        // std_msgs::Header header;
        // double time = 0.0;
        if (!depth_buf.empty()){
            // time = depth_buf.front()->header.stamp.toSec();
            // header = depth_buf.front()->header;
            // printf("got pic\n");
            depth_img = getImageFromMsg(depth_buf.front());
            depth_buf.pop();
        }
        d_buf.unlock();

        // Dp.dep_contours(depth_img);]]
        auto start   = system_clock::now();
        Dp.dep_afkmc2(depth_img);
        auto mid   = system_clock::now();
        Dp.dep_k_means(depth_img); 
        auto end   = system_clock::now();
        auto duration1 = duration_cast<microseconds>(mid - start);
        auto duration2 = duration_cast<microseconds>(end - mid);

        cout <<  "afkmc2花费了"     << double(duration1.count()) * microseconds::period::num / microseconds::period::den     << "秒" << endl;
        cout <<  "k-means++花费了"     << double(duration2.count()) * microseconds::period::num / microseconds::period::den     << "秒" << endl;
        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
    
}
int main(int argc, char ** argv)
{
    ros::init(argc, argv, "test_node");
    ros::NodeHandle nh;
    //"/camera/infra1/image_rect_raw" 
    ros::Subscriber sub_depth = nh.subscribe("/camera/aligned_depth_to_color/image_raw", 100, depth_callback);
    std::thread sync_thread{sync_process};

    ros::spin();
    return 0;
}