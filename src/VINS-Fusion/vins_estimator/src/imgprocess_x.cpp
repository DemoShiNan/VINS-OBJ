// 创建图片修改测试节点，接受rawimg 
// 经过处理之后发布处理后的img

#include <stdio.h>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include "estimator/parameters.h"

//设置缓存以及锁
queue<sensor_msgs::ImageConstPtr> img0_buf;
queue<sensor_msgs::ImageConstPtr> img1_buf;
std::mutex m_buf;

void img0_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    m_buf.lock();
    img0_buf.push(img_msg);
    m_buf.unlock();
}

void img1_callback(const sensor_msgs::ImageConstPtr &img_msg)
{
    m_buf.lock();
    img1_buf.push(img_msg);
    m_buf.unlock();
}

//将Msg转化为可处理cv对象

cv::Mat getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg)
{
    cv_bridge::CvImageConstPtr ptr;
    if (img_msg->encoding == "8UC1")
    {
        sensor_msgs::Image img;
        img.header = img_msg->header;
        img.height = img_msg->height;
        img.width = img_msg->width;
        img.is_bigendian = img_msg->is_bigendian;
        img.step = img_msg->step;
        img.data = img_msg->data;
        img.encoding = "mono8";
        ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::MONO8);
    }
    else
    {
        ptr = cv_bridge::toCvCopy(img_msg, sensor_msgs::image_encodings::RGB8);
    }
    cv::Mat img = ptr->image.clone();
    return img;
}


//自定义的图像处理函数
cv::Mat process_img(cv::Mat img)
{
    return img;
}


//将image0_x和image1_x还原成Msg


int main(int argc, char **argv)
{
    ros::init(argc, argv, "img_process");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

#ifdef EIGEN_DONT_PARALLELIZE
    ROS_DEBUG("EIGEN_DONT_PARALLELIZE");
#endif
    // 从无人机上直接订阅图片消息
    ros::Subscriber sub_img0 = n.subscribe<sensor_msgs::Image>("/iris_0/stereo_camera/left/image_raw", 100, img0_callback);
    ros::Subscriber sub_img1 = n.subscribe<sensor_msgs::Image>("/iris_0/stereo_camera/right/image_raw", 100, img1_callback);

    //创建发布者 发布给自定义的TOPIC
    ros::Publisher pub_img0 = n.advertise<sensor_msgs::Image>("IMAGE0_x_topic", 100);
    ros::Publisher pub_img1 = n.advertise<sensor_msgs::Image>("IMAGE1_x_topic", 100);

    cv::Mat image0 ,image1;
    cv::Mat image0_x, image1_x;
    std_msgs::Header header;

    //发布
    ros::Rate loop_rate(10);
    while (ros::ok())
    {
        //读取并修改图片
        m_buf.lock();
        if (!img0_buf.empty() && !img1_buf.empty())
        {
            header = img0_buf.front()->header;
            image0 = getImageFromMsg(img0_buf.front());
            img0_buf.pop();
            image1 = getImageFromMsg(img1_buf.front());
            img1_buf.pop();

            //自定义的图像处理
            image0_x = process_img(image0);
            image1_x = process_img(image1);
            //printf("find img0 and img1\n");
        }
        m_buf.unlock();

        pub_img0.publish(cv_bridge::CvImage(header, "rgb8", image0_x).toImageMsg());
        pub_img1.publish(cv_bridge::CvImage(header, "rgb8", image1_x).toImageMsg());
        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}