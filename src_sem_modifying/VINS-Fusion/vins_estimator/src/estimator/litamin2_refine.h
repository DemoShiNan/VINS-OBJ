#pragma once

#include <cstdio>
#include <iostream>
#include <chrono>

// #include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/approximate_voxel_grid.h>

#include <pcl/registration/ndt.h>
#include <pcl/registration/gicp.h>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/point_cloud_color_handlers.h>


#include <fast_gicp/gicp/fast_gicp.hpp>
#include "fast_gicp/gicp/impl/fast_gicp_impl.hpp"
#include <litamin2/litamin2point2voxelnewton.hpp>


using namespace std;

class PointCloudRefine
{
public:
    PointCloudRefine(/* args */);
    ~PointCloudRefine();
    
    Matrix4 voxelnewton(pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud,pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud);
};