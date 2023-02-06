#include "litamin2_refine.h"
#include <cstdio>
#include <iostream>


using namespace std;

PointCloudRefine::PointCloudRefine(/* args */){};
PointCloudRefine::~PointCloudRefine(){};


template <typename Registration>
//需要传入初值initial_guess,通过外部传入
Matrix4 test(Registration& reg, const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& target, const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& source,Matrix4 initial_guess) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>);

  double fitness_score = 0.0;

  std::cout << "source_cloud size: " << source->size() << std::endl;
  std::cout << "target_cloud size: " << target->size() << std::endl;
  
  // single run
  auto t1 = std::chrono::high_resolution_clock::now();
  // fast_gicp reuses calculated covariances if an input cloud is the same as the previous one
  // to prevent this for benchmarking, force clear source and target clouds
  
  reg.clearTarget();
  reg.clearSource();
  reg.setInputTarget(target);
  reg.setInputSource(source);
  //需要在里面加上第二项initial guess,形式为Matrix4
  reg.align(*aligned,initial_guess);
  auto t2 = std::chrono::high_resolution_clock::now();
  fitness_score = reg.getFitnessScore();
  double single = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() / 1e6;
  // std::cout << "align result: \n" << reg.getFinalTransformation() << std::endl; 
  std::cout << "single:" << single << "[msec] " << std::endl;
  return reg.getFinalTransformation();
};

Matrix4 PointCloudRefine::voxelnewton(pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud,pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud,Matrix4 initial_guess){
    //初始化target 和 source
    pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    //文件io 或者赋值操作

    int a = 0;
  // remove invalid points around origin  清理无效点
  source_cloud->erase(
    std::remove_if(source_cloud->begin(), source_cloud->end(), [=](const pcl::PointXYZ& pt) { return pt.getVector3fMap().squaredNorm() < 1e-3; }),
    source_cloud->end());
  target_cloud->erase(
    std::remove_if(target_cloud->begin(), target_cloud->end(), [=](const pcl::PointXYZ& pt) { return pt.getVector3fMap().squaredNorm() < 1e-3; }),
    target_cloud->end());

  // downsampling  降采样
  pcl::VoxelGrid<pcl::PointXYZ> voxelgrid;
  voxelgrid.setLeafSize(0.1f, 0.1f, 0.1f);

  pcl::PointCloud<pcl::PointXYZ>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZ>());
  voxelgrid.setInputCloud(target_cloud);
  voxelgrid.filter(*filtered);
  target_cloud = filtered;

  filtered.reset(new pcl::PointCloud<pcl::PointXYZ>());
  voxelgrid.setInputCloud(source_cloud);
  voxelgrid.filter(*filtered);
  source_cloud = filtered;

  std::cout << "target:" << target_cloud->size() << "[pts] source:" << source_cloud->size() << "[pts]" << std::endl;

  std::cout << "--- LiTAMIN2Point2VoxelNewton_test ---" << std::endl;
  litamin::LiTAMIN2Point2VoxelNewton<pcl::PointXYZ, pcl::PointXYZ> litamin2_test;
  // fast_gicp uses all the CPU cores by default
  litamin2_test.setNumThreads(4);
  litamin2_test.setResolution(3.0);
  litamin2_test.setMaxCorrespondenceDistance(1.0);
  litamin2_test.setTransformationEpsilon(1e-2);
  litamin2_test.setMaximumIterations(64);
  Matrxi4 Rt = test(litamin2_test, target_cloud, source_cloud,initial_guess);

//test应该返回计算出来的位姿然后再返回到estimator中的位姿中，因此需要修改函数类型
  return Rt;
  // std::cout << "--- fgicp_ceres ---" << std::endl;
  // fast_gicp::FastGICP<pcl::PointXYZ, pcl::PointXYZ> fgicp_ceres;
  // // fast_gicp uses all the CPU cores by default
  // fgicp_ceres.setNumThreads(8);
  // fgicp_ceres.setTransformationEpsilon(1e-2);
  // fgicp_ceres.setMaxCorrespondenceDistance(0.5);
  // fgicp_ceres.setLocalParameterization(true);
  // fgicp_ceres.setLSQType(fast_gicp::LSQ_OPTIMIZER_TYPE::CeresDogleg);
  // test(fgicp_ceres, target_cloud, source_cloud);

//   std::cout << std::endl << std::endl << "Visualizing ......." << std::endl;


//   pcl::visualization::PCLVisualizer vis;
//   vis.initCameraParameters();
//   vis.setCameraPosition(15.5219, 6.13405, 22.536,   8.258, -0.376825, -0.895555,    0.0226091, 0.961419, -0.274156);
//   vis.setCameraFieldOfView(0.523599);
//   vis.setCameraClipDistances(0.00522511, 50); 

//   vis.addPointCloud<pcl::PointXYZ>(source_cloud, 
//     pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(source_cloud, 255.0, 255.0, 255.0), 
//     "source");
//   vis.addPointCloud<pcl::PointXYZ>(target_cloud, 
//     pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(source_cloud, 0.0, 255.0, 0.0), 
//     "target");

//   pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>);
//   litamin2_test.align(*aligned);
//   vis.addPointCloud<pcl::PointXYZ>(aligned, 
//     pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(aligned, 0.0, 0.0, 255.0), 
//     "aligned");


//   vis.spin();

//   return ;

};