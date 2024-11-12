/*
 * Copyright (C) 2021 by Autonomous Driving Group, Shanghai AI Laboratory
 * Limited. All rights reserved.
 * Yan Guohang <yanguohang@pjlab.org.cn>
 */

#include <Eigen/Core>
#include <boost/filesystem.hpp>
#include <pangolin/pangolin.h>
#include <pcl/common/transforms.h>
#include <pcl/conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/octree/octree_search.h>
#include <pcl/point_cloud.h>
#include <stdio.h>
#include <sys/ioctl.h>
#include <termios.h>
#include <unistd.h>

#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>

#include "extrinsic_param.hpp"

using namespace std;

#define GL_GPU_MEM_INFO_CURRENT_AVAILABLE_MEM_NVX 0x9049
#define MAX_RADAR_TIME_GAP 15 * 1e6

pangolin::GlBuffer *source_vertexBuffer_;
pangolin::GlBuffer *source_colorBuffer_;
pangolin::GlBuffer *target_vertexBuffer_;
pangolin::GlBuffer *target_colorBuffer_;

double cali_scale_degree_ = 0.3;
double cali_scale_trans_ = 0.06;
static Eigen::Matrix4d calibration_matrix_ = Eigen::Matrix4d::Identity();
static Eigen::Matrix4d orign_calibration_matrix_ = Eigen::Matrix4d::Identity();
std::vector<Eigen::Matrix4d> modification_list_;
bool display_mode_ = false;
int point_size_ = 2;

struct RGB {
  unsigned char r;
  unsigned char g;
  unsigned char b;
};

struct PointCloudBbox {
  int min_x = 0;
  int min_y = 0;
  int min_z = 0;

  int max_x = 0;
  int max_y = 0;
  int max_z = 0;
};

pcl::PointCloud<pcl::PointXYZI>::Ptr
    cloudLidar(new pcl::PointCloud<pcl::PointXYZI>);
;
pcl::octree::OctreePointCloudSearch<pcl::PointXYZI>::Ptr
    all_octree(new pcl::octree::OctreePointCloudSearch<pcl::PointXYZI>(0.05));

bool kbhit() {
  termios term;
  tcgetattr(0, &term);
  termios term2 = term;
  term2.c_lflag &= ~ICANON;
  tcsetattr(0, TCSANOW, &term2);
  int byteswaiting;
  ioctl(0, FIONREAD, &byteswaiting);
  tcsetattr(0, TCSANOW, &term);
  return byteswaiting > 0;
}

void CalibrationInit(Eigen::Matrix4d json_param) {
  Eigen::Matrix4d init_cali;
  init_cali << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1;
  calibration_matrix_ = json_param; //读取初始化矩阵
  orign_calibration_matrix_ = json_param;//记录原始的标定矩阵
  modification_list_.reserve(12);
  for (int32_t i = 0; i < 12; i++) {
    std::vector<int> transform_flag(6, 0);
    transform_flag[i / 2] = (i % 2) ? (-1) : 1;
    Eigen::Matrix4d tmp = Eigen::Matrix4d::Identity();
    Eigen::Matrix3d rot_tmp;
    rot_tmp =
        Eigen::AngleAxisd(transform_flag[0] * cali_scale_degree_ / 180.0 * M_PI,
                          Eigen::Vector3d::UnitX()) *
        Eigen::AngleAxisd(transform_flag[1] * cali_scale_degree_ / 180.0 * M_PI,
                          Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(transform_flag[2] * cali_scale_degree_ / 180.0 * M_PI,
                          Eigen::Vector3d::UnitZ());
    tmp.block(0, 0, 3, 3) = rot_tmp;
    tmp(0, 3) = transform_flag[3] * cali_scale_trans_;
    tmp(1, 3) = transform_flag[4] * cali_scale_trans_;
    tmp(2, 3) = transform_flag[5] * cali_scale_trans_;
    modification_list_[i] = tmp;
  }
  std::cout << "=>Calibration scale Init!\n";
}

void CalibrationScaleChange() {
  Eigen::Matrix4d init_cali;
  init_cali << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1;
  modification_list_.reserve(12);
  for (int32_t i = 0; i < 12; i++) {
    std::vector<int> transform_flag(6, 0);
    transform_flag[i / 2] = (i % 2) ? (-1) : 1;
    Eigen::Matrix4d tmp = Eigen::Matrix4d::Identity();
    Eigen::Matrix3d rot_tmp;
    rot_tmp =
        Eigen::AngleAxisd(transform_flag[0] * cali_scale_degree_ / 180.0 * M_PI,
                          Eigen::Vector3d::UnitX()) *
        Eigen::AngleAxisd(transform_flag[1] * cali_scale_degree_ / 180.0 * M_PI,
                          Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(transform_flag[2] * cali_scale_degree_ / 180.0 * M_PI,
                          Eigen::Vector3d::UnitZ());
    tmp.block(0, 0, 3, 3) = rot_tmp;
    tmp(0, 3) = transform_flag[3] * cali_scale_trans_;
    tmp(1, 3) = transform_flag[4] * cali_scale_trans_;
    tmp(2, 3) = transform_flag[5] * cali_scale_trans_;
    modification_list_[i] = tmp;
  }
  std::cout << "=>Calibration scale update done!\n";
}

void saveResult(const int &frame_id) {
  std::string file_name =
      "lidar2lidar_extrinsic_" + std::to_string(frame_id) + ".txt";
  std::ofstream fCalib(file_name);
  if (!fCalib.is_open()) {
    std::cerr << "open file " << file_name << " failed." << std::endl;
    return;
  }
  fCalib << "Extrinsic:" << std::endl;
  fCalib << "R:\n"
         << calibration_matrix_(0, 0) << " " << calibration_matrix_(0, 1) << " "
         << calibration_matrix_(0, 2) << "\n"
         << calibration_matrix_(1, 0) << " " << calibration_matrix_(1, 1) << " "
         << calibration_matrix_(1, 2) << "\n"
         << calibration_matrix_(2, 0) << " " << calibration_matrix_(2, 1) << " "
         << calibration_matrix_(2, 2) << std::endl;
  fCalib << "t: " << calibration_matrix_(0, 3) << " "
         << calibration_matrix_(1, 3) << " " << calibration_matrix_(2, 3)
         << std::endl;

  fCalib << "************* json format *************" << std::endl;
  fCalib << "Extrinsic:" << std::endl;
  fCalib << "[" << calibration_matrix_(0, 0) << "," << calibration_matrix_(0, 1)
         << "," << calibration_matrix_(0, 2) << "," << calibration_matrix_(0, 3)
         << "],"
         << "[" << calibration_matrix_(1, 0) << "," << calibration_matrix_(1, 1)
         << "," << calibration_matrix_(1, 2) << "," << calibration_matrix_(1, 3)
         << "],"
         << "[" << calibration_matrix_(2, 0) << "," << calibration_matrix_(2, 1)
         << "," << calibration_matrix_(2, 2) << "," << calibration_matrix_(2, 3)
         << "],"
         << "[" << calibration_matrix_(3, 0) << "," << calibration_matrix_(3, 1)
         << "," << calibration_matrix_(3, 2) << "," << calibration_matrix_(3, 3)
         << "]" << std::endl;
  fCalib.close();
}

bool ManualCalibration(int key_input) {
  char table[] = {'q', 'a', 'w', 's', 'e', 'd', 'r', 'f', 't', 'g', 'y', 'h'};
  bool real_hit = false;
  for (int32_t i = 0; i < 12; i++) {
    if (key_input == table[i]) {//标定人员点击了哪个键，就对应需要修改标定矩阵的一个“更改”
      calibration_matrix_ = calibration_matrix_ * modification_list_[i];
      real_hit = true;
    }
  }
  return real_hit;
}

RGB GreyToColorMix(int val) {//对点云的颜色进行处理（改色）这个处理我觉得不太适用于我们的情况。
  int r, g, b;
  if (val < 128) {
    r = 0;
  } else if (val < 192) {
    r = 255 / 64 * (val - 128);
  } else {
    r = 255;
  }
  if (val < 64) {
    g = 255 / 64 * val;
  } else if (val < 192) {
    g = 255;
  } else {
    g = -255 / 63 * (val - 192) + 255;
  }
  if (val < 64) {
    b = 255;
  } else if (val < 128) {
    b = -255 / 63 * (val - 192) + 255;
  } else {
    b = 0;
  }
  RGB rgb;
  rgb.b = b;
  rgb.g = g;
  rgb.r = r;
  return rgb;
}

bool is_exists(const std::string &name) {
  std::ifstream f(name.c_str());
  return f.good();
}

void LoadOdometerData(const std::string odometer_file,
                      std::vector<std::string> &timestamp,
                      std::vector<Eigen::Matrix4d> &lidar_poses) {

  std::ifstream file(odometer_file);
  if (!file.is_open()) {
    std::cout << "can not open " << odometer_file << std::endl;
    return;
  }
  std::string line;
  while (getline(file, line)) {
    std::stringstream ss(line);
    std::string timeStr;
    ss >> timeStr;
    timestamp.emplace_back(timeStr);
    Eigen::Matrix4d Ti = Eigen::Matrix4d::Identity(); // 初始化一个变换矩阵，就是标定数据中的一行
    ss >> Ti(0, 0) >> Ti(0, 1) >> Ti(0, 2) >> Ti(0, 3) >> Ti(1, 0) >>
        Ti(1, 1) >> Ti(1, 2) >> Ti(1, 3) >> Ti(2, 0) >> Ti(2, 1) >> Ti(2, 2) >>
        Ti(2, 3);// 根据这行代码可以看出标定数据的格式是四行四列的矩阵，按行读取
    lidar_poses.emplace_back(Ti);
  }
  file.close();
}

void PointCloudDownSampling(//根据 voxel_size 下采样点云
    const pcl::PointCloud<pcl::PointXYZI>::Ptr &in_cloud, double voxel_size,
    pcl::PointCloud<pcl::PointXYZI>::Ptr &out_cloud) {
  pcl::VoxelGrid<pcl::PointXYZI> sor;
  sor.setInputCloud(in_cloud);
  sor.setLeafSize(voxel_size, voxel_size, voxel_size);
  sor.filter(*out_cloud);
}

void PointCloudFilterByROI(const pcl::PointCloud<pcl::PointXYZI>::Ptr &in_cloud,
                           const PointCloudBbox &roi,
                           pcl::PointCloud<pcl::PointXYZI>::Ptr &out_cloud) {
  out_cloud->clear();
  for (const auto &src_pt : in_cloud->points) {
    if (src_pt.x > roi.min_x && src_pt.x < roi.max_x) {
      if (src_pt.y > roi.min_y && src_pt.y < roi.max_y) {
        if (src_pt.z > roi.min_z && src_pt.z < roi.max_z) {
          out_cloud->points.push_back(src_pt);
        }
      }
    }
  }
}

void LoadLidarPCDs(const std::string &pcds_dir,
                   const std::vector<std::string> &timestamp,
                   const std::vector<Eigen::Matrix4d> &lidar_poses_ori,// 激光雷达位姿数据
                   std::vector<pcl::PointCloud<pcl::PointXYZI>> &pcds,
                   std::vector<Eigen::Matrix4d> &lidar_poses) {
  if (lidar_poses_ori.size() == 0)
    return;
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(
      new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr filter_cloud(
      new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr filter_cloud_roi(
      new pcl::PointCloud<pcl::PointXYZI>);

  Eigen::Matrix4d first_pose = lidar_poses_ori[0];//从激光雷达位姿数据中抽出第一帧 4x4矩阵
  for (size_t i = 0; i < timestamp.size(); ++i) {
    // DEBUG DAI:
    if (i % 10 != 0)//每隔20个时间处理一次
      continue;
    std::string lidar_file_name = pcds_dir + "/" + timestamp[i] + ".pcd";//获取点云帧的文件名
    if (is_exists(lidar_file_name)) {
      if (pcl::io::loadPCDFile(lidar_file_name, *cloud) < 0) {
        std::cout << "can not open " << lidar_file_name << std::endl;
        continue;
      }
    } else//文件不存在
      continue;
    //当成功读取点云数据之后，进行下面的代码
    PointCloudDownSampling(cloud, 0.5, filter_cloud);//下采样
    PointCloudBbox roi;//我认为在初期进行标定的时候，是可以充分利用远端的数据的。远端的数据对于角度的调整更敏感
    roi.max_x = 30;
    roi.min_x = 5;
    std::cout << "Changed max x to 10." << std::endl;
    roi.max_y = 20;
    roi.min_y = -20;
    roi.max_z = 5;
    roi.min_z = -5;
    PointCloudFilterByROI(filter_cloud, roi, filter_cloud_roi);
    pcds.push_back(*filter_cloud_roi);
    lidar_poses.push_back(first_pose.inverse().eval() * lidar_poses_ori[i]);//第一帧姿态的逆矩阵 乘以 后面每帧的姿态
    // lidar_poses[0] 是单位阵
    // lidar_poses[1] 是从第一帧位姿转换到第二帧位姿的增量转换矩阵


    printf("\rload: %lu/%lu, %s", i, timestamp.size() - 1,
           lidar_file_name.c_str());
  }
}

int RemapIntensity(int value)
{
  // remap a int value from 0 to 50 to 0 to 255, and when a velue is bigger than 50, it will be set to 255
  int new_value = 0;
  if (value > 50)
    new_value = 255;
  else
    new_value = int(value * 255 / 50);

  return new_value;
}

int ProcessLidarFrame(const std::vector<pcl::PointCloud<pcl::PointXYZI>> &pcds,
                      const std::vector<Eigen::Matrix4d> &lidar_poses,
                      const Eigen::Matrix4d &calibration_matrix_,
                      const bool &diaplay_mode) {

  // DEBUG DAI: get the intensity range of the first frame
  int min_intensity = 0;
  int max_intensity = 0;
  for (const auto &src_pt : pcds[0].points) {
    if (!std::isfinite(src_pt.x) || !std::isfinite(src_pt.y) ||
        !std::isfinite(src_pt.z))
      continue;
    if (src_pt.intensity < min_intensity)
      min_intensity = src_pt.intensity;
    if (src_pt.intensity > max_intensity)
      max_intensity = src_pt.intensity;
  }
  std::cout << "min_intensity: " << min_intensity << std::endl;
  std::cout << "max_intensity: " << max_intensity << std::endl;
  
  for (size_t i = 0; i < pcds.size(); i++) {// 对每个点云都进行处理
    // i is the index of pcd, corresponding to a pose
    // 取出当前帧的相对于上一帧的增量变换矩阵
    Eigen::Matrix4d T = lidar_poses[i];//相较于上一帧的增量变换矩阵
    // T 是当前帧点云对应的姿态的齐次坐标表示（对应IMU坐标系到世界坐标系的变换矩阵）
    T *= calibration_matrix_;// T = T * calibration_matrix 

    for (const auto &src_pt : pcds[i].points) {
      if (!std::isfinite(src_pt.x) || !std::isfinite(src_pt.y) ||
          !std::isfinite(src_pt.z))
        continue;
      pcl::PointXYZI dst_pt;// destination point = R * src_pt
      Eigen::Vector3d p(src_pt.x, src_pt.y, src_pt.z);
      Eigen::Vector3d p_res;
      p_res = T.block<3, 3>(0, 0) * p + T.block<3, 1>(0, 3);
      // p: point in pointcloud (lidar coornidate system)
      // p_res = lidar_poses[i]<3,3> * calibration_mtrix<3,3> * p + transition
      // 先将点云中的某个点变换到imu坐标系中，然后再将imu坐标系中的一个点转换到原始的坐标系中（本地坐标系中）

      dst_pt.x = p_res(0);
      dst_pt.y = p_res(1);
      dst_pt.z = p_res(2);
      // DEBUG DAI: REMAP INTENSITY
      // dst_pt.intensity = src_pt.intensity;
      dst_pt.intensity = RemapIntensity(src_pt.intensity);

      double min_x, min_y, min_z, max_x, max_y, max_z;
      all_octree->getBoundingBox(min_x, min_y, min_z, max_x, max_y, max_z);
      bool isInBox = (dst_pt.x >= min_x && dst_pt.x <= max_x) && (dst_pt.y >= min_y && dst_pt.y <= max_y) && (dst_pt.z >= min_z && dst_pt.z <= max_z);
      if (!isInBox || !all_octree->isVoxelOccupiedAtPoint(dst_pt))
      {
        all_octree->addPointToCloud(dst_pt, cloudLidar);
      }
      // // DEBUG DAI: add the point [0,0,0] into the cloudlidar
      // Eigen::Vector3d p_pose;
      // p_pose = T.block<3, 1>(0, 3);//转换之后的点
      // pcl::PointXYZI zero_pt;
      // zero_pt.x = p_pose(0);
      // zero_pt.y = p_pose(1);
      // zero_pt.z = p_pose(2);
      // zero_pt.intensity = (int) 255;
      // all_octree->addPointToCloud(zero_pt, cloudLidar);

      // // DEBUG DAI: add the pose point into the cloudlidar
      // pcl::PointXYZI pose_pt;
      // pose_pt.x = dai_trans(0);
      // pose_pt.y = dai_trans(1);
      // pose_pt.z = dai_trans(2)+1.0;// make z bigger to make it visible
      // pose_pt.intensity = (int) 255;
      // all_octree->addPointToCloud(pose_pt, cloudLidar);
    }// end for one frame of pcd
  }// end for all pcds

  if (target_vertexBuffer_ != nullptr)
    delete (target_vertexBuffer_);
  if (target_colorBuffer_ != nullptr)
    delete (target_colorBuffer_);

  int lidarPointNum = cloudLidar->points.size(); // the number of lidar points, not including the zero point and pose points
  // deal with zeros points and pose points
  for (size_t i = 0; i < pcds.size(); i++) {// 对每个点云都进行处理
    // i is the index of pcd, corresponding to a pose
    // 取出当前帧的相对于上一帧的增量变换矩阵
    Eigen::Matrix4d T = lidar_poses[i];//相较于上一帧的增量变换矩阵
    Eigen::Vector3d dai_trans = T.block<3,1>(0, 3); // pose 的平移量
    // T 是当前帧点云对应的姿态的齐次坐标表示（对应IMU坐标系到世界坐标系的变换矩阵）
    T *= calibration_matrix_;// T = T * calibration_matrix 

    // DEBUG DAI: add the point [0,0,0] into the cloudlidar
    Eigen::Vector3d p_pose;
    p_pose = T.block<3, 1>(0, 3);//转换之后的点
    pcl::PointXYZI zero_pt;
    zero_pt.x = p_pose(0);
    zero_pt.y = p_pose(1);
    zero_pt.z = p_pose(2);
    zero_pt.intensity = (int) 255;
    all_octree->addPointToCloud(zero_pt, cloudLidar);

  }// end for all pcds
  int zeroPointNum = cloudLidar->points.size() - lidarPointNum; // the number of zero points
  for (size_t i = 0; i < pcds.size(); i++) {// 对每个点云都进行处理
    // i is the index of pcd, corresponding to a pose
    // 取出当前帧的相对于上一帧的增量变换矩阵
    Eigen::Matrix4d T = lidar_poses[i];//相较于上一帧的增量变换矩阵
    Eigen::Vector3d dai_trans = T.block<3,1>(0, 3); // pose 的平移量
    // T 是当前帧点云对应的姿态的齐次坐标表示（对应IMU坐标系到世界坐标系的变换矩阵）
    T *= calibration_matrix_;// T = T * calibration_matrix 

    // DEBUG DAI: add the pose point into the cloudlidar
    pcl::PointXYZI pose_pt;
    pose_pt.x = dai_trans(0);
    pose_pt.y = dai_trans(1);
    pose_pt.z = dai_trans(2)+2.0;// make z bigger to make it visible
    pose_pt.intensity = (int) 255;
    all_octree->addPointToCloud(pose_pt, cloudLidar);

  }// end for all pcds
  int posePointNum = cloudLidar->points.size() - lidarPointNum - zeroPointNum; // the number of pose points

  // Part one: render the lidar points
  int pointsNum = cloudLidar->points.size();

  pangolin::GlBuffer *vertexbuffer = new pangolin::GlBuffer(
      pangolin::GlArrayBuffer, pointsNum, GL_FLOAT, 3, GL_DYNAMIC_DRAW);
  pangolin::GlBuffer *colorbuffer = new pangolin::GlBuffer(
      pangolin::GlArrayBuffer, pointsNum, GL_UNSIGNED_BYTE, 3, GL_DYNAMIC_DRAW);

  float *dataUpdate = new float[pointsNum * 3];
  unsigned char *colorUpdate = new unsigned char[pointsNum * 3];
  // 对所有点分别进行处理，存入dataUpdate和colorUpdate中
  for (int ipt = 0; ipt < lidarPointNum; ipt++) {
    Eigen::Vector4d pointPos(cloudLidar->points[ipt].x,
                             cloudLidar->points[ipt].y,
                             cloudLidar->points[ipt].z, 1.0);
    dataUpdate[ipt * 3 + 0] = pointPos.x();
    dataUpdate[ipt * 3 + 1] = pointPos.y();
    dataUpdate[ipt * 3 + 2] = pointPos.z();

    if (diaplay_mode) {// display mode is ont intensity mode

      colorUpdate[ipt * 3 + 0] = static_cast<unsigned char>(0);
      colorUpdate[ipt * 3 + 1] = static_cast<unsigned char>(0);
      colorUpdate[ipt * 3 + 2] = static_cast<unsigned char>(255);
    } else {
      for (int k = 0; k < 3; k++) {
        colorUpdate[ipt * 3 + k] =
            static_cast<unsigned char>(cloudLidar->points[ipt].intensity);
      }
    }
  }
  // Part two: render the zeros points and pose points with different color
  for (int ipt = lidarPointNum; ipt < lidarPointNum + zeroPointNum; ipt++) {
    Eigen::Vector4d pointPos(cloudLidar->points[ipt].x,
                             cloudLidar->points[ipt].y,
                             cloudLidar->points[ipt].z, 1.0);
    dataUpdate[ipt * 3 + 0] = pointPos.x();
    dataUpdate[ipt * 3 + 1] = pointPos.y();
    dataUpdate[ipt * 3 + 2] = pointPos.z();
    if (diaplay_mode) {// display mode is ont intensity mode

      colorUpdate[ipt * 3 + 0] = static_cast<unsigned char>(0);
      colorUpdate[ipt * 3 + 1] = static_cast<unsigned char>(0);
      colorUpdate[ipt * 3 + 2] = static_cast<unsigned char>(255);
    } else {// [255, 0, 0]
      for (int k = 0; k < 3; k++) {
        if (k == 0)
        {
          colorUpdate[ipt * 3 + k] = static_cast<unsigned char>(255); // red
        }
        else if (k == 1)
        {
          colorUpdate[ipt * 3 + k] = static_cast<unsigned char>(0); 
        }
        else
        {
          colorUpdate[ipt * 3 + k] = static_cast<unsigned char>(0); 
        }
      }
    }
  }

  for (int ipt = lidarPointNum + zeroPointNum; ipt < pointsNum; ipt++) {
    Eigen::Vector4d pointPos(cloudLidar->points[ipt].x,
                             cloudLidar->points[ipt].y,
                             cloudLidar->points[ipt].z, 1.0);
    dataUpdate[ipt * 3 + 0] = pointPos.x();
    dataUpdate[ipt * 3 + 1] = pointPos.y();
    dataUpdate[ipt * 3 + 2] = pointPos.z();
    if (diaplay_mode) {// display mode is ont intensity mode

      colorUpdate[ipt * 3 + 0] = static_cast<unsigned char>(0);
      colorUpdate[ipt * 3 + 1] = static_cast<unsigned char>(0);
      colorUpdate[ipt * 3 + 2] = static_cast<unsigned char>(255);
    } else {// [255, 0, 0]
      for (int k = 0; k < 3; k++) {
        if (k == 0)
        {
          colorUpdate[ipt * 3 + k] = static_cast<unsigned char>(0); // red
        }
        else if (k == 1)
        {
          colorUpdate[ipt * 3 + k] = static_cast<unsigned char>(255); 
        }
        else
        {
          colorUpdate[ipt * 3 + k] = static_cast<unsigned char>(0); 
        }
      }
    }
  }


  // 对所有数据完成处理，并上传数据，用于渲染
  (vertexbuffer)->Upload(dataUpdate, sizeof(float) * 3 * pointsNum, 0);
  (colorbuffer)->Upload(colorUpdate, sizeof(unsigned char) * 3 * pointsNum, 0);

  target_vertexBuffer_ = vertexbuffer;
  target_colorBuffer_ = colorbuffer;
  std::cout << "Process target lidar frame!\n";
  int points_size = cloudLidar->points.size();
  cloudLidar->clear();
  all_octree->deleteTree();

  return points_size;
}

int main(int argc, char **argv) {
  if (argc != 4) {
    cout << "Usage: ./run_lidar2imu <lidar_pcds_dir> <lidar_pose_file> "
            "<extrinsic_json> "
            "\nexample:\n\t"
            "./bin/run_lidar2imu data/top_center_lidar "
            "data/top_center_lidar-pose.txt "
            "data/gnss-to-top_center_lidar-extrinsic.json"
         << endl;
    return 0;
  }

  string lidar_dir = argv[1];
  string lidar_pose_dir = argv[2];
  string extrinsic_json = argv[3];
  Eigen::Matrix4d json_param;
  LoadExtrinsic(extrinsic_json, json_param);
  std::vector<std::string> timestamp;
  std::vector<Eigen::Matrix4d> lidar_poses_ori;// 4x4矩阵
  // LoadOdometerData(lidar_pose_dir, json_param, timestamp, lidar_poses_ori);
  LoadOdometerData(lidar_pose_dir, timestamp, lidar_poses_ori);// 将激光雷达位姿数据从文件中读取到 lidar_poses_ori
  std::cout << json_param << std::endl;

  std::vector<pcl::PointCloud<pcl::PointXYZI>> pcds;
  std::vector<Eigen::Matrix4d> lidar_poses;
  LoadLidarPCDs(lidar_dir, timestamp, lidar_poses_ori, pcds, lidar_poses);
  std::cout << pcds.size() << std::endl;
  std::cout << lidar_poses.size() << std::endl;
  all_octree->setInputCloud(cloudLidar);

  cout << "Loading data completed!" << endl;
  CalibrationInit(json_param);
  const int width = 1920, height = 1280;
  pangolin::CreateWindowAndBind("lidar2imu player", width, height);

  glEnable(GL_DEPTH_TEST);
  glDepthMask(GL_TRUE);
  glDepthFunc(GL_LESS);

  pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
      // pangolin::ModelViewLookAt(0, 0, 100, 0, 0, 0, 0.0, 1.0, 0.0));
      pangolin::ModelViewLookAt(0, 100, 0, 0, 0, 0, 0.0, 0.0, 1.0));
  pangolin::View &d_cam = pangolin::CreateDisplay()
                              .SetBounds(0.0, 1.0, pangolin::Attach::Pix(150),
                                         1.0, -1.0 * width / height)
                              .SetHandler(new pangolin::Handler3D(s_cam));
  glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
  pangolin::OpenGlMatrix Twc; // camera to world
  Twc.SetIdentity();

  // control panel
  pangolin::CreatePanel("cp").SetBounds(pangolin::Attach::Pix(30), 1.0, 0.0,
                                        pangolin::Attach::Pix(150));
  pangolin::Var<bool> displayMode("cp.Intensity Color", display_mode_,
                                  true); // logscale
  pangolin::Var<int> pointSize("cp.Point Size", 2, 0, 8);
  pangolin::Var<double> degreeStep("cp.deg step", cali_scale_degree_, 0,
                                   1); // logscale
  pangolin::Var<double> tStep("cp.t step(cm)", 6, 0, 15);

  pangolin::Var<bool> addXdegree("cp.+ x degree", false, false);
  pangolin::Var<bool> minusXdegree("cp.- x degree", false, false);
  pangolin::Var<bool> addYdegree("cp.+ y degree", false, false);
  pangolin::Var<bool> minusYdegree("cp.- y degree", false, false);
  pangolin::Var<bool> addZdegree("cp.+ z degree", false, false);
  pangolin::Var<bool> minusZdegree("cp.- z degree", false, false);
  pangolin::Var<bool> addXtrans("cp.+ x trans", false, false);
  pangolin::Var<bool> minusXtrans("cp.- x trans", false, false);
  pangolin::Var<bool> addYtrans("cp.+ y trans", false, false);
  pangolin::Var<bool> minusYtrans("cp.- y trans", false, false);
  pangolin::Var<bool> addZtrans("cp.+ z trans", false, false);
  pangolin::Var<bool> minusZtrans("cp.- z trans", false, false);

  pangolin::Var<bool> resetButton("cp.Reset", false, false);
  pangolin::Var<bool> saveImg("cp.Save Result", false, false);

  std::vector<pangolin::Var<bool>> mat_calib_box;
  mat_calib_box.push_back(addXdegree);
  mat_calib_box.push_back(minusXdegree);
  mat_calib_box.push_back(addYdegree);
  mat_calib_box.push_back(minusYdegree);
  mat_calib_box.push_back(addZdegree);
  mat_calib_box.push_back(minusZdegree);
  mat_calib_box.push_back(addXtrans);
  mat_calib_box.push_back(minusXtrans);
  mat_calib_box.push_back(addYtrans);
  mat_calib_box.push_back(minusYtrans);
  mat_calib_box.push_back(addZtrans);
  mat_calib_box.push_back(minusZtrans);

  int frame_num = 0;
  int points_size = 0;
  points_size =
      ProcessLidarFrame(pcds, lidar_poses, calibration_matrix_, display_mode_);

  std::cout << "\n=>START\n";
  while (!pangolin::ShouldQuit()) {
    s_cam.Follow(Twc);
    d_cam.Activate(s_cam);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    if (displayMode) {
      if (display_mode_ == false) {
        display_mode_ = true;
        // ProcessTargetFrame(target_cloud, display_mode_);
        // ProcessSourceFrame(source_cloud, calibration_matrix_, display_mode_);
        points_size = ProcessLidarFrame(pcds, lidar_poses, calibration_matrix_,
                                        display_mode_);
      }
    } else {
      if (display_mode_ == true) {
        display_mode_ = false;
        // ProcessTargetFrame(target_cloud, display_mode_);
        // ProcessSourceFrame(source_cloud, calibration_matrix_, display_mode_);
        points_size = ProcessLidarFrame(pcds, lidar_poses, calibration_matrix_,
                                        display_mode_);
      }
    }
    if (pointSize.GuiChanged()) {
      point_size_ = pointSize.Get();
      std::cout << "Point size changed to " << point_size_ << " degree\n";
    }

    if (degreeStep.GuiChanged()) {
      cali_scale_degree_ = degreeStep.Get();
      CalibrationScaleChange();
      std::cout << "Degree calib scale changed to " << cali_scale_degree_
                << " degree\n";
    }
    if (tStep.GuiChanged()) {
      cali_scale_trans_ = tStep.Get() / 100.0;
      CalibrationScaleChange();
      std::cout << "Trans calib scale changed to " << cali_scale_trans_ * 100
                << " cm\n";
    }
    for (int i = 0; i < 12; i++) {
      if (pangolin::Pushed(mat_calib_box[i])) {
        calibration_matrix_ = calibration_matrix_ * modification_list_[i];
        // ProcessSourceFrame(source_cloud, calibration_matrix_, display_mode_);
        points_size = ProcessLidarFrame(pcds, lidar_poses, calibration_matrix_,
                                        display_mode_);
        std::cout << "Changed!\n";
      }
    }

    if (pangolin::Pushed(resetButton)) {
      calibration_matrix_ = orign_calibration_matrix_;
      // ProcessSourceFrame(source_cloud, calibration_matrix_, display_mode_);
      points_size = ProcessLidarFrame(pcds, lidar_poses, calibration_matrix_,
                                      display_mode_);
      std::cout << "Reset!\n";
    }
    if (pangolin::Pushed(saveImg)) {
      saveResult(frame_num);
      std::cout << "\n==>Save Result " << frame_num << std::endl;
      Eigen::Matrix4d transform = calibration_matrix_;
      cout << "Transfromation Matrix:\n" << transform << std::endl;
      frame_num++;
    }

    if (kbhit()) {
      int c = getchar();
      if (ManualCalibration(c)) {
        Eigen::Matrix4d transform = calibration_matrix_;
        // ProcessSourceFrame(source_cloud, calibration_matrix_, display_mode_);
        points_size = ProcessLidarFrame(pcds, lidar_poses, calibration_matrix_,
                                        display_mode_);
        cout << "\nTransfromation Matrix:\n" << transform << std::endl;
      }
    }

    // draw lidar points
    glDisable(GL_LIGHTING);
    glPointSize(point_size_);
    // draw target lidar points
    target_colorBuffer_->Bind();
    glColorPointer(target_colorBuffer_->count_per_element,
                   target_colorBuffer_->datatype, 0, 0);
    glEnableClientState(GL_COLOR_ARRAY);
    target_vertexBuffer_->Bind();
    glVertexPointer(target_vertexBuffer_->count_per_element,
                    target_vertexBuffer_->datatype, 0, 0);
    glEnableClientState(GL_VERTEX_ARRAY);
    glDrawArrays(GL_POINTS, 0, points_size);
    glDisableClientState(GL_VERTEX_ARRAY);
    target_vertexBuffer_->Unbind();
    glDisableClientState(GL_COLOR_ARRAY);
    target_colorBuffer_->Unbind();
    pangolin::FinishFrame();
    usleep(100);
    glFinish();
  }

  // delete[] imageArray;

  Eigen::Matrix4d transform = calibration_matrix_;
  cout << "\nFinal Transfromation Matrix:\n" << transform << std::endl;

  return 0;
}
