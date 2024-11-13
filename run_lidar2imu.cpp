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

//定义了一个函数 kbhit()，用于检测键盘输入是否有按键被按下
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

//使用Eigen库进行的相机标定，通过对初始的相机标定矩阵进行一系列变换来实现不同的校正效果。
void CalibrationInit(Eigen::Matrix4d json_param) {
  Eigen::Matrix4d init_cali;
  init_cali << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1;
  calibration_matrix_ = json_param;
  orign_calibration_matrix_ = json_param;
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

//使用了Eigen库进行矩阵运算，实现了对相机标定结果的调整和校正。
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

//saveResult(const int &frame_id)函数用于保存相机标定结果到文件中。它首先构造了保存文件的文件名，然后打开文件流。
//如果文件无法打开，则输出错误信息并返回。接着，它将标定结果以矩阵形式写入文件，并以普通格式和 JSON 格式分别输出。最后，关闭文件流。
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

//ManualCalibration(int key_input)函数实现了手动调整相机标定的功能。它通过接收按键输入来选择要应用的标定变换。
//具体来说，它根据输入的按键来选择对应的标定变换，并将当前标定矩阵乘以该变换矩阵。如果输入的按键与已定义的标定变换匹配，则返回 true，否则返回 false。
bool ManualCalibration(int key_input) {
  char table[] = {'q', 'a', 'w', 's', 'e', 'd', 'r', 'f', 't', 'g', 'y', 'h'};
  bool real_hit = false;
  for (int32_t i = 0; i < 12; i++) {
    if (key_input == table[i]) {
      calibration_matrix_ = calibration_matrix_ * modification_list_[i];
      real_hit = true;
    }
  }
  return real_hit;
}

//将灰度值转换为RGB颜色值的函数。该函数接受一个0到255之间的整数参数val，表示灰度值。
RGB GreyToColorMix(int val) {
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

//用于检查指定文件是否存在
bool is_exists(const std::string &name) {
  std::ifstream f(name.c_str());
  return f.good();
}

//加载里程计数据，提取时间戳和位姿信息，并存储在相应的数据结构中，以便后续处理和分析
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
    Eigen::Matrix4d Ti = Eigen::Matrix4d::Identity();
    ss >> Ti(0, 0) >> Ti(0, 1) >> Ti(0, 2) >> Ti(0, 3) >> Ti(1, 0) >>
        Ti(1, 1) >> Ti(1, 2) >> Ti(1, 3) >> Ti(2, 0) >> Ti(2, 1) >> Ti(2, 2) >>
        Ti(2, 3);
    lidar_poses.emplace_back(Ti);
  }
  file.close();
}

//对输入的点云数据进行下采样，使用 VoxelGrid 滤波器将原始的点云数据根据指定的体素大小进行降采样，并将结果存储在输出点云out_cloud中。
void PointCloudDownSampling(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr &in_cloud, double voxel_size,
    pcl::PointCloud<pcl::PointXYZI>::Ptr &out_cloud) {
  pcl::VoxelGrid<pcl::PointXYZI> sor;
  sor.setInputCloud(in_cloud);
  sor.setLeafSize(voxel_size, voxel_size, voxel_size);
  sor.filter(*out_cloud);
}

//基于给定的区域（ROI）对输入的点云数据进行过滤，并将符合条件的点云数据存储在输出点云数据中。
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

//加载激光雷达的点云数据，对数据进行下采样和ROI过滤，并存储处理后的点云数据和对应的激光雷达姿态变换矩阵。
void LoadLidarPCDs(const std::string &pcds_dir,
                   const std::vector<std::string> &timestamp,
                   const std::vector<Eigen::Matrix4d> &lidar_poses_ori,
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

  Eigen::Matrix4d first_pose = lidar_poses_ori[0];
  for (size_t i = 0; i < timestamp.size(); ++i) {
    if (i % 20 != 0)
      continue;
    std::string lidar_file_name = pcds_dir + "/" + timestamp[i] + ".pcd";
    if (is_exists(lidar_file_name)) {
      if (pcl::io::loadPCDFile(lidar_file_name, *cloud) < 0) {
        std::cout << "can not open " << lidar_file_name << std::endl;
        continue;
      }
    } else
      continue;

    //对激光雷达数据进行下采样处理，然后根据指定的 ROI 区域进行过滤，并保存过滤后的点云数据和更新的激光雷达位置信息。
    PointCloudDownSampling(cloud, 0.7, filter_cloud); //传参：激光雷达数据点云、采样率、存储下采样后数据的变量
    PointCloudBbox roi; //表示一个区域roi的边界框,指定边界框在三维空间中的最大和最小的 x、y、z 坐标值
    roi.max_x = 20;
    roi.min_x = -20;
    //std::cout << "changed max x to 10."  << std::endl;
    roi.max_y = 20;
    roi.min_y = -20;
    roi.max_z = 5;
    roi.min_z = -5;
    PointCloudFilterByROI(filter_cloud, roi, filter_cloud_roi);  
    pcds.push_back(*filter_cloud_roi); 
    lidar_poses.push_back(first_pose.inverse().eval() * lidar_poses_ori[i]);

    printf("\rload: %lu/%lu, %s", i, timestamp.size() - 1,
           lidar_file_name.c_str());
  }
}

//实现了激光雷达数据的处理和可视化操作，包括点云变换、边界检查、数据上传等功能。
int ProcessLidarFrame(const std::vector<pcl::PointCloud<pcl::PointXYZI>> &pcds,
                      const std::vector<Eigen::Matrix4d> &lidar_poses,
                      const Eigen::Matrix4d &calibration_matrix_,
                      const bool &diaplay_mode) {
  for (size_t i = 0; i < pcds.size(); i++) {
    Eigen::Matrix4d T = lidar_poses[i];
    T *= calibration_matrix_;

    for (const auto &src_pt : pcds[i].points) {
      if (!std::isfinite(src_pt.x) || !std::isfinite(src_pt.y) ||
          !std::isfinite(src_pt.z))
        continue;
      pcl::PointXYZI dst_pt;
      Eigen::Vector3d p(src_pt.x, src_pt.y, src_pt.z);
      Eigen::Vector3d p_res;
      p_res = T.block<3, 3>(0, 0) * p + T.block<3, 1>(0, 3);

      dst_pt.x = p_res(0);
      dst_pt.y = p_res(1);
      dst_pt.z = p_res(2);
      dst_pt.intensity = src_pt.intensity;

      double min_x, min_y, min_z, max_x, max_y, max_z;
      all_octree->getBoundingBox(min_x, min_y, min_z, max_x, max_y, max_z);
      bool isInBox = (dst_pt.x >= min_x && dst_pt.x <= max_x) && (dst_pt.y >= min_y && dst_pt.y <= max_y) && (dst_pt.z >= min_z && dst_pt.z <= max_z);
      if (!isInBox || !all_octree->isVoxelOccupiedAtPoint(dst_pt))
      {
        all_octree->addPointToCloud(dst_pt, cloudLidar);
      }
    }
  }

  if (target_vertexBuffer_ != nullptr)
    delete (target_vertexBuffer_);
  if (target_colorBuffer_ != nullptr)
    delete (target_colorBuffer_);

  int pointsNum = cloudLidar->points.size();

  pangolin::GlBuffer *vertexbuffer = new pangolin::GlBuffer(
      pangolin::GlArrayBuffer, pointsNum, GL_FLOAT, 3, GL_DYNAMIC_DRAW);
  pangolin::GlBuffer *colorbuffer = new pangolin::GlBuffer(
      pangolin::GlArrayBuffer, pointsNum, GL_UNSIGNED_BYTE, 3, GL_DYNAMIC_DRAW);

  float *dataUpdate = new float[pointsNum * 3];
  unsigned char *colorUpdate = new unsigned char[pointsNum * 3];
  for (int ipt = 0; ipt < pointsNum; ipt++) {
    Eigen::Vector4d pointPos(cloudLidar->points[ipt].x,
                             cloudLidar->points[ipt].y,
                             cloudLidar->points[ipt].z, 1.0);
    dataUpdate[ipt * 3 + 0] = pointPos.x();
    dataUpdate[ipt * 3 + 1] = pointPos.y();
    dataUpdate[ipt * 3 + 2] = pointPos.z();

    if (diaplay_mode) {

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
  //根据命令行传入的参数，加载外部的激光雷达数据、位姿数据和外部参数，并输出加载的外部参数。
  string lidar_dir = argv[1];
  string lidar_pose_dir = argv[2];
  string extrinsic_json = argv[3];
  Eigen::Matrix4d json_param;
  LoadExtrinsic(extrinsic_json, json_param);
  std::vector<std::string> timestamp;
  std::vector<Eigen::Matrix4d> lidar_poses_ori;
  // LoadOdometerData(lidar_pose_dir, json_param, timestamp, lidar_poses_ori);
  LoadOdometerData(lidar_pose_dir, timestamp, lidar_poses_ori);
  std::cout << json_param << std::endl;

  //加载激光雷达的点云数据和位姿数据
  std::vector<pcl::PointCloud<pcl::PointXYZI>> pcds;
  std::vector<Eigen::Matrix4d> lidar_poses;
  LoadLidarPCDs(lidar_dir, timestamp, lidar_poses_ori, pcds, lidar_poses);
  std::cout << pcds.size() << std::endl;
  std::cout << lidar_poses.size() << std::endl;
  all_octree->setInputCloud(cloudLidar);

  //为接下来的处理准备了数据，并创建了一个窗口用于可视化展示。
  cout << "Loading data completed!" << endl;
  CalibrationInit(json_param);
  const int width = 1920, height = 1280;
  pangolin::CreateWindowAndBind("lidar2imu player", width, height);

  //使用 OpenGL 函数来设置深度测试的相关参数
  glEnable(GL_DEPTH_TEST);
  glDepthMask(GL_TRUE);
  glDepthFunc(GL_LESS);

  //使用了 Pangolin 库来设置相机的投影矩阵和观察矩阵，并创建了一个显示窗口。实现了对相机投影矩阵和观察矩阵的设置，并创建了一个名为 d_cam 的显示窗口，最后设置了背景颜色并创建了一个单位变换矩阵。
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
  //创建了一个名为 "cp" 的控制面板，并在面板上添加了四个控件。
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
  //主要用于图形界面的交互和控制，通过不同的按钮和控件来实现对点云显示效果和校准参数的调整。
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

    //用于绘制激光雷达点云的部分。
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
  //使用了Eigen库中的Matrix4d类型，创建了一个名为transform的4x4双精度浮点型变换矩阵，并将其初始化为calibration_matrix_。
  //接着，通过cout打印输出了最终的变换矩阵transform的数值，输出格式为4x4的矩阵形式。这样就可以在控制台上查看和验证最终的变换矩阵。
  Eigen::Matrix4d transform = calibration_matrix_;
  cout << "\nFinal Transfromation Matrix:\n" << transform << std::endl;

  return 0;
}
