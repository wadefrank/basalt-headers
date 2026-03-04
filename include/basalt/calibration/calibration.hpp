/**
BSD 3-Clause License

This file is part of the Basalt project.
https://gitlab.com/VladyslavUsenko/basalt-headers.git

Copyright (c) 2019, Vladyslav Usenko and Nikolaus Demmel.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

@file
@brief Calibration datatypes for muticam-IMU and motion capture calibration
*/

#pragma once

#include <memory>

#include <basalt/spline/rd_spline.h>
#include <basalt/calibration/calib_bias.hpp>
#include <basalt/camera/generic_camera.hpp>

namespace basalt {

/// @brief Struct to store camera-IMU calibration 存储相机-IMU标定参数的结构体
/// @tparam Scalar 标量类型（如float/double），用于适配不同精度需求
template <class Scalar>
struct Calibration {
  
  // 定义智能指针类型，方便内存管理和对象共享
  using Ptr = std::shared_ptr<Calibration>;
  
  // 定义SE3类型（李群），表示刚体变换（旋转+平移）
  using SE3 = Sophus::SE3<Scalar>;
  
  // 定义3维向量类型，用于表示点、角速度、加速度等
  using Vec3 = Eigen::Matrix<Scalar, 3, 1>;

  /// @brief Default constructor. 默认构造函数
  /// 初始化标定参数为合理的默认值（适用于大多数IMU的经验值）
  Calibration() {
    // 相机与IMU的时间偏移初始化为0（单位：纳秒）
    cam_time_offset_ns = 0;

    // IMU更新频率初始化为200Hz（常见的IMU采样频率）
    imu_update_rate = 200;

    // reasonable defaults
    // 初始化IMU噪声和偏置的默认标准差（经验值，单位：rad/s 或 m/s²）
    gyro_noise_std.setConstant(0.000282); // 陀螺仪噪声标准差（连续时间）
    accel_noise_std.setConstant(0.016);   // 加速度计噪声标准差（连续时间）
    accel_bias_std.setConstant(0.001);    // 加速度计偏置随机游走标准差
    gyro_bias_std.setConstant(0.0001);    // 陀螺仪偏置随机游走标准差
  }

  /// @brief Cast to other scalar type 将标定参数转换为其他标量类型
  /// @tparam Scalar2 目标标量类型（如从float转double）
  /// @return 转换后的Calibration对象
  template <class Scalar2>
  Calibration<Scalar2> cast() const {
    // 创建目标类型的标定对象
    Calibration<Scalar2> new_cam;

    // 转换相机到IMU的位姿（SE3）
    for (const auto& v : T_i_c)
      new_cam.T_i_c.emplace_back(v.template cast<Scalar2>());

    // 转换相机内参
    for (const auto& v : intrinsics)
      new_cam.intrinsics.emplace_back(v.template cast<Scalar2>());

    // 转换渐晕样条曲线
    for (const auto& v : vignette)
      new_cam.vignette.emplace_back(v.template cast<Scalar2>());

    // 转换相机分辨率（整数类型，无需cast）
    new_cam.resolution = resolution;

    // 转换时间偏移（整数类型，无需cast）
    new_cam.cam_time_offset_ns = cam_time_offset_ns;

    // 转换加速度计静态偏置参数
    new_cam.calib_accel_bias.getParam() =
        calib_accel_bias.getParam().template cast<Scalar2>();

    // 转换陀螺仪静态偏置参数
    new_cam.calib_gyro_bias.getParam() =
        calib_gyro_bias.getParam().template cast<Scalar2>();

    // 转换IMU更新频率
    new_cam.imu_update_rate = imu_update_rate;

    // 转换IMU噪声和偏置标准差
    new_cam.gyro_noise_std = gyro_noise_std.template cast<Scalar2>();
    new_cam.accel_noise_std = accel_noise_std.template cast<Scalar2>();
    new_cam.gyro_bias_std = gyro_bias_std.template cast<Scalar2>();
    new_cam.accel_bias_std = accel_bias_std.template cast<Scalar2>();

    return new_cam;
  }

  /// @brief Vector of transformations from camera to IMU
  ///        相机到IMU的位姿变换列表（支持多相机）
  /// 
  /// Point in camera coordinate frame \f$ p_c \f$ can be transformed to the
  /// point in IMU coordinate frame as \f$ p_i = T_{ic} p_c, T_{ic} \in SE(3)\f$
  /// 将相机坐标系下的点p_c转换到IMU坐标系p_i
  /// 公式：p_i = T_ic * p_c
  Eigen::aligned_vector<SE3> T_i_c;

  /// @brief Vector of camera intrinsics. Can store different camera models. See
  /// \ref GenericCamera.
  /// 相机内参列表（与T_i_c一一对应），支持多种相机模型（如针孔、鱼眼、等距投影），具体由GenericCamera实现
  Eigen::aligned_vector<GenericCamera<Scalar>> intrinsics;

  /// @brief Camera resolutions.
  ///        相机分辨率列表（宽度、高度，单位：像素）
  Eigen::aligned_vector<Eigen::Vector2i> resolution;

  /// @brief Vector of splines representing radially symmetric vignetting for each of the camera.
  ///        每个相机的径向渐晕样条曲线
  /// 
  /// Splines use time in nanoseconds for evaluation, but in this case we use
  /// distance from the optical center in pixels multiplied by 1e9 as a "time"
  /// parameter.
  /// 特殊用法：样条原本用于时间序列（纳秒），此处将光学中心的像素距离×1e9作为"时间"参数
  /// 作用：精确建模相机成像的亮度随像素位置的衰减（渐晕）
  std::vector<basalt::RdSpline<1, 4, Scalar>> vignette;

  /// @brief Time offset between cameras and IMU in nanoseconds.
  ///        相机与IMU的时间偏移（单位：纳秒）
  ///
  /// With raw image timestamp \f$ t_r \f$ and this offset \f$ o \f$ we cam get a timestamp aligned with IMU clock as \f$ t_c = t_r + o \f$.
  /// 时间对齐公式：t_c（IMU时钟）= t_r（原始图像时间戳） + o（该偏移量）
  int64_t cam_time_offset_ns;

  /// @brief Static accelerometer bias from calibration.
  ///        加速度计的静态偏置（标定得到），作用：用于补偿IMU加速度计的固有零偏
  CalibAccelBias<Scalar> calib_accel_bias;

  /// @brief Static gyroscope bias from calibration.
  ///        陀螺仪的静态偏置（标定得到），作用：用于补偿IMU陀螺仪的固有零偏
  CalibGyroBias<Scalar> calib_gyro_bias;

  /// @brief IMU update rate.
  ///        IMU的更新频率（单位：Hz）
  Scalar imu_update_rate;

  /// @brief Continuous time gyroscope noise standard deviation.
  ///        陀螺仪连续时间噪声标准差（单位：rad/s），作用：描述陀螺仪测量值的高斯白噪声强度
  Vec3 gyro_noise_std;

  /// @brief Continuous time accelerometer noise standard deviation.
  ///        加速度计连续时间噪声标准差（单位：m/s²），作用：描述加速度计测量值的高斯白噪声强度
  Vec3 accel_noise_std;

  /// @brief Continuous time bias random walk standard deviation for gyroscope.
  ///        陀螺仪偏置随机游走标准差（单位：rad/(s·√Hz)），作用：描述陀螺仪偏置随时间的缓慢变化特性
  Vec3 gyro_bias_std;

  /// @brief Continuous time bias random walk standard deviation for accelerometer.
  ///        加速度计偏置随机游走标准差（单位：m/(s²·√Hz)），作用：描述加速度计偏置随时间的缓慢变化特性
  Vec3 accel_bias_std;

  /// @brief Dicrete time gyroscope noise standard deviation.
  ///        计算陀螺仪离散时间噪声标准差
  ///
  /// \f$ \sigma_d = \sigma_c \sqrt{r} \f$, where \f$ r \f$ is IMU update rate.
  /// 公式：σ_d = σ_c * √r，其中σ_c是连续噪声，r是IMU更新频率
  /// 作用：将连续时间噪声转换为离散时间（适配IMU采样）
  ///
  /// @return 离散时间陀螺仪噪声标准差
  inline Vec3 dicrete_time_gyro_noise_std() const {
    return gyro_noise_std * std::sqrt(imu_update_rate);
  }

  /// @brief Dicrete time accelerometer noise standard deviation.
  ///        计算加速度计离散时间噪声标准差
  ///
  /// \f$ \sigma_d = \sigma_c \sqrt{r} \f$, where \f$ r \f$ is IMU update rate.
  /// 公式：σ_d = σ_c * √r，其中σ_c是连续噪声，r是IMU更新频率
  /// 作用：将连续时间噪声转换为离散时间（适配IMU采样）
  ///
  /// @return 离散时间加速度计噪声标准差
  inline Vec3 dicrete_time_accel_noise_std() const {
    return accel_noise_std * std::sqrt(imu_update_rate);
  }

  // Eigen库宏：确保Eigen对象的内存对齐，提升计算效率
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

/// @brief Struct to store motion capture to IMU calibration
///        存储运动捕捉（MoCap）到IMU标定参数的结构体
///
/// @tparam Scalar 标量类型（如float/double）
template <class Scalar>
struct MocapCalibration {
  // 定义智能指针类型
  using Ptr = std::shared_ptr<MocapCalibration>;
  
  // 定义SE3类型，表示刚体变换
  using SE3 = Sophus::SE3<Scalar>;

  /// @brief Default constructor.
  ///        默认构造函数
  MocapCalibration() {
    // MoCap与IMU初始时间偏移（基于消息到达时间），初始化时间偏移为0
    mocap_time_offset_ns = 0;

    // MoCap到IMU的最终时间偏移（标定后），初始化时间偏移为0
    mocap_to_imu_offset_ns = 0;
  }

  /// @brief Transformation from motion capture origin to the world (calibration pattern).
  ///        运动捕捉原点到世界坐标系（标定板）的位姿变换
  SE3 T_moc_w;

  /// @brief Transformation from the coordinate frame of the markers attached to the object to the IMU.
  ///        物体上标记点坐标系到IMU坐标系的位姿变换
  SE3 T_i_mark;

  /// @brief Initial time alignment between IMU and MoCap clocks based on message arrival time.
  ///        基于消息接收时间的粗略对齐，后续会通过标定优化
  int64_t mocap_time_offset_ns;

  /// @brief Time offset between IMU and motion capture clock.
  ///        标定优化后的精确时间对齐参数
  int64_t mocap_to_imu_offset_ns;
};

}  // namespace basalt