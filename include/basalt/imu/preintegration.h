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
@brief IMU preintegration
*/

#pragma once

#include <basalt/imu/imu_types.h>
#include <basalt/utils/assert.h>
#include <basalt/utils/sophus_utils.hpp>

namespace basalt {

/// @brief Integrated pseudo-measurement that combines several consecutive IMU
/// measurements.
template <class Scalar_>
class IntegratedImuMeasurement {
 public:
  using Scalar = Scalar_;

  using Ptr = std::shared_ptr<IntegratedImuMeasurement<Scalar>>;

  using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
  using VecN = Eigen::Matrix<Scalar, POSE_VEL_SIZE, 1>;
  using Mat3 = Eigen::Matrix<Scalar, 3, 3>;
  using MatNN = Eigen::Matrix<Scalar, POSE_VEL_SIZE, POSE_VEL_SIZE>;
  using MatN3 = Eigen::Matrix<Scalar, POSE_VEL_SIZE, 3>;
  using MatN6 = Eigen::Matrix<Scalar, POSE_VEL_SIZE, 6>;
  using SO3 = Sophus::SO3<Scalar>;

  /// @brief Propagate current state given ImuData and optionally compute Jacobians.
  ///        单步IMU积分：基于给定ImuData，传播当前状态到下一时刻，并计算雅可比矩阵（可选）
  ///
  /// @param[in] curr_state current state
  /// @param[in] data IMU data
  /// @param[out] next_state predicted state
  /// @param[out] d_next_d_curr Jacobian of the predicted state with respect
  /// to current state
  /// @param[out] d_next_d_accel Jacobian of the predicted state with respect
  /// accelerometer measurement
  /// @param[out] d_next_d_gyro Jacobian of the predicted state with respect
  /// gyroscope measurement

    /**
   * @brief IMU状态递推
   * 
   * @param curr_state      当前状态，s_t
   * @param data            t+1时刻IMU观测：加速度 a_{t+1} 和 角速度 w_{t+1}
   * @param next_state      预测的状态，s_{t+1}  
   * @param d_next_d_curr   d(s_{t+1}) / d(s_t)     ，维度 9 x 9
   * @param d_next_d_accel  d(s_{t+1}) / d(a_{t+1}) ，维度 9 x 3
   * @param d_next_d_gyro   d(s_{t+1}) / d(w_{t+1}) ，维度 9 x 3
   */
  inline static void propagateState(const PoseVelState<Scalar>& curr_state,
                                    const ImuData<Scalar>& data,
                                    PoseVelState<Scalar>& next_state,
                                    MatNN* d_next_d_curr = nullptr,
                                    MatN3* d_next_d_accel = nullptr,
                                    MatN3* d_next_d_gyro = nullptr) {
    // 断言：IMU数据时间戳必须晚于当前状态时间戳
    BASALT_ASSERT_STREAM(
        data.t_ns > curr_state.t_ns,
        "data.t_ns " << data.t_ns << " curr_state.t_ns " << curr_state.t_ns);

    // 计算时间差（纳秒转秒）
    int64_t dt_ns = data.t_ns - curr_state.t_ns;  // 时间差（纳秒）
    Scalar dt = dt_ns * Scalar(1e-9);             // 时间差（秒）

    // -------------------------- 核心积分计算 --------------------------
    // 1. 中值法计算旋转增量：先用半时间步的角速度计算旋转（用于加速度计的坐标系转换）
    // R_{(2t+1)/2} = R_t * Exp(0.5 * dt * w_{t+1})
    SO3 R_w_i_new_2 = curr_state.T_w_i.so3() * SO3::exp(Scalar(0.5) * dt * data.gyro);

    // R_{(2t+1)/2}的矩阵形式
    Mat3 RR_w_i_new_2 = R_w_i_new_2.matrix();

    // 2. 加速度计测量值转换到世界坐标系（中值旋转）
     // R_{(2t+1)/2} * a_{t+1}
    Vec3 accel_world = RR_w_i_new_2 * data.accel;

    // 3. 预测下一时刻状态
    // dt
    next_state.t_ns = data.t_ns;                                                  // 时间戳更新
    
    // 旋转：使用的是后向积分法
    // R_{t+1} = R_t * Exp(dt * w_{t+1})
    next_state.T_w_i.so3() = curr_state.T_w_i.so3() * SO3::exp(dt * data.gyro);   // 旋转更新（李群指数映射）
    
    // 速度：使用的是中值积分法
    // v_{t+1} = v_t + R_{(2t+1)/2} * a_{t+1} * dt
    next_state.vel_w_i = curr_state.vel_w_i + accel_world * dt;                   // 速度更新
    
    // 平移：使用的是中值积分法
    // p_{t+1} = p_t + v_t * dt + 0.5 * R_{(2t+1)/2} * a_{t+1} * dt^2
    next_state.T_w_i.translation() = curr_state.T_w_i.translation() +             // 位置更新
                                     curr_state.vel_w_i * dt +
                                     0.5 * accel_world * dt * dt;

    // -------------------------- 雅可比矩阵计算 --------------------------
    // 雅可比矩阵用于优化过程中的残差求导
    
    // 计算下一状态对当前状态的雅可比：d(s_{t+1}) / d(s_t)
    if (d_next_d_curr) {

      // 初始化为单位矩阵（大部分元素不变）
      d_next_d_curr->setIdentity();
      
      // 位置对速度的偏导：dx/dv = dt（位置由速度积分而来）
      // d(p_{t+1}) / d(v_t)
      d_next_d_curr->template block<3, 3>(0, 6).diagonal().setConstant(dt);

      // 速度对旋转的偏导：dv/dR = -[a_w ×] * dt（旋转变化影响加速度在世界系的表示）
      // [a_w ×] 是加速度的反对称矩阵（叉乘等价）
      // d(v_{t+1}) / d(R_t)
      d_next_d_curr->template block<3, 3>(6, 3) = SO3::hat(-accel_world * dt);

      // 位置对旋转的偏导：dx/dR = 0.5 * dt * dv/dR（位置由速度积分，速度对旋转偏导的积分）
      // d(p_{t+1}) / d(R_t)
      d_next_d_curr->template block<3, 3>(0, 3) =
          d_next_d_curr->template block<3, 3>(6, 3) * dt * Scalar(0.5);
    }

    // 计算下一状态对加速度计测量值的雅可比：d(s_{t+1}) / d(a_{t+1})
    if (d_next_d_accel) {

      // 初始化为零矩阵
      d_next_d_accel->setZero();

      // 位置对加速度的偏导：dx/da = 0.5 * R_mid * dt²（中值旋转矩阵）
      // d(p_{t+1}) / d(a_{t+1})
      d_next_d_accel->template block<3, 3>(0, 0) =
          Scalar(0.5) * RR_w_i_new_2 * dt * dt;
      
      // 速度对加速度的偏导：dv/da = R_mid * dt
      // d(p_{v+1}) / d(a_{t+1})
      d_next_d_accel->template block<3, 3>(6, 0) = RR_w_i_new_2 * dt;
    }

    // 计算下一状态对陀螺仪测量值的雅可比：d(s_{t+1}) / d(w_{t+1})
    if (d_next_d_gyro) {
      d_next_d_gyro->setZero(); // 初始化为零矩阵

      // 计算SO(3)右雅可比矩阵（处理李群指数映射的非线性）
      Mat3 Jr;
      Sophus::rightJacobianSO3(dt * data.gyro, Jr); // 全时间步角速度的右雅可比
      Mat3 Jr2;
      Sophus::rightJacobianSO3(Scalar(0.5) * dt * data.gyro, Jr2);  // 半时间步角速度的右雅可比

      // 旋转对陀螺仪的偏导：dR/dω = R_new * Jr * dt
      d_next_d_gyro->template block<3, 3>(3, 0) =
          next_state.T_w_i.so3().matrix() * Jr * dt;

      // 速度对陀螺仪的偏导：dv/dω = -[a_w ×] * R_mid * Jr2 * 0.5 * dt
      d_next_d_gyro->template block<3, 3>(6, 0) =
          SO3::hat(-accel_world * dt) * RR_w_i_new_2 * Jr2 * Scalar(0.5) * dt;

      // 位置对陀螺仪的偏导：dx/dω = 0.5 * dt * dv/dω
      d_next_d_gyro->template block<3, 3>(0, 0) =
          Scalar(0.5) * dt * d_next_d_gyro->template block<3, 3>(6, 0);
    }
  }

  /// @brief Default constructor. 默认构造函数
  IntegratedImuMeasurement() noexcept {
    cov_.setZero();               // 协方差矩阵初始化为零
    d_state_d_ba_.setZero();      // 状态对加速度计偏置的雅可比初始化为零
    d_state_d_bg_.setZero();      // 状态对陀螺仪偏置的雅可比初始化为零
    bias_gyro_lin_.setZero();     // 陀螺仪线性偏置初始化为零
    bias_accel_lin_.setZero();    // 加速度计线性偏置初始化为零
    sqrt_cov_inv_.setZero();      // 协方差逆的平方根初始化为零
  }

  /// @brief Constructor with start time and bias estimates. 带初始时间和偏置的构造函数
  /// @param[in] start_t_ns     积分起始时间戳（纳秒）
  /// @param[in] bias_gyro_lin  陀螺仪线性偏置
  /// @param[in] bias_accel_lin 加速度计线性偏置
  IntegratedImuMeasurement(int64_t start_t_ns, const Vec3& bias_gyro_lin,
                           const Vec3& bias_accel_lin) noexcept
      : start_t_ns_(start_t_ns),
        bias_gyro_lin_(bias_gyro_lin),
        bias_accel_lin_(bias_accel_lin) {
    cov_.setZero();               // 协方差矩阵初始化为零
    d_state_d_ba_.setZero();      // 状态对加速度计偏置的雅可比初始化为零
    d_state_d_bg_.setZero();      // 状态对陀螺仪偏置的雅可比初始化为零
    sqrt_cov_inv_.setZero();      // 协方差逆的平方根初始化为零
  }

  /// @brief Integrate IMU data 积分单个IMU数据（预积分核心函数）
  ///        累计IMU测量值，更新增量状态、协方差矩阵、偏置雅可比
  /// 
  /// @param[in] data IMU data IMU测量数据
  /// @param[in] accel_cov diagonal of accelerometer noise covariance matrix 加速度计噪声协方差（对角矩阵）
  /// @param[in] gyro_cov diagonal of gyroscope noise covariance matrix 陀螺仪噪声协方差（对角矩阵）
  void integrate(const ImuData<Scalar>& data, const Vec3& accel_cov,
                 const Vec3& gyro_cov) {
    // 1. 修正IMU数据：减去偏置，调整时间戳（相对起始时间）
    ImuData<Scalar> data_corrected = data;
    data_corrected.t_ns -= start_t_ns_;       // 时间戳转为相对起始时间
    data_corrected.accel -= bias_accel_lin_;  // 加速度计测量值减去偏置
    data_corrected.gyro -= bias_gyro_lin_;    // 陀螺仪测量值减去偏置

    // 2. 临时变量存储新状态
    PoseVelState<Scalar> new_state;

    // 3. 雅可比矩阵临时变量
    MatNN F;  // 状态转移矩阵（下一状态对当前状态的雅可比）
    MatN3 A;  // 下一状态对加速度计测量值的雅可比
    MatN3 G;  // 下一状态对陀螺仪测量值的雅可比

    // 4. 传播状态：积分当前增量状态到新状态
    propagateState(delta_state_, data_corrected, new_state, &F, &A, &G);

    // 5. 更新增量状态
    delta_state_ = new_state;

    // 6. 更新协方差矩阵（误差传播：P = F*P*F^T + A*Q_a*A^T + G*Q_g*G^T）
    // Q_a: 加速度计噪声协方差，Q_g: 陀螺仪噪声协方差
    cov_ = F * cov_ * F.transpose() +
           A * accel_cov.asDiagonal() * A.transpose() +
           G * gyro_cov.asDiagonal() * G.transpose();
    sqrt_cov_inv_computed_ = false; // 协方差变化，需要重新计算逆的平方根

    // 7. 更新状态对偏置的雅可比（考虑偏置变化的影响）
    // 偏置雅可比传播：d_state/d_ba = -A + F*d_state/d_ba
    d_state_d_ba_ = -A + F * d_state_d_ba_;
    d_state_d_bg_ = -G + F * d_state_d_bg_;
  }

  /// @brief Predict state given this pseudo-measurement 基于预积分结果预测下一时刻状态
  ///        从初始状态state0，结合预积分的增量和重力，预测state1
  /// 
  /// @param[in] state0 current state 初始状态
  /// @param[in] g gravity vector 重力向量（世界坐标系）
  /// @param[out] state1 predicted state 预测的下一状态
  void predictState(const PoseVelState<Scalar>& state0, const Vec3& g,
                    PoseVelState<Scalar>& state1) const {
    // 计算预积分的总时间（秒）
    Scalar dt = delta_state_.t_ns * Scalar(1e-9);

    // 1. 旋转预测：初始旋转 * 预积分旋转增量
    state1.T_w_i.so3() = state0.T_w_i.so3() * delta_state_.T_w_i.so3();

    // 2. 速度预测：初始速度 + 重力增量 + 预积分速度增量（旋转到世界系）
    state1.vel_w_i =
        state0.vel_w_i + g * dt + state0.T_w_i.so3() * delta_state_.vel_w_i;
    
    // 3. 位置预测：初始位置 + 匀速项 + 重力项 + 预积分位置增量（旋转到世界系）
    state1.T_w_i.translation() =
        state0.T_w_i.translation() + state0.vel_w_i * dt +
        Scalar(0.5) * g * dt * dt +
        state0.T_w_i.so3() * delta_state_.T_w_i.translation();
  }

  /// @brief Compute residual between two states given this pseudo-measurement and optionally compute Jacobians.
  ///        计算两个状态之间的残差（用于SLAM优化中的IMU预积分约束项），可选计算雅可比
  ///        残差定义：预积分的增量 与 两状态之间的实际增量 的差值
  /// 
  /// @param[in] state0 initial state                                                                     初始状态
  /// @param[in] g gravity vector                                                                         重力向量
  /// @param[in] state1 next state                                                                        下一状态
  /// @param[in] curr_bg current estimate of gyroscope bias                                               当前陀螺仪偏置估计值
  /// @param[in] curr_ba current estimate of accelerometer bias                                           当前加速度计偏置估计值
  /// @param[out] d_res_d_state0 if not nullptr, Jacobian of the residual with respect to state0          残差对state0的雅可比（9x9）
  /// @param[out] d_res_d_state1 if not nullptr, Jacobian of the residual with respect to state1          残差对state1的雅可比（9x9）
  /// @param[out] d_res_d_bg if not nullptr, Jacobian of the residual with respect to gyroscope bias      残差对陀螺仪偏置的雅可比（9x3）
  /// @param[out] d_res_d_ba if not nullptr, Jacobian of the residual with respect to accelerometer bias  残差对加速度计偏置的雅可比（9x3）
  /// @return residual                                                                                    残差向量（9维：位置3+旋转3+速度3）
  VecN residual(const PoseVelState<Scalar>& state0, const Vec3& g,
                const PoseVelState<Scalar>& state1, const Vec3& curr_bg,
                const Vec3& curr_ba, MatNN* d_res_d_state0 = nullptr,
                MatNN* d_res_d_state1 = nullptr, MatN3* d_res_d_bg = nullptr,
                MatN3* d_res_d_ba = nullptr) const {
    
    Scalar dt = delta_state_.t_ns * Scalar(1e-9); // 预积分总时间
    VecN res; // 残差向量

    // 1. 计算偏置变化带来的状态增量
    VecN bg_diff;
    VecN ba_diff;
    bg_diff = d_state_d_bg_ * (curr_bg - bias_gyro_lin_);     // 陀螺仪偏置变化的影响
    ba_diff = d_state_d_ba_ * (curr_ba - bias_accel_lin_);    // 加速度计偏置变化的影响

    // 断言：加速度计偏置变化不影响旋转部分（ba_diff的3-5维应为0）
    BASALT_ASSERT(ba_diff.template segment<3>(3).isApproxToConstant(0));

    // 2. 计算state0到state1的实际位置增量（转换到state0的坐标系）
    Mat3 R0_inv = state0.T_w_i.so3().inverse().matrix();    // state0旋转的逆矩阵
    Vec3 tmp =
        R0_inv * (state1.T_w_i.translation() - state0.T_w_i.translation() -
                  state0.vel_w_i * dt - Scalar(0.5) * g * dt * dt);
    
    // -------------------------- 残差计算 --------------------------
    // 位置残差：实际位置增量 - 预积分位置增量 - 偏置影响
    res.template segment<3>(0) =
        tmp - (delta_state_.T_w_i.translation() +
               bg_diff.template segment<3>(0) + ba_diff.template segment<3>(0));
    
    // 旋转残差：用李代数表示旋转误差（预积分旋转增量 与 实际旋转增量 的差值）
    // 公式：log( exp(bg_diff_rot) * delta_R * R1_inv * R0 )
    res.template segment<3>(3) =
        (SO3::exp(bg_diff.template segment<3>(3)) * delta_state_.T_w_i.so3() *
         state1.T_w_i.so3().inverse() * state0.T_w_i.so3())
            .log();

    // 速度残差：实际速度增量 - 预积分速度增量 - 偏置影响
    Vec3 tmp2 = R0_inv * (state1.vel_w_i - state0.vel_w_i - g * dt);
    res.template segment<3>(6) =
        tmp2 - (delta_state_.vel_w_i + bg_diff.template segment<3>(6) +
                ba_diff.template segment<3>(6));


    // -------------------------- 雅可比矩阵计算 --------------------------
    if (d_res_d_state0 || d_res_d_state1) {
      // 计算旋转残差的右雅可比逆（处理李代数到李群的非线性）
      Mat3 J;
      Sophus::rightJacobianInvSO3(res.template segment<3>(3), J);

      // 残差对state0的雅可比
      if (d_res_d_state0) {
        d_res_d_state0->setZero();
        d_res_d_state0->template block<3, 3>(0, 0) = -R0_inv;                   // 位置残差对state0位置的偏导：-R0_inv
        d_res_d_state0->template block<3, 3>(0, 3) = SO3::hat(tmp) * R0_inv;    // 位置残差对state0旋转的偏导：[tmp ×] * R0_inv
        d_res_d_state0->template block<3, 3>(3, 3) = J * R0_inv;                // 旋转残差对state0旋转的偏导：J * R0_inv
        d_res_d_state0->template block<3, 3>(6, 3) = SO3::hat(tmp2) * R0_inv;   // 速度残差对state0旋转的偏导：[tmp2 ×] * R0_inv

        d_res_d_state0->template block<3, 3>(0, 6) = -R0_inv * dt;              // 位置残差对state0速度的偏导：-R0_inv * dt
        d_res_d_state0->template block<3, 3>(6, 6) = -R0_inv;                   // 速度残差对state0速度的偏导：-R0_inv
      }

      // 残差对state1的雅可比
      if (d_res_d_state1) {
        d_res_d_state1->setZero();                                              
        d_res_d_state1->template block<3, 3>(0, 0) = R0_inv;                    // 位置残差对state1位置的偏导：R0_inv
        d_res_d_state1->template block<3, 3>(3, 3) = -J * R0_inv;               // 旋转残差对state1旋转的偏导：-J * R0_inv

        d_res_d_state1->template block<3, 3>(6, 6) = R0_inv;                    // 速度残差对state1速度的偏导：R0_inv
      }
    }

    // 残差对加速度计偏置的雅可比
    if (d_res_d_ba) {
      *d_res_d_ba = -d_state_d_ba_;
    }

    // 残差对陀螺仪偏置的雅可比
    if (d_res_d_bg) {
      d_res_d_bg->setZero();
      *d_res_d_bg = -d_state_d_bg_;

      // 修正旋转残差对陀螺仪偏置的雅可比（左雅可比逆）
      Mat3 J;
      Sophus::leftJacobianInvSO3(res.template segment<3>(3), J);
      d_res_d_bg->template block<3, 3>(3, 0) =
          J * d_state_d_bg_.template block<3, 3>(3, 0);
    }

    return res;
  }

  /// @brief Time duretion of preintegrated measurement in nanoseconds. 获取预积分的总时间（纳秒）
  int64_t get_dt_ns() const { return delta_state_.t_ns; }

  /// @brief Start time of preintegrated measurement in nanoseconds.  获取预积分的起始时间戳（纳秒）
  int64_t get_start_t_ns() const { return start_t_ns_; }

  /// @brief Inverse of the measurement covariance matrix 获取协方差矩阵的逆矩阵
  ///        内部缓存协方差逆的平方根，避免重复计算
  inline MatNN get_cov_inv() const {
    if (!sqrt_cov_inv_computed_) {
      compute_sqrt_cov_inv();
      sqrt_cov_inv_computed_ = true;
    }

    return sqrt_cov_inv_.transpose() * sqrt_cov_inv_;
  }

  /// @brief Square root inverse of the measurement covariance matrix
  inline const MatNN& get_sqrt_cov_inv() const {
    if (!sqrt_cov_inv_computed_) {
      compute_sqrt_cov_inv();
      sqrt_cov_inv_computed_ = true;
    }

    return sqrt_cov_inv_;   // 协方差逆 = (sqrt_cov_inv)^T * sqrt_cov_inv
  }

  /// @brief Measurement covariance matrix
  const MatNN& get_cov() const { return cov_; }

  // Just for testing...
  /// @brief Delta state
  const PoseVelState<Scalar>& getDeltaState() const { return delta_state_; }

  /// @brief Jacobian of delta state with respect to accelerometer bias
  const MatN3& get_d_state_d_ba() const { return d_state_d_ba_; }

  /// @brief Jacobian of delta state with respect to gyroscope bias
  const MatN3& get_d_state_d_bg() const { return d_state_d_bg_; }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 private:
  /// @brief Helper function to compute square root of the inverse covariance
  void compute_sqrt_cov_inv() const {
    sqrt_cov_inv_.setIdentity();
    auto ldlt = cov_.ldlt();

    sqrt_cov_inv_ = ldlt.transpositionsP() * sqrt_cov_inv_;
    ldlt.matrixL().solveInPlace(sqrt_cov_inv_);

    VecN D_inv_sqrt;
    for (size_t i = 0; i < POSE_VEL_SIZE; i++) {
      if (ldlt.vectorD()[i] < std::numeric_limits<Scalar>::min()) {
        D_inv_sqrt[i] = 0;
      } else {
        D_inv_sqrt[i] = Scalar(1.0) / sqrt(ldlt.vectorD()[i]);
      }
    }
    sqrt_cov_inv_ = D_inv_sqrt.asDiagonal() * sqrt_cov_inv_;
  }

  int64_t start_t_ns_{0};  ///< Integration start time in nanoseconds 预积分起始时间戳（纳秒）

  PoseVelState<Scalar> delta_state_;  ///< Delta state 预积分的增量状态（相对起始时间的时间、位姿和速度增量）

  MatNN cov_;  ///< Measurement covariance
  mutable MatNN
      sqrt_cov_inv_;  ///< Cached square root inverse of measurement covariance
  mutable bool sqrt_cov_inv_computed_{
      false};  ///< If the cached square root inverse
               ///< covariance is computed

  MatN3 d_state_d_ba_, d_state_d_bg_;

  Vec3 bias_gyro_lin_, bias_accel_lin_;
};

}  // namespace basalt
