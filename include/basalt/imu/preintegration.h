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

  /// @brief Propagate current state given ImuData and optionally compute
  /// Jacobians.
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
    BASALT_ASSERT_STREAM(
        data.t_ns > curr_state.t_ns,
        "data.t_ns " << data.t_ns << " curr_state.t_ns " << curr_state.t_ns);

    int64_t dt_ns = data.t_ns - curr_state.t_ns;
    Scalar dt = dt_ns * Scalar(1e-9);

    // R_{(2t+1)/2} = R_t * Exp(0.5 * dt * w_{t+1})
    SO3 R_w_i_new_2 = curr_state.T_w_i.so3() * SO3::exp(Scalar(0.5) * dt * data.gyro);
    
    // R_{(2t+1)/2}的矩阵形式
    Mat3 RR_w_i_new_2 = R_w_i_new_2.matrix();

    // R_{(2t+1)/2} * a_{t+1}
    Vec3 accel_world = RR_w_i_new_2 * data.accel;

    /******* s_{t+1} ********/
    
    // dt
    next_state.t_ns = data.t_ns;
    
    // 旋转：使用的是后向积分法
    // R_{t+1} = R_t * Exp(dt * w_{t+1})
    next_state.T_w_i.so3() = curr_state.T_w_i.so3() * SO3::exp(dt * data.gyro);

    // 速度：使用的是中值积分法
    // v_{t+1} = v_t + R_{(2t+1)/2} * a_{t+1} * dt
    next_state.vel_w_i = curr_state.vel_w_i + accel_world * dt;

    // 平移：使用的是中值积分法
    // p_{t+1} = p_t + v_t * dt + 0.5 * R_{(2t+1)/2} * a_{t+1} * dt^2
    next_state.T_w_i.translation() = curr_state.T_w_i.translation() +
                                     curr_state.vel_w_i * dt +
                                     0.5 * accel_world * dt * dt;

    // d(s_{t+1}) / d(a_{t+1})
    if (d_next_d_curr) {
      d_next_d_curr->setIdentity();

      // d(p_{t+1}) / d(v_t)
      d_next_d_curr->template block<3, 3>(0, 6).diagonal().setConstant(dt);

      // d(v_{t+1}) / d(R_t)
      d_next_d_curr->template block<3, 3>(6, 3) = SO3::hat(-accel_world * dt);
      
      // d(p_{t+1}) / d(R_t)
      d_next_d_curr->template block<3, 3>(0, 3) = d_next_d_curr->template block<3, 3>(6, 3) * dt * Scalar(0.5);
    }
    
    // d(s_{t+1}) / d(a_{t+1})
    if (d_next_d_accel) {
      d_next_d_accel->setZero();

      // d(p_{t+1}) / d(a_{t+1})
      d_next_d_accel->template block<3, 3>(0, 0) = Scalar(0.5) * RR_w_i_new_2 * dt * dt;

      // d(p_{v+1}) / d(a_{t+1})
      d_next_d_accel->template block<3, 3>(6, 0) = RR_w_i_new_2 * dt;
    }

    // d(s_{t+1}) / d(w_{t+1})
    if (d_next_d_gyro) {
      d_next_d_gyro->setZero();

      Mat3 Jr;
      Sophus::rightJacobianSO3(dt * data.gyro, Jr);

      Mat3 Jr2;
      Sophus::rightJacobianSO3(Scalar(0.5) * dt * data.gyro, Jr2);

      d_next_d_gyro->template block<3, 3>(3, 0) =
          next_state.T_w_i.so3().matrix() * Jr * dt;
      d_next_d_gyro->template block<3, 3>(6, 0) =
          SO3::hat(-accel_world * dt) * RR_w_i_new_2 * Jr2 * Scalar(0.5) * dt;

      d_next_d_gyro->template block<3, 3>(0, 0) =
          Scalar(0.5) * dt * d_next_d_gyro->template block<3, 3>(6, 0);
    }
  }

  /// @brief Default constructor.
  IntegratedImuMeasurement() noexcept {
    cov_.setZero();
    d_state_d_ba_.setZero();
    d_state_d_bg_.setZero();
    bias_gyro_lin_.setZero();
    bias_accel_lin_.setZero();
    sqrt_cov_inv_.setZero();
  }

  /// @brief Constructor with start time and bias estimates.
  IntegratedImuMeasurement(int64_t start_t_ns, const Vec3& bias_gyro_lin,
                           const Vec3& bias_accel_lin) noexcept
      : start_t_ns_(start_t_ns),
        bias_gyro_lin_(bias_gyro_lin),
        bias_accel_lin_(bias_accel_lin) {
    cov_.setZero();
    d_state_d_ba_.setZero();
    d_state_d_bg_.setZero();
    sqrt_cov_inv_.setZero();
  }

  /// @brief Integrate IMU data
  ///
  /// @param[in] data IMU data
  /// @param[in] accel_cov diagonal of accelerometer noise covariance matrix
  /// @param[in] gyro_cov diagonal of gyroscope noise covariance matrix
  void integrate(const ImuData<Scalar>& data, const Vec3& accel_cov,
                 const Vec3& gyro_cov) {
    ImuData<Scalar> data_corrected = data;
    data_corrected.t_ns -= start_t_ns_;
    data_corrected.accel -= bias_accel_lin_;
    data_corrected.gyro -= bias_gyro_lin_;

    PoseVelState<Scalar> new_state;

    MatNN F;
    MatN3 A;
    MatN3 G;

    propagateState(delta_state_, data_corrected, new_state, &F, &A, &G);

    delta_state_ = new_state;
    cov_ = F * cov_ * F.transpose() +
           A * accel_cov.asDiagonal() * A.transpose() +
           G * gyro_cov.asDiagonal() * G.transpose();
    sqrt_cov_inv_computed_ = false;

    d_state_d_ba_ = -A + F * d_state_d_ba_;
    d_state_d_bg_ = -G + F * d_state_d_bg_;
  }

  /// @brief Predict state given this pseudo-measurement
  ///
  /// @param[in] state0 current state
  /// @param[in] g gravity vector
  /// @param[out] state1 predicted state
  void predictState(const PoseVelState<Scalar>& state0, const Vec3& g,
                    PoseVelState<Scalar>& state1) const {
    Scalar dt = delta_state_.t_ns * Scalar(1e-9);

    state1.T_w_i.so3() = state0.T_w_i.so3() * delta_state_.T_w_i.so3();
    state1.vel_w_i =
        state0.vel_w_i + g * dt + state0.T_w_i.so3() * delta_state_.vel_w_i;
    state1.T_w_i.translation() =
        state0.T_w_i.translation() + state0.vel_w_i * dt +
        Scalar(0.5) * g * dt * dt +
        state0.T_w_i.so3() * delta_state_.T_w_i.translation();
  }

  /// @brief Compute residual between two states given this pseudo-measurement
  /// and optionally compute Jacobians.
  ///
  /// @param[in] state0 initial state
  /// @param[in] g gravity vector
  /// @param[in] state1 next state
  /// @param[in] curr_bg current estimate of gyroscope bias
  /// @param[in] curr_ba current estimate of accelerometer bias
  /// @param[out] d_res_d_state0 if not nullptr, Jacobian of the residual with
  /// respect to state0
  /// @param[out] d_res_d_state1 if not nullptr, Jacobian of the residual with
  /// respect to state1
  /// @param[out] d_res_d_bg if not nullptr, Jacobian of the residual with
  /// respect to gyroscope bias
  /// @param[out] d_res_d_ba if not nullptr, Jacobian of the residual with
  /// respect to accelerometer bias
  /// @return residual
  VecN residual(const PoseVelState<Scalar>& state0, const Vec3& g,
                const PoseVelState<Scalar>& state1, const Vec3& curr_bg,
                const Vec3& curr_ba, MatNN* d_res_d_state0 = nullptr,
                MatNN* d_res_d_state1 = nullptr, MatN3* d_res_d_bg = nullptr,
                MatN3* d_res_d_ba = nullptr) const {
    Scalar dt = delta_state_.t_ns * Scalar(1e-9);
    VecN res;

    VecN bg_diff;
    VecN ba_diff;
    bg_diff = d_state_d_bg_ * (curr_bg - bias_gyro_lin_);
    ba_diff = d_state_d_ba_ * (curr_ba - bias_accel_lin_);

    BASALT_ASSERT(ba_diff.template segment<3>(3).isApproxToConstant(0));

    Mat3 R0_inv = state0.T_w_i.so3().inverse().matrix();
    Vec3 tmp =
        R0_inv * (state1.T_w_i.translation() - state0.T_w_i.translation() -
                  state0.vel_w_i * dt - Scalar(0.5) * g * dt * dt);

    res.template segment<3>(0) =
        tmp - (delta_state_.T_w_i.translation() +
               bg_diff.template segment<3>(0) + ba_diff.template segment<3>(0));
    res.template segment<3>(3) =
        (SO3::exp(bg_diff.template segment<3>(3)) * delta_state_.T_w_i.so3() *
         state1.T_w_i.so3().inverse() * state0.T_w_i.so3())
            .log();

    Vec3 tmp2 = R0_inv * (state1.vel_w_i - state0.vel_w_i - g * dt);
    res.template segment<3>(6) =
        tmp2 - (delta_state_.vel_w_i + bg_diff.template segment<3>(6) +
                ba_diff.template segment<3>(6));

    if (d_res_d_state0 || d_res_d_state1) {
      Mat3 J;
      Sophus::rightJacobianInvSO3(res.template segment<3>(3), J);

      if (d_res_d_state0) {
        d_res_d_state0->setZero();
        d_res_d_state0->template block<3, 3>(0, 0) = -R0_inv;
        d_res_d_state0->template block<3, 3>(0, 3) = SO3::hat(tmp) * R0_inv;
        d_res_d_state0->template block<3, 3>(3, 3) = J * R0_inv;
        d_res_d_state0->template block<3, 3>(6, 3) = SO3::hat(tmp2) * R0_inv;

        d_res_d_state0->template block<3, 3>(0, 6) = -R0_inv * dt;
        d_res_d_state0->template block<3, 3>(6, 6) = -R0_inv;
      }

      if (d_res_d_state1) {
        d_res_d_state1->setZero();
        d_res_d_state1->template block<3, 3>(0, 0) = R0_inv;
        d_res_d_state1->template block<3, 3>(3, 3) = -J * R0_inv;

        d_res_d_state1->template block<3, 3>(6, 6) = R0_inv;
      }
    }

    if (d_res_d_ba) {
      *d_res_d_ba = -d_state_d_ba_;
    }

    if (d_res_d_bg) {
      d_res_d_bg->setZero();
      *d_res_d_bg = -d_state_d_bg_;

      Mat3 J;
      Sophus::leftJacobianInvSO3(res.template segment<3>(3), J);
      d_res_d_bg->template block<3, 3>(3, 0) =
          J * d_state_d_bg_.template block<3, 3>(3, 0);
    }

    return res;
  }

  /// @brief Time duretion of preintegrated measurement in nanoseconds.
  int64_t get_dt_ns() const { return delta_state_.t_ns; }

  /// @brief Start time of preintegrated measurement in nanoseconds.
  int64_t get_start_t_ns() const { return start_t_ns_; }

  /// @brief Inverse of the measurement covariance matrix
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

    return sqrt_cov_inv_;
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

  int64_t start_t_ns_{0};  ///< Integration start time in nanoseconds

  PoseVelState<Scalar> delta_state_;  ///< Delta state

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
