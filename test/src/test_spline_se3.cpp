/**
BSD 3-Clause License

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
*/

#include <basalt/spline/se3_spline.h>

#include <iostream>

#include "gtest/gtest.h"
#include "test_utils.h"

/**
 * @brief Test helper function for gyroscope residual computation and its
 * Jacobians
 *
 * Tests the computation of gyroscope residuals and their Jacobians for SE(3)
 * splines. For each knot, verifies:
 * 1. Residual between measured and predicted angular velocity
 * 2. Jacobians with respect to spline control points
 * 3. Jacobians with respect to gyroscope bias parameters
 *
 * @param s SE(3) spline to test
 * @param t_ns Timestamp in nanoseconds for evaluation
 */
template <int N>
void testGyroRes(const basalt::Se3Spline<N> &s, int64_t t_ns) {
  typename basalt::Se3Spline<N>::SO3JacobianStruct J_spline;
  Eigen::Matrix<double, 3, 12> J_bias;

  // Create random gyroscope bias for testing
  basalt::CalibGyroBias<double> bias;
  bias.setRandom();

  // Get angular velocity in body frame
  Eigen::Vector3d measurement = s.rotVelBody(t_ns);

  // Compute residual and Jacobians
  s.gyroResidual(t_ns, measurement, bias, &J_spline, &J_bias);

  // Test Jacobians for each knot
  for (size_t i = 0; i < s.numKnots(); i++) {
    Sophus::Vector3d x0;
    x0.setZero();

    std::stringstream ss;
    ss << "Spline order " << N << " d_gyro_res_d_knot" << i << " time " << t_ns;

    // Get analytical Jacobian for this knot
    Sophus::Matrix3d J_a;
    if (i >= J_spline.start_idx && i < J_spline.start_idx + N) {
      J_a = J_spline.d_val_d_knot[i - J_spline.start_idx];
    } else {
      J_a.setZero();
    }

    // Verify Jacobian using numerical differentiation
    test_jacobian(
        ss.str(), J_a,
        [&](const Sophus::Vector3d &x) {
          basalt::Se3Spline<N> s1 = s;
          s1.getKnotSO3(i) = Sophus::SO3d::exp(x) * s.getKnotSO3(i);
          return s1.gyroResidual(t_ns, measurement, bias);
        },
        x0);
  }

  // Test Jacobian with respect to bias parameters
  {
    Eigen::Matrix<double, 12, 1> x0;
    x0.setZero();

    std::stringstream ss;
    ss << "Spline order " << N << " d_gyro_res_d_bias";

    test_jacobian(
        ss.str(), J_bias,
        [&](const Eigen::Matrix<double, 12, 1> &x) {
          auto b1 = bias;
          b1 += x;
          return s.gyroResidual(t_ns, measurement, b1);
        },
        x0);
  }
}

/**
 * @brief Test helper function for accelerometer residual computation and its
 * Jacobians
 *
 * Tests the computation of accelerometer residuals and their Jacobians for
 * SE(3) splines. For each knot, verifies:
 * 1. Residual between measured and predicted linear acceleration
 * 2. Jacobians with respect to spline control points
 * 3. Jacobians with respect to accelerometer bias parameters
 * 4. Jacobians with respect to gravity direction
 *
 * @param s SE(3) spline to test
 * @param t_ns Timestamp in nanoseconds for evaluation
 */
template <int N>
void testAccelRes(const basalt::Se3Spline<N> &s, int64_t t_ns) {
  typename basalt::Se3Spline<N>::AccelPosSO3JacobianStruct J_spline;
  Eigen::Matrix3d J_g;
  Eigen::Matrix<double, 3, 9> J_bias;

  // Create random accelerometer bias for testing
  basalt::CalibAccelBias<double> bias;
  bias.setRandom();

  // Define gravity vector and compute measurement
  Eigen::Vector3d g(0, 0, -9.81);  // Gravity in world frame
  Eigen::Vector3d measurement =
      s.transAccelWorld(t_ns) + g;  // Acceleration + gravity

  // Compute residual and Jacobians
  s.accelResidual(t_ns, measurement, bias, g, &J_spline, &J_bias, &J_g);

  // Test Jacobians for each knot
  for (size_t i = 0; i < s.numKnots(); i++) {
    Sophus::Vector6d x0;
    x0.setZero();

    std::stringstream ss;
    ss << "Spline order " << N << " d_accel_res_d_knot" << i;

    // Get analytical Jacobian for this knot
    typename basalt::Se3Spline<N>::Mat36 J_a;
    if (i >= J_spline.start_idx && i < J_spline.start_idx + N) {
      J_a = J_spline.d_val_d_knot[i - J_spline.start_idx];
    } else {
      J_a.setZero();
    }

    // Verify Jacobian using numerical differentiation
    test_jacobian(
        ss.str(), J_a,
        [&](const Sophus::Vector6d &x) {
          basalt::Se3Spline<N> s1 = s;
          s1.applyInc(i, x);
          return s1.accelResidual(t_ns, measurement, bias, g);
        },
        x0);
  }

  // Test Jacobian with respect to bias parameters
  {
    Eigen::Matrix<double, 9, 1> x0;
    x0.setZero();

    std::stringstream ss;
    ss << "Spline order " << N << " d_accel_res_d_bias";

    test_jacobian(
        ss.str(), J_bias,
        [&](const Eigen::Matrix<double, 9, 1> &x) {
          auto b1 = bias;
          b1 += x;
          return s.accelResidual(t_ns, measurement, b1, g);
        },
        x0);
  }

  // Test Jacobian with respect to gravity direction
  {
    Sophus::Vector3d x0;
    x0.setZero();

    std::stringstream ss;
    ss << "Spline order " << N << " d_accel_res_d_g";

    test_jacobian(
        ss.str(), J_g,
        [&](const Sophus::Vector3d &x) {
          return s.accelResidual(t_ns, measurement, bias, g + x);
        },
        x0);
  }
}

/**
 * @brief Test helper function for orientation residual computation and its
 * Jacobians
 *
 * Tests the computation of orientation residuals and their Jacobians for SE(3)
 * splines. For each knot, verifies:
 * 1. Residual between measured and predicted orientation
 * 2. Jacobians with respect to spline control points
 *
 * @param s SE(3) spline to test
 * @param t_ns Timestamp in nanoseconds for evaluation
 */
template <int N>
void testOrientationRes(const basalt::Se3Spline<N> &s, int64_t t_ns) {
  typename basalt::Se3Spline<N>::SO3JacobianStruct J_spline;

  // Get orientation measurement from spline
  Sophus::SO3d measurement = s.pose(t_ns).so3();

  // Compute residual and Jacobians
  s.orientationResidual(t_ns, measurement, &J_spline);

  // Test Jacobians for each knot
  for (size_t i = 0; i < s.numKnots(); i++) {
    std::stringstream ss;
    ss << "Spline order " << N << " d_rot_res_d_knot" << i;

    // Get analytical Jacobian for this knot
    Sophus::Matrix3d J_a;
    if (i >= J_spline.start_idx && i < J_spline.start_idx + N) {
      J_a = J_spline.d_val_d_knot[i - J_spline.start_idx];
    } else {
      J_a.setZero();
    }

    Sophus::Vector3d x0;
    x0.setZero();

    // Verify Jacobian using numerical differentiation
    test_jacobian(
        ss.str(), J_a,
        [&](const Sophus::Vector3d &x_rot) {
          Sophus::Vector6d x;
          x.setZero();
          x.tail<3>() = x_rot;  // Only modify rotation part

          basalt::Se3Spline<N> s1 = s;
          s1.applyInc(i, x);

          return s1.orientationResidual(t_ns, measurement);
        },
        x0);
  }
}

/**
 * @brief Test helper function for position residual computation and its
 * Jacobians
 *
 * Tests the computation of position residuals and their Jacobians for SE(3)
 * splines. For each knot, verifies:
 * 1. Residual between measured and predicted position
 * 2. Jacobians with respect to spline control points
 *
 * @param s SE(3) spline to test
 * @param t_ns Timestamp in nanoseconds for evaluation
 */
template <int N>
void testPositionRes(const basalt::Se3Spline<N> &s, int64_t t_ns) {
  typename basalt::Se3Spline<N>::PosJacobianStruct J_spline;

  // Get position measurement from spline
  Eigen::Vector3d measurement = s.pose(t_ns).translation();

  // Compute residual and Jacobians
  s.positionResidual(t_ns, measurement, &J_spline);

  // Test Jacobians for each knot
  for (size_t i = 0; i < s.numKnots(); i++) {
    std::stringstream ss;
    ss << "Spline order " << N << " d_pos_res_d_knot" << i;

    // Get analytical Jacobian for this knot
    Sophus::Matrix3d J_a;
    J_a.setZero();
    if (i >= J_spline.start_idx && i < J_spline.start_idx + N) {
      J_a.diagonal().setConstant(J_spline.d_val_d_knot[i - J_spline.start_idx]);
    }

    Sophus::Vector3d x0;
    x0.setZero();

    // Verify Jacobian using numerical differentiation
    test_jacobian(
        ss.str(), J_a,
        [&](const Sophus::Vector3d &x_rot) {
          Sophus::Vector6d x;
          x.setZero();
          x.head<3>() = x_rot;  // Only modify position part

          basalt::Se3Spline<N> s1 = s;
          s1.applyInc(i, x);

          return s1.positionResidual(t_ns, measurement);
        },
        x0);
  }
}

/**
 * @brief Test helper function for pose computation and its Jacobians
 *
 * Tests the computation of SE(3) poses and their Jacobians from the spline.
 * For each knot, verifies:
 * 1. Full SE(3) pose evaluation
 * 2. Jacobians with respect to spline control points
 * 3. Time derivative of the pose
 *
 * @param s SE(3) spline to test
 * @param t_ns Timestamp in nanoseconds for evaluation
 */
template <int N>
void testPose(const basalt::Se3Spline<N> &s, int64_t t_ns) {
  typename basalt::Se3Spline<N>::PosePosSO3JacobianStruct J_spline;

  // Get pose and Jacobians
  Sophus::SE3d res = s.pose(t_ns, &J_spline);

  Sophus::Vector6d x0;
  x0.setZero();

  // Test Jacobians for each knot
  for (size_t i = 0; i < s.numKnots(); i++) {
    std::stringstream ss;
    ss << "Spline order " << N << " d_pose_d_knot" << i;

    // Get analytical Jacobian for this knot
    typename basalt::Se3Spline<N>::Mat6 J_a;
    if (i >= J_spline.start_idx && i < J_spline.start_idx + N) {
      J_a = J_spline.d_val_d_knot[i - J_spline.start_idx];
    } else {
      J_a.setZero();
    }

    // Verify Jacobian using numerical differentiation
    test_jacobian(
        ss.str(), J_a,
        [&](const Sophus::Vector6d &x) {
          basalt::Se3Spline<N> s1 = s;
          s1.applyInc(i, x);
          return Sophus::se3_logd(res.inverse() * s1.pose(t_ns));
        },
        x0);
  }

  // Test time derivative of pose
  {
    Eigen::Matrix<double, 1, 1> x0;
    x0[0] = 0;

    typename basalt::Se3Spline<N>::Vec6 J_a;
    s.d_pose_d_t(t_ns, J_a);

    test_jacobian(
        "J_pose_time", J_a,
        [&](const Eigen::Matrix<double, 1, 1> &x) {
          int64_t t_ns_new = t_ns;
          t_ns_new += x[0] * 1e9;  // Convert to nanoseconds
          return Sophus::se3_logd(res.inverse() * s.pose(t_ns_new));
        },
        x0);
  }
}

/**
 * @brief Tests gyroscope residuals for SE(3) splines
 *
 * Creates a random SE(3) spline trajectory and tests gyroscope residuals
 * and their Jacobians at multiple time points along the trajectory.
 */
TEST(SplineSE3, GyroResidualTest) {
  static constexpr int N = 5;  // Quintic spline

  const int num_knots = 3 * N;           // Use 3N knots for sufficient testing
  basalt::Se3Spline<N> s(int64_t(2e9));  // 2 second trajectory
  s.genRandomTrajectory(num_knots);

  // Test at 100ms intervals
  for (int64_t t_ns = 0; t_ns < s.maxTimeNs(); t_ns += 1e8) {
    testGyroRes<N>(s, t_ns);
  }
}

/**
 * @brief Tests accelerometer residuals for SE(3) splines
 *
 * Creates a random SE(3) spline trajectory and tests accelerometer residuals
 * and their Jacobians at multiple time points along the trajectory.
 */
TEST(SplineSE3, AccelResidualTest) {
  static constexpr int N = 5;  // Quintic spline

  const int num_knots = 3 * N;           // Use 3N knots for sufficient testing
  basalt::Se3Spline<N> s(int64_t(2e9));  // 2 second trajectory
  s.genRandomTrajectory(num_knots);

  // Test at 100ms intervals
  for (int64_t t_ns = 0; t_ns < s.maxTimeNs(); t_ns += 1e8) {
    testAccelRes<N>(s, t_ns);
  }
}

/**
 * @brief Tests position residuals for SE(3) splines
 *
 * Creates a random SE(3) spline trajectory and tests position residuals
 * and their Jacobians at multiple time points along the trajectory.
 */
TEST(SplineSE3, PositionResidualTest) {
  static constexpr int N = 5;  // Quintic spline

  const int num_knots = 3 * N;           // Use 3N knots for sufficient testing
  basalt::Se3Spline<N> s(int64_t(2e9));  // 2 second trajectory
  s.genRandomTrajectory(num_knots);

  // Test at 100ms intervals
  for (int64_t t_ns = 0; t_ns < s.maxTimeNs(); t_ns += 1e8) {
    testPositionRes<N>(s, t_ns);
  }
}

/**
 * @brief Tests orientation residuals for SE(3) splines
 *
 * Creates a random SE(3) spline trajectory and tests orientation residuals
 * and their Jacobians at multiple time points along the trajectory.
 */
TEST(SplineSE3, OrientationResidualTest) {
  static constexpr int N = 5;  // Quintic spline

  const int num_knots = 3 * N;           // Use 3N knots for sufficient testing
  basalt::Se3Spline<N> s(int64_t(2e9));  // 2 second trajectory
  s.genRandomTrajectory(num_knots);

  // Test at 100ms intervals
  for (int64_t t_ns = 0; t_ns < s.maxTimeNs(); t_ns += 1e8) {
    testOrientationRes<N>(s, t_ns);
  }
}

/**
 * @brief Tests pose computation for SE(3) splines
 *
 * Creates a random SE(3) spline trajectory and tests pose computation
 * and its Jacobians at multiple time points along the trajectory.
 * Avoids testing too close to trajectory boundaries.
 */
TEST(SplineSE3, PoseTest) {
  static constexpr int N = 5;  // Quintic spline

  const int num_knots = 3 * N;           // Use 3N knots for sufficient testing
  basalt::Se3Spline<N> s(int64_t(2e9));  // 2 second trajectory
  s.genRandomTrajectory(num_knots);

  int64_t offset = 100;  // Avoid testing too close to boundaries

  // Test at 100ms intervals
  for (int64_t t_ns = offset; t_ns < s.maxTimeNs() - offset; t_ns += 1e8) {
    testPose<N>(s, t_ns);
  }
}
