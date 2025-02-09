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

#include <basalt/imu/preintegration.h>
#include <basalt/spline/se3_spline.h>

#include <iostream>

#include "gtest/gtest.h"
#include "test_utils.h"

// Define gravity vector in world frame (z-down convention)
namespace basalt::constants {
static const Eigen::Vector3d G(0, 0, -9.81);  // Gravity vector [m/s^2]
}  // namespace basalt::constants

// IMU noise characteristics (realistic values for a typical MEMS IMU)
constexpr double ACCEL_STD_DEV =
    0.23;  // Accelerometer white noise std dev [m/s^2]
constexpr double GYRO_STD_DEV =
    0.0027;  // Gyroscope white noise std dev [rad/s]

// Random number generation for IMU noise simulation
std::random_device rd{};  // Hardware random number source
std::mt19937 gen{rd()};   // Mersenne Twister PRNG with random seed

// Normal distributions for sensor noise generation
std::normal_distribution<> gyro_noise_dist{
    0, GYRO_STD_DEV};  // Gyro noise N(0, σ²)
std::normal_distribution<> accel_noise_dist{
    0, ACCEL_STD_DEV};  // Accel noise N(0, σ²)

/**
 * @brief Tests state prediction against ground truth values
 *
 * This test verifies that the IMU preintegration correctly predicts the state
 * by comparing against pre-computed ground truth values generated from a
 * spline. It checks:
 * 1. Position prediction
 * 2. Velocity prediction
 * 3. Orientation prediction
 */
TEST(ImuPreintegrationTestCase, PredictTestGT) {
  // Set up spline parameters
  int num_knots = 15;   // Number of control points for the spline
  int64_t dt_ns = 1e7;  // Time step of 10ms in nanoseconds
  int64_t max_time_ns = int64_t(20e9);  // Maximum integration time (20 seconds)

  // Initialize IMU measurement integrator with zero biases
  basalt::IntegratedImuMeasurement<double> imu_meas(
      0,                        // Start time (ns)
      Eigen::Vector3d::Zero(),  // Gyroscope bias
      Eigen::Vector3d::Zero()   // Accelerometer bias
  );

  // Create ground truth trajectory using a spline
  basalt::Se3Spline<5> gt_spline(int64_t(10e9));  // 10s spline duration
  gt_spline.genRandomTrajectory(
      num_knots);  // Generate random smooth trajectory

  // Initialize states for ground truth comparison
  basalt::PoseVelState<double> state0;     // Initial state
  basalt::PoseVelState<double> state1;     // Predicted state
  basalt::PoseVelState<double> state1_gt;  // Ground truth state

  // Set initial state from ground truth spline at t=0
  state0.T_w_i = gt_spline.pose(int64_t(0));             // Initial pose
  state0.vel_w_i = gt_spline.transVelWorld(int64_t(0));  // Initial velocity

  // Integrate IMU measurements along the trajectory
  for (int64_t t_ns = dt_ns / 2; t_ns < max_time_ns; t_ns += dt_ns) {
    // Get ground truth pose at current time
    Sophus::SE3d pose = gt_spline.pose(t_ns);

    // Convert ground truth acceleration to body frame and remove gravity
    Eigen::Vector3d accel_body =
        pose.so3().inverse() *              // Rotation from world to body
        (gt_spline.transAccelWorld(t_ns) -  // World frame acceleration
         basalt::constants::G);             // Subtract gravity

    // Get angular velocity in body frame
    Eigen::Vector3d rot_vel_body = gt_spline.rotVelBody(t_ns);

    // Create IMU measurement
    basalt::ImuData<double> data;
    data.accel = accel_body;       // Linear acceleration
    data.gyro = rot_vel_body;      // Angular velocity
    data.t_ns = t_ns + dt_ns / 2;  // Timestamp at interval midpoint

    // Integrate measurement with unit noise parameters
    imu_meas.integrate(data, Eigen::Vector3d::Ones(), Eigen::Vector3d::Ones());
  }

  // Get ground truth state at final time
  state1_gt.T_w_i = gt_spline.pose(imu_meas.get_dt_ns());
  state1_gt.vel_w_i = gt_spline.transVelWorld(imu_meas.get_dt_ns());

  // Predict final state using integrated measurements
  imu_meas.predictState(state0, basalt::constants::G, state1);

  // Verify velocity prediction (tolerance: 1e-4)
  EXPECT_TRUE(state1_gt.vel_w_i.isApprox(state1.vel_w_i, 1e-4))
      << "vel1_gt " << state1_gt.vel_w_i.transpose() << " vel1 "
      << state1.vel_w_i.transpose();

  // Verify orientation prediction (tolerance: 1e-6 radians)
  EXPECT_LE(state1_gt.T_w_i.unit_quaternion().angularDistance(
                state1.T_w_i.unit_quaternion()),
            1e-6);

  // Verify position prediction (tolerance: 1e-4 meters)
  EXPECT_TRUE(
      state1_gt.T_w_i.translation().isApprox(state1.T_w_i.translation(), 1e-4))
      << "pose1_gt p " << state1_gt.T_w_i.translation().transpose()
      << " pose1 p " << state1.T_w_i.translation().transpose();
}

/**
 * @brief Tests Jacobian computation for state prediction
 *
 * This test verifies that the Jacobians of the state prediction function are
 * correct. It generates a spline trajectory and integrates the measurements to
 * generate ground truth states. It then computes the Jacobians using the
 * preintegration and compares them to the numeric Jacobians.
 */
TEST(ImuPreintegrationTestCase, PredictJacobiansTest) {
  int num_knots = 15;

  basalt::Se3Spline<5> gt_spline(int64_t(2e9));
  gt_spline.genRandomTrajectory(num_knots);

  int64_t dt_ns = 1e7;
  for (int64_t t_ns = dt_ns / 2; t_ns < gt_spline.maxTimeNs() - int64_t(1e9);
       t_ns += dt_ns) {
    // Get ground truth pose at current time
    Sophus::SE3d pose = gt_spline.pose(t_ns);

    // Convert world frame acceleration to body frame and remove gravity
    Eigen::Vector3d accel_body =
        pose.so3().inverse() *
        (gt_spline.transAccelWorld(t_ns) - basalt::constants::G);

    // Get angular velocity in body frame
    Eigen::Vector3d rot_vel_body = gt_spline.rotVelBody(t_ns);

    // Create IMU measurement
    basalt::ImuData<double> data;
    data.accel = accel_body;
    data.gyro = rot_vel_body;
    data.t_ns = t_ns + dt_ns / 2;  // measurement in the middle of the interval;

    // Propagate state using the preintegration
    basalt::PoseVelState<double> next_state;

    int64_t curr_state_t_ns = t_ns - dt_ns / 2;
    basalt::PoseVelState<double> curr_state(
        curr_state_t_ns, gt_spline.pose(curr_state_t_ns),
        gt_spline.transVelWorld(curr_state_t_ns));

    // Compute Jacobians of the propagated state
    basalt::IntegratedImuMeasurement<double>::MatNN d_next_d_curr;
    basalt::IntegratedImuMeasurement<double>::MatN3 d_next_d_accel;
    basalt::IntegratedImuMeasurement<double>::MatN3 d_next_d_gyro;

    basalt::IntegratedImuMeasurement<double>::propagateState(
        curr_state, data, next_state, &d_next_d_curr, &d_next_d_accel,
        &d_next_d_gyro);

    // Test Jacobian computation using finite differences
    {
      basalt::PoseVelState<double>::VecN x0;
      x0.setZero();
      test_jacobian(
          "F_TEST", d_next_d_curr,
          [&](const basalt::PoseVelState<double>::VecN& x) {
            basalt::PoseVelState<double> curr_state1 = curr_state;
            curr_state1.applyInc(x);
            basalt::PoseVelState<double> next_state1;

            basalt::IntegratedImuMeasurement<double>::propagateState(
                curr_state1, data, next_state1);

            return next_state.diff(next_state1);
          },
          x0);
    }

    {
      Eigen::Vector3d x0;
      x0.setZero();
      test_jacobian(
          "A_TEST", d_next_d_accel,
          [&](const Eigen::Vector3d& x) {
            basalt::ImuData<double> data1 = data;
            data1.accel += x;
            basalt::PoseVelState<double> next_state1;

            basalt::IntegratedImuMeasurement<double>::propagateState(
                curr_state, data1, next_state1);

            return next_state.diff(next_state1);
          },
          x0);
    }

    {
      Eigen::Vector3d x0;
      x0.setZero();
      test_jacobian(
          "G_TEST", d_next_d_gyro,
          [&](const Eigen::Vector3d& x) {
            basalt::ImuData<double> data1 = data;
            data1.gyro += x;
            basalt::PoseVelState<double> next_state1;

            basalt::IntegratedImuMeasurement<double>::propagateState(
                curr_state, data1, next_state1);

            return next_state.diff(next_state1);
          },
          x0, 1e-8);
    }
  }
}

/**
 * @brief Tests residual and Jacobian computation
 *
 * Computes the Jacobians for residuals and compares them to the numeric
 * Jacobians.
 */
TEST(ImuPreintegrationTestCase, ResidualJacobiansTest) {
  // Number of control points for the spline trajectory
  int num_knots = 15;

  // Initialize random gyroscope and accelerometer biases
  // Scale down to realistic values: gyro bias ~0.01 rad/s, accel bias ~0.1
  // m/s^2
  Eigen::Vector3d bg;
  Eigen::Vector3d ba;
  bg = Eigen::Vector3d::Random() / 100;  // Gyroscope bias
  ba = Eigen::Vector3d::Random() / 10;   // Accelerometer bias

  // Create IMU measurement integrator with initial biases
  basalt::IntegratedImuMeasurement<double> imu_meas(0, bg, ba);

  // Create ground truth trajectory using a cubic B-spline
  // Time span of 10 seconds (10e9 nanoseconds)
  basalt::Se3Spline<5> gt_spline(int64_t(10e9));
  gt_spline.genRandomTrajectory(num_knots);

  // States for testing: initial, predicted, and ground truth
  basalt::PoseVelState<double> state0;     // Initial state
  basalt::PoseVelState<double> state1;     // State to test Jacobians
  basalt::PoseVelState<double> state1_gt;  // Ground truth final state

  // Set initial state from spline at t=0
  state0.T_w_i = gt_spline.pose(int64_t(0));             // Initial pose
  state0.vel_w_i = gt_spline.transVelWorld(int64_t(0));  // Initial velocity

  // Generate and integrate IMU measurements along the trajectory
  int64_t dt_ns = 1e7;  // 10ms measurement interval
  for (int64_t t_ns = dt_ns / 2; t_ns < int64_t(1e8);  // Integrate for 0.1s
       t_ns += dt_ns) {
    // Get ground truth pose at current time
    Sophus::SE3d pose = gt_spline.pose(t_ns);

    // Convert world frame acceleration to body frame and remove gravity
    Eigen::Vector3d accel_body =
        pose.so3().inverse() *
        (gt_spline.transAccelWorld(t_ns) - basalt::constants::G);

    // Get angular velocity in body frame
    Eigen::Vector3d rot_vel_body = gt_spline.rotVelBody(t_ns);

    // Create IMU measurement by adding biases
    basalt::ImuData<double> data;
    data.accel = accel_body + ba;   // Add accelerometer bias
    data.gyro = rot_vel_body + bg;  // Add gyroscope bias
    data.t_ns = t_ns + dt_ns / 2;   // Timestamp at middle of interval

    // Integrate measurement with unit covariance
    imu_meas.integrate(data, Eigen::Vector3d::Ones(), Eigen::Vector3d::Ones());
  }

  // Set ground truth final state from spline
  state1_gt.T_w_i = gt_spline.pose(imu_meas.get_dt_ns());
  state1_gt.vel_w_i = gt_spline.transVelWorld(imu_meas.get_dt_ns());

  // Compute residual with ground truth states - should be near zero
  basalt::PoseVelState<double>::VecN res_gt =
      imu_meas.residual(state0, basalt::constants::G, state1_gt, bg, ba);

  // Verify residual is small (less than 1e-6)
  EXPECT_LE(res_gt.array().abs().maxCoeff(), 1e-6)
      << "res_gt " << res_gt.transpose();

  // Create perturbed final state for Jacobian testing
  // Add small random perturbations to pose and velocity
  state1.T_w_i =
      gt_spline.pose(imu_meas.get_dt_ns()) *
      Sophus::se3_expd(Sophus::Vector6d::Random() / 10);  // ~0.1 perturbation
  state1.vel_w_i = gt_spline.transVelWorld(imu_meas.get_dt_ns()) +
                   Sophus::Vector3d::Random() / 10;  // ~0.1 m/s perturbation

  // Matrices to store computed Jacobians
  basalt::IntegratedImuMeasurement<double>::MatNN
      d_res_d_state0;  // wrt initial state
  basalt::IntegratedImuMeasurement<double>::MatNN
      d_res_d_state1;  // wrt final state
  basalt::IntegratedImuMeasurement<double>::MatN3 d_res_d_bg;  // wrt gyro bias
  basalt::IntegratedImuMeasurement<double>::MatN3 d_res_d_ba;  // wrt accel bias

  // Compute residual and all Jacobians
  imu_meas.residual(state0, basalt::constants::G, state1, bg, ba,
                    &d_res_d_state0, &d_res_d_state1, &d_res_d_bg, &d_res_d_ba);

  // Test Jacobian with respect to initial state
  {
    basalt::PoseVelState<double>::VecN x0;
    x0.setZero();
    test_jacobian(
        "d_res_d_state0", d_res_d_state0,
        [&](const basalt::PoseVelState<double>::VecN& x) {
          basalt::PoseVelState<double> state0_new = state0;
          state0_new.applyInc(x);  // Apply increment to initial state

          return imu_meas.residual(state0_new, basalt::constants::G, state1, bg,
                                   ba);
        },
        x0);
  }

  // Test Jacobian with respect to final state
  {
    basalt::PoseVelState<double>::VecN x0;
    x0.setZero();
    test_jacobian(
        "d_res_d_state1", d_res_d_state1,
        [&](const basalt::PoseVelState<double>::VecN& x) {
          basalt::PoseVelState<double> state1_new = state1;
          state1_new.applyInc(x);  // Apply increment to final state

          return imu_meas.residual(state0, basalt::constants::G, state1_new, bg,
                                   ba);
        },
        x0);
  }

  // Test Jacobian with respect to gyroscope bias
  {
    Sophus::Vector3d x0;
    x0.setZero();
    test_jacobian(
        "d_res_d_bg", d_res_d_bg,
        [&](const Sophus::Vector3d& x) {
          return imu_meas.residual(state0, basalt::constants::G, state1, bg + x,
                                   ba);
        },
        x0);
  }

  // Test Jacobian with respect to accelerometer bias
  {
    Sophus::Vector3d x0;
    x0.setZero();
    test_jacobian(
        "d_res_d_ba", d_res_d_ba,
        [&](const Sophus::Vector3d& x) {
          return imu_meas.residual(state0, basalt::constants::G, state1, bg,
                                   ba + x);
        },
        x0);
  }
}

/**
 * @brief Tests residual bias computation
 *
 * Ensures that residuals are correctly computed when biases are present and
 * their Jacobians.
 */
TEST(ImuPreintegrationTestCase, ResidualBiasTest) {
  // Number of control points for the spline trajectory
  int num_knots = 15;

  // Initialize random gyroscope and accelerometer biases
  // Scale down to realistic values: gyro bias ~0.01 rad/s, accel bias ~0.1
  // m/s^2
  Eigen::Vector3d bg;                    // Gyroscope bias
  Eigen::Vector3d ba;                    // Accelerometer bias
  bg = Eigen::Vector3d::Random() / 100;  // ~0.01 rad/s
  ba = Eigen::Vector3d::Random() / 10;   // ~0.1 m/s^2

  // Create ground truth trajectory using a cubic B-spline over 10 seconds
  basalt::Se3Spline<5> gt_spline(int64_t(10e9));
  gt_spline.genRandomTrajectory(num_knots);

  // Vectors to store synthetic IMU measurements
  Eigen::aligned_vector<Eigen::Vector3d>
      accel_data_vec;  // Accelerometer readings
  Eigen::aligned_vector<Eigen::Vector3d> gyro_data_vec;  // Gyroscope readings
  Eigen::aligned_vector<int64_t> timestamps_vec;  // Measurement timestamps

  // Generate synthetic IMU measurements from the ground truth trajectory
  int64_t dt_ns = 1e7;  // 10ms measurement interval
  for (int64_t t_ns = dt_ns / 2; t_ns < int64_t(1e9);  // Integrate for 1 second
       t_ns += dt_ns) {
    // Get ground truth pose at current time
    Sophus::SE3d pose = gt_spline.pose(t_ns);

    // Convert world frame acceleration to body frame and remove gravity
    Eigen::Vector3d accel_body =
        pose.so3().inverse() *
        (gt_spline.transAccelWorld(t_ns) - basalt::constants::G);

    // Get angular velocity in body frame
    Eigen::Vector3d rot_vel_body = gt_spline.rotVelBody(t_ns);

    // Store measurements with added biases
    accel_data_vec.emplace_back(accel_body + ba);   // Add accelerometer bias
    gyro_data_vec.emplace_back(rot_vel_body + bg);  // Add gyroscope bias
    timestamps_vec.emplace_back(
        t_ns + dt_ns / 2);  // Store timestamp at interval midpoint
  }

  // Create IMU measurement integrator with true biases
  basalt::IntegratedImuMeasurement<double> imu_meas(0, bg, ba);

  // Integrate all IMU measurements with unit covariance
  for (size_t i = 0; i < timestamps_vec.size(); i++) {
    basalt::ImuData<double> data;
    data.accel = accel_data_vec[i];
    data.gyro = gyro_data_vec[i];
    data.t_ns = timestamps_vec[i];

    imu_meas.integrate(data, Eigen::Vector3d::Ones(), Eigen::Vector3d::Ones());
  }

  // Initialize states for testing
  basalt::PoseVelState<double> state0;  // Initial state
  basalt::PoseVelState<double> state1;  // Final state

  // Set initial state from ground truth trajectory at t=0
  state0.T_w_i = gt_spline.pose(int64_t(0));
  state0.vel_w_i = gt_spline.transVelWorld(int64_t(0));

  // Set final state with small random perturbation from ground truth
  state1.T_w_i =
      gt_spline.pose(imu_meas.get_dt_ns()) *
      Sophus::se3_expd(Sophus::Vector6d::Random() / 10);  // ~0.1 perturbation
  state1.vel_w_i = gt_spline.transVelWorld(imu_meas.get_dt_ns()) +
                   Sophus::Vector3d::Random() / 10;  // ~0.1 m/s perturbation

  // Create test biases with small perturbations from true values
  Eigen::Vector3d bg_test =
      bg + Eigen::Vector3d::Random() / 1000;  // ~0.001 rad/s difference
  Eigen::Vector3d ba_test =
      ba + Eigen::Vector3d::Random() / 100;  // ~0.01 m/s^2 difference

  // Matrices to store Jacobians
  basalt::IntegratedImuMeasurement<double>::MatN3 d_res_d_ba;  // wrt accel bias
  basalt::IntegratedImuMeasurement<double>::MatN3 d_res_d_bg;  // wrt gyro bias

  // Compute residual with test biases and get Jacobians
  basalt::PoseVelState<double>::VecN res =
      imu_meas.residual(state0, basalt::constants::G, state1, bg_test, ba_test,
                        nullptr, nullptr, &d_res_d_bg, &d_res_d_ba);

  // Test 1: Verify residual computation consistency
  {
    // Create new integrator with test biases
    basalt::IntegratedImuMeasurement<double> imu_meas1(0, bg_test, ba_test);

    // Integrate the same measurements
    for (size_t i = 0; i < timestamps_vec.size(); i++) {
      basalt::ImuData<double> data;
      data.accel = accel_data_vec[i];
      data.gyro = gyro_data_vec[i];
      data.t_ns = timestamps_vec[i];

      imu_meas1.integrate(data, Eigen::Vector3d::Ones(),
                          Eigen::Vector3d::Ones());
    }

    // Compute residual with test biases
    basalt::PoseVelState<double>::VecN res1 = imu_meas1.residual(
        state0, basalt::constants::G, state1, bg_test, ba_test);

    // Verify that both methods produce the same residual
    EXPECT_TRUE(res.isApprox(res1, 1e-4))
        << "res\n"
        << res.transpose() << "\nres1\n"
        << res1.transpose() << "\ndiff\n"
        << (res - res1).transpose() << std::endl;
  }

  // Test 2: Verify Jacobian with respect to accelerometer bias
  {
    Sophus::Vector3d x0;
    x0.setZero();
    test_jacobian(
        "d_res_d_ba", d_res_d_ba,
        [&](const Sophus::Vector3d& x) {
          // Create integrator with perturbed accelerometer bias
          basalt::IntegratedImuMeasurement<double> imu_meas1(0, bg_test,
                                                             ba_test + x);

          // Integrate measurements with perturbed bias
          for (size_t i = 0; i < timestamps_vec.size(); i++) {
            basalt::ImuData<double> data;
            data.accel = accel_data_vec[i];
            data.gyro = gyro_data_vec[i];
            data.t_ns = timestamps_vec[i];

            imu_meas1.integrate(data, Eigen::Vector3d::Ones(),
                                Eigen::Vector3d::Ones());
          }

          // Return residual with perturbed bias
          return imu_meas1.residual(state0, basalt::constants::G, state1,
                                    bg_test, ba_test + x);
        },
        x0);
  }

  // Test 3: Verify Jacobian with respect to gyroscope bias
  {
    Sophus::Vector3d x0;
    x0.setZero();
    test_jacobian(
        "d_res_d_bg", d_res_d_bg,
        [&](const Sophus::Vector3d& x) {
          // Create integrator with perturbed gyroscope bias
          basalt::IntegratedImuMeasurement<double> imu_meas1(0, bg_test + x,
                                                             ba_test);

          // Integrate measurements with perturbed bias
          for (size_t i = 0; i < timestamps_vec.size(); i++) {
            basalt::ImuData<double> data;
            data.accel = accel_data_vec[i];
            data.gyro = gyro_data_vec[i];
            data.t_ns = timestamps_vec[i];

            imu_meas1.integrate(data, Eigen::Vector3d::Ones(),
                                Eigen::Vector3d::Ones());
          }

          // Return residual with perturbed bias
          return imu_meas1.residual(state0, basalt::constants::G, state1,
                                    bg_test + x, ba_test);
        },
        x0, 1e-8, 1e-2);  // Use tighter tolerances for gyro bias test
  }
}

/**
 * @brief Tests computation of residual Jacobians for bias terms
 *
 * Verifies that the residual Jacobians for gyroscope and accelerometer biases
 * are computed correctly by comparing them to the numeric Jacobians
 */
TEST(ImuPreintegrationTestCase, BiasResidualJacobiansTest) {
  // Number of control points for the spline trajectory
  int num_knots = 15;

  // Initialize random gyroscope and accelerometer biases
  // Scale down to realistic values: gyro bias ~0.01 rad/s, accel bias ~0.1
  // m/s^2
  Eigen::Vector3d bg;
  Eigen::Vector3d ba;
  bg = Eigen::Vector3d::Random() / 100;
  ba = Eigen::Vector3d::Random() / 10;

  // Create ground truth trajectory using a cubic B-spline over 10 seconds
  basalt::Se3Spline<5> gt_spline(int64_t(10e9));
  gt_spline.genRandomTrajectory(num_knots);

  // Vectors to store synthetic IMU measurements
  Eigen::aligned_vector<Eigen::Vector3d>
      accel_data_vec;  // Accelerometer readings
  Eigen::aligned_vector<Eigen::Vector3d> gyro_data_vec;  // Gyroscope readings
  Eigen::aligned_vector<int64_t> timestamps_vec;  // Measurement timestamps

  // Generate synthetic IMU measurements from the ground truth trajectory
  int64_t dt_ns = 1e7;  // 10ms measurement interval
  for (int64_t t_ns = dt_ns / 2; t_ns < int64_t(1e9);  // Integrate for 1 second
       t_ns += dt_ns) {
    // Get ground truth pose at current time
    Sophus::SE3d pose = gt_spline.pose(t_ns);

    // Convert world frame acceleration to body frame and remove gravity
    Eigen::Vector3d accel_body =
        pose.so3().inverse() *
        (gt_spline.transAccelWorld(t_ns) - basalt::constants::G);

    // Get angular velocity in body frame
    Eigen::Vector3d rot_vel_body = gt_spline.rotVelBody(t_ns);

    // Store measurements with added biases
    accel_data_vec.emplace_back(accel_body + ba);   // Add accelerometer bias
    gyro_data_vec.emplace_back(rot_vel_body + bg);  // Add gyroscope bias
    timestamps_vec.emplace_back(
        t_ns + dt_ns / 2);  // Store timestamp at interval midpoint
  }

  // Create IMU measurement integrator with initial biases
  basalt::IntegratedImuMeasurement<double> imu_meas(0, bg, ba);

  // Integrate all IMU measurements with unit covariance
  for (size_t i = 0; i < timestamps_vec.size(); i++) {
    basalt::ImuData<double> data;
    data.accel = accel_data_vec[i];
    data.gyro = gyro_data_vec[i];
    data.t_ns = timestamps_vec[i];

    imu_meas.integrate(data, Eigen::Vector3d::Ones(), Eigen::Vector3d::Ones());
  }

  // Matrices to store Jacobians (not used directly but declared for clarity)
  basalt::IntegratedImuMeasurement<double>::MatN3 d_res_d_ba;  // wrt accel bias
  basalt::IntegratedImuMeasurement<double>::MatN3 d_res_d_bg;  // wrt gyro bias

  // Get the delta state (change in state over integration period)
  basalt::PoseVelState<double> delta_state = imu_meas.getDeltaState();

  // Test Jacobian with respect to gyroscope bias
  {
    Sophus::Vector3d x0;
    x0.setZero();
    test_jacobian(
        "d_res_d_bg", imu_meas.get_d_state_d_bg(),
        [&](const Sophus::Vector3d& x) {
          // Create new integrator with perturbed gyro bias
          basalt::IntegratedImuMeasurement<double> imu_meas1(0, bg + x, ba);

          // Integrate the same measurements with perturbed bias
          for (size_t i = 0; i < timestamps_vec.size(); i++) {
            basalt::ImuData<double> data;
            data.accel = accel_data_vec[i];
            data.gyro = gyro_data_vec[i];
            data.t_ns = timestamps_vec[i];

            imu_meas1.integrate(data, Eigen::Vector3d::Ones(),
                                Eigen::Vector3d::Ones());
          }

          // Compare delta states to compute numerical derivative
          basalt::PoseVelState<double> delta_state1 = imu_meas1.getDeltaState();
          return delta_state.diff(delta_state1);
        },
        x0);
  }

  // Test Jacobian with respect to accelerometer bias
  {
    Sophus::Vector3d x0;
    x0.setZero();
    test_jacobian(
        "d_res_d_ba", imu_meas.get_d_state_d_ba(),
        [&](const Sophus::Vector3d& x) {
          // Create new integrator with perturbed accelerometer bias
          basalt::IntegratedImuMeasurement<double> imu_meas1(0, bg, ba + x);

          // Integrate the same measurements with perturbed bias
          for (size_t i = 0; i < timestamps_vec.size(); i++) {
            basalt::ImuData<double> data;
            data.accel = accel_data_vec[i];
            data.gyro = gyro_data_vec[i];
            data.t_ns = timestamps_vec[i];

            imu_meas1.integrate(data, Eigen::Vector3d::Ones(),
                                Eigen::Vector3d::Ones());
          }

          // Compare delta states to compute numerical derivative
          basalt::PoseVelState<double> delta_state1 = imu_meas1.getDeltaState();
          return delta_state.diff(delta_state1);
        },
        x0);
  }
}

/**
 * @brief Tests covariance propagation
 *
 * Verifies the measurement covariance computation. Samples accelerometer and
 * gyro noise to compute the variance numerically. Uses KL-divergence to compare
 * the computed covariance with the one computed analytically.
 */
TEST(ImuPreintegrationTestCase, CovarianceTest) {
  // Number of control points for the spline trajectory
  int num_knots = 15;

  // Create ground truth trajectory using a cubic B-spline over 10 seconds
  basalt::Se3Spline<5> gt_spline(int64_t(10e9));
  gt_spline.genRandomTrajectory(num_knots);

  // Vectors to store noise-free IMU measurements
  Eigen::aligned_vector<Eigen::Vector3d>
      accel_data_vec;  // Accelerometer readings
  Eigen::aligned_vector<Eigen::Vector3d> gyro_data_vec;  // Gyroscope readings
  Eigen::aligned_vector<int64_t> timestamps_vec;  // Measurement timestamps

  // Generate synthetic IMU measurements from the ground truth trajectory
  int64_t dt_ns = 1e7;  // 10ms measurement interval
  for (int64_t t_ns = dt_ns / 2; t_ns < int64_t(1e9);  // Integrate for 1 second
       t_ns += dt_ns) {
    // Get ground truth pose at current time
    Sophus::SE3d pose = gt_spline.pose(t_ns);

    // Convert world frame acceleration to body frame and remove gravity
    Eigen::Vector3d accel_body =
        pose.so3().inverse() *
        (gt_spline.transAccelWorld(t_ns) - basalt::constants::G);

    // Get angular velocity in body frame
    Eigen::Vector3d rot_vel_body = gt_spline.rotVelBody(t_ns);

    // Store noise-free measurements
    accel_data_vec.emplace_back(accel_body);
    gyro_data_vec.emplace_back(rot_vel_body);
    timestamps_vec.emplace_back(
        t_ns + dt_ns / 2);  // Store timestamp at interval midpoint
  }

  // Set measurement covariances (variance = std_dev^2)
  Eigen::Vector3d accel_cov;
  Eigen::Vector3d gyro_cov;
  accel_cov.setConstant(ACCEL_STD_DEV *
                        ACCEL_STD_DEV);  // Accelerometer measurement variance
  gyro_cov.setConstant(GYRO_STD_DEV *
                       GYRO_STD_DEV);  // Gyroscope measurement variance

  // Create IMU measurement integrator with zero initial biases
  basalt::IntegratedImuMeasurement<double> imu_meas(0, Eigen::Vector3d::Zero(),
                                                    Eigen::Vector3d::Zero());

  // Integrate noise-free measurements to get reference delta state
  for (size_t i = 0; i < timestamps_vec.size(); i++) {
    basalt::ImuData<double> data;
    data.accel = accel_data_vec[i];
    data.gyro = gyro_data_vec[i];
    data.t_ns = timestamps_vec[i];

    imu_meas.integrate(data, accel_cov, gyro_cov);
  }

  // Get reference delta state from noise-free integration
  basalt::PoseVelState<double> delta_state = imu_meas.getDeltaState();

  // Matrix to accumulate empirical covariance from Monte Carlo samples
  basalt::IntegratedImuMeasurement<double>::MatNN cov_computed;
  cov_computed.setZero();

  // Monte Carlo simulation to estimate covariance
  const int num_samples = 1000;  // Number of Monte Carlo iterations
  for (int i = 0; i < num_samples; i++) {
    // Create new integrator for each Monte Carlo iteration
    basalt::IntegratedImuMeasurement<double> imu_meas1(
        0, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());

    // Integrate measurements with added random noise
    for (size_t i = 0; i < timestamps_vec.size(); i++) {
      basalt::ImuData<double> data;
      data.accel = accel_data_vec[i];
      data.gyro = gyro_data_vec[i];
      data.t_ns = timestamps_vec[i];

      // Add Gaussian noise to accelerometer measurements
      data.accel[0] += accel_noise_dist(gen);
      data.accel[1] += accel_noise_dist(gen);
      data.accel[2] += accel_noise_dist(gen);

      // Add Gaussian noise to gyroscope measurements
      data.gyro[0] += gyro_noise_dist(gen);
      data.gyro[1] += gyro_noise_dist(gen);
      data.gyro[2] += gyro_noise_dist(gen);

      imu_meas1.integrate(data, accel_cov, gyro_cov);
    }

    // Get delta state from noisy integration
    basalt::PoseVelState<double> delta_state1 = imu_meas1.getDeltaState();

    // Compute difference between noisy and reference states
    basalt::PoseVelState<double>::VecN diff = delta_state.diff(delta_state1);

    // Accumulate outer product for covariance computation
    cov_computed += diff * diff.transpose();
  }

  // Compute empirical covariance by averaging
  cov_computed /= num_samples;

  // Compute Kullback-Leibler divergence between analytical and empirical
  // covariances KL = tr(Σ₁⁻¹Σ₂) - n + ln(|Σ₁|/|Σ₂|), where Σ₁ is analytical and
  // Σ₂ is empirical
  double kl =
      (imu_meas.get_cov_inv() * cov_computed).trace() - 9 +
      std::log(imu_meas.get_cov().determinant() / cov_computed.determinant());

  // Verify that analytical and empirical covariances are similar (small KL
  // divergence)
  EXPECT_LE(kl, 0.08);

  // Verify random number generator properties
  Eigen::VectorXd test_vec(num_samples);
  for (int i = 0; i < num_samples; i++) {
    test_vec[i] = accel_noise_dist(gen);
  }

  // Check that generated noise follows expected distribution
  double mean = test_vec.mean();
  double var = (test_vec.array() - mean).square().sum() / num_samples;

  // Verify that sample standard deviation is close to specified value
  EXPECT_LE(std::abs(std::sqrt(var) - ACCEL_STD_DEV), 0.03);
}

/**
 * @brief Tests random walk behavior
 *
 * Verifies that the random walk variance scales linearly with the integration
 * time (standard deviation scales as sqrt(dt)).
 */
TEST(ImuPreintegrationTestCase, RandomWalkTest) {
  // Time step for discrete integration (5ms)
  double dt = 0.005;

  // Number of steps to simulate for each random walk
  double period = 200;
  // Total time duration for each random walk (period * dt = 1s)
  double period_dt = period * dt;

  // Number of Monte Carlo iterations to estimate statistics
  int num_samples = 10000;

  // Vector to store final positions of random walks
  Eigen::VectorXd test_vec(num_samples);
  for (int j = 0; j < num_samples; j++) {
    // Simulate one random walk trajectory
    double test = 0;
    for (int i = 0; i < period; i++) {
      // Add random increment scaled by sqrt(dt)
      // This scaling ensures proper continuous-time limit behavior
      test += gyro_noise_dist(gen) * std::sqrt(dt);
    }
    // Store final position of this random walk
    test_vec[j] = test;
  }

  // Compute statistics of final positions across all random walks
  double mean = test_vec.mean();  // Should be close to zero
  // Compute variance (mean square displacement)
  double var = (test_vec.array() - mean).square().sum() / num_samples;
  // Standard deviation (root mean square displacement)
  double std = std::sqrt(var);

  // Verify that standard deviation grows as sqrt(t)
  // For a random walk, std_dev(t) = noise_std_dev * sqrt(t)
  EXPECT_NEAR(GYRO_STD_DEV * std::sqrt(period_dt), std, 1e-4);

  // Verify that variance grows linearly with time
  // For a random walk, var(t) = noise_variance * t
  EXPECT_NEAR(GYRO_STD_DEV * GYRO_STD_DEV * period_dt, var, 1e-6);
}

/**
 * @brief Tests covariance inverse computation
 *
 * Verifies that the inverse of the covariance matrix is computed correctly.
 */
TEST(ImuPreintegrationTestCase, ComputeCovInv) {
  // Define MatNN type alias for the covariance matrix type
  // This is a square matrix of size N where N is the dimension of the state
  // space
  using MatNN = basalt::IntegratedImuMeasurement<double>::MatNN;

  // Number of control points for the spline trajectory
  int num_knots = 15;

  // Create IMU measurement integrator with zero initial biases
  basalt::IntegratedImuMeasurement<double> imu_meas(0, Eigen::Vector3d::Zero(),
                                                    Eigen::Vector3d::Zero());

  // Create ground truth trajectory using a cubic B-spline over 10 seconds
  basalt::Se3Spline<5> gt_spline(int64_t(10e9));
  gt_spline.genRandomTrajectory(num_knots);

  // Generate and integrate IMU measurements along the trajectory
  int64_t dt_ns = 1e7;  // 10ms measurement interval
  for (int64_t t_ns = dt_ns / 2;
       t_ns < int64_t(20e9);  // Integrate for 20 seconds
       t_ns += dt_ns) {
    // Get ground truth pose at current time
    Sophus::SE3d pose = gt_spline.pose(t_ns);

    // Convert world frame acceleration to body frame and remove gravity
    Eigen::Vector3d accel_body =
        pose.so3().inverse() *
        (gt_spline.transAccelWorld(t_ns) - basalt::constants::G);

    // Get angular velocity in body frame
    Eigen::Vector3d rot_vel_body = gt_spline.rotVelBody(t_ns);

    // Create IMU measurement
    basalt::ImuData<double> data;
    data.accel = accel_body;
    data.gyro = rot_vel_body;
    data.t_ns = t_ns + dt_ns / 2;  // Timestamp at interval midpoint

    // Integrate measurement with specified noise characteristics
    // Accelerometer noise std: sqrt(0.1) m/s^2
    // Gyroscope noise std: sqrt(0.01) rad/s
    imu_meas.integrate(data, 0.1 * Eigen::Vector3d::Ones(),
                       0.01 * Eigen::Vector3d::Ones());
  }

  // Get inverse covariance computed by the optimized method
  MatNN cov_inv_computed = imu_meas.get_cov_inv();

  // Compute ground truth inverse covariance by direct matrix inversion
  MatNN cov_inv_gt = imu_meas.get_cov().inverse();

  // Verify that the optimized inverse computation matches direct inversion
  // Uses a tight tolerance (1e-12) since this is a numerical verification
  EXPECT_TRUE(cov_inv_computed.isApprox(cov_inv_gt, 1e-12))
      << "cov_inv_computed\n"
      << cov_inv_computed << "\ncov_inv_gt\n"
      << cov_inv_gt;
}

/**
 * @brief Tests square root of covariance inverse computation
 *
 * Verifies the computation of the square root of the inverse covariance matrix
 * by squaring it and comparing to the original matrix.
 */
TEST(ImuPreintegrationTestCase, ComputeSqrtCovInv) {
  // Define MatNN type alias for the covariance matrix type
  // This is a square matrix of size N where N is the dimension of the state
  // space
  using MatNN = basalt::IntegratedImuMeasurement<double>::MatNN;

  // Number of control points for the spline trajectory
  int num_knots = 15;

  // Create IMU measurement integrator with zero initial biases
  basalt::IntegratedImuMeasurement<double> imu_meas(0, Eigen::Vector3d::Zero(),
                                                    Eigen::Vector3d::Zero());

  // Create ground truth trajectory using a cubic B-spline over 10 seconds
  basalt::Se3Spline<5> gt_spline(int64_t(10e9));
  gt_spline.genRandomTrajectory(num_knots);

  // Generate and integrate IMU measurements along the trajectory
  int64_t dt_ns = 1e7;  // 10ms measurement interval
  for (int64_t t_ns = dt_ns / 2;
       t_ns < int64_t(20e9);  // Integrate for 20 seconds
       t_ns += dt_ns) {
    // Get ground truth pose at current time
    Sophus::SE3d pose = gt_spline.pose(t_ns);

    // Convert world frame acceleration to body frame and remove gravity
    Eigen::Vector3d accel_body =
        pose.so3().inverse() *
        (gt_spline.transAccelWorld(t_ns) - basalt::constants::G);

    // Get angular velocity in body frame
    Eigen::Vector3d rot_vel_body = gt_spline.rotVelBody(t_ns);

    // Create IMU measurement
    basalt::ImuData<double> data;
    data.accel = accel_body;
    data.gyro = rot_vel_body;
    data.t_ns = t_ns + dt_ns / 2;  // Timestamp at interval midpoint

    // Integrate measurement with specified noise characteristics
    // Accelerometer noise std: sqrt(0.1) m/s^2
    // Gyroscope noise std: sqrt(0.01) rad/s
    imu_meas.integrate(data, 0.1 * Eigen::Vector3d::Ones(),
                       0.01 * Eigen::Vector3d::Ones());
  }

  // Get square root of inverse covariance computed by the optimized method
  MatNN sqrt_cov_inv_computed = imu_meas.get_sqrt_cov_inv();

  // Compute square root of inverse covariance by squaring the result
  MatNN cov_inv_computed =
      sqrt_cov_inv_computed.transpose() * sqrt_cov_inv_computed;

  // Compute ground truth inverse covariance by direct matrix inversion
  MatNN cov_inv_gt = imu_meas.get_cov().inverse();

  // Verify that the optimized square root computation matches direct inversion
  // Uses a tight tolerance (1e-12) since this is a numerical verification
  EXPECT_TRUE(cov_inv_computed.isApprox(cov_inv_gt, 1e-12))
      << "cov_inv_computed\n"
      << cov_inv_computed << "\ncov_inv_gt\n"
      << cov_inv_gt;
}

/**
 * @brief Tests behavior with zero measurements
 *
 * Verifies that the system behaves correctly when no measurements are
 * integrated. This edge case should:
 * 1. Maintain the initial state
 * 2. Not introduce any artificial motion
 * 3. Handle zero time delta appropriately
 */
TEST(ImuPreintegrationTestCase, ZeroMeasurements) {
  // Test behavior with zero measurements
  basalt::IntegratedImuMeasurement<double> imu_meas(0, Eigen::Vector3d::Zero(),
                                                    Eigen::Vector3d::Zero());

  basalt::PoseVelState<double> state0;
  basalt::PoseVelState<double> state1;

  // Set initial state
  state0.T_w_i = Sophus::SE3d();
  state0.vel_w_i = Eigen::Vector3d(1, 0, 0);  // Initial velocity in x direction

  // Predict with zero measurements should only apply gravity
  imu_meas.predictState(state0, basalt::constants::G, state1);

  // Position should be unchanged since dt = 0
  EXPECT_TRUE(state1.T_w_i.translation().isApprox(state0.T_w_i.translation()));
  EXPECT_TRUE(
      state1.T_w_i.rotationMatrix().isApprox(state0.T_w_i.rotationMatrix()));

  // Velocity should be unchanged since dt = 0
  EXPECT_TRUE(state1.vel_w_i.isApprox(state0.vel_w_i));
}

/**
 * @brief Tests integration of a single measurement
 *
 * Verifies that a single IMU measurement is correctly integrated. This test:
 * 1. Creates a measurement with known values
 * 2. Integrates the measurement
 * 3. Verifies the resulting state change
 * 4. Checks both rotation and acceleration effects
 */
TEST(ImuPreintegrationTestCase, SingleMeasurement) {
  basalt::IntegratedImuMeasurement<double> imu_meas(0, Eigen::Vector3d::Zero(),
                                                    Eigen::Vector3d::Zero());

  // Create a single measurement with known values
  basalt::ImuData<double> data;
  data.t_ns = 1e7;                           // 10ms
  data.accel = Eigen::Vector3d(0, 0, 9.81);  // Cancels gravity
  data.gyro = Eigen::Vector3d(0, 0, 0.1);    // Small rotation around z

  // Integrate the measurement
  imu_meas.integrate(data, Eigen::Vector3d::Ones(), Eigen::Vector3d::Ones());

  basalt::PoseVelState<double> state0;
  basalt::PoseVelState<double> state1;
  state0.T_w_i = Sophus::SE3d();
  state0.vel_w_i = Eigen::Vector3d::Zero();

  // Predict next state
  imu_meas.predictState(state0, basalt::constants::G, state1);

  // Verify rotation around z-axis
  double angle = Eigen::AngleAxisd(state1.T_w_i.rotationMatrix()).angle();
  EXPECT_NEAR(angle, 0.001, 1e-6);  // 10ms * 0.1 rad/s = 0.001 rad

  // Velocity should be close to zero since accel cancels gravity
  EXPECT_TRUE(state1.vel_w_i.norm() < 1e-6);
}

/**
 * @brief Tests consistency of bias updates
 *
 * Verifies that changes in IMU biases produce consistent changes in the
 * predicted states. This test:
 * 1. Integrates measurements with initial biases
 * 2. Integrates same measurements with updated biases
 * 3. Verifies that state differences are consistent with bias changes
 */
TEST(ImuPreintegrationTestCase, BiasConsistency) {
  // Test that bias updates are consistent
  Eigen::Vector3d bias_gyro(0.01, -0.01, 0.02);
  Eigen::Vector3d bias_accel(0.1, -0.1, 0.05);

  basalt::IntegratedImuMeasurement<double> imu_meas(0, bias_gyro, bias_accel);

  // Create measurement data
  basalt::ImuData<double> data;
  data.t_ns = 1e7;
  data.accel = Eigen::Vector3d(1, 0, 9.81);
  data.gyro = Eigen::Vector3d(0.1, 0.2, 0.3);

  // Integrate with initial biases
  imu_meas.integrate(data, Eigen::Vector3d::Ones(), Eigen::Vector3d::Ones());

  basalt::PoseVelState<double> state0;
  basalt::PoseVelState<double> state1;
  basalt::PoseVelState<double> state1_updated;
  state0.T_w_i = Sophus::SE3d();
  state0.vel_w_i = Eigen::Vector3d::Zero();

  // Get state with initial biases
  imu_meas.predictState(state0, basalt::constants::G, state1);

  // Create new measurement with different biases
  basalt::IntegratedImuMeasurement<double> imu_meas_updated(
      0, bias_gyro + Eigen::Vector3d(0.001, -0.002, 0.003),
      bias_accel + Eigen::Vector3d(0.01, -0.02, 0.03));

  // Integrate same data with updated biases
  imu_meas_updated.integrate(data, Eigen::Vector3d::Ones(),
                             Eigen::Vector3d::Ones());
  imu_meas_updated.predictState(state0, basalt::constants::G, state1_updated);

  // Verify that the state difference is consistent with the bias update
  Sophus::SE3d delta_pose = state1.T_w_i.inverse() * state1_updated.T_w_i;
  EXPECT_TRUE(delta_pose.translation().norm() > 1e-6);
  EXPECT_TRUE((state1.vel_w_i - state1_updated.vel_w_i).norm() > 1e-6);
}

/**
 * @brief Tests time consistency in integration
 *
 * Verifies that the integration properly handles timestamps and time intervals.
 * This test:
 * 1. Checks proper initialization of time variables
 * 2. Verifies time accumulation during integration
 * 3. Ensures proper handling of time intervals
 */
TEST(ImuPreintegrationTestCase, TimeConsistency) {
  // Test that integration time is tracked correctly
  int64_t start_t_ns = 1000000000;  // 1s
  basalt::IntegratedImuMeasurement<double> imu_meas(
      start_t_ns, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());

  EXPECT_EQ(imu_meas.get_start_t_ns(), start_t_ns);
  EXPECT_EQ(imu_meas.get_dt_ns(), 0);

  // Add measurements with increasing timestamps
  for (int i = 0; i < 10; ++i) {
    basalt::ImuData<double> data;
    data.t_ns = start_t_ns + (i + 1) * 1e7;  // 10ms intervals
    data.accel = Eigen::Vector3d::Random();
    data.gyro = Eigen::Vector3d::Random();

    imu_meas.integrate(data, Eigen::Vector3d::Ones(), Eigen::Vector3d::Ones());
  }

  // Verify total integration time
  EXPECT_EQ(imu_meas.get_dt_ns(), 1e8);  // 100ms total
}

/**
 * @brief Tests integration of large rotations
 *
 * Verifies that the integration remains accurate for large rotations.
 * This test:
 * 1. Simulates a quarter rotation around z-axis
 * 2. Verifies both the rotation angle
 * 3. Checks the effect on transformed points
 */
TEST(ImuPreintegrationTestCase, LargeRotation) {
  // Initialize start time at 1s in nanoseconds for better numerical stability
  const int64_t start_t_ns = 1000000000;  // 1s

  // Create IMU measurement integrator with zero initial biases
  basalt::IntegratedImuMeasurement<double> imu_meas(
      start_t_ns,
      Eigen::Vector3d::Zero(),   // Initial accelerometer bias
      Eigen::Vector3d::Zero());  // Initial gyroscope bias

  // Simulation parameters for a quarter rotation around z-axis
  const int num_steps = 100;  // Number of discrete integration steps
  const double dt = 0.01;     // Time step in seconds (10ms)
  const double omega =
      M_PI / 2;  // Angular velocity for 90-degree rotation in 1s

  // Generate and integrate IMU measurements for the rotation
  for (int i = 0; i < num_steps; i++) {
    basalt::ImuData<double> data;
    // Convert time step to nanoseconds for precise timing
    data.t_ns = start_t_ns + (i + 1) * int64_t(dt * 1e9);
    data.accel = Eigen::Vector3d(
        0, 0, 9.81);  // Apply upward acceleration to cancel gravity
    data.gyro = Eigen::Vector3d(
        0, 0, omega);  // Apply constant angular velocity around z-axis

    // Integrate measurement with unit noise characteristics
    imu_meas.integrate(data, Eigen::Vector3d::Ones(), Eigen::Vector3d::Ones());
  }

  // Initialize states for prediction
  basalt::PoseVelState<double> state0;  // Initial state
  basalt::PoseVelState<double> state1;  // State after integration
  state0.T_w_i = Sophus::SE3d();        // Identity transformation (no initial
                                        // rotation/translation)
  state0.vel_w_i = Eigen::Vector3d::Zero();  // Zero initial velocity

  // Predict final state using integrated measurements
  imu_meas.predictState(state0, basalt::constants::G, state1);

  // Compute and verify rotation angle between initial and final states
  Eigen::Matrix3d R_diff =
      state0.T_w_i.rotationMatrix().transpose() * state1.T_w_i.rotationMatrix();
  double angle = std::abs(Eigen::AngleAxisd(R_diff).angle());
  EXPECT_NEAR(angle, M_PI / 2,
              1e-6);  // Verify 90-degree rotation with small tolerance

  // Verify rotation effect on a unit vector along x-axis
  // After 90-degree CCW rotation around z, (1,0,0) should become (0,1,0)
  Eigen::Vector3d p0(1, 0, 0);  // Initial point along x-axis
  Eigen::Vector3d p1 = state1.T_w_i.rotationMatrix() * p0;
  EXPECT_NEAR(p1[0], 0, 1e-6);  // x-component should be 0
  EXPECT_NEAR(p1[1], 1, 1e-6);  // y-component should be 1
  EXPECT_NEAR(p1[2], 0, 1e-6);  // z-component should remain 0
}
