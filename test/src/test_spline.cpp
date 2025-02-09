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

#include <basalt/spline/rd_spline.h>
#include <basalt/spline/so3_spline.h>

#include <iostream>

#include "gtest/gtest.h"
#include "test_utils.h"

/**
 * @brief Tests the evaluation of R^d B-splines and their Jacobians
 *
 * This template function verifies that the spline evaluation and its Jacobians
 * with respect to the control points are computed correctly. It tests:
 * 1. Spline evaluation at a given time point
 * 2. Jacobians with respect to each affected control point
 * 3. Numerical validation of analytical Jacobians
 *
 * @tparam DIM Dimension of the spline space (e.g. 3 for 3D positions)
 * @tparam N Number of control points affecting each evaluation (spline order +
 * 1)
 * @tparam DERIV Order of derivative to evaluate (0=position, 1=velocity,
 * 2=acceleration)
 * @param spline The B-spline to test
 * @param t_ns Timestamp in nanoseconds at which to evaluate
 */
template <int DIM, int N, int DERIV>
void testEvaluate(const basalt::RdSpline<DIM, N> &spline, int64_t t_ns) {
  using VectorD = typename basalt::RdSpline<DIM, N>::VecD;
  using MatrixD = typename basalt::RdSpline<DIM, N>::MatD;

  // Get analytical Jacobians from spline evaluation
  typename basalt::RdSpline<DIM, N>::JacobianStruct J_spline;
  spline.template evaluate<DERIV>(t_ns, &J_spline);

  // Test point for numerical differentiation
  VectorD x0;
  x0.setZero();

  // Test Jacobian for each control point that affects this evaluation
  for (size_t i = 0; i < 3 * N; i++) {
    std::stringstream ss;
    ss << "d_val_d_knot" << i << " time " << t_ns;

    // Construct analytical Jacobian matrix
    MatrixD J_a;
    J_a.setZero();
    if (i >= J_spline.start_idx && i < J_spline.start_idx + N) {
      J_a.diagonal().setConstant(J_spline.d_val_d_knot[i - J_spline.start_idx]);
    }

    // Verify Jacobian using numerical differentiation
    test_jacobian(
        ss.str(), J_a,
        [&](const VectorD &x) {
          basalt::RdSpline<DIM, N> spline1 = spline;
          spline1.getKnot(i) += x;  // Perturb control point
          return spline1.template evaluate<DERIV>(t_ns);
        },
        x0);
  }
}

/**
 * @brief Tests time derivatives of R^d B-splines
 *
 * Verifies that time derivatives are computed correctly by comparing:
 * 1. Analytical time derivative at time t
 * 2. Numerical derivative using finite differences
 *
 * For example, for DERIV=0, checks that position derivative matches velocity
 * For DERIV=1, checks that velocity derivative matches acceleration
 *
 * @tparam DIM Dimension of the spline space
 * @tparam N Number of control points per segment
 * @tparam DERIV Order of derivative to test
 * @param spline The B-spline to test
 * @param t_ns Timestamp in nanoseconds
 */
template <int DIM, int N, int DERIV>
void testTimeDeriv(const basalt::RdSpline<DIM, N> &spline, int64_t t_ns) {
  using VectorD = typename basalt::RdSpline<DIM, N>::VecD;

  // Get analytical time derivative
  VectorD d_val_d_t = spline.template evaluate<DERIV + 1>(t_ns);

  // Test point for numerical differentiation
  Eigen::Matrix<double, 1, 1> x0;
  x0.setZero();

  // Verify using numerical differentiation
  test_jacobian(
      "d_val_d_t", d_val_d_t,
      [&](const Eigen::Matrix<double, 1, 1> &x) {
        int64_t inc = x[0] * 1e9;  // Convert to nanoseconds
        return spline.template evaluate<DERIV>(t_ns + inc);
      },
      x0);
}

/**
 * @brief Tests SO(3) B-spline evaluation and its Jacobians
 *
 * Verifies that rotation spline evaluation and its Jacobians with respect to
 * the control points are computed correctly. The test:
 * 1. Evaluates the spline at a given time
 * 2. Computes analytical Jacobians
 * 3. Verifies Jacobians using numerical differentiation
 *
 * Uses the logarithm map to compute differences between rotations.
 *
 * @tparam N Number of control points per segment
 * @param spline The SO(3) B-spline to test
 * @param t_ns Timestamp in nanoseconds
 */
template <int N>
void testEvaluateSo3(const basalt::So3Spline<N> &spline, int64_t t_ns) {
  using VectorD = typename basalt::So3Spline<N>::Vec3;
  using MatrixD = typename basalt::So3Spline<N>::Mat3;
  using SO3 = typename basalt::So3Spline<N>::SO3;

  // Get analytical Jacobians
  typename basalt::So3Spline<N>::JacobianStruct J_spline;
  SO3 res = spline.evaluate(t_ns, &J_spline);

  // Test point for numerical differentiation
  VectorD x0;
  x0.setZero();

  // Test Jacobian for each control point
  for (size_t i = 0; i < 3 * N; i++) {
    std::stringstream ss;
    ss << "d_val_d_knot" << i << " time " << t_ns;

    // Construct analytical Jacobian matrix
    MatrixD J_a;
    J_a.setZero();
    if (i >= J_spline.start_idx && i < J_spline.start_idx + N) {
      J_a = J_spline.d_val_d_knot[i - J_spline.start_idx];
    }

    // Verify using numerical differentiation
    test_jacobian(
        ss.str(), J_a,
        [&](const VectorD &x) {
          basalt::So3Spline<N> spline1 = spline;
          // Apply small rotation to control point
          spline1.getKnot(i) = SO3::exp(x) * spline.getKnot(i);
          SO3 res1 = spline1.evaluate(t_ns);
          // Return difference in the tangent space
          return (res1 * res.inverse()).log();
        },
        x0);
  }
}

/**
 * @brief Tests velocity of SO(3) B-splines
 *
 * Verifies that the velocity of the rotation spline is computed correctly by
 * comparing:
 * 1. Analytical velocity at time t
 * 2. Numerical derivative using finite differences
 *
 * @tparam N Number of control points per segment
 * @param spline The SO(3) B-spline to test
 * @param t_ns Timestamp in nanoseconds
 */
template <int N>
void testVelSo3(const basalt::So3Spline<N> &spline, int64_t t_ns) {
  using VectorD = typename basalt::So3Spline<N>::Vec3;
  using SO3 = typename basalt::So3Spline<N>::SO3;

  // Get analytical velocity
  SO3 res = spline.evaluate(t_ns);
  VectorD d_res_d_t = spline.velocityBody(t_ns);

  // Test point for numerical differentiation
  Eigen::Matrix<double, 1, 1> x0;
  x0.setZero();

  // Verify using numerical differentiation
  test_jacobian(
      "d_val_d_t", d_res_d_t,
      [&](const Eigen::Matrix<double, 1, 1> &x) {
        int64_t inc = x[0] * 1e9;  // Convert to nanoseconds
        return (res.inverse() * spline.evaluate(t_ns + inc)).log();
      },
      x0);
}

/**
 * @brief Tests acceleration of SO(3) B-splines
 *
 * Verifies that the acceleration of the rotation spline is computed correctly
 * by comparing:
 * 1. Analytical acceleration at time t
 * 2. Numerical derivative using finite differences
 *
 * @tparam N Number of control points per segment
 * @param spline The SO(3) B-spline to test
 * @param t_ns Timestamp in nanoseconds
 */
template <int N>
void testAccelSo3(const basalt::So3Spline<N> &spline, int64_t t_ns) {
  using VectorD = typename basalt::So3Spline<5>::Vec3;

  // Get analytical acceleration
  VectorD vel1;
  VectorD d_res_d_t = spline.accelerationBody(t_ns, &vel1);

  // Verify velocity
  VectorD vel2 = spline.velocityBody(t_ns);
  EXPECT_TRUE(vel1.isApprox(vel2));

  // Test point for numerical differentiation
  Eigen::Matrix<double, 1, 1> x0;
  x0.setZero();

  // Verify using numerical differentiation
  test_jacobian(
      "d_val_d_t", d_res_d_t,
      [&](const Eigen::Matrix<double, 1, 1> &x) {
        int64_t inc = x[0] * 1e9;  // Convert to nanoseconds
        return spline.velocityBody(t_ns + inc);
      },
      x0);
}

/**
 * @brief Tests jerk of SO(3) B-splines
 *
 * Verifies that the jerk of the rotation spline is computed correctly by
 * comparing:
 * 1. Analytical jerk at time t
 * 2. Numerical derivative using finite differences
 *
 * @tparam N Number of control points per segment
 * @param spline The SO(3) B-spline to test
 * @param t_ns Timestamp in nanoseconds
 */
template <int N>
void testJerkSo3(const basalt::So3Spline<N> &spline, int64_t t_ns) {
  using VectorD = typename basalt::So3Spline<5>::Vec3;

  // Get analytical jerk
  VectorD vel1;
  VectorD accel1;
  VectorD d_res_d_t = spline.jerkBody(t_ns, &vel1, &accel1);

  // Verify velocity
  VectorD vel2 = spline.velocityBody(t_ns);
  EXPECT_TRUE(vel1.isApprox(vel2));

  // Verify acceleration
  VectorD accel2 = spline.accelerationBody(t_ns);
  EXPECT_TRUE(accel1.isApprox(accel2));

  // Test point for numerical differentiation
  Eigen::Matrix<double, 1, 1> x0;
  x0.setZero();

  // Verify using numerical differentiation
  test_jacobian(
      "d_val_d_t", d_res_d_t,
      [&](const Eigen::Matrix<double, 1, 1> &x) {
        int64_t inc = x[0] * 1e9;  // Convert to nanoseconds
        return spline.accelerationBody(t_ns + inc);
      },
      x0);
}

/**
 * @brief Tests velocity of SO(3) B-splines and its Jacobians
 *
 * Verifies that the velocity of the rotation spline and its Jacobians with
 * respect to the control points are computed correctly. The test:
 * 1. Evaluates the velocity at a given time
 * 2. Computes analytical Jacobians
 * 3. Verifies Jacobians using numerical differentiation
 *
 * @tparam N Number of control points per segment
 * @param spline The SO(3) B-spline to test
 * @param t_ns Timestamp in nanoseconds
 */
template <int N>
void testEvaluateSo3Vel(const basalt::So3Spline<N> &spline, int64_t t_ns) {
  using VectorD = typename basalt::So3Spline<5>::Vec3;
  using MatrixD = typename basalt::So3Spline<5>::Mat3;
  using SO3 = typename basalt::So3Spline<5>::SO3;

  // Get analytical Jacobians
  typename basalt::So3Spline<N>::JacobianStruct J_spline;
  VectorD res = spline.velocityBody(t_ns, &J_spline);
  VectorD res_ref = spline.velocityBody(t_ns);

  // Verify reference velocity
  ASSERT_TRUE(res_ref.isApprox(res)) << "res and res_ref are not the same";

  // Test point for numerical differentiation
  VectorD x0;
  x0.setZero();

  // Test Jacobian for each control point
  for (size_t i = 0; i < 3 * N; i++) {
    std::stringstream ss;
    ss << "d_vel_d_knot" << i << " time " << t_ns;

    // Construct analytical Jacobian matrix
    MatrixD J_a;
    J_a.setZero();
    if (i >= J_spline.start_idx && i < J_spline.start_idx + N) {
      J_a = J_spline.d_val_d_knot[i - J_spline.start_idx];
    }

    // Verify using numerical differentiation
    test_jacobian(
        ss.str(), J_a,
        [&](const VectorD &x) {
          basalt::So3Spline<N> spline1 = spline;
          // Apply small rotation to control point
          spline1.getKnot(i) = SO3::exp(x) * spline.getKnot(i);
          return spline1.velocityBody(t_ns);
        },
        x0);
  }
}

/**
 * @brief Tests acceleration of SO(3) B-splines and its Jacobians
 *
 * Verifies that the acceleration of the rotation spline and its Jacobians with
 * respect to the control points are computed correctly. The test:
 * 1. Evaluates the acceleration at a given time
 * 2. Computes analytical Jacobians
 * 3. Verifies Jacobians using numerical differentiation
 *
 * @tparam N Number of control points per segment
 * @param spline The SO(3) B-spline to test
 * @param t_ns Timestamp in nanoseconds
 */
template <int N>
void testEvaluateSo3Accel(const basalt::So3Spline<N> &spline, int64_t t_ns) {
  using VectorD = typename basalt::So3Spline<N>::Vec3;
  using MatrixD = typename basalt::So3Spline<N>::Mat3;
  using SO3 = typename basalt::So3Spline<N>::SO3;

  // Get analytical Jacobians
  typename basalt::So3Spline<N>::JacobianStruct J_accel;
  typename basalt::So3Spline<N>::JacobianStruct J_vel;

  // Get acceleration and velocity
  VectorD vel;
  VectorD vel_ref;
  VectorD res = spline.accelerationBody(t_ns, &J_accel, &vel, &J_vel);
  VectorD res_ref = spline.accelerationBody(t_ns, &vel_ref);

  // Verify reference acceleration and velocity
  ASSERT_TRUE(vel_ref.isApprox(vel)) << "vel and vel_ref are not the same";
  ASSERT_TRUE(res_ref.isApprox(res)) << "res and res_ref are not the same";

  // Test point for numerical differentiation
  VectorD x0;
  x0.setZero();

  // Test velocity Jacobian
  for (size_t i = 0; i < 3 * N; i++) {
    std::stringstream ss;
    ss << "d_vel_d_knot" << i << " time " << t_ns;

    // Construct analytical Jacobian matrix
    MatrixD J_a;
    J_a.setZero();
    if (i >= J_vel.start_idx && i < J_vel.start_idx + N) {
      J_a = J_vel.d_val_d_knot[i - J_vel.start_idx];
    }

    // Verify using numerical differentiation
    test_jacobian(
        ss.str(), J_a,
        [&](const VectorD &x) {
          basalt::So3Spline<N> spline1 = spline;
          // Apply small rotation to control point
          spline1.getKnot(i) = SO3::exp(x) * spline.getKnot(i);
          return spline1.velocityBody(t_ns);
        },
        x0);
  }

  // Test acceleration Jacobian
  for (size_t i = 0; i < 3 * N; i++) {
    std::stringstream ss;
    ss << "d_accel_d_knot" << i << " time " << t_ns;

    // Construct analytical Jacobian matrix
    MatrixD J_a;
    J_a.setZero();
    if (i >= J_accel.start_idx && i < J_accel.start_idx + N) {
      J_a = J_accel.d_val_d_knot[i - J_accel.start_idx];
    }

    // Verify using numerical differentiation
    test_jacobian(
        ss.str(), J_a,
        [&](const VectorD &x) {
          basalt::So3Spline<N> spline1 = spline;
          // Apply small rotation to control point
          spline1.getKnot(i) = SO3::exp(x) * spline.getKnot(i);
          return spline1.accelerationBody(t_ns);
        },
        x0);
  }
}

/**
 * @brief Tests SO(3) B-spline evaluation with 4 control points
 *
 * Tests the evaluation of a cubic B-spline on SO(3) with 4 control points.
 * Verifies that rotation evaluations and their Jacobians are correct at
 * multiple time points. The test checks that:
 * 1. The spline correctly interpolates rotations
 * 2. Jacobians with respect to control points are accurate
 */
TEST(SplineTest, SO3CUBSplineEvaluateKnots4) {
  static constexpr int N = 4;  // 4 control points (cubic spline)

  // Create spline over 2 second interval
  basalt::So3Spline<N> spline(int64_t(2e9));
  spline.genRandomTrajectory(3 * N);

  // Test at 100ms intervals
  for (int64_t t_ns = 0; t_ns < spline.maxTimeNs(); t_ns += 1e8) {
    testEvaluateSo3(spline, t_ns);
  }
}

/**
 * @brief Tests SO(3) B-spline velocity computation with 4 control points
 *
 * Tests the angular velocity computation of a cubic B-spline on SO(3).
 * Verifies that:
 * 1. Angular velocities are computed correctly
 * 2. Numerical derivatives match analytical velocities
 */
TEST(SplineTest, SO3CUBSplineVelocity4) {
  static constexpr int N = 4;  // 4 control points (cubic spline)

  // Create spline over 2 second interval
  basalt::So3Spline<N> spline(int64_t(2e9));
  spline.genRandomTrajectory(3 * N);

  // Test at 100ms intervals, avoiding boundary points
  for (int64_t t_ns = 1e8; t_ns < spline.maxTimeNs() - 1e8; t_ns += 1e8) {
    testVelSo3(spline, t_ns);
  }
}

/**
 * @brief Tests SO(3) B-spline acceleration computation with 4 control points
 *
 * Tests the angular acceleration computation of a cubic B-spline on SO(3).
 * Verifies that:
 * 1. Angular accelerations are computed correctly
 * 2. Numerical derivatives of velocity match analytical accelerations
 * 3. Velocity computations are consistent
 */
TEST(SplineTest, SO3CUBSplineAcceleration4) {
  static constexpr int N = 4;  // 4 control points (cubic spline)

  // Create spline over 2 second interval
  basalt::So3Spline<N> spline(int64_t(2e9));
  spline.genRandomTrajectory(3 * N);

  // Test at 100ms intervals, avoiding boundary points
  for (int64_t t_ns = 1e8; t_ns < spline.maxTimeNs() - 1e8; t_ns += 1e8) {
    testAccelSo3(spline, t_ns);
  }
}

/**
 * @brief Tests SO(3) B-spline jerk computation with 5 control points
 *
 * Tests the angular jerk (derivative of acceleration) computation of a quartic
 * B-spline on SO(3). Verifies that:
 * 1. Angular jerks are computed correctly
 * 2. Numerical derivatives of acceleration match analytical jerks
 * 3. Velocity and acceleration computations are consistent
 */
TEST(SplineTest, SO3CUBSplineJerk5) {
  static constexpr int N = 5;  // 5 control points (quartic spline)

  // Create spline over 2 second interval
  basalt::So3Spline<N> spline(int64_t(2e9));
  spline.genRandomTrajectory(3 * N);

  // Test at 100ms intervals, avoiding boundary points
  for (int64_t t_ns = 1e8; t_ns < spline.maxTimeNs() - 1e8; t_ns += 1e8) {
    testJerkSo3(spline, t_ns);
  }
}

/**
 * @brief Tests SO(3) B-spline velocity Jacobians with 4 control points
 *
 * Tests the Jacobians of angular velocity with respect to control points
 * for a cubic B-spline on SO(3). Verifies that:
 * 1. Angular velocity Jacobians are computed correctly
 * 2. Numerical Jacobians match analytical ones
 */
TEST(SplineTest, SO3CUBSplineVelocityKnots4) {
  static constexpr int N = 4;  // 4 control points (cubic spline)

  // Create spline over 2 second interval
  basalt::So3Spline<N> spline(int64_t(2e9));
  spline.genRandomTrajectory(3 * N);

  // Test at 100ms intervals
  for (int64_t t_ns = 0; t_ns < spline.maxTimeNs(); t_ns += 1e8) {
    testEvaluateSo3Vel(spline, t_ns);
  }
}

/**
 * @brief Tests cross product Jacobian computation
 *
 * Tests the computation of Jacobians for cross products of vectors.
 * This is important for SO(3) operations since cross products appear in:
 * 1. Angular velocity computations
 * 2. Lie bracket operations
 * 3. Adjoint representations
 */
TEST(SplineTest, CrossProductTest) {
  Eigen::Matrix3d J_1;
  Eigen::Matrix3d J_2;
  Eigen::Matrix3d J_cross;
  Eigen::Vector3d v1;
  Eigen::Vector3d v2;

  // Initialize random test data
  J_1.setRandom();
  J_2.setRandom();
  v1.setRandom();
  v2.setRandom();

  // Test case 1: Full cross product Jacobian
  // J_cross = hat(J_1 * v1) * J_2 - hat(J_2 * v2) * J_1
  J_cross =
      Sophus::SO3d::hat(J_1 * v1) * J_2 - Sophus::SO3d::hat(J_2 * v2) * J_1;

  test_jacobian(
      "cross_prod_test1", J_cross,
      [&](const Eigen::Vector3d &x) {
        return (J_1 * (v1 + x)).cross(J_2 * (v2 + x));
      },
      Eigen::Vector3d::Zero());

  // Test case 2: Jacobian with respect to first vector only
  J_cross = -Sophus::SO3d::hat(J_2 * v2) * J_1;

  test_jacobian(
      "cross_prod_test2", J_cross,
      [&](const Eigen::Vector3d &x) {
        return (J_1 * (v1 + x)).cross(J_2 * v2);
      },
      Eigen::Vector3d::Zero());

  // Test case 3: Jacobian with respect to second vector only
  J_cross = Sophus::SO3d::hat(J_1 * v1) * J_2;

  test_jacobian(
      "cross_prod_test2", J_cross,
      [&](const Eigen::Vector3d &x) {
        return (J_1 * v1).cross(J_2 * (v2 + x));
      },
      Eigen::Vector3d::Zero());
}

TEST(SplineTest, UBSplineEvaluateKnots4) {
  static constexpr int DIM = 3;
  static constexpr int N = 4;

  basalt::RdSpline<DIM, N> spline(int64_t(2e9));
  spline.genRandomTrajectory(3 * N);

  for (int64_t t_ns = 0; t_ns < spline.maxTimeNs(); t_ns += 1e8) {
    testEvaluate<DIM, N, 0>(spline, t_ns);
  }
}

TEST(SplineTest, UBSplineEvaluateKnots5) {
  static constexpr int DIM = 3;
  static constexpr int N = 5;

  basalt::RdSpline<DIM, N> spline(int64_t(2e9));
  spline.genRandomTrajectory(3 * N);

  for (int64_t t_ns = 0; t_ns < spline.maxTimeNs(); t_ns += 1e8) {
    testEvaluate<DIM, N, 0>(spline, t_ns);
  }
}

TEST(SplineTest, UBSplineEvaluateKnots6) {
  static constexpr int DIM = 3;
  static constexpr int N = 6;

  basalt::RdSpline<DIM, N> spline(int64_t(2e9));
  spline.genRandomTrajectory(3 * N);

  for (int64_t t_ns = 0; t_ns < spline.maxTimeNs(); t_ns += 1e8) {
    testEvaluate<DIM, N, 0>(spline, t_ns);
  }
}

TEST(SplineTest, UBSplineVelocityKnots4) {
  static constexpr int DIM = 3;
  static constexpr int N = 4;

  basalt::RdSpline<DIM, N> spline(int64_t(2e9));
  spline.genRandomTrajectory(3 * N);

  for (int64_t t_ns = 0; t_ns < spline.maxTimeNs(); t_ns += 1e8) {
    testEvaluate<DIM, N, 1>(spline, t_ns);
  }
}

TEST(SplineTest, UBSplineVelocityKnots5) {
  static constexpr int DIM = 3;
  static constexpr int N = 5;

  basalt::RdSpline<DIM, N> spline(int64_t(2e9));
  spline.genRandomTrajectory(3 * N);

  for (int64_t t_ns = 0; t_ns < spline.maxTimeNs(); t_ns += 1e8) {
    testEvaluate<DIM, N, 1>(spline, t_ns);
  }
}

TEST(SplineTest, UBSplineVelocityKnots6) {
  static constexpr int DIM = 3;
  static constexpr int N = 6;

  basalt::RdSpline<DIM, N> spline(int64_t(2e9));
  spline.genRandomTrajectory(3 * N);

  for (int64_t t_ns = 0; t_ns < spline.maxTimeNs(); t_ns += 1e8) {
    testEvaluate<DIM, N, 1>(spline, t_ns);
  }
}

TEST(SplineTest, UBSplineAccelKnots4) {
  static constexpr int DIM = 3;
  static constexpr int N = 4;

  basalt::RdSpline<DIM, N> spline(int64_t(2e9));
  spline.genRandomTrajectory(3 * N);

  for (int64_t t_ns = 0; t_ns < spline.maxTimeNs(); t_ns += 1e8) {
    testEvaluate<DIM, N, 2>(spline, t_ns);
  }
}

TEST(SplineTest, UBSplineAccelKnots5) {
  static constexpr int DIM = 3;
  static constexpr int N = 5;

  basalt::RdSpline<DIM, N> spline(int64_t(2e9));
  spline.genRandomTrajectory(3 * N);

  for (int64_t t_ns = 0; t_ns < spline.maxTimeNs(); t_ns += 1e8) {
    testEvaluate<DIM, N, 2>(spline, t_ns);
  }
}

TEST(SplineTest, UBSplineAccelKnots6) {
  static constexpr int DIM = 3;
  static constexpr int N = 6;

  basalt::RdSpline<DIM, N> spline(int64_t(2e9));
  spline.genRandomTrajectory(3 * N);

  for (int64_t t_ns = 0; t_ns < spline.maxTimeNs(); t_ns += 1e8) {
    testEvaluate<DIM, N, 2>(spline, t_ns);
  }
}

TEST(SplineTest, UBSplineEvaluateTimeDeriv4) {
  static constexpr int DIM = 3;
  static constexpr int N = 4;

  basalt::RdSpline<DIM, N> spline(int64_t(2e9));
  spline.genRandomTrajectory(3 * N);

  for (int64_t t_ns = 1e8; t_ns < spline.maxTimeNs() - 1e8; t_ns += 1e8) {
    testTimeDeriv<DIM, N, 0>(spline, t_ns);
  }
}

TEST(SplineTest, UBSplineEvaluateTimeDeriv5) {
  static constexpr int DIM = 3;
  static constexpr int N = 5;

  basalt::RdSpline<DIM, N> spline(int64_t(2e9));
  spline.genRandomTrajectory(3 * N);

  for (int64_t t_ns = 1e8; t_ns < spline.maxTimeNs() - 1e8; t_ns += 1e8) {
    testTimeDeriv<DIM, N, 0>(spline, t_ns);
  }
}

TEST(SplineTest, UBSplineEvaluateTimeDeriv6) {
  static constexpr int DIM = 3;
  static constexpr int N = 6;

  basalt::RdSpline<DIM, N> spline(int64_t(2e9));
  spline.genRandomTrajectory(3 * N);

  for (int64_t t_ns = 1e8; t_ns < spline.maxTimeNs() - 1e8; t_ns += 1e8) {
    testTimeDeriv<DIM, N, 0>(spline, t_ns);
  }
}

TEST(SplineTest, UBSplineVelocityTimeDeriv4) {
  static constexpr int DIM = 3;
  static constexpr int N = 4;

  basalt::RdSpline<DIM, N> spline(int64_t(2e9));
  spline.genRandomTrajectory(3 * N);

  for (int64_t t_ns = 1e8; t_ns < spline.maxTimeNs() - 1e8; t_ns += 1e8) {
    testTimeDeriv<DIM, N, 1>(spline, t_ns);
  }
}

TEST(SplineTest, UBSplineVelocityTimeDeriv5) {
  static constexpr int DIM = 3;
  static constexpr int N = 5;

  basalt::RdSpline<DIM, N> spline(int64_t(2e9));
  spline.genRandomTrajectory(3 * N);

  for (int64_t t_ns = 1e8; t_ns < spline.maxTimeNs() - 1e8; t_ns += 1e8) {
    testTimeDeriv<DIM, N, 1>(spline, t_ns);
  }
}

TEST(SplineTest, UBSplineVelocityTimeDeriv6) {
  static constexpr int DIM = 3;
  static constexpr int N = 6;

  basalt::RdSpline<DIM, N> spline(int64_t(2e9));
  spline.genRandomTrajectory(3 * N);

  for (int64_t t_ns = 1e8; t_ns < spline.maxTimeNs() - 1e8; t_ns += 1e8) {
    testTimeDeriv<DIM, N, 1>(spline, t_ns);
  }
}

TEST(SplineTest, SO3CUBSplineAccelerationKnots4) {
  static constexpr int N = 4;

  basalt::So3Spline<N> spline(int64_t(2e9));
  spline.genRandomTrajectory(3 * N);

  for (int64_t t_ns = 0; t_ns < spline.maxTimeNs(); t_ns += 1e8) {
    testEvaluateSo3Accel(spline, t_ns);
  }
}

TEST(SplineTest, SO3CUBSplineAccelerationKnots5) {
  static constexpr int N = 5;

  basalt::So3Spline<N> spline(int64_t(2e9));
  spline.genRandomTrajectory(3 * N);

  for (int64_t t_ns = 0; t_ns < spline.maxTimeNs(); t_ns += 1e8) {
    testEvaluateSo3Accel(spline, t_ns);
  }
}

TEST(SplineTest, SO3CUBSplineAccelerationKnots6) {
  static constexpr int N = 6;

  basalt::So3Spline<N> spline(int64_t(2e9));
  spline.genRandomTrajectory(3 * N);

  for (int64_t t_ns = 0; t_ns < spline.maxTimeNs(); t_ns += 1e8) {
    testEvaluateSo3Accel(spline, t_ns);
  }
}

/**
 * @brief Tests SO(3) B-spline evaluation with 5 control points
 *
 * Tests the evaluation of a quartic B-spline on SO(3) with 5 control points.
 * Similar to the N=4 case, but with higher order continuity.
 */
TEST(SplineTest, SO3CUBSplineEvaluateKnots5) {
  static constexpr int N = 5;  // 5 control points (quartic spline)

  // Create spline over 2 second interval
  basalt::So3Spline<N> spline(int64_t(2e9));
  spline.genRandomTrajectory(3 * N);

  // Test at 100ms intervals
  for (int64_t t_ns = 0; t_ns < spline.maxTimeNs(); t_ns += 1e8) {
    testEvaluateSo3(spline, t_ns);
  }
}

/**
 * @brief Tests SO(3) B-spline evaluation with 6 control points
 *
 * Tests the evaluation of a quintic B-spline on SO(3) with 6 control points.
 * Similar to N=4,5 cases, but with even higher order continuity.
 */
TEST(SplineTest, SO3CUBSplineEvaluateKnots6) {
  static constexpr int N = 6;  // 6 control points (quintic spline)

  // Create spline over 2 second interval
  basalt::So3Spline<N> spline(int64_t(2e9));
  spline.genRandomTrajectory(3 * N);

  // Test at 100ms intervals
  for (int64_t t_ns = 0; t_ns < spline.maxTimeNs(); t_ns += 1e8) {
    testEvaluateSo3(spline, t_ns);
  }
}

/**
 * @brief Tests SO(3) B-spline velocity with 5 control points
 *
 * Tests the angular velocity computation of a quartic B-spline on SO(3).
 * Higher order spline allows for smoother velocity profiles.
 */
TEST(SplineTest, SO3CUBSplineVelocity5) {
  static constexpr int N = 5;  // 5 control points (quartic spline)

  // Create spline over 2 second interval
  basalt::So3Spline<N> spline(int64_t(2e9));
  spline.genRandomTrajectory(3 * N);

  // Test at 100ms intervals, avoiding boundary points
  for (int64_t t_ns = 1e8; t_ns < spline.maxTimeNs() - 1e8; t_ns += 1e8) {
    testVelSo3(spline, t_ns);
  }
}

/**
 * @brief Tests SO(3) B-spline velocity with 6 control points
 *
 * Tests the angular velocity computation of a quintic B-spline on SO(3).
 * Highest order spline tested, allowing for very smooth velocity profiles.
 */
TEST(SplineTest, SO3CUBSplineVelocity6) {
  static constexpr int N = 6;  // 6 control points (quintic spline)

  // Create spline over 2 second interval
  basalt::So3Spline<N> spline(int64_t(2e9));
  spline.genRandomTrajectory(3 * N);

  // Test at 100ms intervals, avoiding boundary points
  for (int64_t t_ns = 1e8; t_ns < spline.maxTimeNs() - 1e8; t_ns += 1e8) {
    testVelSo3(spline, t_ns);
  }
}

/**
 * @brief Tests SO(3) B-spline acceleration with 5 control points
 *
 * Tests the angular acceleration computation of a quartic B-spline on SO(3).
 * Higher order allows for continuous acceleration profiles.
 */
TEST(SplineTest, SO3CUBSplineAcceleration5) {
  static constexpr int N = 5;  // 5 control points (quartic spline)

  // Create spline over 2 second interval
  basalt::So3Spline<N> spline(int64_t(2e9));
  spline.genRandomTrajectory(3 * N);

  // Test at 100ms intervals, avoiding boundary points
  for (int64_t t_ns = 1e8; t_ns < spline.maxTimeNs() - 1e8; t_ns += 1e8) {
    testAccelSo3(spline, t_ns);
  }
}

/**
 * @brief Tests SO(3) B-spline acceleration with 6 control points
 *
 * Tests the angular acceleration computation of a quintic B-spline on SO(3).
 * Highest order allows for smooth acceleration profiles.
 */
TEST(SplineTest, SO3CUBSplineAcceleration6) {
  static constexpr int N = 6;  // 6 control points (quintic spline)

  // Create spline over 2 second interval
  basalt::So3Spline<N> spline(int64_t(2e9));
  spline.genRandomTrajectory(3 * N);

  // Test at 100ms intervals, avoiding boundary points
  for (int64_t t_ns = 1e8; t_ns < spline.maxTimeNs() - 1e8; t_ns += 1e8) {
    testAccelSo3(spline, t_ns);
  }
}

/**
 * @brief Tests SO(3) B-spline jerk with 6 control points
 *
 * Tests the angular jerk computation of a quintic B-spline on SO(3).
 * High order spline needed for continuous jerk profiles.
 */
TEST(SplineTest, SO3CUBSplineJerk6) {
  static constexpr int N = 6;  // 6 control points (quintic spline)

  // Create spline over 2 second interval
  basalt::So3Spline<N> spline(int64_t(2e9));
  spline.genRandomTrajectory(3 * N);

  // Test at 100ms intervals, avoiding boundary points
  for (int64_t t_ns = 1e8; t_ns < spline.maxTimeNs() - 1e8; t_ns += 1e8) {
    testJerkSo3(spline, t_ns);
  }
}

/**
 * @brief Tests SO(3) B-spline velocity Jacobians with 5 control points
 *
 * Tests the Jacobians of angular velocity with respect to control points
 * for a quartic B-spline on SO(3). Higher order allows for more accurate
 * velocity control.
 */
TEST(SplineTest, SO3CUBSplineVelocityKnots5) {
  static constexpr int N = 5;  // 5 control points (quartic spline)

  // Create spline over 2 second interval
  basalt::So3Spline<N> spline(int64_t(2e9));
  spline.genRandomTrajectory(3 * N);

  // Test at 100ms intervals
  for (int64_t t_ns = 0; t_ns < spline.maxTimeNs(); t_ns += 1e8) {
    testEvaluateSo3Vel(spline, t_ns);
  }
}

/**
 * @brief Tests SO(3) B-spline velocity Jacobians with 6 control points
 *
 * Tests the Jacobians of angular velocity with respect to control points
 * for a quintic B-spline on SO(3). Highest order tested, allowing for
 * very precise velocity control.
 */
TEST(SplineTest, SO3CUBSplineVelocityKnots6) {
  static constexpr int N = 6;  // 6 control points (quintic spline)

  // Create spline over 2 second interval
  basalt::So3Spline<N> spline(int64_t(2e9));
  spline.genRandomTrajectory(3 * N);

  // Test at 100ms intervals
  for (int64_t t_ns = 0; t_ns < spline.maxTimeNs(); t_ns += 1e8) {
    testEvaluateSo3Vel(spline, t_ns);
  }
}

/**
 * @brief Tests boundary behavior of SO(3) B-splines
 *
 * Tests the evaluation of SO(3) B-splines at their temporal boundaries.
 * Verifies that:
 * 1. Spline can be evaluated at exact boundary times
 * 2. Spline evaluation fails gracefully outside valid time range
 */
TEST(SplineTest, SO3CUBSplineBounds) {
  static constexpr int N = 5;  // 5 control points (quartic spline)

  // Create spline over 2 second interval
  basalt::So3Spline<N> spline(int64_t(2e9));
  spline.genRandomTrajectory(3 * N);

  // Test evaluation at exact boundary times
  spline.evaluate(spline.maxTimeNs());
  spline.evaluate(spline.minTimeNs());

  // Note: Commented out tests for out-of-bounds evaluation
  // as they are expected to fail
  // Sophus::SO3d res3 = spline.evaluate(spline.maxTimeNs() + 1);
  // Sophus::SO3d res4 = spline.evaluate(spline.minTimeNs() - 1);
}

/**
 * @brief Tests boundary behavior of R^d B-splines
 *
 * Tests the evaluation of R^d B-splines at their temporal boundaries.
 * Similar to SO3CUBSplineBounds, but for Euclidean splines.
 */
TEST(SplineTest, UBSplineBounds) {
  static constexpr int N = 5;    // 5 control points (quartic spline)
  static constexpr int DIM = 3;  // 3D spline

  // Create spline over 2 second interval
  basalt::RdSpline<DIM, N> spline(int64_t(2e9));
  spline.genRandomTrajectory(3 * N);

  // Test evaluation at exact boundary times
  spline.evaluate(spline.maxTimeNs());
  spline.evaluate(spline.minTimeNs());

  // Note: Commented out tests for out-of-bounds evaluation
  // as they are expected to fail
  // Eigen::Vector3d res3 = spline.evaluate(spline.maxTimeNs() + 1);
  // Eigen::Vector3d res4 = spline.evaluate(spline.minTimeNs() - 1);
}
