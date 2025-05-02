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

#include <basalt/spline/spline_common.h>
#include "basalt/spline/rd_bezier.h"
#include "gtest/gtest.h"
#include "test_utils.h"



template <int DIM, int N, int DERIV>
void testEvaluate(const basalt::RdBezier<DIM, N> &spline, int64_t t_ns) {
  using VectorD = typename basalt::RdBezier<DIM, N>::VecD;
  using MatrixD = typename basalt::RdBezier<DIM, N>::MatD;

  typename basalt::RdBezier<DIM, N>::JacobianStruct J_spline;

  spline.template evaluate<DERIV>(t_ns, &J_spline);

  VectorD x0;
  x0.setZero();

  for (size_t i = 0; i < N; i++) {
    std::stringstream ss;

    ss << "d_val_d_knot" << i << " time " << t_ns;

    MatrixD J_a;
    J_a.setZero();

    if (i >= J_spline.start_idx && i < J_spline.start_idx + N) {
      J_a.diagonal().setConstant(J_spline.d_val_d_knot[i - J_spline.start_idx]);
    }

    test_jacobian(
        ss.str(), J_a,
        [&](const VectorD &x) {
          basalt::RdBezier<DIM, N> spline1 = spline;
          spline1.getKnot(i) += x;

          return spline1.template evaluate<DERIV>(t_ns);
        },
        x0);
  }
}

template <int DIM, int N, int DERIV>
void testTimeDeriv(const basalt::RdBezier<DIM, N> &spline, int64_t t_ns) {
  using VectorD = typename basalt::RdBezier<DIM, N>::VecD;

  VectorD d_val_d_t = spline.template evaluate<DERIV + 1>(t_ns);

  Eigen::Matrix<double, 1, 1> x0;
  x0.setZero();

  test_jacobian(
      "d_val_d_t", d_val_d_t,
      [&](const Eigen::Matrix<double, 1, 1> &x) {
        int64_t inc = x[0] * 1e9;
        return spline.template evaluate<DERIV>(t_ns + inc);
      },
      x0);
}


TEST(BezierTest, UBBezierEvaluateKnots4) {
  static constexpr int DIM = 3;
  static constexpr int N = 4;

  basalt::RdBezier<DIM, N> spline(int64_t(2e9));
  spline.genRandomTrajectory();

  for (int64_t t_ns = 0; t_ns < spline.maxTimeNs(); t_ns += 1e8) {
    testEvaluate<DIM, N, 0>(spline, t_ns);
  }
}

TEST(BezierTest, UBBezierEvaluateKnots5) {
  static constexpr int DIM = 3;
  static constexpr int N = 5;

  basalt::RdBezier<DIM, N> spline(int64_t(2e9));
  spline.genRandomTrajectory();

  for (int64_t t_ns = 0; t_ns < spline.maxTimeNs(); t_ns += 1e8) {
    testEvaluate<DIM, N, 0>(spline, t_ns);
  }
}

TEST(BezierTest, UBBezierEvaluateKnots6) {
  static constexpr int DIM = 3;
  static constexpr int N = 6;

  basalt::RdBezier<DIM, N> spline(int64_t(2e9));
  spline.genRandomTrajectory();

  for (int64_t t_ns = 0; t_ns < spline.maxTimeNs(); t_ns += 1e8) {
    testEvaluate<DIM, N, 0>(spline, t_ns);
  }
}

TEST(BezierTest, UBBezierVelocityKnots4) {
  static constexpr int DIM = 3;
  static constexpr int N = 4;

  basalt::RdBezier<DIM, N> spline(int64_t(2e9));
  spline.genRandomTrajectory();

  for (int64_t t_ns = 0; t_ns < spline.maxTimeNs(); t_ns += 1e8) {
    testEvaluate<DIM, N, 1>(spline, t_ns);
  }
}

TEST(BezierTest, UBBezierVelocityKnots5) {
  static constexpr int DIM = 3;
  static constexpr int N = 5;

  basalt::RdBezier<DIM, N> spline(int64_t(2e9));
  spline.genRandomTrajectory();

  for (int64_t t_ns = 0; t_ns < spline.maxTimeNs(); t_ns += 1e8) {
    testEvaluate<DIM, N, 1>(spline, t_ns);
  }
}

TEST(BezierTest, UBBezierVelocityKnots6) {
  static constexpr int DIM = 3;
  static constexpr int N = 6;

  basalt::RdBezier<DIM, N> spline(int64_t(2e9));
  spline.genRandomTrajectory();

  for (int64_t t_ns = 0; t_ns < spline.maxTimeNs(); t_ns += 1e8) {
    testEvaluate<DIM, N, 1>(spline, t_ns);
  }
}

TEST(BezierTest, UBBezierAccelKnots4) {
  static constexpr int DIM = 3;
  static constexpr int N = 4;

  basalt::RdBezier<DIM, N> spline(int64_t(2e9));
  spline.genRandomTrajectory();

  for (int64_t t_ns = 0; t_ns < spline.maxTimeNs(); t_ns += 1e8) {
    testEvaluate<DIM, N, 2>(spline, t_ns);
  }
}

TEST(BezierTest, UBBezierAccelKnots5) {
  static constexpr int DIM = 3;
  static constexpr int N = 5;

  basalt::RdBezier<DIM, N> spline(int64_t(2e9));
  spline.genRandomTrajectory();

  for (int64_t t_ns = 0; t_ns < spline.maxTimeNs(); t_ns += 1e8) {
    testEvaluate<DIM, N, 2>(spline, t_ns);
  }
}

TEST(BezierTest, UBBezierAccelKnots6) {
  static constexpr int DIM = 3;
  static constexpr int N = 6;

  basalt::RdBezier<DIM, N> spline(int64_t(2e9));
  spline.genRandomTrajectory();

  for (int64_t t_ns = 0; t_ns < spline.maxTimeNs(); t_ns += 1e8) {
    testEvaluate<DIM, N, 2>(spline, t_ns);
  }
}

TEST(BezierTest, UBBezierEvaluateTimeDeriv4) {
  static constexpr int DIM = 3;
  static constexpr int N = 4;

  basalt::RdBezier<DIM, N> spline(int64_t(2e9));
  spline.genRandomTrajectory();

  for (int64_t t_ns = 1e8; t_ns < spline.maxTimeNs() - 1e8; t_ns += 1e8) {
    testTimeDeriv<DIM, N, 0>(spline, t_ns);
  }
}

TEST(BezierTest, UBBezierEvaluateTimeDeriv5) {
  static constexpr int DIM = 3;
  static constexpr int N = 5;

  basalt::RdBezier<DIM, N> spline(int64_t(2e9));
  spline.genRandomTrajectory();

  for (int64_t t_ns = 1e8; t_ns < spline.maxTimeNs() - 1e8; t_ns += 1e8) {
    testTimeDeriv<DIM, N, 0>(spline, t_ns);
  }
}

TEST(BezierTest, UBBezierEvaluateTimeDeriv6) {
  static constexpr int DIM = 3;
  static constexpr int N = 6;

  basalt::RdBezier<DIM, N> spline(int64_t(2e9));
  spline.genRandomTrajectory();

  for (int64_t t_ns = 1e8; t_ns < spline.maxTimeNs() - 1e8; t_ns += 1e8) {
    testTimeDeriv<DIM, N, 0>(spline, t_ns);
  }
}

TEST(BezierTest, UBBezierVelocityTimeDeriv4) {
  static constexpr int DIM = 3;
  static constexpr int N = 4;

  basalt::RdBezier<DIM, N> spline(int64_t(2e9));
  spline.genRandomTrajectory();

  for (int64_t t_ns = 1e8; t_ns < spline.maxTimeNs() - 1e8; t_ns += 1e8) {
    testTimeDeriv<DIM, N, 1>(spline, t_ns);
  }
}

TEST(BezierTest, UBBezierVelocityTimeDeriv5) {
  static constexpr int DIM = 3;
  static constexpr int N = 5;

  basalt::RdBezier<DIM, N> spline(int64_t(2e9));
  spline.genRandomTrajectory();

  for (int64_t t_ns = 1e8; t_ns < spline.maxTimeNs() - 1e8; t_ns += 1e8) {
    testTimeDeriv<DIM, N, 1>(spline, t_ns);
  }
}

TEST(BezierTest, UBBezierVelocityTimeDeriv6) {
  static constexpr int DIM = 3;
  static constexpr int N = 6;

  basalt::RdBezier<DIM, N> spline(int64_t(2e9));
  spline.genRandomTrajectory();

  for (int64_t t_ns = 1e8; t_ns < spline.maxTimeNs() - 1e8; t_ns += 1e8) {
    testTimeDeriv<DIM, N, 1>(spline, t_ns);
  }
}
