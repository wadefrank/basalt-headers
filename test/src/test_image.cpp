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

#include <Eigen/Dense>

#include <basalt/image/image.h>

#include "gtest/gtest.h"
#include "test_utils.h"

void setImageData(uint16_t* image_array, int size) {
  double norm = RAND_MAX;
  norm /= (double)std::numeric_limits<uint16_t>::max();

  for (int i = 0; i < size; i++) {
    image_array[i] = (uint16_t)(rand() / norm);
  }
}

TEST(Image, ImageInterpolate) {
  Eigen::Vector2i offset(231, 123);

  basalt::ManagedImage<uint16_t> img(640, 480);
  setImageData(img.ptr, img.size());

  double eps = 1e-12;
  double threshold = 1e-6;

  {
    const Eigen::Vector2i& pi = offset;
    Eigen::Vector2d pd = pi.cast<double>() + Eigen::Vector2d(eps, eps);

    uint16_t val1 = img(pi);
    double val2 = img.interp(pd);
    double val3 = img.interpGrad(pd)[0];

    EXPECT_LE(std::abs(val2 - val1), threshold);
    EXPECT_FLOAT_EQ(val2, val3);
  }

  {
    const Eigen::Vector2i& pi = offset;
    Eigen::Vector2d pd = pi.cast<double>() + Eigen::Vector2d(eps, eps);

    uint16_t val1 = img(pi);
    double val2 = img.interp(pd);
    double val3 = img.interpGrad(pd)[0];

    EXPECT_LE(std::abs(val2 - val1), threshold);
    EXPECT_FLOAT_EQ(val2, val3);
  }

  {
    Eigen::Vector2i pi = offset + Eigen::Vector2i(1, 0);
    Eigen::Vector2d pd = pi.cast<double>() + Eigen::Vector2d(-eps, eps);

    uint16_t val1 = img(pi);
    double val2 = img.interp(pd);
    double val3 = img.interpGrad(pd)[0];

    EXPECT_LE(std::abs(val2 - val1), threshold);
    EXPECT_FLOAT_EQ(val2, val3);
  }

  {
    Eigen::Vector2i pi = offset + Eigen::Vector2i(0, 1);
    Eigen::Vector2d pd = pi.cast<double>() + Eigen::Vector2d(eps, -eps);

    uint16_t val1 = img(pi);
    double val2 = img.interp(pd);
    double val3 = img.interpGrad(pd)[0];

    EXPECT_LE(std::abs(val2 - val1), threshold);
    EXPECT_FLOAT_EQ(val2, val3);
  }

  {
    Eigen::Vector2i pi = offset + Eigen::Vector2i(1, 1);
    Eigen::Vector2d pd = pi.cast<double>() + Eigen::Vector2d(-eps, -eps);

    uint16_t val1 = img(pi);
    double val2 = img.interp(pd);
    double val3 = img.interpGrad(pd)[0];

    EXPECT_LE(std::abs(val2 - val1), threshold);
    EXPECT_FLOAT_EQ(val2, val3);
  }
}

TEST(Image, ImageInterpolateGrad) {
  Eigen::Vector2i offset(231, 123);

  basalt::ManagedImage<uint16_t> img(640, 480);
  setImageData(img.ptr, img.size());

  Eigen::Vector2d pd = offset.cast<double>() + Eigen::Vector2d(0.4, 0.34345);

  Eigen::Vector3d val_grad = img.interpGrad<double>(pd);
  Eigen::Matrix<double, 1, 2> J_x = val_grad.tail<2>();

  test_jacobian(
      "d_res_d_x", J_x,
      [&](const Eigen::Vector2d& x) {
        return Eigen::Matrix<double, 1, 1>(img.interp<double>(pd + x));
      },
      Eigen::Vector2d::Zero(),
      1);  // only works with eps=1 with this gradient interpolation
}

TEST(Image, ImageInterpolateGradBilinearExact) {
  Eigen::Vector2i offset(231, 123);

  basalt::ManagedImage<uint16_t> img(640, 480);
  setImageData(img.ptr, img.size());

  Eigen::Vector2d pd = offset.cast<double>() + Eigen::Vector2d(0.4, 0.34345);

  Eigen::Vector3d val_grad = img.interpGradBilinearExact<double>(pd);
  Eigen::Matrix<double, 1, 2> J_x = val_grad.tail<2>();

  test_jacobian(
      "d_res_d_x", J_x,
      [&](const Eigen::Vector2d& x) {
        return Eigen::Matrix<double, 1, 1>(img.interp<double>(pd + x));
      },
      Eigen::Vector2d::Zero(), 1e-4);
}

TEST(Image, ImageInterpolateGradCubicSplines) {
  Eigen::Vector2i offset(231, 123);

  basalt::ManagedImage<uint16_t> img(640, 480);
  setImageData(img.ptr, img.size());

  Eigen::Vector2d pd = offset.cast<double>() + Eigen::Vector2d(0.4, 0.34345);

  Eigen::Vector3d val_grad = img.interpGradCubicSplines<double>(pd);
  Eigen::Matrix<double, 1, 2> J_x = val_grad.tail<2>();

  test_jacobian(
      "d_res_d_x", J_x,
      [&](const Eigen::Vector2d& x) {
        return Eigen::Matrix<double, 1, 1>(
            img.interpCubicSplines<double>(pd + x));
      },
      Eigen::Vector2d::Zero(), 1e-4);
}

TEST(Image, BasicOperations) {
  basalt::ManagedImage<uint16_t> img(4, 3);

  // Test Fill
  img.Fill(42);
  for (size_t i = 0; i < img.size(); ++i) {
    EXPECT_EQ(img.ptr[i], 42);
  }

  // Test Replace
  img.Replace(42, 100);
  for (size_t i = 0; i < img.size(); ++i) {
    EXPECT_EQ(img.ptr[i], 100);
  }

  // Test Memset
  img.Memset(0);
  for (size_t i = 0; i < img.size(); ++i) {
    EXPECT_EQ(img.ptr[i], 0);
  }
}

TEST(Image, SizeAndValidity) {
  basalt::ManagedImage<uint16_t> img;
  EXPECT_FALSE(img.IsValid());
  EXPECT_EQ(img.size(), 0);
  EXPECT_EQ(img.Area(), 0);

  img = basalt::ManagedImage<uint16_t>(4, 3);
  EXPECT_TRUE(img.IsValid());
  EXPECT_EQ(img.size(), 12);
  EXPECT_EQ(img.Area(), 12);
  EXPECT_TRUE(img.IsContiguous());
}

TEST(Image, CopyAndMove) {
  basalt::ManagedImage<uint16_t> img1(4, 3);
  setImageData(img1.ptr, img1.size());

  // Test move constructor
  basalt::ManagedImage<uint16_t> img2(std::move(img1));
  EXPECT_FALSE(img1.IsValid());
  EXPECT_TRUE(img2.IsValid());
  EXPECT_EQ(img2.w, 4);
  EXPECT_EQ(img2.h, 3);

  // Test move assignment
  basalt::ManagedImage<uint16_t> img3;
  img3 = std::move(img2);
  EXPECT_FALSE(img2.IsValid());
  EXPECT_TRUE(img3.IsValid());
  EXPECT_EQ(img3.w, 4);
  EXPECT_EQ(img3.h, 3);
}

TEST(Image, SubImageOperations) {
  basalt::ManagedImage<uint16_t> img(6, 4);
  setImageData(img.ptr, img.size());

  // Create a sub-image
  auto sub_img = img.SubImage(1, 1, 3, 2);
  EXPECT_EQ(sub_img.w, 3);
  EXPECT_EQ(sub_img.h, 2);

  // Verify sub-image data is correctly referenced
  for (size_t y = 0; y < sub_img.h; ++y) {
    for (size_t x = 0; x < sub_img.w; ++x) {
      EXPECT_EQ(sub_img(x, y), img(x + 1, y + 1));
    }
  }

  // Modify sub-image and verify original is affected
  sub_img.Fill(42);
  for (size_t y = 0; y < sub_img.h; ++y) {
    for (size_t x = 0; x < sub_img.w; ++x) {
      EXPECT_EQ(img(x + 1, y + 1), 42);
    }
  }
}

TEST(Image, Transformations) {
  basalt::ManagedImage<uint16_t> img(4, 3);
  setImageData(img.ptr, img.size());

  // Test transform (multiply by 2)
  img.Transform([](const uint16_t& val) { return val * 2; });

  // Test accumulate (sum all elements)
  uint32_t sum =
      img.Accumulate(0, [](uint32_t acc, uint16_t val) { return acc + val; });
  EXPECT_GT(sum, 0);

  // Test MinMax
  auto [min_val, max_val] = img.MinMax();
  EXPECT_LE(min_val, max_val);
}
