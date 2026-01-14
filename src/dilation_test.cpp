#include "dilation.h"

#include <gtest/gtest.h>

#include <Eigen/Dense>

#include "sh_coeffs.h"

namespace sh_baker {

TEST(DilationTest, SimpleFill) {
  int w = 3;
  int h = 3;
  std::vector<SHCoeffs> pixels(w * h);
  std::vector<uint8_t> mask(w * h, 0);

  // Center pixel is valid (Red)
  int center = 1 * w + 1;
  mask[center] = 1;
  pixels[center].coeffs[0] = Eigen::Vector3f(1.0f, 0.0f, 0.0f);  // Band 0 Red

  // Dilate 1 pass
  Dilate(w, h, pixels, mask, 1);

  // Neighbors should now be valid and red
  // (0, 1) -> index 1
  EXPECT_EQ(mask[1], 1);
  EXPECT_NEAR(pixels[1].coeffs[0].x(), 1.0f, 0.001f);

  // Corner (0,0) -> index 0. Neighbor of center? Yes (diagonal).
  EXPECT_EQ(mask[0], 1);
  EXPECT_NEAR(pixels[0].coeffs[0].x(), 1.0f, 0.001f);
}

TEST(DilationTest, AverageNeighbors) {
  int w = 3;
  int h = 3;
  std::vector<SHCoeffs> pixels(w * h);
  std::vector<uint8_t> mask(w * h, 0);

  // Left (0,1) Red, Right (2,1) Blue. Center (1,1) Invalid.
  int left = 1 * w + 0;
  int right = 1 * w + 2;
  int center = 1 * w + 1;

  mask[left] = 1;
  pixels[left].coeffs[0] = Eigen::Vector3f(1.0f, 0.0f, 0.0f);

  mask[right] = 1;
  pixels[right].coeffs[0] = Eigen::Vector3f(0.0f, 0.0f, 1.0f);

  Dilate(w, h, pixels, mask, 1);

  // Center should be (0.5, 0, 0.5)
  EXPECT_EQ(mask[center], 1);
  EXPECT_NEAR(pixels[center].coeffs[0].x(), 0.5f, 0.001f);
  EXPECT_NEAR(pixels[center].coeffs[0].z(), 0.5f, 0.001f);
}

}  // namespace sh_baker
