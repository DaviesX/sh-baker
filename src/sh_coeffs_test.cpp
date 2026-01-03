#include "sh_coeffs.h"

#include <gtest/gtest.h>

namespace sh_baker {

TEST(SHCoeffsTest, BasisFunctionCheck) {
  // Test specific basis values for known directions.
  // Direction +Z (0, 0, 1)
  // l=0: Y00 = 0.282095
  // l=1: Y10 = 0.488603 * z = 0.488603
  // l=2: Y20 = 0.315392 * (3*1 - 1) = 0.630784
  // Others should be 0.

  SHCoeffs result;
  Eigen::Vector3f radiance(1.0f, 1.0f, 1.0f);
  Eigen::Vector3f direction(0.0f, 0.0f, 1.0f);

  AccumulateRadiance(radiance, direction, &result);

  // Allow small epsilon for float precision
  constexpr float kEpsilon = 1e-5f;

  // Band 0
  EXPECT_NEAR(result.coeffs[0].x(), 0.282095f, kEpsilon);

  // Band 1
  EXPECT_NEAR(result.coeffs[1].x(), 0.0f, kEpsilon);       // Y1-1 (y)
  EXPECT_NEAR(result.coeffs[2].x(), 0.488603f, kEpsilon);  // Y10  (z)
  EXPECT_NEAR(result.coeffs[3].x(), 0.0f, kEpsilon);       // Y11  (x)

  // Band 2
  EXPECT_NEAR(result.coeffs[4].x(), 0.0f, kEpsilon);       // Y2-2 (xy)
  EXPECT_NEAR(result.coeffs[5].x(), 0.0f, kEpsilon);       // Y2-1 (yz)
  EXPECT_NEAR(result.coeffs[6].x(), 0.630784f, kEpsilon);  // Y20  (3z^2-1)
  EXPECT_NEAR(result.coeffs[7].x(), 0.0f, kEpsilon);       // Y21  (xz)
  EXPECT_NEAR(result.coeffs[8].x(), 0.0f, kEpsilon);       // Y22  (x^2-y^2)
}

TEST(SHCoeffsTest, Accumulate) {
  SHCoeffs c1;
  c1.coeffs[0] = Eigen::Vector3f(1.0f, 2.0f, 3.0f);

  SHCoeffs c2;
  c2.coeffs[0] = Eigen::Vector3f(0.5f, 0.5f, 0.5f);

  c1 += c2;
  EXPECT_FLOAT_EQ(c1.coeffs[0].x(), 1.5f);
  EXPECT_FLOAT_EQ(c1.coeffs[0].y(), 2.5f);
  EXPECT_FLOAT_EQ(c1.coeffs[0].z(), 3.5f);
}

TEST(SHCoeffsTest, Scale) {
  SHCoeffs c1;
  c1.coeffs[0] = Eigen::Vector3f(1.0f, 2.0f, 3.0f);

  SHCoeffs c2 = c1 * 2.0f;
  EXPECT_FLOAT_EQ(c2.coeffs[0].x(), 2.0f);
  EXPECT_FLOAT_EQ(c2.coeffs[0].y(), 4.0f);
  EXPECT_FLOAT_EQ(c2.coeffs[0].z(), 6.0f);
}

}  // namespace sh_baker
