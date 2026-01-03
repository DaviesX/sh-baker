#include "sh_coeffs.h"

#include <cmath>

namespace sh_baker {

namespace {

// Real Spherical Harmonic Basis Functions (l <= 2)
// Inputs: normalized direction (x, y, z)
//
// l=0:
// Y_0,0  = 0.282095
//
// l=1:
// Y_1,-1 = 0.488603 * y
// Y_1,0  = 0.488603 * z
// Y_1,1  = 0.488603 * x
//
// l=2:
// Y_2,-2 = 1.092548 * x * y
// Y_2,-1 = 1.092548 * y * z
// Y_2,0  = 0.315392 * (3z^2 - 1)
// Y_2,1  = 1.092548 * x * z
// Y_2,2  = 0.546274 * (x^2 - y^2)

constexpr float kBasis0 = 0.282095f;
constexpr float kBasis1 = 0.488603f;
constexpr float kBasis2_1 = 1.092548f;
constexpr float kBasis2_0 = 0.315392f;
constexpr float kBasis2_2 = 0.546274f;

}  // namespace

SHCoeffs SHCoeffs::operator*(float scalar) const {
  SHCoeffs result;
  for (int i = 0; i < 9; ++i) {
    result.coeffs[i] = coeffs[i] * scalar;
  }
  return result;
}

SHCoeffs& SHCoeffs::operator+=(const SHCoeffs& other) {
  for (int i = 0; i < 9; ++i) {
    coeffs[i] += other.coeffs[i];
  }
  return *this;
}

void AccumulateRadiance(const Eigen::Vector3f& radiance,
                        const Eigen::Vector3f& direction, SHCoeffs* result) {
  if (!result) return;

  float x = direction.x();
  float y = direction.y();
  float z = direction.z();

  // Band 0
  float sh[9];
  sh[0] = kBasis0;

  // Band 1
  sh[1] = kBasis1 * y;
  sh[2] = kBasis1 * z;
  sh[3] = kBasis1 * x;

  // Band 2
  sh[4] = kBasis2_1 * x * y;
  sh[5] = kBasis2_1 * y * z;
  sh[6] = kBasis2_0 * (3.0f * z * z - 1.0f);
  sh[7] = kBasis2_1 * x * z;
  sh[8] = kBasis2_2 * (x * x - y * y);

  // Accumulate
  for (int i = 0; i < 9; ++i) {
    result->coeffs[i] += radiance * sh[i];
  }
}

}  // namespace sh_baker
