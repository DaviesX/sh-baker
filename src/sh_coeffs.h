#ifndef SH_BAKER_SRC_SH_COEFFS_H_
#define SH_BAKER_SRC_SH_COEFFS_H_

#include <Eigen/Dense>

namespace sh_baker {

// Stores 3rd-order Spherical Harmonics coefficients (9 bands).
// The coefficients represent the projection of the incident radiance field L_i.
struct SHCoeffs {
  SHCoeffs() = default;
  SHCoeffs(float val) {
    for (int i = 0; i < 9; i++) {
      coeffs[i] = Eigen::Vector3f(val, val, val);
    }
  }

  // 9 coefficients. Each coefficient is an RGB vector (3 floats).
  Eigen::Vector3f coeffs[9] = {Eigen::Vector3f::Zero(), Eigen::Vector3f::Zero(),
                               Eigen::Vector3f::Zero(), Eigen::Vector3f::Zero(),
                               Eigen::Vector3f::Zero(), Eigen::Vector3f::Zero(),
                               Eigen::Vector3f::Zero(), Eigen::Vector3f::Zero(),
                               Eigen::Vector3f::Zero()};

  // Irradiance is the integral of the incident radiance over the hemisphere.
  Eigen::Vector3f irradiance = Eigen::Vector3f::Zero();

  // Scalar multiplication (scaling brightness).
  SHCoeffs operator*(float scalar) const;

  // Addition (accumulation).
  SHCoeffs& operator+=(const SHCoeffs& other);
};

// Accumulates the contribution of incident radiance from a specific direction
// into the SH coefficients.
//
// Arguments:
//   radiance: The generic incoming radiance L_i(w).
//   direction: The normalized direction vector w pointing TOWARDS the light
//              (away from the surface point).
//   normal: The normalized surface normal vector n.
//   result: Pointer to the SHCoeffs to accumulate into.
void AccumulateRadiance(const Eigen::Vector3f& radiance,
                        const Eigen::Vector3f& direction,
                        const Eigen::Vector3f& normal, SHCoeffs* result);

}  // namespace sh_baker

#endif  // SH_BAKER_SRC_SH_COEFFS_H_
