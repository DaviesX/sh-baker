#ifndef SH_BAKER_SRC_MATERIAL_H_
#define SH_BAKER_SRC_MATERIAL_H_

#include <Eigen/Dense>

#include "scene.h"

namespace sh_baker {

// Uniformly sample a direction on the hemisphere (Z-up local frame).
// u1, u2 are uniform random numbers in [0, 1).
Eigen::Vector3f SampleHemisphereUniform(float u1, float u2);

// Evaluates the material (albedo + emission) at the given UV coordinate.
Eigen::Vector3f EvalMaterial(const Material& mat, const Eigen::Vector2f& uv);

// Returns the alpha (transparency) value at the given UV coordinate.
// Returns 1.0f if the texture has no alpha channel.
float GetAlpha(const Material& mat, const Eigen::Vector2f& uv);

}  // namespace sh_baker

#endif  // SH_BAKER_SRC_MATERIAL_H_
