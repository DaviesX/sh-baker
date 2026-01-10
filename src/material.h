#ifndef SH_BAKER_SRC_MATERIAL_H_
#define SH_BAKER_SRC_MATERIAL_H_

#include <Eigen/Dense>
#include <random>

#include "scene.h"

namespace sh_baker {

// Uniformly sample a direction on the hemisphere (Z-up local frame).
// u1, u2 are uniform random numbers in [0, 1).
// Output of material sampling
struct ReflectionSample {
  Eigen::Vector3f direction;  // Sampled outgoing direction
  float pdf = 0.f;            // Probability density of this sample
};

// Samples an outgoing direction based on the material BRDF and reflected
// direction.
// reflected: outgoing direction, away from the surface (pointing to
// sensor/previous bounce).
ReflectionSample SampleMaterial(const Material& mat, const Eigen::Vector2f& uv,
                                const Eigen::Vector3f& normal,
                                const Eigen::Vector3f& reflected,
                                std::mt19937& rng);

// Evaluates the material BRDF f_r(p, wr, wi).
// Returns the BRDF value (color).
// incident: incoming direction, away from the surface (pointing to light
// sources/next bounce).
// reflected: outgoing direction, away from the surface (pointing to
// sensor/previous bounce).
Eigen::Vector3f EvalMaterial(const Material& mat, const Eigen::Vector2f& uv,
                             const Eigen::Vector3f& normal,
                             const Eigen::Vector3f& incident,
                             const Eigen::Vector3f& reflected);

// Helper to retrieve albedo from texture or default.
Eigen::Vector3f GetAlbedo(const Material& mat, const Eigen::Vector2f& uv);

// Helper to retrieve emission (radiance).
Eigen::Vector3f GetEmission(const Material& mat, const Eigen::Vector2f& uv);

// Returns the alpha (transparency) value at the given UV coordinate.
// Returns 1.0f if the texture has no alpha channel.
float GetAlpha(const Material& mat, const Eigen::Vector2f& uv);

// Helper to retrieve metallic and roughness.
void GetMetallicRoughness(const Material& mat, const Eigen::Vector2f& uv,
                          float& metallic, float& roughness);

}  // namespace sh_baker

#endif  // SH_BAKER_SRC_MATERIAL_H_
