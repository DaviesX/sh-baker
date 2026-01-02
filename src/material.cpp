#include "material.h"

#include <algorithm>
#include <cmath>

namespace sh_baker {

Eigen::Vector3f SampleHemisphereUniform(float u1, float u2) {
  float r = std::sqrt(1.0f - u1 * u1);
  float phi = 2.0f * M_PI * u2;
  return Eigen::Vector3f(r * std::cos(phi), r * std::sin(phi), u1);
}

Eigen::Vector3f EvalMaterial(const Material& mat, const Eigen::Vector2f& uv) {
  // Simple diffuse only + emission
  // Retrieve albedo from texture if present
  Eigen::Vector3f albedo(0.8f, 0.8f, 0.8f);  // Default gray
  if (!mat.albedo.pixel_data.empty()) {
    // Bilinear sample or nearest? Nearest is fine for now.
    int tx = std::clamp((int)(uv.x() * mat.albedo.width), 0,
                        (int)mat.albedo.width - 1);
    int ty = std::clamp((int)(uv.y() * mat.albedo.height), 0,
                        (int)mat.albedo.height - 1);
    int idx = (ty * mat.albedo.width + tx) * mat.albedo.channels;
    float r = mat.albedo.pixel_data[idx] / 255.0f;
    float g = mat.albedo.pixel_data[idx + 1] / 255.0f;
    float b = mat.albedo.pixel_data[idx + 2] / 255.0f;
    albedo = Eigen::Vector3f(r, g, b);
  }

  // Emission
  if (mat.emission_intensity > 0.0f) {
    return albedo * mat.emission_intensity;
  }
  return albedo;
}

float GetAlpha(const Material& mat, const Eigen::Vector2f& uv) {
  if (mat.albedo.pixel_data.empty()) return 1.0f;
  if (mat.albedo.channels < 4) return 1.0f;

  int tx = std::clamp((int)(uv.x() * mat.albedo.width), 0,
                      (int)mat.albedo.width - 1);
  int ty = std::clamp((int)(uv.y() * mat.albedo.height), 0,
                      (int)mat.albedo.height - 1);
  int idx = (ty * mat.albedo.width + tx) * mat.albedo.channels;
  return mat.albedo.pixel_data[idx + 3] / 255.0f;
}

}  // namespace sh_baker
