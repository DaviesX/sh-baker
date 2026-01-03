#include "material.h"

#include <algorithm>
#include <cmath>
#include <random>

namespace sh_baker {

Eigen::Vector3f GetAlbedo(const Material& mat, const Eigen::Vector2f& uv) {
  Eigen::Vector3f albedo(0.8f, 0.8f, 0.8f);  // Default gray
  if (!mat.albedo.pixel_data.empty()) {
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
  return albedo;
}

Eigen::Vector3f GetEmission(const Material& mat, const Eigen::Vector2f& uv) {
  if (mat.emission_intensity > 0.0f) {
    return GetAlbedo(mat, uv) * mat.emission_intensity;
  }
  return Eigen::Vector3f::Zero();
}

ReflectionSample SampleMaterial(const Material& mat, const Eigen::Vector2f& uv,
                                const Eigen::Vector3f& normal,
                                const Eigen::Vector3f& incident,
                                std::mt19937& rng) {
  // Cosine-weighted hemisphere sampling for Lambertian
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  float r1 = dist(rng);
  float r2 = dist(rng);

  // Local Cosine Sample
  float phi = 2.0f * M_PI * r1;
  float theta = std::acos(std::sqrt(r2));
  float sin_theta = std::sin(theta);
  Eigen::Vector3f local_dir(std::cos(phi) * sin_theta,
                            std::sin(phi) * sin_theta, std::cos(theta));

  // Transform to World
  Eigen::Vector3f t, b;
  if (std::abs(normal.x()) > std::abs(normal.z())) {
    t = Eigen::Vector3f(-normal.y(), normal.x(), 0.0f);
  } else {
    t = Eigen::Vector3f(0.0f, -normal.z(), normal.y());
  }
  t.normalize();
  b = normal.cross(t);

  Eigen::Vector3f world_dir =
      t * local_dir.x() + b * local_dir.y() + normal * local_dir.z();

  ReflectionSample sample;
  sample.direction = world_dir.normalized();
  // PDF = cos(theta) / PI
  // cos(theta) is local_dir.z()
  sample.pdf = local_dir.z() / M_PI;
  if (sample.pdf < 1e-6f) sample.pdf = 1e-6f;

  return sample;
}

Eigen::Vector3f EvalMaterial(const Material& mat, const Eigen::Vector2f& uv,
                             const Eigen::Vector3f& normal,
                             const Eigen::Vector3f& incident,
                             const Eigen::Vector3f& reflected) {
  // Lambertian BRDF: rho / PI
  // Note: incident direction is not used for perfect Lambertian
  // reflected direction is also not strictly needed if we assume valid
  // hemisphere

  // Check if reflected is in the same hemisphere as normal
  if (normal.dot(reflected) <= 0.0f) {
    return Eigen::Vector3f::Zero();
  }

  Eigen::Vector3f albedo = GetAlbedo(mat, uv);
  return albedo * (1.0f / M_PI);
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
