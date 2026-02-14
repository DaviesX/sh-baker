#include "sensor.h"

#include <cmath>
#include <optional>
#include <random>

#include "occlusion.h"
#include "sh_coeffs.h"

namespace sh_baker {
namespace sensor_internal {

Eigen::Vector3f SampleHemisphereUniform(std::mt19937& rng) {
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  float u1 = dist(rng);
  float u2 = dist(rng);

  float r = std::sqrt(1.0f - u1 * u1);
  float phi = 2.0f * M_PI * u2;
  return Eigen::Vector3f(r * std::cos(phi), r * std::sin(phi), u1);
}

}  // namespace sensor_internal

void AddSample(const SHCoeffs& sample, Sensor* sensor) {
  sensor->sample_count++;
  sensor->sh_coeffs_sum += sample;

  // L0 luminance (DC component brightness).
  // SH basis 0 is (1/sqrt(4pi)) * Y00. Y00 is constant.
  // We can just use the norm of the first coefficient vector (RGB).
  float lum = sample.coeffs[0].norm();

  // Welford's algorithm
  float delta = lum - sensor->mean_luminance;
  sensor->mean_luminance += delta / sensor->sample_count;
  float delta2 = lum - sensor->mean_luminance;
  sensor->m2_luminance += delta * delta2;
}

SHCoeffs GetEstimation(const Sensor& sensor) {
  if (sensor.sample_count == 0) {
    return SHCoeffs(0);
  }
  return sensor.sh_coeffs_sum *
         (1.0f / static_cast<float>(sensor.sample_count));
}

std::optional<Ray> SampleRay(const Sensor& sensor, std::mt19937& rng) {
  // 1. Check max samples
  if (sensor.sample_count >= sensor.max_samples) {
    return std::nullopt;
  }

  // 2. Check convergence (Adaptive Sampling)
  constexpr int kMinSamples = 32;
  // if (sensor.sample_count >= kMinSamples) {
  //   // Calculate Standard Error of the Mean
  //   if (sensor.mean_luminance > 1e-3f) {
  //     float variance = sensor.m2_luminance / (sensor.sample_count - 1);
  //     float std_dev = std::sqrt(variance);
  //     float sem = 3.f * std_dev / std::sqrt((float)sensor.sample_count);

  //     // Coefficient of Variation of the Mean = SEM / Mean
  //     // If error is small enough relative to the signal, we stop.
  //     if (sem < sensor.confidence_threshold * sensor.mean_luminance) {
  //       return std::nullopt;
  //     }
  //   } else {
  //     // If it's pitch black (or very close), we can stop early too.
  //     return std::nullopt;
  //   }
  // }

  // 3. Generate Ray
  Ray ray;
  ray.tnear = 0.005f;
  ray.tfar = 1e4f;
  ray.origin = sensor.sp.position;

  Eigen::Vector3f dir_local = sensor_internal::SampleHemisphereUniform(rng);

  // Transform to World
  // Calculate bitangent (using w for handedness)
  Eigen::Vector3f bitangent =
      (sensor.sp.normal.cross(sensor.sp.tangent.head<3>()) *
       sensor.sp.tangent.w())
          .normalized();

  ray.direction = sensor.sp.tangent.head<3>() * dir_local.x() +
                  bitangent * dir_local.y() + sensor.sp.normal * dir_local.z();

  return ray;
}

}  // namespace sh_baker
