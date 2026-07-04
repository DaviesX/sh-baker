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

  // 2. Check convergence (Adaptive Sampling).
  //
  // Estimate variance from more than a handful of samples: the integrand is
  // heavy-tailed, so a small warm-up under-estimates variance and stops right
  // before the tail appears. 64 gives the tail a chance to register.
  constexpr int kMinSamples = 64;
  if (sensor.sample_count >= kMinSamples) {
    // 3-sigma standard error of the mean of the per-sample luminance. Guard
    // against a slightly-negative m2 from Welford float error, which would make
    // std::sqrt return NaN and stall convergence (NaN < tolerance is false).
    float m2 = std::max(0.0f, sensor.m2_luminance);
    float variance = m2 / (sensor.sample_count - 1);
    float std_dev = std::sqrt(variance);
    float sem = 3.f * std_dev / std::sqrt((float)sensor.sample_count);

    // Relative test (SEM small vs. signal) for lit texels, with an absolute
    // noise floor so genuinely black texels (mean and variance ~0) still
    // terminate. This replaces the old `else return nullopt` dark branch, which
    // force-stopped every dim indirect texel at kMinSamples *independent of the
    // threshold* — the reason a smaller threshold never reduced the noise.
    constexpr float kAbsNoiseFloor = 1e-4f;
    float tolerance = std::max(
        sensor.confidence_threshold * sensor.mean_luminance, kAbsNoiseFloor);
    if (sem < tolerance) {
      return std::nullopt;
    }
  }

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
