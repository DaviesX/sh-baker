#ifndef SH_BAKER_SRC_SENSOR_H_
#define SH_BAKER_SRC_SENSOR_H_

#include <Eigen/Dense>
#include <optional>
#include <random>

#include "occlusion.h"
#include "rasterizer.h"
#include "sh_coeffs.h"

namespace sh_baker {
namespace sensor_internal {

// A uniform direction on the hemisphere around +Z.
Eigen::Vector3f SampleHemisphereUniform(std::mt19937& rng);

// A uniform direction on the whole unit sphere (pdf 1 / (4 pi)).
Eigen::Vector3f SampleSphereUniform(std::mt19937& rng);

}  // namespace sensor_internal

// A sensor is a point on the surface of the object that we are baking.
struct Sensor {
  Sensor(const SurfacePoint& surface_point, const unsigned max_samples,
         float confidence_threshold)
      : sp(surface_point),
        max_samples(max_samples),
        confidence_threshold(confidence_threshold),
        sh_coeffs_sum(SHCoeffs(0)) {}

  const SurfacePoint sp;
  const unsigned max_samples;
  const float confidence_threshold;

  SHCoeffs sh_coeffs_sum;
  unsigned sample_count = 0;
  float mean_luminance = 0.0f;
  float m2_luminance = 0.0f;
};

// Add a sample to the sensor.
void AddSample(const SHCoeffs& sample, Sensor* sensor);

// Get the sample-averaged SH coefficients from the sensor.
SHCoeffs GetEstimation(const Sensor& sensor);

// Sample a ray from the sensor if we need more samples (sample count <
// max_samples or confidence threshold not met).
// TODO: We will perform uniform sampling on the hemisphere for now.
// A full-sphere point (SurfacePoint::full_sphere) samples the whole sphere
// uniformly in world space instead, and ignores its normal and tangent. The
// adaptive stop is the same for both: it tests the L0 luminance, which suits a
// probe as well as a surface point.
std::optional<Ray> SampleRay(const Sensor& sensor, std::mt19937& rng);

// The Monte Carlo weight of one ray from SampleRay(), 1 / pdf: 2 pi for a
// surface point's hemisphere, 4 pi for a full-sphere point's sphere.
float SampleRayInvPdf(const Sensor& sensor);

}  // namespace sh_baker

#endif  // SH_BAKER_SRC_SENSOR_H_