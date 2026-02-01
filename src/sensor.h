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

Eigen::Vector3f SampleHemisphereUniform(std::mt19937& rng);

float ComputeLuminance(const SHCoeffs& coeffs);

// Computes the covariance matrix for the direction distribution from SH
// coefficients. This corresponds to the "Intensity Tensor" or "Orientation
// Tensor".
Eigen::Matrix3f ComputeCovariance(const SHCoeffs& coeffs);

// Samples a direction from the angular Gaussian distribution defined by the
// covariance matrix.
Eigen::Vector3f SampleAngularGaussian(const Eigen::Matrix3f& cov,
                                      std::mt19937& rng);

// Evaluates the PDF of the angular Gaussian distribution for a given direction.
float PdfAngularGaussian(const Eigen::Matrix3f& cov,
                         const Eigen::Vector3f& dir);

}  // namespace sensor_internal

// A sensor is a point on the surface of the object that we are baking.
struct Sensor {
  Sensor(const SurfacePoint& surface_point, const unsigned max_samples,
         float confidence_threshold)
      : sp(surface_point),
        max_samples(max_samples),
        confidence_threshold(confidence_threshold),
        sh_direct_coeffs_sum(SHCoeffs(0)),
        sh_indirect_coeffs_sum(SHCoeffs(0)) {}

  const SurfacePoint sp;
  const unsigned max_samples;
  const float confidence_threshold;

  SHCoeffs sh_direct_coeffs_sum;
  SHCoeffs sh_indirect_coeffs_sum;
  unsigned sample_count = 0;
  float mean_luminance = 0.0f;
  float m2_luminance = 0.0f;
};

// Add a sample to the sensor.
void AddSample(const SHCoeffs& direct_sh, const SHCoeffs& indirect_sh,
               Sensor* sensor);

// Get the sample-averaged SH coefficients from the sensor.
SHCoeffs GetEstimation(const Sensor& sensor);

// Sample a ray from the sensor if we need more samples (sample count <
// max_samples or confidence threshold not met).
// TODO: We will perform uniform sampling on the hemisphere for now.
// Sample a ray from the sensor.
// Returns std::nullopt if the sensor has enough samples and variance is low.
// Populates `pdf` with the probability density of sampling the returned ray.
std::optional<Ray> SampleRay(const Sensor& sensor, std::mt19937& rng,
                             float* pdf);

}  // namespace sh_baker

#endif  // SH_BAKER_SRC_SENSOR_H_