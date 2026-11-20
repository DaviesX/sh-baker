#include "sensor.h"

#include <gtest/gtest.h>

#include <cmath>
#include <optional>
#include <random>

#include "glog/logging.h"
#include "rasterizer.h"

namespace sh_baker {

TEST(SensorTest, SampleHemisphereUniform) {
  std::mt19937 rng(42);
  for (int i = 0; i < 100; ++i) {
    Eigen::Vector3f v = sensor_internal::SampleHemisphereUniform(rng);
    // Check normalization
    EXPECT_NEAR(v.norm(), 1.0f, 1e-4f);
    // Check hemisphere (z >= 0 assumption based on code, usually it's Z-up)
    // Code says: "return Eigen::Vector3f(r * std::cos(phi), r * std::sin(phi),
    // u1);" u1 is in [0, 1). So Z is positive.
    EXPECT_GE(v.z(), 0.0f);
  }
}

TEST(SensorTest, EstimationConvergence) {
  // Test that the sensor stops sampling when variance is low
  SurfacePoint sp;
  sp.position = Eigen::Vector3f(0, 0, 0);
  Sensor sensor(sp, 100, 0.01f);

  SHCoeffs const_val(1.0f);  // Bright constant value

  // Feed constant values -> Variance = 0. Must exceed the adaptive-sampling
  // warm-up (kMinSamples) before convergence is evaluated.
  for (int i = 0; i < 65; ++i) {
    AddSample(const_val, &sensor);
  }

  std::mt19937 rng(42);
  auto ray = SampleRay(sensor, rng);
  // Should stop: variance is 0 (SEM below threshold) and samples > kMinSamples.
  EXPECT_FALSE(ray.has_value());

  SHCoeffs est = GetEstimation(sensor);
  // Coeff[0] should be roughly what we put in.
  // const_val is 1.0 everywhere.
  EXPECT_NEAR(est.coeffs[0].x(), 1.0f, 1e-4f);
}

TEST(SensorTest, MaxSamples) {
  SurfacePoint sp;
  sp.position = Eigen::Vector3f(0, 0, 0);
  Sensor sensor(sp, 100, 0.01f);

  // Force high variance so it doesn't stop early
  SHCoeffs v1(0.0f);
  SHCoeffs v2(10.0f);

  for (int i = 0; i < 100; ++i) {
    AddSample(i % 2 == 0 ? v1 : v2, &sensor);
  }

  std::mt19937 rng(42);
  auto ray = SampleRay(sensor, rng);
  EXPECT_FALSE(ray.has_value());
  EXPECT_EQ(sensor.sample_count, 100);
}

// Takes rays until SampleRay() stops (with a constant, zero-variance sample) or
// `cap` samples are taken, and returns the sample count.
unsigned RunToStop(Sensor* sensor, unsigned cap) {
  std::mt19937 rng(42);
  const SHCoeffs const_val(1.0f);
  while (sensor->sample_count < cap && SampleRay(*sensor, rng).has_value()) {
    AddSample(const_val, sensor);
  }
  return sensor->sample_count;
}

SurfacePoint UpPoint() {
  SurfacePoint sp;
  sp.position = Eigen::Vector3f(0, 0, 0);
  sp.normal = Eigen::Vector3f(0, 0, 1);
  sp.tangent = Eigen::Vector4f(1, 0, 0, 1);
  return sp;
}

// The default threshold stops a zero-variance sensor right after the warm-up
// (kMinSamples = 64), however many samples it may take.
TEST(SensorTest, DefaultThresholdStopsZeroVarianceAtWarmUp) {
  Sensor sensor(UpPoint(), 4096, 0.01f);
  EXPECT_EQ(RunToStop(&sensor, 10000), 64u);
}

// A threshold of 0 disables the adaptive stop: even a zero-variance sensor
// takes exactly max_samples rays.
TEST(SensorTest, ZeroThresholdTakesEveryRay) {
  Sensor sensor(UpPoint(), 1000, 0.0f);
  EXPECT_EQ(RunToStop(&sensor, 10000), 1000u);
  std::mt19937 rng(1);
  EXPECT_FALSE(SampleRay(sensor, rng).has_value());
}

// So does a negative threshold.
TEST(SensorTest, NegativeThresholdTakesEveryRay) {
  Sensor sensor(UpPoint(), 1000, -1.0f);
  EXPECT_EQ(RunToStop(&sensor, 10000), 1000u);
}

TEST(SensorTest, SampleSphereUniformCoversTheSphere) {
  std::mt19937 rng(42);
  constexpr int kN = 20000;
  Eigen::Vector3f mean = Eigen::Vector3f::Zero();
  int below = 0;
  for (int i = 0; i < kN; ++i) {
    Eigen::Vector3f v = sensor_internal::SampleSphereUniform(rng);
    EXPECT_NEAR(v.norm(), 1.0f, 1e-4f);
    mean += v;
    if (v.z() < 0.0f) ++below;
  }
  mean /= float(kN);
  // Uniform: zero mean (each component has variance 1/3, so the mean's sigma
  // is 0.004) and half the samples below the equator (sigma 0.0035).
  EXPECT_LT(mean.norm(), 0.02f);
  EXPECT_NEAR(below / float(kN), 0.5f, 0.02f);
}

// A full-sphere point samples world-space directions over the whole sphere,
// with a zero normal and tangent, and weighs each by 4 pi; a surface point
// keeps the hemisphere and 2 pi.
TEST(SensorTest, FullSphereRaysIgnoreTheFrame) {
  SurfacePoint sp;
  sp.position = Eigen::Vector3f(1, 2, 3);
  sp.material_id = 0;
  sp.full_sphere = true;
  Sensor sensor(sp, 1000, 0.01f);
  EXPECT_FLOAT_EQ(SampleRayInvPdf(sensor), 4.0f * float(M_PI));

  std::mt19937 rng(7);
  std::mt19937 reference_rng(7);
  int below = 0;
  for (int i = 0; i < 200; ++i) {
    std::optional<Ray> ray = SampleRay(sensor, rng);
    ASSERT_TRUE(ray.has_value());
    EXPECT_TRUE(ray->direction.allFinite());
    EXPECT_NEAR(ray->direction.norm(), 1.0f, 1e-4f);
    EXPECT_EQ(ray->origin, sp.position);
    // Exactly the world-space sphere sample, with the same draws.
    EXPECT_EQ(ray->direction,
              sensor_internal::SampleSphereUniform(reference_rng));
    if (ray->direction.z() < 0.0f) ++below;
  }
  EXPECT_GT(below, 50);
  EXPECT_LT(below, 150);

  SurfacePoint surface = sp;
  surface.full_sphere = false;
  surface.normal = Eigen::Vector3f(0, 0, 1);
  surface.tangent = Eigen::Vector4f(1, 0, 0, 1);
  EXPECT_EQ(SampleRayInvPdf(Sensor(surface, 1000, 0.01f)),
            static_cast<float>(2.0f * M_PI));
}

}  // namespace sh_baker
