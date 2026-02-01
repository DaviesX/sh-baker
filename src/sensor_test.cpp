#include "sensor.h"

#include <gtest/gtest.h>

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

  // Feed constant values -> Variance = 0
  for (int i = 0; i < 20; ++i) {
    AddSample(const_val, &sensor);
  }

  std::mt19937 rng(42);
  auto ray = SampleRay(sensor, rng);
  // Should stop because variance is 0 (below threshold) and samples > 16.
  EXPECT_FALSE(ray.has_value());

  SHCoeffs est = GetEstimation(sensor);
  // Coeff[0] should be roughly what we put in.
  // const_val is 1.0 everywhere.
  EXPECT_NEAR(est.coeffs[0].x(), 1.0f, 1e-4f);
}

TEST(SensorTest, MaxSamples) {
  SurfacePoint sp;
  sp.position = Eigen::Vector3f(0, 0, 0);
  Sensor sensor(sp, 5, 0.0001f);

  // Force high variance so it doesn't stop early
  SHCoeffs v1(0.0f);
  SHCoeffs v2(10.0f);

  for (int i = 0; i < 5; ++i) {
    AddSample(i % 2 == 0 ? v1 : v2, &sensor);
  }

  std::mt19937 rng(42);
  auto ray = SampleRay(sensor, rng);
  EXPECT_FALSE(ray.has_value());
  EXPECT_EQ(sensor.sample_count, 5);
}

}  // namespace sh_baker
