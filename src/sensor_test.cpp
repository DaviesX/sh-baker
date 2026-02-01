#include "sensor.h"

#include <gtest/gtest.h>

#include <iostream>

#include "rasterizer.h"
#include "sh_coeffs.h"

namespace sh_baker {

TEST(SensorTest, SampleHemisphereUniform) {
  std::mt19937 rng(42);
  for (int i = 0; i < 100; ++i) {
    Eigen::Vector3f v = sensor_internal::SampleHemisphereUniform(rng);
    // Check normalization
    EXPECT_NEAR(v.norm(), 1.0f, 1e-4f);
    // Check hemisphere (z >= 0 assumption based on code, usually it's Z-up)
    EXPECT_GE(v.z(), 0.0f);
  }
}

TEST(SensorTest, CovarianceConstruction) {
  // Construct SH coeffs that correspond to strong Z directional light.
  SHCoeffs coeffs;
  // Use AccumulateRadiance to build it?
  Eigen::Vector3f light_dir(0, 0, 1);
  AccumulateRadiance(Eigen::Vector3f(1, 1, 1), light_dir, &coeffs);

  Eigen::Matrix3f cov = sensor_internal::ComputeCovariance(coeffs);

  // Z direction should have highest value on diagonal relative to X/Y?
  // Our logic: M_zz = L0 + 2*c0. M_xx = L0 - c0 + c2.
  // For Z-dir light, c0 (3z^2-1) is positive. c2 (x^2-y^2) is 0.
  // So M_zz > M_xx approx.
  EXPECT_GT(cov(2, 2), cov(0, 0));
  EXPECT_GT(cov(2, 2), cov(1, 1));
}

TEST(SensorTest, EstimationConvergence) {
  // Test that the sensor stops sampling when variance is low
  SurfacePoint sp;
  sp.position = Eigen::Vector3f(0, 0, 0);
  Sensor sensor(sp, 100, 0.01f);

  SHCoeffs const_val(1.0f);   // Bright constant value
  SHCoeffs const_val_2(.0f);  // Bright constant value

  // Feed constant values -> Variance = 0
  for (int i = 0; i < 20; ++i) {
    AddSample(const_val, const_val_2, &sensor);
  }

  std::mt19937 rng(42);
  float pdf;
  auto ray = SampleRay(sensor, rng, &pdf);
  // Should stop because variance is 0 (below threshold) and samples > 16.
  EXPECT_FALSE(ray.has_value());
}

TEST(SensorTest, MaxSamples) {
  SurfacePoint sp;
  sp.position = Eigen::Vector3f(0, 0, 0);
  Sensor sensor(sp, 5, 0.0001f);

  // Force high variance so it doesn't stop early
  SHCoeffs v1(0.0f);
  SHCoeffs v2(10.0f);
  SHCoeffs v_dummy(0.0f);

  for (int i = 0; i < 5; ++i) {
    AddSample(i % 2 == 0 ? v1 : v2, v_dummy, &sensor);
  }

  std::mt19937 rng(42);
  float pdf;
  auto ray = SampleRay(sensor, rng, &pdf);
  EXPECT_FALSE(ray.has_value());
  EXPECT_EQ(sensor.sample_count, 5);
}

// Distribution Fitting Test
// Draw samples from a known distribution and fit SH coefficients.
// Then decode and compare.
TEST(SensorTest, DistributionFitting) {
  std::mt19937 rng(1234);

  // 1. Define Ground Truth Covariance (stretched in X) and Mean
  Eigen::Matrix3f cov_gt = Eigen::Matrix3f::Identity() * 0.2f;
  cov_gt(0, 0) = 2.0f;  // Strongly expected in X

  Eigen::Vector3f mean_gt(1.0f, 0.0f, 0.0f);  // Mean direction also X

  // 2. Generate samples and accumulate SH
  // We simulate "light" coming from directions distributed according to cov_gt.
  // We use SampleAngularGaussian logic here manually or re-use helper?
  // We can use the helper to GENERATE samples.

  SHCoeffs accumulated_sh(0.0f);
  int num_samples = 10000;

  for (int i = 0; i < num_samples; ++i) {
    Eigen::Vector3f dir_local =
        sensor_internal::SampleAngularGaussian(cov_gt, mean_gt, rng);
    // SampleAngularGaussian gives random directions based on cov.
    // We assume each "photon" carries unit energy.
    AccumulateRadiance(Eigen::Vector3f(1, 1, 1), dir_local, &accumulated_sh);
  }

  SHCoeffs avg_sh = accumulated_sh * (1.0f / num_samples);

  // 3. Decode Covariance
  Eigen::Matrix3f cov_est = sensor_internal::ComputeCovariance(avg_sh);
  Eigen::Vector3f mean_est = sensor_internal::ComputeMean(avg_sh);

  // 4. Verification
  // The estimated covariance should share principal axes with GT.
  // GT principal axis is X (1, 0, 0).
  // cov_est(0,0) should be largest diagonal.

  std::cout << "Gt Cov:\n" << cov_gt << "\n";
  std::cout << "Est Cov:\n" << cov_est << "\n";
  std::cout << "Gt Mean:\n" << mean_gt.transpose() << "\n";
  std::cout << "Est Mean (SH L1):\n" << mean_est.transpose() << "\n";

  EXPECT_GT(cov_est(0, 0), cov_est(1, 1));
  EXPECT_GT(cov_est(0, 0), cov_est(2, 2));

  // Check that off-diagonals are small relative to X diagonal
  EXPECT_LT(std::abs(cov_est(0, 1)), cov_est(0, 0) * 0.1f);
  EXPECT_LT(std::abs(cov_est(0, 2)), cov_est(0, 0) * 0.1f);

  // Verify Mean Direction matches X axis
  // It should be roughly (1, 0, 0) * scale?
  // Our ComputeMean extracts L1 moments.
  // With X-aligned distribution, M_x should be largest.
  EXPECT_GT(std::abs(mean_est.x()), std::abs(mean_est.y()));
  EXPECT_GT(std::abs(mean_est.x()), std::abs(mean_est.z()));
  EXPECT_GT(mean_est.x(), 0.0f);  // Should be positive X
}

}  // namespace sh_baker
