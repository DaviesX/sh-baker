#include "light.h"

#include <gtest/gtest.h>

#include "loader.h"

namespace sh_baker {
namespace {

class LightTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Setup a simple scene
    scene_.lights.clear();

    // Add Point Light
    Light point;
    point.type = Light::Type::Point;
    point.position = Eigen::Vector3f(0, 10, 0);
    point.color = Eigen::Vector3f(1, 1, 1);
    point.intensity = 100.0f;
    scene_.lights.push_back(point);

    // Add Area Light
    Light area;
    area.type = Light::Type::Area;
    area.center = Eigen::Vector3f(10, 0, 0);
    area.normal = Eigen::Vector3f(-1, 0, 0);
    area.intensity = 10.0f;
    area.area = 5.0f;
    area.flux = 50.0f * M_PI;  // Approx
    scene_.lights.push_back(area);
  }

  Scene scene_;
  std::mt19937 rng_{12345};
};

TEST_F(LightTest, SampleLights_Count) {
  Eigen::Vector3f P(0, 0, 0);
  Eigen::Vector3f N(0, 1, 0);

  auto samples = SampleLights(scene_, P, N, 1, rng_);
  EXPECT_EQ(samples.size(), 1);

  auto samples2 = SampleLights(scene_, P, N, 5, rng_);
  EXPECT_EQ(samples2.size(), 5);
}

TEST_F(LightTest, SampleLights_PDF) {
  Eigen::Vector3f P(0, 0, 0);
  Eigen::Vector3f N(0, 1, 0);

  // Point light at (0, 10, 0). Dist=10. Dir=(0, 1, 0). N=(0, 1, 0)
  // Area light at (10, 0, 0). Dist=10. Dir=(1, 0, 0). N=(0, 1, 0) -> cos=0.
  // Area light should have 0 weight (cos_surface = 0).
  // So Point light should be sampled 100%.

  auto samples = SampleLights(scene_, P, N, 10, rng_);
  for (const auto& s : samples) {
    EXPECT_EQ(s.light->type, Light::Type::Point);
    EXPECT_NEAR(s.pdf, 1.0f, 1e-5f);
  }
}

TEST_F(LightTest, SampleLights_BothVisible) {
  Eigen::Vector3f P(0, 0, 0);
  Eigen::Vector3f N(0.7071f, 0.7071f, 0);  // 45 degrees

  // Both lights should be visible/weighted.
  auto samples = SampleLights(scene_, P, N, 100, rng_);

  int point_count = 0;
  int area_count = 0;
  for (const auto& s : samples) {
    if (s.light->type == Light::Type::Point)
      point_count++;
    else
      area_count++;
    EXPECT_GT(s.pdf, 0.0f);
  }

  EXPECT_GT(point_count, 0);
  EXPECT_GT(area_count, 0);
}

}  // namespace
}  // namespace sh_baker
