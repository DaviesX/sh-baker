#include "tracer.h"

#include <gtest/gtest.h>

#include "occlusion.h"
#include "scene.h"

namespace sh_baker {

class TracerTest : public ::testing::Test {
 protected:
  void SetUp() override { device = rtcNewDevice(nullptr); }

  void TearDown() override { rtcReleaseDevice(device); }

  RTCDevice device;
};

TEST_F(TracerTest, SampleHemisphereUniform) {
  std::mt19937 rng(42);
  for (int i = 0; i < 100; ++i) {
    Eigen::Vector3f v = SampleHemisphereUniform(rng);
    // Check normalization
    EXPECT_NEAR(v.norm(), 1.0f, 1e-4f);
    // Check hemisphere (z >= 0 assumption based on code, usually it's Z-up)
    // Code says: "return Eigen::Vector3f(r * std::cos(phi), r * std::sin(phi),
    // u1);" u1 is in [0, 1). So Z is positive.
    EXPECT_GE(v.z(), 0.0f);
  }
}

TEST_F(TracerTest, TraceSky) {
  Scene scene;
  // Empty scene
  RTCScene rtc_scene = BuildBVH(scene, device);

  bool sky_hit = false;
  auto on_sky = [&]() { sky_hit = true; };

  TraceConfig config(rtc_scene, scene, 1, 1, on_sky);

  std::mt19937 rng(42);
  Eigen::Vector3f origin(0, 0, 0);
  Eigen::Vector3f dir(0, 0, 1);

  Eigen::Vector3f result = Trace(config, origin, dir, 0, rng);

  EXPECT_EQ(result, Eigen::Vector3f::Zero());
  EXPECT_TRUE(sky_hit);

  rtcReleaseScene(rtc_scene);
}

TEST_F(TracerTest, TraceObjectNoLight) {
  Scene scene;
  Geometry geo;
  // Triangle at Z=2
  geo.vertices = {{-1, -1, 2}, {1, -1, 2}, {0, 1, 2}};
  geo.indices = {0, 1, 2};
  geo.material_id = 0;
  scene.geometries.push_back(geo);

  Material mat;
  mat.name = "default";
  scene.materials.push_back(mat);

  RTCScene rtc_scene = BuildBVH(scene, device);

  bool sky_hit = false;
  auto on_sky = [&]() { sky_hit = true; };

  TraceConfig config(rtc_scene, scene, 1, 1, on_sky);

  std::mt19937 rng(42);
  Eigen::Vector3f origin(0, 0, 0);
  Eigen::Vector3f dir(0, 0, 1);  // Towards Z=2

  // With no lights and default material (black?), it should return black.
  Eigen::Vector3f result = Trace(config, origin, dir, 0, rng);

  EXPECT_EQ(result, Eigen::Vector3f::Zero());
  EXPECT_FALSE(sky_hit);

  rtcReleaseScene(rtc_scene);
}

}  // namespace sh_baker
