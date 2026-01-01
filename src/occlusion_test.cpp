#include "occlusion.h"

#include <gtest/gtest.h>

#include "scene.h"

namespace sh_baker {

class OcclusionTest : public ::testing::Test {
 protected:
  void SetUp() override { device = rtcNewDevice(nullptr); }

  void TearDown() override { rtcReleaseDevice(device); }

  RTCDevice device;
};

TEST_F(OcclusionTest, TriangleIntersection) {
  Scene scene;
  Geometry geo;

  // Triangle at Z=0, covering [-1, -1] to [1, 1] rough area
  geo.vertices = {
      {-1.0f, -1.0f, 0.0f}, {1.0f, -1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}};
  geo.indices = {0, 1, 2};

  scene.geometries.push_back(geo);

  RTCScene rtc_scene = BuildBVH(scene, device);
  ASSERT_NE(rtc_scene, nullptr);

  // Ray from front (Z=1) to back (Z=-1) -> Should Hit
  Ray ray_hit;
  ray_hit.origin = Eigen::Vector3f(0.0f, 0.0f, 1.0f);
  ray_hit.direction = Eigen::Vector3f(0.0f, 0.0f, -1.0f);

  auto hit = FindOcclusion(rtc_scene, ray_hit);
  ASSERT_TRUE(hit.has_value());
  // Hit should be roughly at 0,0,0
  EXPECT_NEAR(hit->position.z(), 0.0f, 1e-4f);

  // Check Normal (should be 0,0,1 because the triangle is flat on Z=0)
  // Wait, the triangle is {-1, -1, 0}, {1, -1, 0}, {0, 1, 0}.
  // Normal is indeed +Z (0, 0, 1) or -Z depending on winding.
  // 0->1 is (2, 0, 0). 0->2 is (1, 2, 0). Cross((2,0,0), (1,2,0)) = (0, 0, 4).
  // Normalized is (0,0,1).
  EXPECT_NEAR(hit->normal.z(), 1.0f, 1e-4f);

  // Ray from side, missing the triangle -> Should Miss
  Ray ray_miss;
  ray_miss.origin = Eigen::Vector3f(2.0f, 0.0f, 1.0f);
  ray_miss.direction = Eigen::Vector3f(0.0f, 0.0f, -1.0f);
  EXPECT_FALSE(FindOcclusion(rtc_scene, ray_miss));

  // Ray pointing away -> Should Miss
  Ray ray_away;
  ray_away.origin = Eigen::Vector3f(0.0f, 0.0f, 1.0f);
  ray_away.direction = Eigen::Vector3f(0.0f, 0.0f, 1.0f);
  EXPECT_FALSE(FindOcclusion(rtc_scene, ray_away));

  rtcReleaseScene(rtc_scene);
}

}  // namespace sh_baker
