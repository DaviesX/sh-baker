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
  EXPECT_TRUE(IsOccluded(rtc_scene, ray_hit));

  // Ray from side, missing the triangle -> Should Miss
  Ray ray_miss;
  ray_miss.origin = Eigen::Vector3f(2.0f, 0.0f, 1.0f);
  ray_miss.direction = Eigen::Vector3f(0.0f, 0.0f, -1.0f);
  EXPECT_FALSE(IsOccluded(rtc_scene, ray_miss));

  // Ray pointing away -> Should Miss
  Ray ray_away;
  ray_away.origin = Eigen::Vector3f(0.0f, 0.0f, 1.0f);
  ray_away.direction = Eigen::Vector3f(0.0f, 0.0f, 1.0f);
  EXPECT_FALSE(IsOccluded(rtc_scene, ray_away));

  rtcReleaseScene(rtc_scene);
}

}  // namespace sh_baker
