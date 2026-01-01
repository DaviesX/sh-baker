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

TEST_F(OcclusionTest, RayClipping) {
  Scene scene;
  Geometry geo;
  // Triangle at Z=0
  geo.vertices = {
      {-1.0f, -1.0f, 0.0f}, {1.0f, -1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}};
  geo.indices = {0, 1, 2};
  scene.geometries.push_back(geo);

  RTCScene rtc_scene = BuildBVH(scene, device);
  ASSERT_NE(rtc_scene, nullptr);

  // Ray aiming at triangle (0,0,0) from (0,0,1)
  // Distance is 1.0.

  // 1. Tfar clipping: Ray is too short
  {
    Ray ray;
    ray.origin = Eigen::Vector3f(0.0f, 0.0f, 1.0f);
    ray.direction = Eigen::Vector3f(0.0f, 0.0f, -1.0f);
    ray.tfar = 0.5f;  // Stops before Z=0
    EXPECT_FALSE(FindOcclusion(rtc_scene, ray));
  }

  // 2. Tnear clipping: Ray starts "after" the triangle
  {
    Ray ray;
    ray.origin = Eigen::Vector3f(0.0f, 0.0f, 1.0f);
    ray.direction = Eigen::Vector3f(0.0f, 0.0f, -1.0f);
    ray.tnear =
        1.2f;  // Starts at Z=-0.2 (past 0) - wait, distance is 1.0.
               // Origin Z=1, Dir -Z.
               // t=1 => Z=0. t=1.2 => Z=-0.2.
               // If tnear=1.2, we only check intersections AFTER Z=-0.2.
               // The intersection is at t=1.0. So 1.0 < 1.2, should miss.
    EXPECT_FALSE(FindOcclusion(rtc_scene, ray));
  }

  // 3. Valid hit within range
  {
    Ray ray;
    ray.origin = Eigen::Vector3f(0.0f, 0.0f, 1.0f);
    ray.direction = Eigen::Vector3f(0.0f, 0.0f, -1.0f);
    ray.tnear = 0.1f;
    ray.tfar = 10.0f;
    auto hit = FindOcclusion(rtc_scene, ray);
    ASSERT_TRUE(hit.has_value());
    EXPECT_NEAR(hit->position.z(), 0.0f, 1e-4f);
  }

  rtcReleaseScene(rtc_scene);
}

TEST_F(OcclusionTest, MultiGeometryMaterialTest) {
  Scene scene;

  // Geometry 1: Plane at X = -2. Material ID = 10.
  // 2 Triangles.
  {
    Geometry geo;
    geo.material_id = 10;
    geo.vertices = {
        {-3.0f, -1.0f, 0.0f},
        {-1.0f, -1.0f, 0.0f},  // Bottom-Left, Bottom-Right
        {-3.0f, 1.0f, 0.0f},
        {-1.0f, 1.0f, 0.0f}  // Top-Left, Top-Right
    };
    // Tri 1: 0, 1, 2. Tri 2: 1, 3, 2.
    geo.indices = {0, 1, 2, 1, 3, 2};
    scene.geometries.push_back(geo);
  }

  // Geometry 2: Plane at X = 2. Material ID = 20.
  // 2 Triangles.
  // We want to hit the second triangle.
  // Vertices range x in [1, 3], y in [-1, 1].
  {
    Geometry geo;
    geo.material_id = 20;
    geo.vertices = {
        {1.0f, -1.0f, 0.0f},
        {3.0f, -1.0f, 0.0f},  // 0, 1
        {1.0f, 1.0f, 0.0f},
        {3.0f, 1.0f, 0.0f}  // 2, 3
    };
    // Tri 1: 0, 1, 2 (Bottom-Left triangle).
    // Tri 2: 1, 3, 2 (Top-Right triangle).
    geo.indices = {0, 1, 2, 1, 3, 2};
    scene.geometries.push_back(geo);
  }

  RTCScene rtc_scene = BuildBVH(scene, device);
  ASSERT_NE(rtc_scene, nullptr);

  // Target: Geometry 2, Triangle 2.
  // Triangle 2 indices are 1(3,-1), 3(3,1), 2(1,1).
  // Centroid roughly at x=2.33, y=0.33.
  // Let's aim at (2.5, 0.5, 0).
  // Point (2.5, 0.5) is inside the box [1,3]x[-1,1].
  // Check if it's in the top-right triangle.
  // Diagonal is from (1,1) to (3,-1)? No, indices 1(3,-1) to 2(1,1).
  // The diagonal splits the quad.
  // Let's just aim clearly at the top right corner area: (2.8, 0.8).

  Ray ray;
  ray.origin = Eigen::Vector3f(2.8f, 0.8f, 5.0f);
  ray.direction = Eigen::Vector3f(0.0f, 0.0f, -1.0f);

  auto hit = FindOcclusion(rtc_scene, ray);
  ASSERT_TRUE(hit.has_value());

  // Verify Material ID
  EXPECT_EQ(hit->material_id, 20);

  // Verify Position (approx)
  EXPECT_NEAR(hit->position.x(), 2.8f, 1e-4f);
  EXPECT_NEAR(hit->position.y(), 0.8f, 1e-4f);
  EXPECT_NEAR(hit->position.z(), 0.0f, 1e-4f);

  rtcReleaseScene(rtc_scene);
}

}  // namespace sh_baker
