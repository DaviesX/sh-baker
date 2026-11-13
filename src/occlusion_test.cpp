#include "occlusion.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "material.h"
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

TEST_F(OcclusionTest, ReportsMaterialLessOccluder) {
  Scene scene;

  // A normal (materialed) triangle at Z=0.
  Geometry lit;
  lit.vertices = {
      {-1.0f, -1.0f, 0.0f}, {1.0f, -1.0f, 0.0f}, {0.0f, 1.0f, 0.0f}};
  lit.indices = {0, 1, 2};
  lit.material_id = 0;
  scene.geometries.push_back(lit);

  // A material-less pure-occluder triangle at Z=-1 (behind the lit one).
  Geometry shell;
  shell.vertices = {
      {-1.0f, -1.0f, -1.0f}, {1.0f, -1.0f, -1.0f}, {0.0f, 1.0f, -1.0f}};
  shell.indices = {0, 1, 2};
  shell.material_id = -1;
  scene.geometries.push_back(shell);

  RTCScene rtc_scene = BuildBVH(scene, device);
  ASSERT_NE(rtc_scene, nullptr);

  // Hitting the front (materialed) triangle: a real surface.
  Ray front_ray;
  front_ray.origin = Eigen::Vector3f(0.0f, 0.0f, 1.0f);
  front_ray.direction = Eigen::Vector3f(0.0f, 0.0f, -1.0f);
  auto front_hit = FindOcclusion(rtc_scene, front_ray);
  ASSERT_TRUE(front_hit.has_value());
  EXPECT_NEAR(front_hit->position.z(), 0.0f, 1e-4f);
  EXPECT_GE(front_hit->material_id, 0);

  // Hitting the shell directly (from below, missing the lit triangle's front):
  // reported material-less so the tracer blocks transport without shading.
  Ray shell_ray;
  shell_ray.origin = Eigen::Vector3f(0.0f, 0.0f, -5.0f);
  shell_ray.direction = Eigen::Vector3f(0.0f, 0.0f, 1.0f);
  auto shell_hit = FindOcclusion(rtc_scene, shell_ray);
  ASSERT_TRUE(shell_hit.has_value());
  EXPECT_NEAR(shell_hit->position.z(), -1.0f, 1e-4f);
  EXPECT_LT(shell_hit->material_id, 0);

  rtcReleaseScene(rtc_scene);
}

namespace {

// A 1x1 albedo texture. channels 4 carries `alpha` (0..255); channels 3 has
// no alpha channel.
Material SolidAlbedo(int channels, uint8_t alpha) {
  Material mat;
  mat.albedo.width = 1;
  mat.albedo.height = 1;
  mat.albedo.channels = channels;
  mat.albedo.pixel_data.assign(channels, 200);
  if (channels == 4) mat.albedo.pixel_data[3] = alpha;
  return mat;
}

// An axis-aligned quad covering [-2, 2]^2 in the plane Z = z.
Geometry QuadAtZ(float z, int material_id) {
  Geometry geo;
  geo.vertices = {
      {-2.0f, -2.0f, z}, {2.0f, -2.0f, z}, {2.0f, 2.0f, z}, {-2.0f, 2.0f, z}};
  geo.indices = {0, 1, 2, 0, 2, 3};
  geo.material_id = material_id;
  return geo;
}

// A closed, material-less box [lo, hi]: a pure occluder, as q3map2 passes a
// solid brush hull. Embree hits both sides, so winding does not matter here.
Geometry HullBox(const Eigen::Vector3f& lo, const Eigen::Vector3f& hi) {
  Geometry geo;
  for (int i = 0; i < 8; ++i) {
    geo.vertices.push_back(Eigen::Vector3f((i & 1) ? hi.x() : lo.x(),
                                           (i & 2) ? hi.y() : lo.y(),
                                           (i & 4) ? hi.z() : lo.z()));
  }
  // Two triangles per face; corner index bits are (x, y, z).
  geo.indices = {0, 2, 1, 1, 2, 3, 4, 5, 6, 5, 7, 6, 0, 1, 4, 1, 5, 4,
                 2, 6, 3, 3, 6, 7, 0, 4, 2, 2, 4, 6, 1, 3, 5, 3, 7, 5};
  geo.material_id = -1;
  return geo;
}

// A ray from (0, 0, 10) straight down to Z = 0.
Ray DownRay() {
  Ray ray;
  ray.origin = Eigen::Vector3f(0.0f, 0.0f, 10.0f);
  ray.direction = Eigen::Vector3f(0.0f, 0.0f, -1.0f);
  ray.tnear = 0.005f;
  ray.tfar = 10.0f - 0.005f;
  return ray;
}

constexpr float kHalf = 1.0f - 128.0f / 255.0f;

}  // namespace

class TransmittanceTest : public OcclusionTest {
 protected:
  void SetUp() override {
    OcclusionTest::SetUp();
    materials_ = {SolidAlbedo(4, 128), SolidAlbedo(4, 0), SolidAlbedo(4, 255),
                  SolidAlbedo(3, 0)};
  }

  float Transmittance(const std::vector<Geometry>& geometries, const Ray& ray) {
    Scene scene;
    scene.geometries = geometries;
    scene.materials = materials_;
    RTCScene rtc_scene = BuildBVH(scene, device);
    EXPECT_NE(rtc_scene, nullptr);
    float t = FindTransmittance(rtc_scene, scene.materials, ray);
    rtcReleaseScene(rtc_scene);
    return t;
  }

  static constexpr int kHalfAlpha = 0;
  static constexpr int kCutOut = 1;
  static constexpr int kOpaqueRgba = 2;
  static constexpr int kRgb = 3;
  std::vector<Material> materials_;
};

TEST_F(OcclusionTest, ReportsHitDistance) {
  Scene scene;
  scene.geometries.push_back(QuadAtZ(3.0f, -1));
  RTCScene rtc_scene = BuildBVH(scene, device);
  ASSERT_NE(rtc_scene, nullptr);
  auto hit = FindOcclusion(rtc_scene, DownRay());
  ASSERT_TRUE(hit.has_value());
  EXPECT_NEAR(hit->t, 7.0f, 1e-4f);
  rtcReleaseScene(rtc_scene);
}

TEST_F(TransmittanceTest, MissIsFullyVisible) {
  EXPECT_EQ(Transmittance({}, DownRay()), 1.0f);
  EXPECT_EQ(Transmittance({QuadAtZ(-1.0f, kOpaqueRgba)}, DownRay()), 1.0f);
}

TEST_F(TransmittanceTest, HalfCoveredCard) {
  EXPECT_NEAR(Transmittance({QuadAtZ(5.0f, kHalfAlpha)}, DownRay()), kHalf,
              1e-6f);
}

TEST_F(TransmittanceTest, TwoCardsMultiply) {
  EXPECT_NEAR(
      Transmittance({QuadAtZ(3.0f, kHalfAlpha), QuadAtZ(6.0f, kHalfAlpha)},
                    DownRay()),
      kHalf * kHalf, 1e-6f);
}

TEST_F(TransmittanceTest, CutOutCardPassesEverything) {
  EXPECT_EQ(Transmittance({QuadAtZ(5.0f, kCutOut)}, DownRay()), 1.0f);
}

TEST_F(TransmittanceTest, OpaqueCardsBlock) {
  EXPECT_EQ(Transmittance({QuadAtZ(5.0f, kOpaqueRgba)}, DownRay()), 0.0f);
  EXPECT_EQ(Transmittance({QuadAtZ(5.0f, kRgb)}, DownRay()), 0.0f);
  // An opaque card behind translucent ones still blocks.
  EXPECT_EQ(Transmittance({QuadAtZ(7.0f, kHalfAlpha), QuadAtZ(6.0f, kCutOut),
                           QuadAtZ(2.0f, kOpaqueRgba)},
                          DownRay()),
            0.0f);
}

TEST_F(TransmittanceTest, MaterialLessOccluderBehindCutOutBlocks) {
  EXPECT_EQ(
      Transmittance({QuadAtZ(7.0f, kCutOut), HullBox(Eigen::Vector3f(-1, -1, 2),
                                                     Eigen::Vector3f(1, 1, 4))},
                    DownRay()),
      0.0f);
}

TEST_F(TransmittanceTest, IgnoresSurfacesBeyondTfar) {
  Ray ray = DownRay();
  ray.tfar = 4.0f;  // Ends at Z = 6.
  EXPECT_EQ(Transmittance({QuadAtZ(5.0f, kOpaqueRgba)}, ray), 1.0f);
  EXPECT_NEAR(Transmittance(
                  {QuadAtZ(8.0f, kHalfAlpha), QuadAtZ(5.0f, kOpaqueRgba)}, ray),
              kHalf, 1e-6f);
}

TEST_F(TransmittanceTest, ManyCrossingsCountAsBlocked) {
  std::vector<Geometry> cards;
  for (int i = 0; i < 100; ++i) {
    cards.push_back(QuadAtZ(0.5f + 0.09f * i, kCutOut));
  }
  EXPECT_EQ(Transmittance(cards, DownRay()), 0.0f);

  // At the limit the walk still passes.
  cards.resize(kMaxTransmittanceCrossings);
  EXPECT_EQ(Transmittance(cards, DownRay()), 1.0f);
}

}  // namespace sh_baker
