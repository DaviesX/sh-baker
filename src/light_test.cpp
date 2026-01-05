#include "light.h"

#include <embree4/rtcore.h>
#include <gtest/gtest.h>

#include <cmath>  // For M_PI

#include "loader.h"
#include "scene.h"

namespace sh_baker {
namespace {

class LightTest : public ::testing::Test {
 protected:
  void SetUp() override {
    device_ = rtcNewDevice(nullptr);

    // Setup a simple scene
    scene_.lights.clear();
    scene_.materials.clear();
    scene_.geometries.clear();

    // Dummy Material
    Material mat;
    mat.name = "default";
    scene_.materials.push_back(mat);

    // Add Point Light
    Light point;
    point.type = Light::Type::Point;
    point.position = Eigen::Vector3f(0, 10, 0);
    point.color = Eigen::Vector3f(1, 1, 1);
    point.intensity = 100.0f;
    scene_.lights.push_back(point);

    // Add Area Light
    // Need dummy geometry
    Geometry area_geo;
    area_geo.vertices = {Eigen::Vector3f(10, -1, 1), Eigen::Vector3f(10, 1, 0),
                         Eigen::Vector3f(10, -1, -1)};
    area_geo.normals = {Eigen::Vector3f(-1, 0, 0), Eigen::Vector3f(-1, 0, 0),
                        Eigen::Vector3f(-1, 0, 0)};
    area_geo.indices = {0, 1, 2};
    area_geo.material_id = 0;

    scene_.geometries.push_back(area_geo);

    Light area;
    area.type = Light::Type::Area;
    area.geometry_index = 0;  // First geometry
    area.intensity = 10.0f;
    area.area = 2.0f;
    // Pointers must be set manually as Loader does it
    // Wait, Loader sets pointers at end of LoadScene.
    // Here we construct manually. So we must set pointers.
    // Warning: scene_.geometries vector might reallocate if we push more.
    // But we only push once here.
    // However, scene_.lights copy 'area'. If 'area' has pointers to
    // &scene_.geometries[0], it's fine.
    scene_.lights.push_back(area);

    // Fix pointers for test
    scene_.lights[1].geometry = &scene_.geometries[0];
    scene_.lights[1].material = &scene_.materials[0];
  }

  void TearDown() override { rtcReleaseDevice(device_); }

  RTCDevice device_;
  Scene scene_;
  std::mt19937 rng_{12345};
};

TEST_F(LightTest, EvaluatePointLight) {
  // Setup
  Light point;
  point.type = Light::Type::Point;
  point.position = Eigen::Vector3f(0, 10, 0);
  point.color = Eigen::Vector3f(1, 1, 1);
  point.intensity = 100.0f;

  std::vector<Light> lights = {point};

  // Hit Point at Origin, Normal Up.
  Eigen::Vector3f P(0, 0, 0);
  Eigen::Vector3f N(0, 1, 0);
  Eigen::Vector3f wo(0, 1, 0);  // View direction
  Eigen::Vector2f uv(0, 0);

  // Empty BVH
  RTCScene rtc_scene = rtcNewScene(device_);

  // Eval
  // Material default (white albedo, rough).
  // EvalMaterial should return albedo/PI.
  // Radiance = I/d^2 * cos * BRDF
  // d=10, d^2=100. I=100. -> Irradiance = 1 * cos(0) = 1.
  // BRDF = 1/PI.
  // Result = 1/PI.

  Eigen::Vector3f result =
      EvaluateLightSamples(scene_.sky, lights, rtc_scene, P, N, wo,
                           scene_.materials[0], uv, 1, rng_);

  rtcReleaseScene(rtc_scene);

  // 1/PI approx 0.318. Albedo is 0.8 => 0.2546.
  // PBR Fresnel Loss reduces this to ~0.249.
  EXPECT_NEAR(result.x(), 0.249f, 0.01f);
}

TEST_F(LightTest, EvaluateAreaLight) {
  // Area Light Geometry
  Geometry area_geo;
  // A triangle at y=10, large enough to be easily hit?
  // Or just a small one.
  // Let's define a triangle directly above P.
  // Vertices: (-1, 10, 1), (1, 10, 1), (0, 10, -1).
  // Center roughly (0, 10, 0).
  // Normal pointing Down (0, -1, 0).
  area_geo.vertices = {Eigen::Vector3f(-1, 10, 1), Eigen::Vector3f(1, 10, 1),
                       Eigen::Vector3f(0, 10, -1)};
  area_geo.normals = {Eigen::Vector3f(0, -1, 0), Eigen::Vector3f(0, -1, 0),
                      Eigen::Vector3f(0, -1, 0)};
  // Indices
  area_geo.indices = {0, 1, 2};  // CCW?
  // V1-V0 = (2, 0, 0). V2-V0 = (1, 0, -2). Cross: (0, 4, 0). Up?
  // We want Normal Down.
  // Let's just trust normals provided.
  area_geo.material_id = 0;

  scene_.geometries.push_back(area_geo);  // Index 0.

  Light area;
  area.type = Light::Type::Area;
  area.geometry_index = 0;
  area.geometry = &scene_.geometries[0];
  area.material = &scene_.materials[0];  // Self-emission comes from material
  scene_.materials[0].emission_intensity = 10.0f;  // Enable emission
  area.intensity = 10.0f;
  area.area = 2.0f;

  std::vector<Light> lights = {area};

  Eigen::Vector3f P(0, 0, 0);
  Eigen::Vector3f N(1, 0, 0);  // Pointing towards Area Light at X=10
  Eigen::Vector3f wo(1, 0,
                     0);  // View direction along normal to avoid grazing angle
  Eigen::Vector2f uv(0, 0);

  RTCScene rtc_scene = rtcNewScene(device_);

  // Eval 100 samples to average
  Eigen::Vector3f result =
      EvaluateLightSamples(scene_.sky, lights, rtc_scene, P, N, wo,
                           scene_.materials[0], uv, 100, rng_);

  rtcReleaseScene(rtc_scene);

  // Check non-zero.
  // Result should be > 0.
  // Approx: 8 * 0.02 * 1 * 1 / PI = 0.05.
  EXPECT_GT(result.x(), 0.01f);
  EXPECT_LT(result.x(), 1.0f);
}

TEST_F(LightTest, DirectionalLightRadiance) {
  Light dir_light;
  dir_light.type = Light::Type::Directional;
  dir_light.direction = Eigen::Vector3f(0, -1, 0);  // Down
  dir_light.color = Eigen::Vector3f(1, 1, 1);
  dir_light.intensity = 2.0f;

  // Need to ensure non-zero return, check BRDF inputs.
  // DirectionalLightRadiance uses brdf(-dir).
  // -dir = (0, 1, 0).

  Eigen::Vector3f P(0, 0, 0);
  Eigen::Vector3f N(0, 1, 0);  // Up
  Ray visibility_ray;

  auto simple_brdf = [](const Eigen::Vector3f&) {
    return Eigen::Vector3f(1, 1, 1);
  };

  Eigen::Vector3f L_out = light_internal::DirectionalLightRadiance(
      dir_light, P, N, simple_brdf, &visibility_ray);

  // Expected: Intensity * Color * BRDF * cos(theta)
  // cos(theta) = dot(N, -dir) = dot((0,1,0), (0,1,0)) = 1
  // 2.0 * (1,1,1) * (1,1,1) * 1 = (2,2,2)
  EXPECT_NEAR(L_out.x(), 2.0f, 1e-5f);

  // Check Ray
  EXPECT_TRUE(visibility_ray.direction.isApprox(-dir_light.direction));
  EXPECT_FLOAT_EQ(visibility_ray.tfar, 1.0e10f);
}

TEST_F(LightTest, SpotLightRadiance_Falloff) {
  Light spot;
  spot.type = Light::Type::Spot;
  spot.position = Eigen::Vector3f(0, 10, 0);
  spot.direction = Eigen::Vector3f(0, -1, 0);
  spot.color = Eigen::Vector3f(1, 1, 1);
  spot.intensity = 100.0f;
  // Inner: 45 deg (cos ~0.707), Outer: 60 deg (cos 0.5)
  spot.cos_inner_cone = 0.70710678f;
  spot.cos_outer_cone = 0.5f;

  Eigen::Vector3f P(0, 0, 0);
  Eigen::Vector3f N(0, 1, 0);
  Ray visibility_ray;
  auto simple_brdf = [](const Eigen::Vector3f&) {
    return Eigen::Vector3f(1, 1, 1);
  };

  // 1. Directly below (Angle 0, cos 1). Inside Inner. Falloff = 1.
  {
    Eigen::Vector3f L_out = light_internal::SpotLightRadiance(
        spot, P, N, simple_brdf, &visibility_ray);
    // 100 / 100 * 1 * 1 * 1 = 1.0
    EXPECT_NEAR(L_out.x(), 1.0f, 1e-5f);
  }

  // 2. Angle 50 deg. Cos(50) ~= 0.642.
  // Between Inner (0.707) and Outer (0.5).
  // Falloff = (cos_l - outer) / (inner - outer)
  // (0.642 - 0.5) / (0.707 - 0.5) = 0.142 / 0.207 ~= 0.685
  {
    float angle_deg = 50.0f;
    float angle_rad = angle_deg * M_PI / 180.0f;
    // Move P to create this angle wrt (0,10,0) looking down.
    // L vector is (P - LightPos).Normalized.
    // We want angle between L and SpotDir(0,-1,0) to be 50 deg.
    // So L should be (sin(50), -cos(50), 0).
    // P = LightPos + L * dist. Let dist = 10.
    float dist = 10.0f;
    Eigen::Vector3f L_dir(std::sin(angle_rad), -std::cos(angle_rad), 0);
    Eigen::Vector3f P_side = spot.position + L_dir * dist;

    // N should point at light for max cosine response at surface
    Eigen::Vector3f N_side = -L_dir;

    Eigen::Vector3f L_out = light_internal::SpotLightRadiance(
        spot, P_side, N_side, simple_brdf, &visibility_ray);

    // Expected Falloff
    float cos_l = std::cos(angle_rad);  // 0.6427
    float dist_sq = dist * dist;        // 100
    float expected_falloff = (cos_l - spot.cos_outer_cone) /
                             (spot.cos_inner_cone - spot.cos_outer_cone);
    float expected_val = (spot.intensity * expected_falloff * 1.0f /
                          dist_sq);  // * BRDF(1) * cos_n(1)

    EXPECT_NEAR(L_out.x(), expected_val, 1e-4f);
    EXPECT_GT(L_out.x(), 0.0f);
    EXPECT_LT(L_out.x(), 1.0f);
  }

  // 3. Outside Outer Cone (e.g. 70 deg). Cos(70) ~= 0.342 < 0.5.
  {
    float angle_rad = 70.0f * M_PI / 180.0f;
    Eigen::Vector3f L_dir(std::sin(angle_rad), -std::cos(angle_rad), 0);
    Eigen::Vector3f P_side = spot.position + L_dir * 10.0f;
    Eigen::Vector3f N_side = -L_dir;

    Eigen::Vector3f L_out = light_internal::SpotLightRadiance(
        spot, P_side, N_side, simple_brdf, &visibility_ray);

    EXPECT_NEAR(L_out.x(), 0.0f, 1e-5f);
  }
}

TEST_F(LightTest, SampleAreaLight_Internal) {
  ASSERT_GE(scene_.lights.size(), 2);
  // Use the Area Light from SetUp (index 1)
  const Light& area = scene_.lights[1];
  ASSERT_NE(area.geometry, nullptr);
  ASSERT_NE(area.material, nullptr);

  // Set Emission
  scene_.materials[0].emission_intensity = 10.0f;

  // Sample
  light_internal::AreaSample sample =
      light_internal::SampleAreaLight(area, rng_);

  // Verify Point is on plane (x=10)
  EXPECT_NEAR(sample.point.x(), 10.0f, 1e-4f);
  // Verify Normal (-1, 0, 0)
  EXPECT_NEAR(sample.normal.x(), -1.0f, 1e-4f);

  // Verify PDF
  // Area = 2.0. NumTriangles = 1.
  // PDF = 1/1 * 1/2 = 0.5.
  EXPECT_NEAR(sample.pdf, 0.5f, 1e-4f);

  // Verify Emission
  // Mat intensity = 10. Albedo default grey (0.8).
  // Radiance = 10 * 0.8 = 8.
  EXPECT_NEAR(sample.radiance.x(), 8.0f, 1e-4f);
}
}  // namespace
}  // namespace sh_baker
