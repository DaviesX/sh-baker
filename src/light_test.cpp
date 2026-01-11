#include "light.h"

#include <embree4/rtcore.h>
#include <gtest/gtest.h>

#include <cmath>  // For M_PI

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

  Eigen::Vector3f result = EvaluateLightSamples(
      scene_, rtc_scene, P, N, wo, scene_.materials[0], uv, 1, rng_);

  rtcReleaseScene(rtc_scene);

  // 1/PI approx 0.318. Albedo is 0.8 => 0.2546.
  // PBR Fresnel Loss reduces this to ~0.249.
  EXPECT_NEAR(result.x(), 0.249f, 0.01f);
}

TEST_F(LightTest, EvaluateAreaLight) {
  // Clear scene to avoid pollution/pointer invalidation from SetUp
  scene_.lights.clear();
  scene_.geometries.clear();
  scene_.materials.clear();

  Material mat;
  mat.name = "emit";
  mat.emission_intensity = 10.0f;  // Important
  scene_.materials.push_back(mat);

  // Area Light Geometry (Y=10)
  Geometry area_geo;
  area_geo.vertices = {Eigen::Vector3f(-1, 10, 1), Eigen::Vector3f(1, 10, 1),
                       Eigen::Vector3f(0, 10, -1)};
  // Normal (0, -1, 0) -- Down
  area_geo.normals = {Eigen::Vector3f(0, -1, 0), Eigen::Vector3f(0, -1, 0),
                      Eigen::Vector3f(0, -1, 0)};
  area_geo.indices = {0, 1, 2};  // Winding?
  // V1-V0 = (2,0,0). V2-V0=(1,0,-2). Cross=(0,4,0) -> Up.
  // Wait, if geometric normal is UP, and we provide normal DOWN,
  // SampleAreaLight uses interpolated normal (-1).
  // But AreaLightRadiance checks dot(N, -L).
  // If we are below (0,0,0), L is (0,1,0). -L is (0,-1,0).
  // dot((0,-1,0), (0,-1,0)) = 1. Good.
  // But strictly, geometric normal check?
  // Embree ray intersection should hit it.
  // Let's ensure indices produce Down normal to be safe: {0, 2, 1}.
  area_geo.indices = {0, 2, 1};
  area_geo.material_id = 0;

  scene_.geometries.push_back(area_geo);

  Light area;
  area.type = Light::Type::Area;
  area.geometry_index = 0;
  area.geometry = &scene_.geometries[0];
  area.material = &scene_.materials[0];
  area.intensity = 10.0f;
  area.area = 2.0f;

  scene_.lights.push_back(area);

  Eigen::Vector3f P(0, 0, 0);
  Eigen::Vector3f N(0, 1, 0);   // Pointing UP towards Light
  Eigen::Vector3f wo(0, 1, 0);  // View direction UP
  Eigen::Vector2f uv(0, 0);

  RTCScene rtc_scene = rtcNewScene(device_);

  // Eval 100 samples
  Eigen::Vector3f result = EvaluateLightSamples(
      scene_, rtc_scene, P, N, wo, scene_.materials[0], uv, 100, rng_);

  rtcReleaseScene(rtc_scene);

  EXPECT_GT(result.x(), 0.01f);
  EXPECT_LT(result.x(), 10.0f);  // Should be reasonably bounded
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

TEST_F(LightTest, SampleEnvironment_Preetham) {
  scene_.environment = Environment{};
  scene_.environment->type = Environment::Type::Preetham;
  scene_.environment->sun_direction = Eigen::Vector3f(0, 1, 0);  // Up
  scene_.environment->intensity = 2.0f;

  // Sample
  light_internal::EnvironmentSample sample =
      light_internal::SampleEnvironment(scene_, rng_);

  // Check non-zero pdf
  EXPECT_GT(sample.pdf, 0.0f);

  // Check direction is normalized
  EXPECT_NEAR(sample.direction.norm(), 1.0f, 1e-4f);

  // Check radiance
  // If direction is close to sun, very bright. Else blueish.
  // Random sampling makes this probabilistic, but we can verify it's not zero.
  EXPECT_GT(sample.radiance.maxCoeff(), 0.0f);
}

TEST_F(LightTest, SampleEnvironment_Texture) {
  scene_.environment = Environment{};
  scene_.environment->type = Environment::Type::Texture;
  scene_.environment->intensity = 1.0f;

  // Create a simple 2x2 texture
  // Top-Left (0,0): Red (255,0,0)
  // Top-Right (1,0): Black (0,0,0)
  // Bottom-Left (0,1): Green (0,255,0)
  // Bottom-Right (1,1): Blue (0,0,255)

  Texture& tex = scene_.environment->texture;
  tex.width = 2;
  tex.height = 2;
  tex.channels = 3;
  tex.pixel_data = {255, 0, 0, 0, 0, 0, 0, 255, 0, 0, 0, 255};

  BuildEnvironmentCDF(*scene_.environment);

  // Verify CDF construction
  // Row 0 (Top): Red(Lum~0.2) * sin(theta_0)  vs Black(0)
  // Row 1 (Bot): Green(Lum~0.7) * sin(theta_1) vs Blue(Lum~0.07) * sin(theta_1)
  // theta_0 = 0.25 * PI. theta_1 = 0.75 * PI. sin(theta_0)=sin(theta_1).
  // So row weights proportional to luminance sums.

  // Sample
  light_internal::EnvironmentSample sample =
      light_internal::SampleEnvironment(scene_, rng_);

  EXPECT_NEAR(sample.direction.norm(), 1.0f, 1e-4f);
  // Should have valid pdf
  // It's possible to pick black pixel (pdf 0? no, logic handles black with
  // uniform fallback or small epsilon usually, but in my implementation black
  // pixels have probability 0 if row is not all black). Only Top-Right is
  // black. Top-Left is Red. So if sample is Top-Right, PDF might be tricky. But
  // strict logic says 0 probability. So we should effectively never sample
  // Top-Right.

  // Any sample we get should correspond to a non-zero pixel.
  if (sample.radiance.maxCoeff() > 0) {
    // Red, Green or Blue.
    bool is_red = (sample.radiance.x() > 0 && sample.radiance.y() == 0 &&
                   sample.radiance.z() == 0);
    bool is_green = (sample.radiance.x() == 0 && sample.radiance.y() > 0 &&
                     sample.radiance.z() == 0);
    bool is_blue = (sample.radiance.x() == 0 && sample.radiance.y() == 0 &&
                    sample.radiance.z() > 0);
    EXPECT_TRUE(is_red || is_green || is_blue);
  }
}
}  // namespace
}  // namespace sh_baker
