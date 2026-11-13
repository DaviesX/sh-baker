#include "light.h"

#include <embree4/rtcore.h>
#include <gtest/gtest.h>

#include <cmath>  // For M_PI

#include "colorspace.h"
#include "light_tree.h"
#include "scene.h"

namespace sh_baker {
namespace {

class LightTest : public ::testing::Test {
 protected:
  void SetUp() override {
    device_ = rtcNewDevice(nullptr);

    // Dummy Material
    mat_.name = "default";
    mat_.albedo =
        Texture{.width = 1,
                .height = 1,
                .channels = 3,
                .pixel_data = std::vector<uint8_t>(3, LinearToSRGB(.8f))};
  }

  void TearDown() override { rtcReleaseDevice(device_); }

  RTCDevice device_;
  Material mat_;
  // Scene scene_;
  // LightTree light_tree_;
  std::mt19937 rng_{12345};
};

TEST_F(LightTest, EvaluatePointLight) {
  // Setup
  Light point;
  point.type = Light::Type::Point;
  point.position = Eigen::Vector3f(0, 10, 0);
  point.color = Eigen::Vector3f(1, 1, 1);
  point.intensity = 100.0f;
  point.flux = 1000.0f;

  std::vector<Light> lights{point};
  LightTree light_tree;
  light_tree.Build(lights);

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

  Eigen::Vector3f result = EvaluateLightSamples(&light_tree, rtc_scene, {}, P,
                                                N, wo, mat_, uv, 1, rng_);

  rtcReleaseScene(rtc_scene);

  // 1/PI approx 0.318. Albedo is 0.8 => 0.2546.
  // PBR Fresnel Loss reduces this to ~0.249.
  // Light tree sampling may introduce small variance.
  EXPECT_NEAR(result.x(), 0.249f, 0.02f);
}

TEST_F(LightTest, EvaluateAreaLight) {
  Material emissive_mat;
  emissive_mat.name = "emit";
  emissive_mat.emissive_strength = 10.0f;  // Important
  emissive_mat.emissive_factor = Eigen::Vector3f::Ones();

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

  Light area;
  area.type = Light::Type::Area;

  area.geometry = &area_geo;
  area.material = &emissive_mat;
  area.intensity = 10.0f;
  area.flux = 20.0f;
  area.area = 2.0f;

  std::vector<Light> lights{area};

  // Rebuild light tree for this test's specific scene
  LightTree test_light_tree;
  test_light_tree.Build(lights);

  Eigen::Vector3f P(0, 0, 0);
  Eigen::Vector3f N(0, 1, 0);   // Pointing UP towards Light
  Eigen::Vector3f wo(0, 1, 0);  // View direction UP
  Eigen::Vector2f uv(0, 0);

  RTCScene rtc_scene = rtcNewScene(device_);

  // Eval 100 samples
  Eigen::Vector3f result = EvaluateLightSamples(&test_light_tree, rtc_scene, {},
                                                P, N, wo, mat_, uv, 100, rng_);

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
  dir_light.flux = 1000.0f;

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
  spot.flux = 100.0f;
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
  Material emissive_mat;
  emissive_mat.name = "emit";
  emissive_mat.emissive_strength = 10.0f;  // Important
  emissive_mat.emissive_factor = Eigen::Vector3f::Ones();

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

  Light area;
  area.type = Light::Type::Area;

  area.geometry = &area_geo;
  area.material = &emissive_mat;
  area.intensity = 10.0f;
  area.flux = 20.0f;
  area.area = 2.0f;

  // Sample
  light_internal::AreaSample sample =
      light_internal::SampleAreaLight(area, rng_);

  // Verify Point is on plane (y=10)
  EXPECT_NEAR(sample.point.y(), 10.0f, 1e-4f);

  // Verify PDF
  // Area = 2.0. NumTriangles = 1.
  // PDF = 1/1 * 1/2 = 0.5.
  EXPECT_NEAR(sample.pdf, 0.5f, 1e-4f);

  // Verify Emission
  // Mat intensity = 10. Factor = 1.
  // Radiance = 10 * 1 = 10.
  EXPECT_NEAR(sample.radiance.x(), 10.0f, 1e-4f);
}

// A 1x1 RGBA albedo with the given alpha (channels 3 when `rgb` is set).
Material CardMaterial(uint8_t alpha, bool rgb = false) {
  Material mat;
  mat.albedo.width = 1;
  mat.albedo.height = 1;
  mat.albedo.channels = rgb ? 3 : 4;
  mat.albedo.pixel_data.assign(mat.albedo.channels, 128);
  if (!rgb) mat.albedo.pixel_data[3] = alpha;
  return mat;
}

// A horizontal quad covering [-3, 3]^2 in X/Z at height y.
Geometry HorizontalCard(float y, int material_id) {
  Geometry geo;
  geo.vertices = {
      {-3.0f, y, -3.0f}, {3.0f, y, -3.0f}, {3.0f, y, 3.0f}, {-3.0f, y, 3.0f}};
  geo.indices = {0, 2, 1, 0, 3, 2};
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

constexpr float kHalfPass = 1.0f - 128.0f / 255.0f;

// Direct light at the origin (normal up) from one light above, through the
// given cards, relative to the same light with no cards.
class LightTransmittanceTest : public LightTest {
 protected:
  void SetUp() override {
    LightTest::SetUp();
    // Material 0 is the receiver's; cards use the others.
    materials_ = {mat_, CardMaterial(128), CardMaterial(0), CardMaterial(255),
                  CardMaterial(0, /*rgb=*/true)};

    // An emissive triangle at y = 10 facing down, for the area-light case. It
    // is not in the BVH: shadow rays end just short of it (tfar).
    emitter_material_.name = "emit";
    emitter_material_.emissive_strength = 10.0f;
    emitter_material_.emissive_factor = Eigen::Vector3f::Ones();
    emitter_.vertices = {Eigen::Vector3f(-1, 10, 1), Eigen::Vector3f(1, 10, 1),
                         Eigen::Vector3f(0, 10, -1)};
    emitter_.normals.assign(3, Eigen::Vector3f(0, -1, 0));
    emitter_.indices = {0, 2, 1};
    emitter_.material_id = 0;
  }

  // Samples on the emitter are drawn from the same seeded generator with and
  // without cards, so the ratio is exact.
  Light AreaAbove() const {
    Light area;
    area.type = Light::Type::Area;
    area.geometry = &emitter_;
    area.material = &emitter_material_;
    area.intensity = 10.0f;
    area.flux = 20.0f;
    area.area = 2.0f;
    return area;
  }

  struct Result {
    Eigen::Vector3f evaluated;  // EvaluateLightSamples (bounce vertices).
    Eigen::Vector3f sh_l0;      // AccumulateIncomingLightSamples (bake points).
  };

  Result Direct(const Light& light, const std::vector<Geometry>& cards) {
    Scene scene;
    scene.geometries = cards;
    scene.materials = materials_;
    RTCScene rtc_scene = BuildBVH(scene, device_);
    std::vector<Light> lights{light};
    LightTree tree;
    tree.Build(lights);

    const Eigen::Vector3f P(0, 0, 0), N(0, 1, 0), wo(0, 1, 0);
    const Eigen::Vector2f uv(0, 0);
    Result r;
    std::mt19937 rng_a(7);
    r.evaluated = EvaluateLightSamples(&tree, rtc_scene, scene.materials, P, N,
                                       wo, mat_, uv, 4, rng_a);
    SHCoeffs sh;
    std::mt19937 rng_b(7);
    AccumulateIncomingLightSamples(&tree, rtc_scene, scene.materials, P, N, 4,
                                   rng_b, &sh);
    r.sh_l0 = sh.coeffs[0];
    rtcReleaseScene(rtc_scene);
    return r;
  }

  void ExpectRatio(const Light& light, const std::vector<Geometry>& cards,
                   float ratio) {
    Result open = Direct(light, {});
    Result covered = Direct(light, cards);
    ASSERT_GT(open.evaluated.x(), 0.0f);
    ASSERT_GT(open.sh_l0.x(), 0.0f);
    EXPECT_NEAR(covered.evaluated.x(), ratio * open.evaluated.x(),
                0.01f * open.evaluated.x());
    EXPECT_NEAR(covered.sh_l0.x(), ratio * open.sh_l0.x(),
                0.01f * open.sh_l0.x());
  }

  static Light PointAbove() {
    Light point;
    point.type = Light::Type::Point;
    point.position = Eigen::Vector3f(0, 10, 0);
    point.color = Eigen::Vector3f(1, 1, 1);
    point.intensity = 100.0f;
    point.flux = 1000.0f;
    return point;
  }

  static Light SpotAbove() {
    Light spot = PointAbove();
    spot.type = Light::Type::Spot;
    spot.direction = Eigen::Vector3f(0, -1, 0);
    return spot;
  }

  static Light SunAbove() {
    Light sun;
    sun.type = Light::Type::Directional;
    sun.direction = Eigen::Vector3f(0, -1, 0);
    sun.color = Eigen::Vector3f(1, 1, 1);
    sun.intensity = 2.0f;
    sun.flux = 1000.0f;
    return sun;
  }

  static constexpr int kHalf = 1, kCutOut = 2, kOpaque = 3, kRgb = 4;
  std::vector<Material> materials_;
  Geometry emitter_;
  Material emitter_material_;
};

TEST_F(LightTransmittanceTest, HalfCoveredCardAttenuatesDirectLight) {
  ExpectRatio(PointAbove(), {HorizontalCard(5.0f, kHalf)}, kHalfPass);
  ExpectRatio(SpotAbove(), {HorizontalCard(5.0f, kHalf)}, kHalfPass);
  ExpectRatio(SunAbove(), {HorizontalCard(5.0f, kHalf)}, kHalfPass);
  ExpectRatio(AreaAbove(), {HorizontalCard(5.0f, kHalf)}, kHalfPass);
}

TEST_F(LightTransmittanceTest, TwoCardsMultiply) {
  ExpectRatio(PointAbove(),
              {HorizontalCard(3.0f, kHalf), HorizontalCard(6.0f, kHalf)},
              kHalfPass * kHalfPass);
}

TEST_F(LightTransmittanceTest, CutOutCardPassesEverything) {
  ExpectRatio(PointAbove(), {HorizontalCard(5.0f, kCutOut)}, 1.0f);
  ExpectRatio(SunAbove(), {HorizontalCard(5.0f, kCutOut)}, 1.0f);
}

TEST_F(LightTransmittanceTest, OpaqueCardsBlock) {
  ExpectRatio(PointAbove(), {HorizontalCard(5.0f, kOpaque)}, 0.0f);
  ExpectRatio(PointAbove(), {HorizontalCard(5.0f, kRgb)}, 0.0f);
  ExpectRatio(PointAbove(),
              {HorizontalCard(7.0f, kCutOut),
               HullBox(Eigen::Vector3f(-1, 4, -1), Eigen::Vector3f(1, 6, 1))},
              0.0f);
}

TEST_F(LightTransmittanceTest, CardBeyondLightIgnored) {
  ExpectRatio(PointAbove(), {HorizontalCard(12.0f, kOpaque)}, 1.0f);
}

}  // namespace
}  // namespace sh_baker
