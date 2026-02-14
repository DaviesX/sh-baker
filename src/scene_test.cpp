#include "scene.h"

#include <embree4/rtcore.h>
#include <gtest/gtest.h>

#include <cmath>

namespace sh_baker {

class SceneTest : public ::testing::Test {
 protected:
  void SetUp() override { device = rtcNewDevice(nullptr); }

  void TearDown() override { rtcReleaseDevice(device); }

  RTCDevice device;
};

TEST_F(SceneTest, BuildBVHWithTriangle) {
  Scene scene;
  Geometry geo;

  // Single triangle
  geo.vertices = {{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}};
  geo.indices = {0, 1, 2};

  scene.geometries.push_back(geo);

  RTCScene rtc_scene = BuildBVH(scene, device);
  ASSERT_NE(rtc_scene, nullptr);

  rtcReleaseScene(rtc_scene);
}

TEST_F(SceneTest, BuildEmptyScene) {
  Scene scene;
  RTCScene rtc_scene = BuildBVH(scene, device);
  ASSERT_NE(rtc_scene, nullptr);
  rtcReleaseScene(rtc_scene);
}

TEST_F(SceneTest, TransformedVertices) {
  Geometry geo;
  geo.vertices = {{1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}};

  // Translate by (1, 2, 3)
  geo.transform = Eigen::Affine3f::Identity();
  geo.transform.translate(Eigen::Vector3f(1.0f, 2.0f, 3.0f));

  auto transformed = TransformedVertices(geo);
  ASSERT_EQ(transformed.size(), 2);

  // (1,0,0) + (1,2,3) = (2,2,3)
  EXPECT_TRUE(transformed[0].isApprox(Eigen::Vector3f(2.0f, 2.0f, 3.0f)));
  // (0,1,0) + (1,2,3) = (1,3,3)
  EXPECT_TRUE(transformed[1].isApprox(Eigen::Vector3f(1.0f, 3.0f, 3.0f)));
}

TEST_F(SceneTest, TransformedNormals) {
  Geometry geo;
  geo.normals = {{1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}};

  // Rotate 90 degrees around Z axis.
  // X axis (1,0,0) becomes Y axis (0,1,0)
  // Y axis (0,1,0) becomes -X axis (-1,0,0)
  geo.transform = Eigen::Affine3f::Identity();
  geo.transform.rotate(
      Eigen::AngleAxisf(M_PI / 2.0f, Eigen::Vector3f::UnitZ()));

  auto transformed = TransformedNormals(geo);
  ASSERT_EQ(transformed.size(), 2);

  EXPECT_TRUE(transformed[0].isApprox(Eigen::Vector3f(0.0f, 1.0f, 0.0f)));
  EXPECT_TRUE(transformed[1].isApprox(Eigen::Vector3f(-1.0f, 0.0f, 0.0f)));
}

TEST_F(SceneTest, TransformedTangents) {
  Geometry geo;
  // Tangent pointing X, with sign 1.0
  geo.tangents = {Eigen::Vector4f(1.0f, 0.0f, 0.0f, 1.0f)};

  // Rotate 90 degrees around Z axis.
  geo.transform = Eigen::Affine3f::Identity();
  geo.transform.rotate(
      Eigen::AngleAxisf(M_PI / 2.0f, Eigen::Vector3f::UnitZ()));

  auto transformed = TransformedTangents(geo);
  ASSERT_EQ(transformed.size(), 1);

  // Should rotate to Y (0,1,0), sign preserved
  EXPECT_TRUE(
      transformed[0].head<3>().isApprox(Eigen::Vector3f(0.0f, 1.0f, 0.0f)));
  EXPECT_EQ(transformed[0].w(), 1.0f);
}

TEST_F(SceneTest, ProjectEnvironmentUniformWhite) {
  // Create a 64x32 uniform white texture
  Texture32F tex;
  tex.width = 64;
  tex.height = 32;
  tex.channels = 3;
  // White (255, 255, 255)
  tex.pixel_data.resize(tex.width * tex.height * 3, 1.0f);

  Environment env;
  env.type = Environment::Type::Texture;
  env.texture = tex;

  SHCoeffs coeffs = ProjectEnvironmentToSH(env);

  // Expected L0: Integral(1 * Y00) = Y00 * 4pi
  // Y00 = 0.282095
  // 4pi = 12.56637
  // L0 = 3.5449
  // However, discrete sampling error exists.
  float expected_L0 = 0.282095f * 4.0f * M_PI;

  EXPECT_NEAR(coeffs.coeffs[0].x(), expected_L0, 0.1f);
  EXPECT_NEAR(coeffs.coeffs[0].y(), expected_L0, 0.1f);
  EXPECT_NEAR(coeffs.coeffs[0].z(), expected_L0, 0.1f);

  // Other bands should be near zero
  for (int i = 1; i < 9; ++i) {
    EXPECT_NEAR(coeffs.coeffs[i].x(), 0.0f, 0.1f) << "Band " << i;
  }
}

TEST_F(SceneTest, ComputeAndSampleEmissionCDF) {
  // Create a 2x2 texture
  // (0,0): 10, (1,0): 50
  // (0,1): 100, (1,1): 250
  // Total Sum = 100.
  // CDF should reflect this distribution.
  Texture tex;
  tex.width = 2;
  tex.height = 2;
  tex.channels = 3;
  // Use linear values directly? No, ComputeTexture uses SRGBToLinear.
  // Let's use float inputs via pixel_data being int, but we can set specific
  // pattern. SRGBToLinear(x) ~ x^2.2.
  // To get cleaner numbers, let's use values that SRGBToLinear maps predictably
  // or just accept approximations.
  // Or: mock the internal logic by knowing 255 -> 1.0.
  // Let's just set some values and verify relative probability.
  // 10 -> small.
  // 255 -> 1.0.
  // Let's use 0, 50, 100, 200.
  tex.pixel_data.resize(2 * 2 * 3);
  auto set_pixel = [&](int x, int y, uint8_t val) {
    int idx = (y * 2 + x) * 3;
    tex.pixel_data[idx + 0] = val;
    tex.pixel_data[idx + 1] = val;
    tex.pixel_data[idx + 2] = val;
  };
  set_pixel(0, 0, 10);
  set_pixel(1, 0, 50);
  set_pixel(0, 1, 100);
  set_pixel(1, 1, 250);

  // Identity Jacobian
  Texture32F jacobian;
  jacobian.width = 2;
  jacobian.height = 2;
  jacobian.channels = 1;
  jacobian.pixel_data = {1.0f, 1.0f, 1.0f, 1.0f};

  auto cdf_opt = ComputeTextureEmissionCDF(tex, jacobian);
  ASSERT_TRUE(cdf_opt.has_value());
  const EmissionCDF& cdf = *cdf_opt;

  // Verify structure
  // Expect size H+1 and W+1
  EXPECT_EQ(cdf.marginal_cdf.size(), 3);
  EXPECT_EQ(cdf.conditional_cdf.size(), 3);
  EXPECT_EQ(cdf.conditional_cdf[0].size(), 3);
  EXPECT_EQ(cdf.conditional_cdf[1].size(), 3);

  // Check monotonicity
  EXPECT_FLOAT_EQ(cdf.marginal_cdf[0], 0.0f);
  EXPECT_LT(cdf.marginal_cdf[1], cdf.marginal_cdf[2]);
  EXPECT_FLOAT_EQ(cdf.marginal_cdf[2], 1.0f);

  // Sample and verify distribution
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  int num_samples = 10000;
  int counts[2][2] = {{0, 0}, {0, 0}};

  for (int i = 0; i < num_samples; ++i) {
    auto [uv, pdf] = SampleEmissionCDF(cdf, dist(rng), dist(rng));
    int u_idx = static_cast<int>(uv.x() * 2);
    int v_idx = static_cast<int>(uv.y() * 2);
    u_idx = std::clamp(u_idx, 0, 1);
    v_idx = std::clamp(v_idx, 0, 1);
    counts[v_idx][u_idx]++;
  }

  // Expected relative weights (approx due to SRGB):
  // 10 -> L(10) ~ 0.003
  // 50 -> L(50) ~ 0.03
  // 100 -> L(100) ~ 0.12
  // 250 -> L(250) ~ 0.95
  // Total ~ 1.103
  // P(1,1) ~ 0.95/1.103 ~ 0.86 (Very high)
  // P(0,0) ~ 0.003 (Very low)

  // Verify (1,1) is dominant
  EXPECT_GT(counts[1][1], 8000);
  EXPECT_LT(counts[0][0], 100);
}

TEST_F(SceneTest, SampleEmissionCDF_Manual) {
  // Manually construct a 2x2 CDF
  // Row 0: 20% probability. Inside Row 0: Left 100%, Right 0%.
  // Row 1: 80% probability. Inside Row 1: Left 25%, Right 75%.
  EmissionCDF cdf;

  // Marginal CDF for rows: [0, 0.2, 1.0]
  cdf.marginal_cdf = {0.0f, 0.2f, 1.0f};

  // Conditional CDFs
  // Row 0 (matches v in [0, 0.2]): Only Col 0 has prob. CDF: [0, 1.0, 1.0]
  // Row 1 (matches v in [0.2, 1.0]): Col 0 (25%), Col 1 (75%). CDF: [0,
  // 0.25, 1.0]
  cdf.conditional_cdf.resize(2);
  cdf.conditional_cdf[0] = {0.0f, 1.0f, 1.0f};
  cdf.conditional_cdf[1] = {0.0f, 0.25f, 1.0f};

  // Case 1: Sample Row 0, Col 0
  // v = 0.1 (in 0.0 - 0.2) -> Row 0
  // u = 0.5 (in 0.0 - 1.0) -> Col 0 (since breakpoint is 1.0)
  {
    auto [uv, pdf] = SampleEmissionCDF(cdf, 0.5f, 0.1f);
    // Expected UV: Center of (0,0) in 2x2 grid -> (0.25, 0.25)
    EXPECT_NEAR(uv.x(), 0.25f, 1e-5f);
    EXPECT_NEAR(uv.y(), 0.25f, 1e-5f);

    // PDF = Marginal PDF * Conditional PDF
    // Marginal PDF for Row 0 = 0.2 - 0 = 0.2
    // Conditional PDF for Col 0 in Row 0 = 1.0 - 0.0 = 1.0
    // Total PDF = 0.2 * 1.0 = 0.2
    EXPECT_NEAR(pdf, 0.2f, 1e-5f);
  }

  // Case 2: Sample Row 1, Col 0
  // v = 0.6 (in 0.2 - 1.0) -> Row 1
  // u = 0.1 (in 0.0 - 0.25) -> Col 0
  {
    auto [uv, pdf] = SampleEmissionCDF(cdf, 0.1f, 0.6f);
    // Expected UV: Center of (0,1) in 2x2 grid -> (0.25, 0.75)
    EXPECT_NEAR(uv.x(), 0.25f, 1e-5f);
    EXPECT_NEAR(uv.y(), 0.75f, 1e-5f);

    // Marginal PDF for Row 1 = 1.0 - 0.2 = 0.8
    // Conditional PDF for Col 0 in Row 1 = 0.25 - 0.0 = 0.25
    // Total PDF = 0.8 * 0.25 = 0.2
    EXPECT_NEAR(pdf, 0.2f, 1e-5f);
  }

  // Case 3: Sample Row 1, Col 1
  // v = 0.6 (in 0.2 - 1.0) -> Row 1
  // u = 0.8 (in 0.25 - 1.0) -> Col 1
  {
    auto [uv, pdf] = SampleEmissionCDF(cdf, 0.8f, 0.6f);
    // Expected UV: Center of (1,1) in 2x2 grid -> (0.75, 0.75)
    EXPECT_NEAR(uv.x(), 0.75f, 1e-5f);
    EXPECT_NEAR(uv.y(), 0.75f, 1e-5f);

    // Marginal PDF for Row 1 = 0.8
    // Conditional PDF for Col 1 in Row 1 = 1.0 - 0.25 = 0.75
    // Total PDF = 0.8 * 0.75 = 0.6
    EXPECT_NEAR(pdf, 0.6f, 1e-5f);
  }
}

}  // namespace sh_baker
