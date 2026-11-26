#include "scene.h"

#include <embree4/rtcore.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <random>

#include "light.h"

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

TEST_F(SceneTest, ComputeTangents) {
  Geometry geo;
  geo.vertices = {{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}};
  geo.normals.assign(3, Eigen::Vector3f(0.0f, 0.0f, 1.0f));
  geo.texture_uvs = {{0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 1.0f}};
  geo.indices = {0, 1, 2};

  const auto textured_tangents = ComputeTangents(geo);
  ASSERT_EQ(textured_tangents.size(), 3u);
  for (const Eigen::Vector4f& tangent : textured_tangents) {
    EXPECT_TRUE(
        tangent.head<3>().isApprox(Eigen::Vector3f(1.0f, 0.0f, 0.0f), 1e-4f));
    EXPECT_NEAR(tangent.w(), 1.0f, 1e-4f);
  }

  geo.texture_uvs.clear();
  const auto fallback_tangents = ComputeTangents(geo);
  ASSERT_EQ(fallback_tangents.size(), 3u);
  for (const Eigen::Vector4f& tangent : fallback_tangents) {
    EXPECT_TRUE(
        tangent.head<3>().isApprox(Eigen::Vector3f(1.0f, 0.0f, 0.0f), 1e-4f));
    EXPECT_NEAR(tangent.w(), 1.0f, 1e-4f);
  }
}

// MikkTSpace writes only the vertices a triangle uses; any other vertex gets
// a zero tangent rather than uninitialized memory.
TEST_F(SceneTest, ComputeTangentsZeroesUnusedVertex) {
  Geometry geo;
  geo.vertices = {{0.0f, 0.0f, 0.0f},
                  {1.0f, 0.0f, 0.0f},
                  {0.0f, 1.0f, 0.0f},
                  {5.0f, 5.0f, 5.0f}};
  geo.normals.assign(4, Eigen::Vector3f(0.0f, 0.0f, 1.0f));
  geo.texture_uvs = {{0.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 1.0f}, {0.5f, 0.5f}};
  geo.indices = {0, 1, 2};

  const auto tangents = ComputeTangents(geo);
  ASSERT_EQ(tangents.size(), 4u);
  EXPECT_EQ(tangents[3], Eigen::Vector4f::Zero());
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
  cdf.conditional_cdf.resize(3);
  cdf.conditional_cdf[0] = {0.0f, 0.0f, 0.0f};
  cdf.conditional_cdf[1] = {0.0f, 1.0f, 1.0f};
  cdf.conditional_cdf[2] = {0.0f, 0.25f, 1.0f};

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

namespace {

// A W x H equirect texture whose texel (x, y) is f(theta, phi) at its centre,
// using ProjectLatLongMap()'s mapping.
template <typename F>
Texture32F MakeEquirect(int w, int h, int channels, F f) {
  Texture32F tex;
  tex.width = w;
  tex.height = h;
  tex.channels = channels;
  tex.pixel_data.resize(w * h * channels, 1.0f);
  for (int y = 0; y < h; ++y) {
    for (int x = 0; x < w; ++x) {
      float theta = (y + 0.5f) * static_cast<float>(M_PI) / h;
      float phi = (x + 0.5f) * 2.0f * static_cast<float>(M_PI) / w;
      Eigen::Vector3f c = f(theta, phi);
      for (int k = 0; k < 3; ++k) {
        tex.pixel_data[(y * w + x) * channels + k] = c[k];
      }
    }
  }
  return tex;
}

// Y-up direction for (theta from +Y, phi), as ProjectLatLongMap() maps it.
Eigen::Vector3f LatLongDir(float theta, float phi) {
  return Eigen::Vector3f(std::sin(theta) * std::sin(phi), std::cos(theta),
                         std::sin(theta) * std::cos(phi));
}

// Integrates EvaluateEnvironment() over the sphere on a fine grid and projects
// it onto SH the way ProjectLatLongMap() does.
SHCoeffs IntegrateLookup(const Environment& env) {
  const int steps_theta = 512;
  const int steps_phi = 1024;
  const double d_theta = M_PI / steps_theta;
  const double d_phi = 2.0 * M_PI / steps_phi;
  const Eigen::Vector3f up(0.0f, 1.0f, 0.0f);
  SHCoeffs coeffs;
  for (int t = 0; t < steps_theta; ++t) {
    float theta = static_cast<float>((t + 0.5) * d_theta);
    float d_omega = static_cast<float>(d_theta * d_phi * std::sin(theta));
    for (int p = 0; p < steps_phi; ++p) {
      float phi = static_cast<float>((p + 0.5) * d_phi);
      Eigen::Vector3f dir = LatLongDir(theta, phi);
      AccumulateRadiance(EvaluateEnvironment(env, dir) * d_omega, dir, up,
                         &coeffs);
    }
  }
  return coeffs;
}

void ExpectSHNear(const SHCoeffs& actual, const SHCoeffs& expected) {
  for (int c = 0; c < 3; ++c) {
    const float tol = 0.01f * std::abs(expected.coeffs[0][c]);
    for (int i = 0; i < 9; ++i) {
      EXPECT_NEAR(actual.coeffs[i][c], expected.coeffs[i][c], tol)
          << "coefficient " << i << " channel " << c;
    }
  }
}

}  // namespace

TEST_F(SceneTest, EvaluateEnvironmentTextureRowsAtPoles) {
  // Top row red, bottom row blue, the rest green.
  const int h = 8;
  Environment env;
  env.type = Environment::Type::Texture;
  env.texture = MakeEquirect(16, h, 3, [&](float theta, float) {
    int row = static_cast<int>(theta / M_PI * h);
    if (row == 0) return Eigen::Vector3f(1, 0, 0);
    if (row == h - 1) return Eigen::Vector3f(0, 0, 1);
    return Eigen::Vector3f(0, 1, 0);
  });
  Eigen::Vector3f up = EvaluateEnvironment(env, Eigen::Vector3f(0, 1, 0));
  Eigen::Vector3f down = EvaluateEnvironment(env, Eigen::Vector3f(0, -1, 0));
  EXPECT_TRUE(up.isApprox(Eigen::Vector3f(1, 0, 0), 1e-5f)) << up.transpose();
  EXPECT_TRUE(down.isApprox(Eigen::Vector3f(0, 0, 1), 1e-5f))
      << down.transpose();
}

TEST_F(SceneTest, EvaluateEnvironmentWrapsLongitude) {
  // Column 0 is 1, the last column 3, the rest 2. Phi = 0 lies on the seam
  // between the last column and column 0, so it averages them.
  const int w = 8;
  Environment env;
  env.type = Environment::Type::Texture;
  env.texture = MakeEquirect(w, 4, 3, [&](float, float phi) {
    int col = static_cast<int>(phi / (2.0 * M_PI) * w);
    float v = col == 0 ? 1.0f : (col == w - 1 ? 3.0f : 2.0f);
    return Eigen::Vector3f(v, v, v);
  });
  // phi = 0 is +Z; a hair either side of the seam stays near 2.
  const float theta = static_cast<float>(M_PI / 2);
  Eigen::Vector3f at_zero = EvaluateEnvironment(env, LatLongDir(theta, 0.0f));
  Eigen::Vector3f at_two_pi = EvaluateEnvironment(
      env, LatLongDir(theta, 2.0f * static_cast<float>(M_PI) - 1e-6f));
  Eigen::Vector3f past_zero =
      EvaluateEnvironment(env, LatLongDir(theta, 1e-6f));
  EXPECT_NEAR(at_zero.x(), 2.0f, 1e-3f);
  EXPECT_NEAR(at_two_pi.x(), 2.0f, 1e-3f);
  EXPECT_NEAR(past_zero.x(), 2.0f, 1e-3f);
  // Column centres return their own texel.
  Eigen::Vector3f col0 = EvaluateEnvironment(
      env, LatLongDir(theta, 0.5f * 2.0f * static_cast<float>(M_PI) / w));
  Eigen::Vector3f col_last = EvaluateEnvironment(
      env, LatLongDir(theta, (w - 0.5f) * 2.0f * static_cast<float>(M_PI) / w));
  EXPECT_NEAR(col0.x(), 1.0f, 1e-4f);
  EXPECT_NEAR(col_last.x(), 3.0f, 1e-4f);
}

TEST_F(SceneTest, EvaluateEnvironmentBilinearMidpoint) {
  // Columns alternate 0 and 4; halfway between two column centres gives 2.
  const int w = 8;
  const int h = 4;
  Environment env;
  env.type = Environment::Type::Texture;
  env.texture = MakeEquirect(w, h, 3, [&](float, float phi) {
    int col = static_cast<int>(phi / (2.0 * M_PI) * w);
    float v = (col % 2 == 0) ? 0.0f : 4.0f;
    return Eigen::Vector3f(v, 0.5f * v, 0.25f * v);
  });
  // Row centre theta of row 1, phi on the boundary between columns 2 and 3.
  const float theta = 1.5f * static_cast<float>(M_PI) / h;
  const float phi = 3.0f * 2.0f * static_cast<float>(M_PI) / w;
  Eigen::Vector3f mid = EvaluateEnvironment(env, LatLongDir(theta, phi));
  EXPECT_NEAR(mid.x(), 2.0f, 1e-4f);
  EXPECT_NEAR(mid.y(), 1.0f, 1e-4f);
  EXPECT_NEAR(mid.z(), 0.5f, 1e-4f);
}

TEST_F(SceneTest, EvaluateEnvironmentFourChannelTexture) {
  const int w = 8;
  const int h = 4;
  Environment env;
  env.type = Environment::Type::Texture;
  env.texture = MakeEquirect(w, h, 4, [&](float theta, float phi) {
    return Eigen::Vector3f(theta, phi, 0.25f);
  });
  // Alpha stays 1 and must be skipped. Look up texel (5, 2)'s centre.
  const float theta = 2.5f * static_cast<float>(M_PI) / h;
  const float phi = 5.5f * 2.0f * static_cast<float>(M_PI) / w;
  Eigen::Vector3f c = EvaluateEnvironment(env, LatLongDir(theta, phi));
  EXPECT_NEAR(c.x(), theta, 1e-4f);
  EXPECT_NEAR(c.y(), phi, 1e-4f);
  EXPECT_NEAR(c.z(), 0.25f, 1e-5f);
}

TEST_F(SceneTest, EvaluateEnvironmentPreethamBlackBelowHorizon) {
  Environment env;
  env.type = Environment::Type::Preetham;
  env.sun_direction = Eigen::Vector3f(0.3f, 0.8f, 0.2f).normalized();
  EXPECT_EQ(EvaluateEnvironment(env, Eigen::Vector3f(0, -1, 0)),
            Eigen::Vector3f::Zero());
  EXPECT_EQ(EvaluateEnvironment(
                env, Eigen::Vector3f(0.7f, -0.01f, 0.7f).normalized()),
            Eigen::Vector3f::Zero());
  Eigen::Vector3f zenith = EvaluateEnvironment(env, Eigen::Vector3f(0, 1, 0));
  // Normalised by the zenith value, so the zenith is the sky colour itself.
  EXPECT_TRUE(zenith.isApprox(Eigen::Vector3f(0.2f, 0.5f, 0.9f), 1e-5f))
      << zenith.transpose();
}

TEST_F(SceneTest, EvaluateEnvironmentTextureMatchesProjection) {
  Environment env;
  env.type = Environment::Type::Texture;
  env.texture = MakeEquirect(64, 32, 3, [](float theta, float phi) {
    return Eigen::Vector3f(1.0f + 0.5f * std::sin(phi) * std::sin(theta),
                           1.0f + 0.8f * std::cos(theta),
                           0.6f + 0.3f * std::cos(phi) * std::sin(theta));
  });
  ExpectSHNear(IntegrateLookup(env), ProjectEnvironmentToSH(env));
}

TEST_F(SceneTest, EvaluateEnvironmentPreethamMatchesProjection) {
  Environment env;
  env.type = Environment::Type::Preetham;
  env.sun_direction = Eigen::Vector3f(0.4f, 0.6f, -0.3f).normalized();
  ExpectSHNear(IntegrateLookup(env), ProjectEnvironmentToSH(env));
}

TEST_F(SceneTest, EvaluateEnvironmentFrameMapsZUpToYUp) {
  // Bright near +Y in the environment frame, dark elsewhere.
  Environment env;
  env.type = Environment::Type::Texture;
  env.texture = MakeEquirect(16, 8, 3, [](float theta, float) {
    float v = theta < 0.5f ? 5.0f : 0.0f;
    return Eigen::Vector3f(v, v, v);
  });
  SHCoeffs before = ProjectEnvironmentToSH(env);
  // World +Z -> environment +Y, world +Y -> environment -Z.
  env.to_environment << 1, 0, 0,  //
      0, 0, 1,                    //
      0, -1, 0;
  EXPECT_GT(EvaluateEnvironment(env, Eigen::Vector3f(0, 0, 1)).x(), 4.0f);
  EXPECT_EQ(EvaluateEnvironment(env, Eigen::Vector3f(0, 1, 0)).x(), 0.0f);
  EXPECT_EQ(EvaluateEnvironment(env, Eigen::Vector3f(0, 0, -1)).x(), 0.0f);
  // The frame doesn't touch the projection.
  SHCoeffs after = ProjectEnvironmentToSH(env);
  for (int i = 0; i < 9; ++i) {
    EXPECT_EQ(after.coeffs[i], before.coeffs[i]) << "coefficient " << i;
  }
}

}  // namespace sh_baker
