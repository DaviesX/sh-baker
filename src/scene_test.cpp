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
  Texture tex;
  tex.width = 64;
  tex.height = 32;
  tex.channels = 3;
  // White (255, 255, 255)
  tex.pixel_data.resize(tex.width * tex.height * 3, 255);

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

}  // namespace sh_baker
