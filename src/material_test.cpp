#include "material.h"

#include <gtest/gtest.h>

#include <cmath>

#include "scene.h"

namespace sh_baker {
namespace {

TEST(MaterialTest, SampleHemisphereUniform) {
  // Check bounds
  Eigen::Vector3f sample = SampleHemisphereUniform(0.5f, 0.5f);
  EXPECT_GE(sample.z(), 0.0f);
  EXPECT_LE(sample.z(), 1.0f);
  EXPECT_NEAR(sample.norm(), 1.0f, 1e-5f);

  // Check 0,0 mapping (top of hemisphere for u1=1, but u1 is cos(theta) or
  // similar?) Logic in code: z = u1. So u1=1.0 -> z=1.0 (top). u1=0.0 -> z=0.0
  // (horizon).
  Eigen::Vector3f top = SampleHemisphereUniform(1.0f, 0.0f);
  EXPECT_NEAR(top.x(), 0.0f, 1e-5f);
  EXPECT_NEAR(top.y(), 0.0f, 1e-5f);
  EXPECT_NEAR(top.z(), 1.0f, 1e-5f);
}

TEST(MaterialTest, EvalMaterialDiffuses) {
  Material mat;
  mat.albedo.width = 1;
  mat.albedo.height = 1;
  mat.albedo.channels = 3;
  mat.albedo.pixel_data = {255, 128, 0};  // Orange-ish
  mat.emission_intensity = 0.0f;

  Eigen::Vector3f res = EvalMaterial(mat, Eigen::Vector2f(0.5f, 0.5f));
  EXPECT_NEAR(res.x(), 1.0f, 1e-5f);
  EXPECT_NEAR(res.y(), 128.0f / 255.0f, 1e-5f);
  EXPECT_NEAR(res.z(), 0.0f, 1e-5f);
}

TEST(MaterialTest, EvalMaterialEmission) {
  Material mat;
  mat.albedo.width = 1;
  mat.albedo.height = 1;
  mat.albedo.channels = 3;
  mat.albedo.pixel_data = {255, 255, 255};
  mat.emission_intensity = 5.0f;

  Eigen::Vector3f res = EvalMaterial(mat, Eigen::Vector2f(0.5f, 0.5f));
  EXPECT_NEAR(res.x(), 5.0f, 1e-5f);
  EXPECT_NEAR(res.y(), 5.0f, 1e-5f);
  EXPECT_NEAR(res.z(), 5.0f, 1e-5f);
}

TEST(MaterialTest, GetAlpha) {
  Material mat;
  mat.albedo.width = 1;
  mat.albedo.height = 1;
  mat.albedo.channels = 4;
  mat.albedo.pixel_data = {255, 255, 255, 128};

  float alpha = GetAlpha(mat, Eigen::Vector2f(0.5f, 0.5f));
  EXPECT_NEAR(alpha, 128.0f / 255.0f, 1e-5f);
}

TEST(MaterialTest, GetAlphaNoAlphaChannel) {
  Material mat;
  mat.albedo.width = 1;
  mat.albedo.height = 1;
  mat.albedo.channels = 3;
  mat.albedo.pixel_data = {255, 255, 255};

  float alpha = GetAlpha(mat, Eigen::Vector2f(0.5f, 0.5f));
  EXPECT_NEAR(alpha, 1.0f, 1e-5f);
}

}  // namespace
}  // namespace sh_baker
