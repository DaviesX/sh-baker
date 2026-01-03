#include "material.h"

#include <gtest/gtest.h>

#include <cmath>

#include "scene.h"

namespace sh_baker {
namespace {

TEST(MaterialTest, SampleMaterial) {
  Material mat;
  Eigen::Vector3f normal(0.0f, 0.0f, 1.0f);
  Eigen::Vector3f incident(0.0f, 0.0f, -1.0f);
  std::mt19937 rng(12345);

  ReflectionSample sample =
      SampleMaterial(mat, Eigen::Vector2f(0.5f, 0.5f), normal, incident, rng);

  EXPECT_GT(sample.pdf, 0.0f);
  EXPECT_NEAR(sample.direction.norm(), 1.0f, 1e-4f);
  EXPECT_GT(sample.direction.z(), -1e-4f);  // Should be in hemisphere
}

TEST(MaterialTest, GetAlbedo) {
  Material mat;
  mat.albedo.width = 1;
  mat.albedo.height = 1;
  mat.albedo.channels = 3;
  mat.albedo.pixel_data = {255, 128, 0};  // Orange-ish

  Eigen::Vector3f res = GetAlbedo(mat, Eigen::Vector2f(0.5f, 0.5f));
  EXPECT_NEAR(res.x(), 1.0f, 1e-5f);
  EXPECT_NEAR(res.y(), 0.21586f, 1e-3f);
  EXPECT_NEAR(res.z(), 0.0f, 1e-5f);
}

TEST(MaterialTest, EvalMaterialBRDF) {
  Material mat;
  // White albedo
  mat.albedo.pixel_data = {255, 255, 255};
  mat.albedo.width = 1;
  mat.albedo.height = 1;
  mat.albedo.channels = 3;

  Eigen::Vector3f normal(0.0f, 0.0f, 1.0f);
  Eigen::Vector3f incident(0.0f, 0.0f, -1.0f);
  Eigen::Vector3f reflected(0.0f, 0.0f, 1.0f);

  Eigen::Vector3f res = EvalMaterial(mat, Eigen::Vector2f(0.5f, 0.5f), normal,
                                     incident, reflected);

  // Lambertian BRDF = rho / PI
  // rho = 1.0
  EXPECT_NEAR(res.x(), 1.0f / M_PI, 1e-5f);
}

TEST(MaterialTest, GetEmission) {
  Material mat;
  mat.albedo.width = 1;
  mat.albedo.height = 1;
  mat.albedo.channels = 3;
  mat.albedo.pixel_data = {255, 255, 255};
  mat.emission_intensity = 5.0f;

  Eigen::Vector3f res = GetEmission(mat, Eigen::Vector2f(0.5f, 0.5f));
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
