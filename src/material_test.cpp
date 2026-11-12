#include "material.h"

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <vector>

#include "scene.h"

namespace sh_baker {
namespace {

TEST(MaterialTest, SampleMaterialAdvanced) {
  Material mat;
  // Initialize textures
  mat.albedo.width = 1;
  mat.albedo.height = 1;
  mat.albedo.channels = 3;
  mat.albedo.pixel_data = {255, 255, 255};

  mat.normal_texture.width = 1;
  mat.normal_texture.height = 1;
  mat.normal_texture.channels = 3;
  mat.normal_texture.pixel_data = {128, 128, 255};

  mat.metallic_roughness_texture.width = 1;
  mat.metallic_roughness_texture.height = 1;
  mat.metallic_roughness_texture.channels = 3;
  mat.metallic_roughness_texture.pixel_data = {
      0, 255, 255};  // R=Unused, G=Roughness, B=Metallic

  Eigen::Vector3f normal(0.0f, 0.0f, 1.0f);
  Eigen::Vector3f incident(0.0f, 0.0f, 1.0f);
  std::mt19937 rng(12345);

  ReflectionSample sample = SampleMaterialAdvanced(
      mat, Eigen::Vector2f(0.5f, 0.5f), normal, incident, rng);

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

TEST(MaterialTest, EvalMaterialAdvancedBRDF) {
  Material mat;
  // White albedo
  mat.albedo.pixel_data = {255, 255, 255};
  mat.albedo.width = 1;
  mat.albedo.height = 1;
  mat.albedo.channels = 3;

  mat.normal_texture.width = 1;
  mat.normal_texture.height = 1;
  mat.normal_texture.channels = 3;
  mat.normal_texture.pixel_data = {128, 128, 255};

  mat.metallic_roughness_texture.width = 1;
  mat.metallic_roughness_texture.height = 1;
  mat.metallic_roughness_texture.channels = 3;
  mat.metallic_roughness_texture.pixel_data = {0, 255, 0};

  Eigen::Vector3f normal(0.0f, 0.0f, 1.0f);
  Eigen::Vector3f incident(0.0f, 0.0f, 1.0f);
  Eigen::Vector3f reflected(0.0f, 0.0f, 1.0f);

  Eigen::Vector3f res = EvalMaterialAdvanced(mat, Eigen::Vector2f(0.5f, 0.5f),
                                             normal, incident, reflected);

  // Lambertian BRDF = rho / PI. However, PBR adds Fresnel (0.04)
  // so Diffuse is scaled by 0.96, plus some Specular.
  // Expected approx 0.97 / PI.
  EXPECT_NEAR(res.x(), 0.97f / M_PI, 1e-2f);
}

TEST(MaterialTest, GetEmission) {
  Material mat;
  mat.albedo.width = 1;
  mat.albedo.height = 1;
  mat.albedo.channels = 3;
  mat.albedo.pixel_data = {255, 255, 255};
  mat.emissive_factor = Eigen::Vector3f::Ones();
  mat.emissive_strength = 5.0f;

  Eigen::Vector3f res = GetEmission(mat, Eigen::Vector2f(0.5f, 0.5f));
  EXPECT_NEAR(res.x(), 5.0f, 1e-5f);
  EXPECT_NEAR(res.y(), 5.0f, 1e-5f);
  EXPECT_NEAR(res.z(), 5.0f, 1e-5f);
}

TEST(MaterialTest, GetEmissionWithTexture) {
  Material mat;
  // Initialize emissive texture (Red)
  mat.emissive_texture = Texture();
  mat.emissive_texture->width = 1;
  mat.emissive_texture->height = 1;
  mat.emissive_texture->channels = 3;
  mat.emissive_texture->pixel_data = {255, 0, 0};

  mat.emissive_strength = 2.0f;
  mat.emissive_factor = Eigen::Vector3f::Ones();

  Eigen::Vector3f res = GetEmission(mat, Eigen::Vector2f(0.5f, 0.5f));
  // Red channel: 1.0 * 1.0 * 2.0 = 2.0
  EXPECT_NEAR(res.x(), 2.0f, 1e-5f);
  // Green/Blue channels: 0.0 * 1.0 * 2.0 = 0.0
  EXPECT_NEAR(res.y(), 0.0f, 1e-5f);
  EXPECT_NEAR(res.z(), 0.0f, 1e-5f);
}

TEST(MaterialTest, GetEmissionTextureWithFactor) {
  Material mat;
  // Initialize emissive texture (Red)
  mat.emissive_texture = Texture();
  mat.emissive_texture->width = 1;
  mat.emissive_texture->height = 1;
  mat.emissive_texture->channels = 3;
  mat.emissive_texture->pixel_data = {255, 0, 0};

  mat.emissive_strength = 2.0f;
  // Factor suppresses Green (even if texture had it) and Red.
  // Here texture is Red. Factor = (0.5, 1.0, 1.0).
  mat.emissive_factor = Eigen::Vector3f(0.5f, 1.0f, 1.0f);

  Eigen::Vector3f res = GetEmission(mat, Eigen::Vector2f(0.5f, 0.5f));
  // Red: Texture(1.0) * Factor(0.5) * Strength(2.0) = 1.0
  EXPECT_NEAR(res.x(), 1.0f, 1e-5f);
  // Green/Blue: 0.0
  EXPECT_NEAR(res.y(), 0.0f, 1e-5f);
  EXPECT_NEAR(res.z(), 0.0f, 1e-5f);
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

TEST(MaterialTest, GetAlbedoWrapping) {
  Material mat;
  mat.albedo.width = 2;
  mat.albedo.height = 2;
  mat.albedo.channels = 3;
  // 2x2 Texture:
  // Red (255,0,0)   | Green (0,255,0)
  // Blue (0,0,255)  | White (255,255,255)
  mat.albedo.pixel_data = {255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255};

  // 1. Normal access (0.25, 0.25) -> Top-Left (Red)
  {
    Eigen::Vector3f res = GetAlbedo(mat, Eigen::Vector2f(0.25f, 0.25f));
    EXPECT_NEAR(res.x(), 1.0f, 1e-5f);
    EXPECT_NEAR(res.y(), 0.0f, 1e-5f);
    EXPECT_NEAR(res.z(), 0.0f, 1e-5f);
  }

  // 2. Wrapped access (1.25, 0.25) -> Should be same as (0.25, 0.25) -> Red
  {
    Eigen::Vector3f res = GetAlbedo(mat, Eigen::Vector2f(1.25f, 0.25f));
    EXPECT_NEAR(res.x(), 1.0f, 1e-5f);
    EXPECT_NEAR(res.y(), 0.0f, 1e-5f);
    EXPECT_NEAR(res.z(), 0.0f, 1e-5f);
  }

  // 3. Negative access (-0.75, 0.25) -> Should map to 0.25 -> Red
  {
    Eigen::Vector3f res = GetAlbedo(mat, Eigen::Vector2f(-0.75f, 0.25f));
    EXPECT_NEAR(res.x(), 1.0f, 1e-5f);
    EXPECT_NEAR(res.y(), 0.0f, 1e-5f);
    EXPECT_NEAR(res.z(), 0.0f, 1e-5f);
  }

  // 4. Large Negative (-10.75, 0.25) -> Red
  {
    Eigen::Vector3f res = GetAlbedo(mat, Eigen::Vector2f(-10.75f, 0.25f));
    EXPECT_NEAR(res.x(), 1.0f, 1e-5f);
    EXPECT_NEAR(res.y(), 0.0f, 1e-5f);
    EXPECT_NEAR(res.z(), 0.0f, 1e-5f);
  }
}

// The layer model is sh-baker's own: a stage stack can be built with
// material.h alone, without any glTF type.
TEST(MaterialLayersTest, BuiltInCode) {
  MaterialLayer layer;
  layer.texture_path = "textures/flame1.png";
  layer.anim_frame_paths = {std::filesystem::path("textures/flame1.png"),
                            std::filesystem::path("textures/flame2.png")};
  layer.anim_freq = 10.0f;
  layer.blend_src = BlendFactor::kOne;
  layer.blend_dst = BlendFactor::kOne;
  layer.rgbgen.type = RgbGenType::kWave;
  layer.rgbgen.wave = WaveType::kTriangle;
  layer.rgbgen.base = 0.5f;
  layer.rgbgen.amplitude = 0.25f;
  layer.rgbgen.frequency = 2.0f;
  layer.tcmods = {
      TcMod{TcModType::kTurb, {0.0f, 0.25f, 0.0f, 1.0f}, WaveType::kSine}};

  MaterialLayers stack;
  stack.surface_blend = SurfaceBlend::kAdd;
  stack.cull_mode = CullMode::kNone;
  stack.layers = {layer};

  Material mat;
  mat.layers = stack;

  ASSERT_TRUE(mat.layers.has_value());
  EXPECT_EQ(mat.layers->surface_blend, SurfaceBlend::kAdd);
  EXPECT_EQ(mat.layers->cull_mode, CullMode::kNone);
  EXPECT_EQ(mat.layers->base_layer, 0);
  ASSERT_EQ(mat.layers->layers.size(), 1u);
  const MaterialLayer& built = mat.layers->layers[0];
  EXPECT_EQ(built.texture_path, std::filesystem::path("textures/flame1.png"));
  ASSERT_EQ(built.anim_frame_paths.size(), 2u);
  EXPECT_EQ(built.anim_frame_paths[1],
            std::filesystem::path("textures/flame2.png"));
  EXPECT_EQ(built.anim_freq, 10.0f);
  EXPECT_EQ(built.blend_dst, BlendFactor::kOne);
  EXPECT_EQ(built.rgbgen.type, RgbGenType::kWave);
  EXPECT_EQ(built.rgbgen.wave, WaveType::kTriangle);
  EXPECT_EQ(built.rgbgen.frequency, 2.0f);
  ASSERT_EQ(built.tcmods.size(), 1u);
  EXPECT_EQ(built.tcmods[0].type, TcModType::kTurb);
  EXPECT_EQ(built.tcmods[0].values,
            (std::vector<float>{0.0f, 0.25f, 0.0f, 1.0f}));
  EXPECT_EQ(built.tcmods[0].wave, WaveType::kSine);
}

// A WAVE rgbGen's function can be absent, which is distinct from any named
// function.
TEST(MaterialLayersTest, WaveFunctionCanBeAbsent) {
  RgbGen gen;
  gen.type = RgbGenType::kWave;
  EXPECT_EQ(gen.wave, WaveType::kSine);
  gen.wave = std::nullopt;
  EXPECT_FALSE(gen.wave.has_value());
}

}  // namespace
}  // namespace sh_baker
