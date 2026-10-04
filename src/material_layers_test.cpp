#include <gtest/gtest.h>

#include "material.h"

namespace sh_baker {
namespace {

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
