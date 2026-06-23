#include "layer_composite.h"

#include <gtest/gtest.h>

#include "scene.h"

namespace sh_baker {
namespace {

// 1x1 texture helper. 3 channels unless an alpha is given (then 4).
Texture Tex(uint8_t r, uint8_t g, uint8_t b, int alpha = -1) {
  Texture t;
  t.width = 1;
  t.height = 1;
  if (alpha < 0) {
    t.channels = 3;
    t.pixel_data = {r, g, b};
  } else {
    t.channels = 4;
    t.pixel_data = {r, g, b, static_cast<uint8_t>(alpha)};
  }
  return t;
}

CompositeLayer Layer(const Texture& tex, BlendFactor src, BlendFactor dst) {
  CompositeLayer l;
  l.texture = tex;
  l.blend_src = src;
  l.blend_dst = dst;
  return l;
}

TEST(LayerComposite, ParseBlendFactor) {
  EXPECT_EQ(ParseBlendFactor("ONE"), BlendFactor::kOne);
  EXPECT_EQ(ParseBlendFactor("ZERO"), BlendFactor::kZero);
  EXPECT_EQ(ParseBlendFactor("SRC_ALPHA"), BlendFactor::kSrcAlpha);
  EXPECT_EQ(ParseBlendFactor("ONE_MINUS_SRC_ALPHA"),
            BlendFactor::kOneMinusSrcAlpha);
  EXPECT_EQ(ParseBlendFactor("DST_COLOR"), BlendFactor::kDstColor);
  EXPECT_EQ(ParseBlendFactor("garbage"), BlendFactor::kOne);  // fallback
}

TEST(LayerComposite, EvalRgbGen) {
  RgbGen identity;
  EXPECT_TRUE(EvalRgbGen(identity).isApprox(Eigen::Vector3f::Ones()));

  RgbGen vertex;
  vertex.type = RgbGenType::kVertex;
  EXPECT_TRUE(EvalRgbGen(vertex).isApprox(Eigen::Vector3f::Ones()));

  // wave sawtooth: base 0, amp 1, phase 0.5 -> 0 + 1*0.5 = 0.5
  RgbGen wave;
  wave.type = RgbGenType::kWave;
  wave.wave = WaveType::kSawtooth;
  wave.base = 0.0f;
  wave.amplitude = 1.0f;
  wave.phase = 0.5f;
  EXPECT_TRUE(EvalRgbGen(wave).isApprox(Eigen::Vector3f::Constant(0.5f), 1e-4f));
}

TEST(LayerComposite, ApplyTcMods) {
  Eigen::Vector2f uv(0.25f, 0.5f);

  TcMod scale{TcModType::kScale, {4.0f, 2.0f}};
  EXPECT_TRUE(ApplyTcMods({scale}, uv).isApprox(Eigen::Vector2f(1.0f, 1.0f)));

  // affine [1 0 0.5; 0 1 0] -> (u+0.5, v)
  TcMod xform{TcModType::kTransform, {1, 0, 0.5f, 0, 1, 0}};
  EXPECT_TRUE(ApplyTcMods({xform}, Eigen::Vector2f(0.1f, 0.2f))
                  .isApprox(Eigen::Vector2f(0.6f, 0.2f), 1e-5f));

  // Time-varying transforms freeze to identity at t=0.
  TcMod scroll{TcModType::kScroll, {}};
  EXPECT_TRUE(ApplyTcMods({scroll}, uv).isApprox(uv));
}

// Single opaque base layer: the composite is the modern albedo; coverage comes
// from the Q3 base texture's alpha (modern is 3-channel).
TEST(LayerComposite, OpaqueBaseReplace) {
  Texture modern = Tex(200, 100, 50);
  std::vector<CompositeLayer> layers = {
      Layer(Tex(10, 20, 30, /*alpha=*/128), BlendFactor::kOne,
            BlendFactor::kZero)};

  Texture out = CompositeAlbedoCoverage(layers, /*base_layer=*/0, modern);
  ASSERT_EQ(out.channels, 4u);
  EXPECT_EQ(out.pixel_data[0], 200);
  EXPECT_EQ(out.pixel_data[1], 100);
  EXPECT_EQ(out.pixel_data[2], 50);
  EXPECT_EQ(out.pixel_data[3], 128);  // coverage from Q3 base alpha
}

// A 4-channel modern albedo overrides the Q3 base coverage.
TEST(LayerComposite, ModernAlphaOverridesCoverage) {
  Texture modern = Tex(200, 100, 50, /*alpha=*/64);
  std::vector<CompositeLayer> layers = {
      Layer(Tex(0, 0, 0, 255), BlendFactor::kOne, BlendFactor::kZero)};

  Texture out = CompositeAlbedoCoverage(layers, 0, modern);
  EXPECT_EQ(out.pixel_data[3], 64);
}

// The base layer's colour is the modern albedo, not its own Q3 texture.
TEST(LayerComposite, BaseUsesModernNotQ3Texture) {
  Texture modern = Tex(11, 22, 33);
  std::vector<CompositeLayer> layers = {
      Layer(Tex(200, 200, 200), BlendFactor::kOne, BlendFactor::kZero)};

  Texture out = CompositeAlbedoCoverage(layers, 0, modern);
  EXPECT_EQ(out.pixel_data[0], 11);
  EXPECT_EQ(out.pixel_data[1], 22);
  EXPECT_EQ(out.pixel_data[2], 33);
}

// Additive detail layer (ONE, ONE) adds onto the base.
TEST(LayerComposite, AdditiveLayer) {
  Texture modern = Tex(100, 100, 100);
  std::vector<CompositeLayer> layers = {
      Layer(Tex(0, 0, 0), BlendFactor::kOne, BlendFactor::kZero),       // base
      Layer(Tex(50, 50, 50), BlendFactor::kOne, BlendFactor::kOne)};    // add

  Texture out = CompositeAlbedoCoverage(layers, /*base_layer=*/0, modern);
  EXPECT_EQ(out.pixel_data[0], 150);
  EXPECT_EQ(out.pixel_data[1], 150);
  EXPECT_EQ(out.pixel_data[2], 150);
}

// Alpha blend layer over a black base: result = a*src.
TEST(LayerComposite, AlphaBlendLayer) {
  Texture modern = Tex(0, 0, 0);
  std::vector<CompositeLayer> layers = {
      Layer(Tex(0, 0, 0), BlendFactor::kOne, BlendFactor::kZero),  // base
      Layer(Tex(200, 200, 200, /*alpha=*/128), BlendFactor::kSrcAlpha,
            BlendFactor::kOneMinusSrcAlpha)};

  Texture out = CompositeAlbedoCoverage(layers, 0, modern);
  // 128/255 * 200/255 * 255 ~= 100
  EXPECT_NEAR(out.pixel_data[0], 100, 1);
}

// Filter layer (DST_COLOR, ZERO) multiplies the accumulator.
TEST(LayerComposite, FilterLayer) {
  Texture modern = Tex(255, 128, 64);
  std::vector<CompositeLayer> layers = {
      Layer(Tex(0, 0, 0), BlendFactor::kOne, BlendFactor::kZero),  // base
      Layer(Tex(128, 128, 128), BlendFactor::kDstColor,
            BlendFactor::kZero)};

  Texture out = CompositeAlbedoCoverage(layers, 0, modern);
  EXPECT_NEAR(out.pixel_data[0], 128, 1);
  EXPECT_NEAR(out.pixel_data[1], 64, 1);
  EXPECT_NEAR(out.pixel_data[2], 32, 1);
}

// Base layer at a non-zero index still draws its colour from the modern albedo.
TEST(LayerComposite, BaseLayerIndexNonZero) {
  Texture modern = Tex(255, 0, 255);  // magenta placeholder
  std::vector<CompositeLayer> layers = {
      Layer(Tex(10, 20, 30), BlendFactor::kOne, BlendFactor::kZero),  // bottom
      Layer(Tex(0, 0, 0), BlendFactor::kSrcAlpha,
            BlendFactor::kOneMinusSrcAlpha)};  // base (modern, opaque)

  Texture out = CompositeAlbedoCoverage(layers, /*base_layer=*/1, modern);
  EXPECT_EQ(out.pixel_data[0], 255);
  EXPECT_EQ(out.pixel_data[1], 0);
  EXPECT_EQ(out.pixel_data[2], 255);
}

// When the modern albedo is a 1x1 placeholder, the output resolution falls back
// to the base Q3 layer's resolution.
TEST(LayerComposite, ResolutionFallsBackToBaseLayer) {
  Texture modern = Tex(255, 0, 255);  // 1x1 placeholder
  Texture base_tex;
  base_tex.width = 2;
  base_tex.height = 2;
  base_tex.channels = 3;
  base_tex.pixel_data.assign(2 * 2 * 3, 128);

  std::vector<CompositeLayer> layers = {
      Layer(base_tex, BlendFactor::kOne, BlendFactor::kZero)};

  Texture out = CompositeAlbedoCoverage(layers, 0, modern);
  EXPECT_EQ(out.width, 2u);
  EXPECT_EQ(out.height, 2u);
}

}  // namespace
}  // namespace sh_baker
