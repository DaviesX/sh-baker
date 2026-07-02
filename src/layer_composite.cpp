#include "layer_composite.h"

#include <glog/logging.h>

#include <algorithm>
#include <cmath>

#include "colorspace.h"

namespace sh_baker {
namespace {

constexpr float kPi = 3.14159265358979323846f;

// Quake 3 unit waveform sampled at fractional phase `x` (period 1), range
// roughly [-1, 1] (sine/triangle/square) or [0, 1] (sawtooth variants).
float WaveValue(WaveType type, float x) {
  float f = x - std::floor(x);  // fractional phase in [0, 1)
  switch (type) {
    case WaveType::kSine:
      return std::sin(f * 2.0f * kPi);
    case WaveType::kSquare:
      return f < 0.5f ? 1.0f : -1.0f;
    case WaveType::kTriangle:
      // Up from -1..1 over [0,0.5], back down over [0.5,1].
      return f < 0.5f ? (-1.0f + 4.0f * f) : (3.0f - 4.0f * f);
    case WaveType::kSawtooth:
      return f;
    case WaveType::kInverseSawtooth:
      return 1.0f - f;
  }
  return 0.0f;
}

// Nearest-neighbour sample with UV wrap, returning gamma-space rgb in [0,1] and
// alpha in [0,1]. An empty texture reads as opaque white.
void SampleTexture(const Texture& tex, const Eigen::Vector2f& uv,
                   Eigen::Vector3f* rgb, float* alpha) {
  if (tex.pixel_data.empty() || tex.width == 0 || tex.height == 0) {
    *rgb = Eigen::Vector3f::Ones();
    *alpha = 1.0f;
    return;
  }
  // Exported layer/albedo textures are always RGB or RGBA; grayscale is not a
  // case we need to support.
  CHECK_GE(tex.channels, 3u);
  float u = uv.x() - std::floor(uv.x());
  float v = uv.y() - std::floor(uv.y());
  int tx = std::clamp(static_cast<int>(u * tex.width), 0,
                      static_cast<int>(tex.width) - 1);
  int ty = std::clamp(static_cast<int>(v * tex.height), 0,
                      static_cast<int>(tex.height) - 1);
  int idx = (ty * static_cast<int>(tex.width) + tx) *
            static_cast<int>(tex.channels);
  *rgb = Eigen::Vector3f(tex.pixel_data[idx] / 255.0f,
                         tex.pixel_data[idx + 1] / 255.0f,
                         tex.pixel_data[idx + 2] / 255.0f);
  *alpha = tex.channels >= 4 ? tex.pixel_data[idx + 3] / 255.0f : 1.0f;
}

// Evaluates a GL blend factor to a per-channel weight, given the source colour
// (src_rgb/src_alpha) and the current accumulator (dst_rgb/dst_alpha).
Eigen::Vector3f BlendWeight(BlendFactor factor, const Eigen::Vector3f& src_rgb,
                            float src_alpha, const Eigen::Vector3f& dst_rgb,
                            float dst_alpha) {
  const Eigen::Vector3f one = Eigen::Vector3f::Ones();
  switch (factor) {
    case BlendFactor::kZero:
      return Eigen::Vector3f::Zero();
    case BlendFactor::kOne:
      return one;
    case BlendFactor::kSrcColor:
      return src_rgb;
    case BlendFactor::kOneMinusSrcColor:
      return one - src_rgb;
    case BlendFactor::kDstColor:
      return dst_rgb;
    case BlendFactor::kOneMinusDstColor:
      return one - dst_rgb;
    case BlendFactor::kSrcAlpha:
      return src_alpha * one;
    case BlendFactor::kOneMinusSrcAlpha:
      return (1.0f - src_alpha) * one;
    case BlendFactor::kDstAlpha:
      return dst_alpha * one;
    case BlendFactor::kOneMinusDstAlpha:
      return (1.0f - dst_alpha) * one;
  }
  return one;
}

// Nearest-neighbour sample with UV wrap, returning LINEAR-space rgb in [0,1]
// and alpha in [0,1]. An empty texture emits nothing (unlike SampleTexture's
// opaque-white albedo default): the additive contribution of a missing stage
// is zero.
void SampleTextureLinear(const Texture& tex, const Eigen::Vector2f& uv,
                         Eigen::Vector3f* rgb, float* alpha) {
  if (tex.pixel_data.empty() || tex.width == 0 || tex.height == 0) {
    *rgb = Eigen::Vector3f::Zero();
    *alpha = 0.0f;
    return;
  }
  CHECK_GE(tex.channels, 3u);
  float u = uv.x() - std::floor(uv.x());
  float v = uv.y() - std::floor(uv.y());
  int tx = std::clamp(static_cast<int>(u * tex.width), 0,
                      static_cast<int>(tex.width) - 1);
  int ty = std::clamp(static_cast<int>(v * tex.height), 0,
                      static_cast<int>(tex.height) - 1);
  int idx = (ty * static_cast<int>(tex.width) + tx) *
            static_cast<int>(tex.channels);
  *rgb = Eigen::Vector3f(SRGBToLinear(tex.pixel_data[idx]),
                         SRGBToLinear(tex.pixel_data[idx + 1]),
                         SRGBToLinear(tex.pixel_data[idx + 2]));
  *alpha = tex.channels >= 4 ? tex.pixel_data[idx + 3] / 255.0f : 1.0f;
}

// Linear sample of a stage, averaging its animMap frames when present (the mean
// glow, since a static SH bake cannot animate).
void SampleLayerLinear(const CompositeLayer& layer, const Eigen::Vector2f& uv,
                       Eigen::Vector3f* rgb, float* alpha) {
  if (layer.anim_frames.empty()) {
    SampleTextureLinear(layer.texture, uv, rgb, alpha);
    return;
  }
  Eigen::Vector3f rgb_sum = Eigen::Vector3f::Zero();
  float alpha_sum = 0.0f;
  for (const Texture& frame : layer.anim_frames) {
    Eigen::Vector3f c;
    float a;
    SampleTextureLinear(frame, uv, &c, &a);
    rgb_sum += c;
    alpha_sum += a;
  }
  float inv = 1.0f / static_cast<float>(layer.anim_frames.size());
  *rgb = rgb_sum * inv;
  *alpha = alpha_sum * inv;
}

}  // namespace

BlendFactor ParseBlendFactor(const std::string& name) {
  if (name == "ZERO") return BlendFactor::kZero;
  if (name == "ONE") return BlendFactor::kOne;
  if (name == "SRC_COLOR") return BlendFactor::kSrcColor;
  if (name == "ONE_MINUS_SRC_COLOR") return BlendFactor::kOneMinusSrcColor;
  if (name == "DST_COLOR") return BlendFactor::kDstColor;
  if (name == "ONE_MINUS_DST_COLOR") return BlendFactor::kOneMinusDstColor;
  if (name == "SRC_ALPHA") return BlendFactor::kSrcAlpha;
  if (name == "ONE_MINUS_SRC_ALPHA") return BlendFactor::kOneMinusSrcAlpha;
  if (name == "DST_ALPHA") return BlendFactor::kDstAlpha;
  if (name == "ONE_MINUS_DST_ALPHA") return BlendFactor::kOneMinusDstAlpha;
  LOG(WARNING) << "Unknown blend factor '" << name << "', defaulting to ONE";
  return BlendFactor::kOne;
}

Eigen::Vector2f ApplyTcMods(const std::vector<TcMod>& tcmods,
                            const Eigen::Vector2f& uv) {
  Eigen::Vector2f out = uv;
  for (const auto& m : tcmods) {
    switch (m.type) {
      case TcModType::kScale:
        if (m.values.size() >= 2) {
          out = Eigen::Vector2f(out.x() * m.values[0], out.y() * m.values[1]);
        }
        break;
      case TcModType::kTransform:
        if (m.values.size() >= 6) {
          // Row-major 2x3 affine [a b c; d e f].
          out = Eigen::Vector2f(
              m.values[0] * out.x() + m.values[1] * out.y() + m.values[2],
              m.values[3] * out.x() + m.values[4] * out.y() + m.values[5]);
        }
        break;
      // Time-varying transforms freeze to identity at t=0.
      case TcModType::kNoOp:
      case TcModType::kScroll:
      case TcModType::kRotate:
      case TcModType::kTurb:
      case TcModType::kStretch:
        break;
    }
  }
  return out;
}

Eigen::Vector3f EvalRgbGen(const RgbGen& gen) {
  switch (gen.type) {
    case RgbGenType::kIdentity:
    case RgbGenType::kIdentityLighting:
      return Eigen::Vector3f::Ones();
    case RgbGenType::kVertex:
    case RgbGenType::kExactVertex:
      // Needs COLOR_0 (not exported yet); treat as identity. Warned at parse.
      return Eigen::Vector3f::Ones();
    case RgbGenType::kWave: {
      // At t=0 the wave argument is the phase. Quake 3 clamps rgbGen to [0,1].
      float s = gen.base + gen.amplitude * WaveValue(gen.wave, gen.phase);
      s = std::clamp(s, 0.0f, 1.0f);
      return Eigen::Vector3f(s, s, s);
    }
  }
  return Eigen::Vector3f::Ones();
}

Texture CompositeAlbedoCoverage(const std::vector<CompositeLayer>& layers,
                                int base_layer, const Texture& modern_albedo) {
  // Output resolution: the modern albedo, unless it is a 1x1 placeholder, in
  // which case fall back to the base Q3 layer's resolution.
  uint32_t out_w = modern_albedo.width;
  uint32_t out_h = modern_albedo.height;
  if ((out_w <= 1 || out_h <= 1) && base_layer >= 0 &&
      base_layer < static_cast<int>(layers.size())) {
    const Texture& base_tex = layers[base_layer].texture;
    if (base_tex.width > 1 && base_tex.height > 1) {
      out_w = base_tex.width;
      out_h = base_tex.height;
    }
  }
  if (out_w == 0) out_w = 1;
  if (out_h == 0) out_h = 1;

  const bool modern_has_alpha = modern_albedo.channels >= 4;

  Texture out;
  out.width = out_w;
  out.height = out_h;
  out.channels = 4;
  out.pixel_data.resize(static_cast<size_t>(out_w) * out_h * 4);

  for (uint32_t ty = 0; ty < out_h; ++ty) {
    for (uint32_t tx = 0; tx < out_w; ++tx) {
      Eigen::Vector2f uv((tx + 0.5f) / out_w, (ty + 0.5f) / out_h);

      Eigen::Vector3f acc = Eigen::Vector3f::Zero();
      float acc_alpha = 1.0f;
      float coverage = 1.0f;

      for (int li = 0; li < static_cast<int>(layers.size()); ++li) {
        const CompositeLayer& layer = layers[li];
        Eigen::Vector2f luv = ApplyTcMods(layer.tcmods, uv);

        Eigen::Vector3f src_rgb;
        float src_alpha;
        if (li == base_layer) {
          // Base layer colour comes from the modern albedo; coverage from the
          // Q3 base texture's alpha (or the modern alpha when authored).
          SampleTexture(modern_albedo, luv, &src_rgb, &src_alpha);
          float q3_alpha;
          Eigen::Vector3f ignore_rgb;
          SampleTexture(layer.texture, luv, &ignore_rgb, &q3_alpha);
          coverage = modern_has_alpha ? src_alpha : q3_alpha;
        } else {
          SampleTexture(layer.texture, luv, &src_rgb, &src_alpha);
        }

        src_rgb = src_rgb.cwiseProduct(EvalRgbGen(layer.rgbgen));

        Eigen::Vector3f sf =
            BlendWeight(layer.blend_src, src_rgb, src_alpha, acc, acc_alpha);
        Eigen::Vector3f df =
            BlendWeight(layer.blend_dst, src_rgb, src_alpha, acc, acc_alpha);
        acc = sf.cwiseProduct(src_rgb) + df.cwiseProduct(acc);
      }

      acc = acc.cwiseMax(0.0f).cwiseMin(1.0f);
      size_t o = (static_cast<size_t>(ty) * out_w + tx) * 4;
      out.pixel_data[o + 0] = static_cast<uint8_t>(std::lround(acc.x() * 255.0f));
      out.pixel_data[o + 1] = static_cast<uint8_t>(std::lround(acc.y() * 255.0f));
      out.pixel_data[o + 2] = static_cast<uint8_t>(std::lround(acc.z() * 255.0f));
      out.pixel_data[o + 3] = static_cast<uint8_t>(
          std::lround(std::clamp(coverage, 0.0f, 1.0f) * 255.0f));
    }
  }

  out.file_path = modern_albedo.file_path;
  return out;
}

Texture CompositeEmissiveRadiance(const std::vector<CompositeLayer>& layers) {
  // Output resolution: the largest stage texture (animMap frames share a size).
  uint32_t out_w = 1;
  uint32_t out_h = 1;
  for (const auto& layer : layers) {
    out_w = std::max(out_w, layer.texture.width);
    out_h = std::max(out_h, layer.texture.height);
    for (const auto& frame : layer.anim_frames) {
      out_w = std::max(out_w, frame.width);
      out_h = std::max(out_h, frame.height);
    }
  }

  Texture out;
  out.width = out_w;
  out.height = out_h;
  out.channels = 4;
  out.pixel_data.resize(static_cast<size_t>(out_w) * out_h * 4);

  for (uint32_t ty = 0; ty < out_h; ++ty) {
    for (uint32_t tx = 0; tx < out_w; ++tx) {
      Eigen::Vector2f uv((tx + 0.5f) / out_w, (ty + 0.5f) / out_h);

      // Blend every stage in LINEAR space (physical additive light). No base
      // layer / modern-albedo substitution: each stage emits its own colour.
      Eigen::Vector3f acc = Eigen::Vector3f::Zero();
      float acc_alpha = 1.0f;
      for (const auto& layer : layers) {
        Eigen::Vector2f luv = ApplyTcMods(layer.tcmods, uv);

        Eigen::Vector3f src_rgb;
        float src_alpha;
        SampleLayerLinear(layer, luv, &src_rgb, &src_alpha);
        src_rgb = src_rgb.cwiseProduct(EvalRgbGen(layer.rgbgen));

        Eigen::Vector3f sf =
            BlendWeight(layer.blend_src, src_rgb, src_alpha, acc, acc_alpha);
        Eigen::Vector3f df =
            BlendWeight(layer.blend_dst, src_rgb, src_alpha, acc, acc_alpha);
        acc = sf.cwiseProduct(src_rgb) + df.cwiseProduct(acc);
      }

      // Clamp to LDR and sRGB-encode; GetEmission re-linearizes and the
      // emissive_strength knob provides the HDR range.
      acc = acc.cwiseMax(0.0f).cwiseMin(1.0f);
      size_t o = (static_cast<size_t>(ty) * out_w + tx) * 4;
      out.pixel_data[o + 0] = LinearToSRGB(acc.x());
      out.pixel_data[o + 1] = LinearToSRGB(acc.y());
      out.pixel_data[o + 2] = LinearToSRGB(acc.z());
      out.pixel_data[o + 3] = 255;
    }
  }
  return out;
}

}  // namespace sh_baker
