#ifndef SH_BAKER_SRC_LAYER_COMPOSITE_H_
#define SH_BAKER_SRC_LAYER_COMPOSITE_H_

#include <Eigen/Dense>
#include <string>
#include <vector>

#include "scene.h"  // Texture

// Self-contained (Eigen + Texture only; no Embree/GL/tinygltf) compositor for
// the Quake 3 shader-stage stack carried by the `SH_material_layers` glTF
// extension. Evaluated at the first frame (t=0): time-varying tcMod/rgbGen and
// animMap are frozen. This is "the renderer's compositor at t=0" and is written
// to be liftable verbatim into a shared sh-scene library later.
namespace sh_baker {

// GL blend factors (subset Quake 3 uses), matching the exporter's emitted names.
enum class BlendFactor {
  kZero,
  kOne,
  kSrcColor,
  kOneMinusSrcColor,
  kDstColor,
  kOneMinusDstColor,
  kSrcAlpha,
  kOneMinusSrcAlpha,
  kDstAlpha,
  kOneMinusDstAlpha,
};

enum class RgbGenType {
  kIdentity,
  kIdentityLighting,
  kVertex,
  kExactVertex,
  kWave,
};

enum class WaveType {
  kSine,
  kTriangle,
  kSquare,
  kSawtooth,
  kInverseSawtooth,
};

struct RgbGen {
  RgbGenType type = RgbGenType::kIdentity;
  WaveType wave = WaveType::kSine;
  float base = 0.0f;
  float amplitude = 0.0f;
  float phase = 0.0f;
  float frequency = 0.0f;
};

enum class TcModType {
  kNoOp,
  kScale,
  kScroll,
  kRotate,
  kTurb,
  kStretch,
  kTransform,
};

struct TcMod {
  TcModType type = TcModType::kNoOp;
  // SCALE: [s_scale, t_scale]; TRANSFORM: [m00,m01,m02,m10,m11,m12]. Unused for
  // the time-varying types, which freeze to identity at t=0.
  std::vector<float> values;
};

struct CompositeLayer {
  // The layer's Quake 3 texture. For the base layer this is kept only for its
  // alpha (coverage); its colour is taken from the modern albedo.
  Texture texture;
  BlendFactor blend_src = BlendFactor::kOne;
  BlendFactor blend_dst = BlendFactor::kZero;
  RgbGen rgbgen;
  std::vector<TcMod> tcmods;
};

// Parses an exporter blend-factor name (e.g. "ONE_MINUS_SRC_ALPHA"). Unknown
// names fall back to kOne.
BlendFactor ParseBlendFactor(const std::string& name);

// t=0 evaluation primitives (exposed for unit testing).
Eigen::Vector2f ApplyTcMods(const std::vector<TcMod>& tcmods,
                            const Eigen::Vector2f& uv);
Eigen::Vector3f EvalRgbGen(const RgbGen& gen);  // rgb multiplier in [0,1]

// Composites the stack at t=0 into an RGBA8 texture (sRGB colour + coverage in
// alpha) in TEXCOORD_0 space. `base_layer` indexes the layer whose colour comes
// from `modern_albedo`; coverage is that layer's Q3 texture alpha, overridden by
// the modern albedo's alpha when it is 4-channel. Output resolution follows the
// modern albedo, falling back to the base Q3 layer when the modern albedo is a
// 1x1 placeholder.
Texture CompositeAlbedoCoverage(const std::vector<CompositeLayer>& layers,
                                int base_layer, const Texture& modern_albedo);

}  // namespace sh_baker

#endif  // SH_BAKER_SRC_LAYER_COMPOSITE_H_
