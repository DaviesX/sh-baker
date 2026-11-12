#ifndef SH_BAKER_SRC_MATERIAL_H_
#define SH_BAKER_SRC_MATERIAL_H_

#include <Eigen/Dense>
#include <filesystem>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "texture.h"

namespace sh_baker {

// GL blend factors (subset Quake 3 uses), matching the exporter's emitted
// names.
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
  // "NONE": the wave field of a TURB or STRETCH tcMod that names no function.
  kNone,
};

struct RgbGen {
  RgbGenType type = RgbGenType::kIdentity;
  // The WAVE function. Absent when the source named none (the exporter omits
  // `func` for noise and unknown waves); compositing evaluates that as SIN.
  std::optional<WaveType> wave = WaveType::kSine;
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
  // SCALE: [s_scale, t_scale]; SCROLL: [s_rate, t_rate]; ROTATE: [degrees per
  // second]; TURB and STRETCH: [base, amplitude, phase, frequency];
  // TRANSFORM: [m00,m01,m02,m10,m11,m12]. Compositing reads only SCALE and
  // TRANSFORM: the time-varying types freeze to identity at t=0.
  std::vector<float> values;
  // TURB and STRETCH only: the wave function, kNone when the source named none.
  WaveType wave = WaveType::kNone;
};

// How the renderer blends a surface's composited stage stack.
enum class SurfaceBlend {
  kOpaque,
  kBlend,
  kAdd,
};

// Quake 3 face culling.
enum class CullMode {
  kFront,
  kBack,
  kNone,
};

// One Quake 3 shader stage.
struct MaterialLayer {
  // Source image of the stage's texture; absent when it has no known source.
  std::optional<std::filesystem::path> texture_path;
  // animMap frames' source images, in order (frame 0 is normally the stage's
  // own texture); empty for a static stage. An absent entry is a frame without
  // a known source, kept so the frames keep their positions.
  std::vector<std::optional<std::filesystem::path>> anim_frame_paths;
  float anim_freq = 0.0f;  // animMap frames per second.
  BlendFactor blend_src = BlendFactor::kOne;
  BlendFactor blend_dst = BlendFactor::kZero;
  RgbGen rgbgen;
  std::vector<TcMod> tcmods;
};

// A material's Quake 3 stage stack: sh-baker's own form of the
// `SH_material_layers` glTF extension, which the glTF loader and saver
// translate to and from. Other loaders, such as q3map2's, fill it directly.
struct MaterialLayers {
  SurfaceBlend surface_blend = SurfaceBlend::kOpaque;
  CullMode cull_mode = CullMode::kFront;
  // The stage whose texture is the albedo source; the modern albedo
  // substitutes for it when compositing.
  int base_layer = 0;
  std::vector<MaterialLayer> layers;
};

// --- Material ---
struct Material {
  std::string name;

  // Albedo / Transparency
  Texture albedo;
  Texture normal_texture;
  Texture metallic_roughness_texture;  // Metallic in B, Roughness in G

  // Emission (for Area Lights).
  Eigen::Vector3f emissive_factor = Eigen::Vector3f::Zero();
  float emissive_strength = 0.f;
  std::optional<Texture> emissive_texture;

  // Additive (order-independent) transparency: every SH_material_layers stage
  // blends with dst factor GL_ONE (flames, glows). Such a material is a
  // non-occluding emitter in the bake -- excluded from the ray-traced occluder
  // scene (BuildBVH) and routed through the area-light path via emissive_*.
  bool additive = false;

  // The Quake 3 stage stack, kept so the saver can re-emit it for the renderer.
  // Absent when the source material had none.
  std::optional<MaterialLayers> layers;
};

// Uniformly sample a direction on the hemisphere (Z-up local frame).
// u1, u2 are uniform random numbers in [0, 1).
// Output of material sampling
struct ReflectionSample {
  Eigen::Vector3f direction;  // Sampled outgoing direction
  float pdf = 0.f;            // Probability density of this sample
};

// Samples an outgoing direction based on the material BRDF and reflected
// direction.
// reflected: outgoing direction, away from the surface (pointing to
// sensor/previous bounce).
ReflectionSample SampleMaterial(const Material& mat, const Eigen::Vector2f& uv,
                                const Eigen::Vector3f& normal,
                                const Eigen::Vector3f& reflected,
                                std::mt19937& rng);

// Evaluates the material BRDF f_r(p, wr, wi).
// Returns the BRDF value (color).
// incident: incoming direction, away from the surface (pointing to light
// sources/next bounce).
// reflected: outgoing direction, away from the surface (pointing to
// sensor/previous bounce).
Eigen::Vector3f EvalMaterial(const Material& mat, const Eigen::Vector2f& uv,
                             const Eigen::Vector3f& normal,
                             const Eigen::Vector3f& incident,
                             const Eigen::Vector3f& reflected);

// Samples an outgoing direction based on the material BRDF and reflected
// direction.
// reflected: outgoing direction, away from the surface (pointing to
// sensor/previous bounce).
ReflectionSample SampleMaterialAdvanced(const Material& mat,
                                        const Eigen::Vector2f& uv,
                                        const Eigen::Vector3f& normal,
                                        const Eigen::Vector3f& reflected,
                                        std::mt19937& rng);

// Evaluates the material BRDF f_r(p, wr, wi).
// Returns the BRDF value (color).
// incident: incoming direction, away from the surface (pointing to light
// sources/next bounce).
// reflected: outgoing direction, away from the surface (pointing to
// sensor/previous bounce).
Eigen::Vector3f EvalMaterialAdvanced(const Material& mat,
                                     const Eigen::Vector2f& uv,
                                     const Eigen::Vector3f& normal,
                                     const Eigen::Vector3f& incident,
                                     const Eigen::Vector3f& reflected);

// Helper to retrieve albedo from texture or default.
Eigen::Vector3f GetAlbedo(const Material& mat, const Eigen::Vector2f& uv);

// Helper to retrieve emission (radiance).
Eigen::Vector3f GetEmission(const Material& mat, const Eigen::Vector2f& uv);

// Returns the alpha (transparency) value at the given UV coordinate.
// Returns 1.0f if the texture has no alpha channel.
float GetAlpha(const Material& mat, const Eigen::Vector2f& uv);

// Helper to retrieve metallic and roughness.
void GetMetallicRoughness(const Material& mat, const Eigen::Vector2f& uv,
                          float& metallic, float& roughness);

}  // namespace sh_baker

#endif  // SH_BAKER_SRC_MATERIAL_H_
