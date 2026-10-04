#ifndef SH_BAKER_SRC_MATERIAL_H_
#define SH_BAKER_SRC_MATERIAL_H_

#include <Eigen/Dense>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "material_layers.h"
#include "texture.h"

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

  // Verbatim SH_material_layers extension, retained so the saver can re-emit it
  // for the renderer. Absent when the source material had no extension.
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
