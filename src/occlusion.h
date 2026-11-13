#ifndef SH_BAKER_SRC_OCCLUSION_H_
#define SH_BAKER_SRC_OCCLUSION_H_

#include <embree4/rtcore.h>

#include <Eigen/Dense>
#include <optional>
#include <vector>

namespace sh_baker {

struct Ray {
  Eigen::Vector3f origin;
  Eigen::Vector3f direction;
  float tnear = 0.0f;
  float tfar = std::numeric_limits<float>::infinity();
};

struct Occlusion {
  Eigen::Vector3f position;
  Eigen::Vector2f uv;
  Eigen::Vector3f normal;
  // < 0 means the hit is a material-less pure occluder (a solidified shell);
  // the tracer blocks transport there without shading.
  int material_id;
  // Ray parameter of the hit: position == origin + t * direction.
  float t;
};

struct Material;

// Returns occlusion information if the ray hits the scene.
// "scene" must be a committed RTCScene.
std::optional<Occlusion> FindOcclusion(RTCScene scene, const Ray& ray);

// The most surface crossings with alpha below 1 that FindTransmittance walks
// before it counts the ray as blocked.
inline constexpr int kMaxTransmittanceCrossings = 64;

// Returns the fraction of light that passes along `ray` between its tnear and
// tfar: the product of (1 - alpha) over every surface it crosses, where alpha
// is GetAlpha() at the hit, the coverage the path tracer's indirect rays read.
// A material-less hit (a pure occluder), a hit with alpha 1, or more than
// kMaxTransmittanceCrossings partially covered crossings give 0. The walk is
// deterministic (no random numbers) and, for a ray that hits only opaque
// geometry or nothing, makes exactly one intersection query and returns
// exactly 0 or 1. `materials` is indexed by Geometry::material_id.
float FindTransmittance(RTCScene scene, const std::vector<Material>& materials,
                        const Ray& ray);

}  // namespace sh_baker

#endif  // SH_BAKER_SRC_OCCLUSION_H_
