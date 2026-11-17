#ifndef SH_BAKER_SRC_OCCLUSION_H_
#define SH_BAKER_SRC_OCCLUSION_H_

#include <embree4/rtcore.h>

#include <Eigen/Dense>
#include <optional>
#include <vector>

#include "material.h"

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

// A material-aware surface hit. Material-less occluders have alpha 1.
struct OcclusionWithAlpha {
  Occlusion hit;
  float alpha;
};

// Returns occlusion information if the ray hits the scene.
// "scene" must be a committed RTCScene.
std::optional<Occlusion> FindOcclusion(RTCScene scene, const Ray& ray);

// Like the above, but it attaches the alpha transparency on to the hit.
std::optional<OcclusionWithAlpha> FindOcclusion(
    RTCScene scene, const std::vector<Material>& scene_materials,
    const Ray& ray);

// Continue along the same ray just beyond the hit, retaining its far bound.
Ray ContinuePastOcclusion(const Ray& ray, const Occlusion& hit);

// Start a new ray at the hit, continuing in the same direction.
Ray NewPastOcclusion(const Ray& ray, const Occlusion& hit);

}  // namespace sh_baker

#endif  // SH_BAKER_SRC_OCCLUSION_H_
