#ifndef SH_BAKER_SRC_OCCLUSION_H_
#define SH_BAKER_SRC_OCCLUSION_H_

#include <embree4/rtcore.h>

#include <Eigen/Dense>
#include <optional>

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
  int material_id;
};

// Returns occlusion information if the ray hits the scene.
// "scene" must be a committed RTCScene.
std::optional<Occlusion> FindOcclusion(RTCScene scene, const Ray& ray);

}  // namespace sh_baker

#endif  // SH_BAKER_SRC_OCCLUSION_H_
