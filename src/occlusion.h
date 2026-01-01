#ifndef SH_BAKER_SRC_OCCLUSION_H_
#define SH_BAKER_SRC_OCCLUSION_H_

#include <embree4/rtcore.h>

#include <Eigen/Dense>

namespace sh_baker {

struct Ray {
  Eigen::Vector3f origin;
  Eigen::Vector3f direction;
  float tnear = 0.0f;
  float tfar = std::numeric_limits<float>::infinity();
};

// Returns true if the ray is occluded by the scene.
// "scene" must be a committed RTCScene.
bool IsOccluded(RTCScene scene, const Ray& ray);

}  // namespace sh_baker

#endif  // SH_BAKER_SRC_OCCLUSION_H_
