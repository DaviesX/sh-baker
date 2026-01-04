#ifndef SH_BAKER_SRC_LIGHT_H_
#define SH_BAKER_SRC_LIGHT_H_

#include <embree4/rtcore.h>

#include <Eigen/Dense>
#include <random>
#include <vector>

#include "scene.h"

namespace sh_baker {

namespace light_internal {

struct LightSample {
  const Light* light = nullptr;
  float pdf = 0.0f;
};

}  // namespace light_internal

// Evaluates the incoming radiance from the set of samples lights + the Sun.
// Handles occlusion internally.
Eigen::Vector3f EvaluateLights(
    const SkyModel& sky_model,
    const std::vector<light_internal::LightSample>& light_samples,
    const Eigen::Vector3f& hit_point, const Eigen::Vector3f& hit_point_normal,
    const Eigen::Vector3f& reflected, const Material& mat,
    const Eigen::Vector2f& uv, RTCScene rtc_scene);

// Samples lights in the scene based on a heuristic (flux / dist^2).
// Returns a set of lights to be evaluated.
// Note: The Sun is strictly handled in EvaluateLights and is not sampled here.
std::vector<light_internal::LightSample> SampleLights(const Scene& scene,
                                                      const Eigen::Vector3f& P,
                                                      const Eigen::Vector3f& N,
                                                      unsigned num_samples,
                                                      std::mt19937& rng);

}  // namespace sh_baker

#endif  // SH_BAKER_SRC_LIGHT_H_
