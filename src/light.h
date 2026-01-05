#ifndef SH_BAKER_SRC_LIGHT_H_
#define SH_BAKER_SRC_LIGHT_H_

#include <embree4/rtcore.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <optional>
#include <random>
#include <vector>

#include "occlusion.h"
#include "scene.h"

namespace sh_baker {
namespace light_internal {

struct AreaSample {
  Eigen::Vector3f point;
  Eigen::Vector3f normal;
  Eigen::Vector3f radiance;
  float pdf = 0.f;
};

// Samples a point on the area light geometry.
AreaSample SampleAreaLight(const Light& light, std::mt19937& rng);

template <typename Brdf>
Eigen::Vector3f DirectionalLightRadiance(const Light& light,
                                         const Eigen::Vector3f& P,
                                         const Eigen::Vector3f& N, Brdf brdf,
                                         Ray* visibility_ray) {
  // Pending visibility ray/shadow ray.
  visibility_ray->origin = P;
  visibility_ray->direction = -light.direction;
  visibility_ray->tnear = 0.001f;
  visibility_ray->tfar = 1.0e10f;

  // Radiance without visibility term.
  float cos_n = std::max(0.0f, N.dot(-light.direction));
  return light.intensity * cos_n *
         light.color.cwiseProduct(brdf(-light.direction));
}

template <typename Brdf>
Eigen::Vector3f PointLightRadiance(const Light& light, const Eigen::Vector3f& P,
                                   const Eigen::Vector3f& N, Brdf brdf,
                                   Ray* visibility_ray) {
  Eigen::Vector3f L = light.position - P;
  float dist_sq = L.squaredNorm();
  float dist = std::sqrt(dist_sq);

  // Pending visibility ray/shadow ray.
  visibility_ray->origin = P;
  visibility_ray->direction = light.position - P;
  visibility_ray->tnear = 0.001f;
  visibility_ray->tfar = dist - 0.001f;

  // Radiance without visibility term.
  L /= dist;
  float cos_n = std::max(0.0f, N.dot(L));
  return (light.intensity * cos_n / (dist * dist)) *
         light.color.cwiseProduct(brdf(L));
}

template <typename Brdf>
Eigen::Vector3f SpotLightRadiance(const Light& light, const Eigen::Vector3f& P,
                                  const Eigen::Vector3f& N, Brdf brdf,
                                  Ray* visibility_ray) {
  Eigen::Vector3f L = light.position - P;
  float dist_sq = L.squaredNorm();
  float dist = std::sqrt(dist_sq);

  // Pending visibility ray/shadow ray.
  visibility_ray->origin = P;
  visibility_ray->direction = light.position - P;
  visibility_ray->tnear = 0.001f;
  visibility_ray->tfar = dist - 0.001f;

  // Radiance without visibility term.
  L /= dist;
  float cos_l = light.direction.dot(-L);
  float falloff = (cos_l - light.cos_outer_cone) /
                  (light.cos_inner_cone - light.cos_outer_cone);
  falloff = std::clamp(falloff, 0.0f, 1.0f);
  float cos_n = std::max(0.0f, N.dot(L));
  return (light.intensity * falloff * cos_n / (dist * dist)) *
         light.color.cwiseProduct(brdf(L));
}

template <typename Brdf>
Eigen::Vector3f AreaLightRadiance(const AreaSample& sample,
                                  const Eigen::Vector3f& P,
                                  const Eigen::Vector3f& N, Brdf brdf,
                                  Ray* visibility_ray) {
  Eigen::Vector3f L = sample.point - P;
  float dist_sq = L.squaredNorm();
  float dist = std::sqrt(dist_sq);

  // Pending visibility ray/shadow ray.
  visibility_ray->origin = P;
  visibility_ray->direction = sample.point - P;
  visibility_ray->tnear = 0.001f;
  visibility_ray->tfar = dist - 0.001f;

  L /= dist;
  float cos_n = std::max(0.0f, N.dot(L));
  float cos_l = std::max(0.0f, sample.normal.dot(-L));
  return (cos_n * cos_l / (dist * dist)) *
         sample.radiance.cwiseProduct(brdf(L));
}

}  // namespace light_internal

// Evaluates the sun and samples `num_samples` lights based on the distribution
// formed from the heuristic:
//  score = L(sample) * brdf * \cos \theta / dist^2.
//
// Then, it computes the radiance L_e(x) combined with the visibility term. The
// estimated radiance is of lower variance if the lights set is potentially
// visible.
Eigen::Vector3f EvaluateLightSamples(
    const SkyModel& sky, const std::vector<Light>& lights, RTCScene rtc_scene,
    const Eigen::Vector3f& hit_point, const Eigen::Vector3f& hit_point_normal,
    const Eigen::Vector3f& reflected, const Material& mat,
    const Eigen::Vector2f& uv, unsigned num_samples, std::mt19937& rng);

}  // namespace sh_baker

#endif  // SH_BAKER_SRC_LIGHT_H_
