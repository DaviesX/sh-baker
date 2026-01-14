#ifndef SH_BAKER_SRC_LIGHT_H_
#define SH_BAKER_SRC_LIGHT_H_

#include <embree4/rtcore.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <random>

#include "occlusion.h"
#include "scene.h"
#include "sh_coeffs.h"

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

struct DirectionalLightIncoming {
  Eigen::Vector3f radiance;
  float cos_n;
  Ray visibility_ray;
};

inline DirectionalLightIncoming DirectionalLightIncomingRadiance(
    const Light& light, const Eigen::Vector3f& P, const Eigen::Vector3f& N,
    Ray* visibility_ray) {
  // Pending visibility ray/shadow ray.
  visibility_ray->origin = P;
  visibility_ray->direction = -light.direction;
  visibility_ray->tnear = 0.001f;
  visibility_ray->tfar = 1.0e10f;

  // Incoming radiance without visibility term.
  float cos_n = N.dot(-light.direction);
  if (cos_n < 0.f) {
    return DirectionalLightIncoming{Eigen::Vector3f::Zero(), cos_n};
  }
  return DirectionalLightIncoming{light.intensity * light.color, cos_n};
}

template <typename Brdf>
Eigen::Vector3f DirectionalLightRadiance(const Light& light,
                                         const Eigen::Vector3f& P,
                                         const Eigen::Vector3f& N, Brdf brdf,
                                         Ray* visibility_ray) {
  DirectionalLightIncoming incoming =
      DirectionalLightIncomingRadiance(light, P, N, visibility_ray);
  if (incoming.cos_n < 0.f) {
    return Eigen::Vector3f::Zero();
  }

  Eigen::Vector3f cos_brdf = incoming.cos_n * brdf(-light.direction);
  return incoming.radiance.cwiseProduct(cos_brdf);
}

struct PointLightIncoming {
  Eigen::Vector3f radiance;
  Eigen::Vector3f incident;
  float cos_n;
};

inline PointLightIncoming PointLightIncomingRadiance(const Light& light,
                                                     const Eigen::Vector3f& P,
                                                     const Eigen::Vector3f& N,
                                                     Ray* visibility_ray) {
  Eigen::Vector3f L = light.position - P;
  float dist_sq = L.squaredNorm();
  float dist = std::sqrt(dist_sq);

  // Pending visibility ray/shadow ray.
  visibility_ray->origin = P;
  visibility_ray->direction = light.position - P;
  visibility_ray->tnear = 0.001f;
  visibility_ray->tfar = dist - 0.001f;

  // Incoming radiance without visibility term.
  L /= dist;

  float cos_n = N.dot(L);
  if (cos_n < 0.f) {
    return PointLightIncoming{Eigen::Vector3f::Zero(), L, cos_n};
  }
  return PointLightIncoming{light.intensity * light.color / (dist * dist), L,
                            cos_n};
}

template <typename Brdf>
Eigen::Vector3f PointLightRadiance(const Light& light, const Eigen::Vector3f& P,
                                   const Eigen::Vector3f& N, Brdf brdf,
                                   Ray* visibility_ray) {
  PointLightIncoming incoming =
      PointLightIncomingRadiance(light, P, N, visibility_ray);
  if (incoming.cos_n < 0.f) {
    return Eigen::Vector3f::Zero();
  }

  Eigen::Vector3f cos_brdf = incoming.cos_n * brdf(incoming.incident);
  return incoming.radiance.cwiseProduct(cos_brdf);
}

struct SpotLightIncoming {
  Eigen::Vector3f radiance;
  Eigen::Vector3f incident;
  float cos_n;
};

inline SpotLightIncoming SpotLightIncomingRadiance(const Light& light,
                                                   const Eigen::Vector3f& P,
                                                   const Eigen::Vector3f& N,
                                                   Ray* visibility_ray) {
  Eigen::Vector3f L = light.position - P;
  float dist_sq = L.squaredNorm();
  float dist = std::sqrt(dist_sq);

  // Pending visibility ray/shadow ray.
  visibility_ray->origin = P;
  visibility_ray->direction = light.position - P;
  visibility_ray->tnear = 0.001f;
  visibility_ray->tfar = dist - 0.001f;

  // Incoming radiance without visibility term.
  L /= dist;

  float cos_n = N.dot(L);
  float cos_l = light.direction.dot(-L);
  float falloff = (cos_l - light.cos_outer_cone) /
                  (light.cos_inner_cone - light.cos_outer_cone);
  falloff = std::clamp(falloff, 0.0f, 1.0f);
  if (cos_n < 0.f) {
    return SpotLightIncoming{Eigen::Vector3f::Zero(), L, cos_n};
  }
  return SpotLightIncoming{
      light.intensity * light.color / (dist * dist) * falloff, L, cos_n};
}

template <typename Brdf>
Eigen::Vector3f SpotLightRadiance(const Light& light, const Eigen::Vector3f& P,
                                  const Eigen::Vector3f& N, Brdf brdf,
                                  Ray* visibility_ray) {
  SpotLightIncoming incoming =
      SpotLightIncomingRadiance(light, P, N, visibility_ray);
  if (incoming.cos_n < 0.f) {
    return Eigen::Vector3f::Zero();
  }

  Eigen::Vector3f cos_brdf = incoming.cos_n * brdf(incoming.incident);
  return incoming.radiance.cwiseProduct(cos_brdf);
}

struct AreaLightIncoming {
  Eigen::Vector3f radiance;
  Eigen::Vector3f incident;
  float cos_n;
};

inline AreaLightIncoming AreaLightIncomingRadiance(const AreaSample& sample,
                                                   const Eigen::Vector3f& P,
                                                   const Eigen::Vector3f& N,
                                                   Ray* visibility_ray) {
  Eigen::Vector3f L = sample.point - P;
  float dist_sq = L.squaredNorm();
  float dist = std::sqrt(dist_sq);

  // Pending visibility ray/shadow ray.
  visibility_ray->origin = P;
  visibility_ray->direction = sample.point - P;
  visibility_ray->tnear = 0.001f;
  visibility_ray->tfar = dist - 0.001f;

  // Incoming radiance without visibility term.
  L /= dist;

  float cos_n = N.dot(L);
  if (cos_n < 0.f) {
    return AreaLightIncoming{Eigen::Vector3f::Zero(), L, cos_n};
  }
  return AreaLightIncoming{sample.radiance / (dist * dist), L, cos_n};
}

template <typename Brdf>
Eigen::Vector3f AreaLightRadiance(const AreaSample& sample,
                                  const Eigen::Vector3f& P,
                                  const Eigen::Vector3f& N, Brdf brdf,
                                  Ray* visibility_ray) {
  AreaLightIncoming incoming =
      AreaLightIncomingRadiance(sample, P, N, visibility_ray);
  if (incoming.cos_n < 0.f) {
    return Eigen::Vector3f::Zero();
  }

  Eigen::Vector3f cos_brdf = incoming.cos_n * brdf(incoming.incident);
  return incoming.radiance.cwiseProduct(cos_brdf);
}

}  // namespace light_internal

// Evaluates the sun and samples `num_samples` lights based on the distribution
// formed from the heuristic:
//  score = L(sample) * brdf * \cos \theta / dist^2.
//
// Then, it computes the radiance L_e(x) combined with the geometric visibility
// term. The estimated radiance is of lower variance if the lights set is
// potentially visible.
Eigen::Vector3f EvaluateLightSamples(const Scene& scene, RTCScene rtc_scene,
                                     const Eigen::Vector3f& hit_point,
                                     const Eigen::Vector3f& hit_point_normal,
                                     const Eigen::Vector3f& reflected,
                                     const Material& mat,
                                     const Eigen::Vector2f& uv,
                                     unsigned num_samples, std::mt19937& rng);

// Computes the direct lighting (Next Event Estimation) and accumulates the
// projected SH coefficients into the accumulator.
//
// Because SH projection depends on the direction of the incident light, we
// cannot simply return a summed radiance covering multiple light sources.
// Instead, we must project each light sample into the SH basis using its
// specific incoming direction.
void AccumulateIncomingLightSamples(const Scene& scene, RTCScene rtc_scene,
                                    const Eigen::Vector3f& hit_point,
                                    const Eigen::Vector3f& hit_point_normal,
                                    unsigned num_samples, std::mt19937& rng,
                                    SHCoeffs* accumulator);

}  // namespace sh_baker

#endif  // SH_BAKER_SRC_LIGHT_H_
