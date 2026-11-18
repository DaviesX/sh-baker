#ifndef SH_BAKER_SRC_LIGHT_H_
#define SH_BAKER_SRC_LIGHT_H_

#include <embree4/rtcore.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "light_tree.h"
#include "occlusion.h"
#include "scene.h"
#include "sh_coeffs.h"

namespace sh_baker {
namespace light_internal {

struct AreaSample {
  Eigen::Vector3f point;
  Eigen::Vector3f radiance;
  // World-space surface normal at the sampled point, used for the emitter-side
  // cosine (single-sided emission, matching the single-sided Flux()).
  Eigen::Vector3f normal = Eigen::Vector3f::UnitZ();
  float pdf = 0.f;
};

// Samples a point on the area light geometry.
AreaSample SampleAreaLight(const Light& light, std::mt19937& rng);

// True when a ranged point or spot light is farther than its range from the
// shaded point, `dist` away.
inline bool OutOfRange(const Light& light, float dist) {
  return light.range > 0.f && dist > light.range;
}

// A spot light's cone factor for cos_l, the cosine of the angle between the
// spot's direction and the direction to the shaded point (Light::ConeFalloff).
inline float SpotConeFactor(const Light& light, float cos_l) {
  const float cos_inner = light.cos_inner_cone;
  const float cos_outer = light.cos_outer_cone;
  const bool hard_edge = light.cone_falloff == Light::ConeFalloff::kSmoothstep
                             ? cos_inner <= cos_outer
                             : cos_inner == cos_outer;
  if (hard_edge) {
    return cos_l >= cos_outer ? 1.0f : 0.0f;
  }
  float t = (cos_l - cos_outer) / (cos_inner - cos_outer);
  t = std::clamp(t, 0.0f, 1.0f);
  if (light.cone_falloff == Light::ConeFalloff::kSmoothstep) {
    return t * t * (3.0f - 2.0f * t);
  }
  return t;
}

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
  visibility_ray->tnear = 0.005f;
  visibility_ray->tfar = 1.0e10f;

  // Incoming radiance without visibility term.
  float cos_n = N.dot(-light.direction);
  if (cos_n <= 0.f) {
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
  if (incoming.cos_n <= 0.f) {
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
  L /= dist;

  // Pending visibility ray/shadow ray.
  visibility_ray->origin = P;
  visibility_ray->direction = L;
  visibility_ray->tnear = 0.005f;
  visibility_ray->tfar = dist - 0.005f;

  // Incoming radiance without visibility term.
  float cos_n = N.dot(L);
  if (cos_n < 0.f || OutOfRange(light, dist)) {
    return PointLightIncoming{Eigen::Vector3f::Zero(), L, cos_n};
  }
  return PointLightIncoming{light.intensity * light.color / dist_sq, L, cos_n};
}

template <typename Brdf>
Eigen::Vector3f PointLightRadiance(const Light& light, const Eigen::Vector3f& P,
                                   const Eigen::Vector3f& N, Brdf brdf,
                                   Ray* visibility_ray) {
  PointLightIncoming incoming =
      PointLightIncomingRadiance(light, P, N, visibility_ray);
  if (incoming.cos_n <= 0.f) {
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
  L /= dist;

  // Pending visibility ray/shadow ray.
  visibility_ray->origin = P;
  visibility_ray->direction = L;
  visibility_ray->tnear = 0.005f;
  visibility_ray->tfar = dist - 0.005f;

  // Incoming radiance without visibility term.
  float cos_n = N.dot(L);
  float cos_l = light.direction.dot(-L);
  float falloff = SpotConeFactor(light, cos_l);
  if (cos_n < 0.f || OutOfRange(light, dist)) {
    return SpotLightIncoming{Eigen::Vector3f::Zero(), L, cos_n};
  }
  return SpotLightIncoming{light.intensity * light.color / dist_sq * falloff, L,
                           cos_n};
}

template <typename Brdf>
Eigen::Vector3f SpotLightRadiance(const Light& light, const Eigen::Vector3f& P,
                                  const Eigen::Vector3f& N, Brdf brdf,
                                  Ray* visibility_ray) {
  SpotLightIncoming incoming =
      SpotLightIncomingRadiance(light, P, N, visibility_ray);
  if (incoming.cos_n <= 0.f) {
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
  L /= dist;

  // Pending visibility ray/shadow ray.
  visibility_ray->origin = P;
  visibility_ray->direction = L;
  visibility_ray->tnear = 0.005f;
  visibility_ray->tfar = dist - 0.005f;

  float cos_n = N.dot(L);
  // Emitter-side cosine (the cos_light factor of the area form-factor / the
  // dA->dw Jacobian for the SH path). The previous estimator omitted it
  // entirely and emitted ISOTROPICALLY -- radiating full L_e/dist^2 in every
  // direction regardless of the emitter's orientation. That over-emits at
  // off-axis/grazing angles, which is what produced fireflies on surfaces away
  // from the light. Applying the cosine falloff is the fix.
  //
  // Kept TWO-SIDED (abs) to match the light tree, whose area-light bounds are
  // two_sided with a full-sphere emission cone (ComputeLightBounds): these Q3
  // emitters are billboard flames -- volumetric emitters faked as quads -- so
  // they radiate from both faces. Estimator and selection pdf must agree on
  // sidedness or samples are wasted / biased.
  float cos_light = std::abs(sample.normal.dot(L));
  if (cos_n < 0.f || cos_light <= 0.f) {
    return AreaLightIncoming{Eigen::Vector3f::Zero(), L, cos_n};
  }
  return AreaLightIncoming{sample.radiance * (cos_light / dist_sq), L, cos_n};
}

template <typename Brdf>
Eigen::Vector3f AreaLightRadiance(const AreaSample& sample,
                                  const Eigen::Vector3f& P,
                                  const Eigen::Vector3f& N, Brdf brdf,
                                  Ray* visibility_ray) {
  AreaLightIncoming incoming =
      AreaLightIncomingRadiance(sample, P, N, visibility_ray);
  if (incoming.cos_n <= 0.f) {
    return Eigen::Vector3f::Zero();
  }

  Eigen::Vector3f cos_brdf = incoming.cos_n * brdf(incoming.incident);
  return incoming.radiance.cwiseProduct(cos_brdf);
}

inline constexpr int kMaxTransmittanceCrossings = 64;

// Product of (1 - alpha) along the ray. Opaque and material-less surfaces
// block light, as do more than kMaxTransmittanceCrossings transparent hits.
float FindTransmittance(RTCScene scene, const std::vector<Material>& materials,
                        const Ray& ray);

}  // namespace light_internal

// Samples lights from the light tree and evaluates the radiance contribution.
// The light tree now handles all lights (including directional) so no separate
// scene iteration is needed.
//
// Evaluates `num_samples` lights based on the importance-weighted distribution
// in the light tree. Returns the estimated radiance combined with the geometric
// visibility term.
//
// Visibility is FindTransmittance() along each shadow ray: surfaces with alpha
// below 1 attenuate the sample by (1 - alpha), opaque surfaces and
// material-less occluders block it. A light whose casts_shadows is false is
// not tested and always visible. `materials` is the scene's material list,
// indexed by Geometry::material_id.
Eigen::Vector3f EvaluateLightSamples(
    const LightTree* light_tree, RTCScene rtc_scene,
    const std::vector<Material>& materials, const Eigen::Vector3f& hit_point,
    const Eigen::Vector3f& hit_point_normal, const Eigen::Vector3f& reflected,
    const Material& mat, const Eigen::Vector2f& uv, unsigned num_samples,
    std::mt19937& rng);

// Computes the direct lighting (Next Event Estimation) and accumulates the
// projected SH coefficients into the accumulator.
//
// Because SH projection depends on the direction of the incident light, we
// cannot simply return a summed radiance covering multiple light sources.
// Instead, we must project each light sample into the SH basis using its
// specific incoming direction.
//
// Visibility is as in EvaluateLightSamples().
void AccumulateIncomingLightSamples(const LightTree* light_tree,
                                    RTCScene rtc_scene,
                                    const std::vector<Material>& materials,
                                    const Eigen::Vector3f& hit_point,
                                    const Eigen::Vector3f& hit_point_normal,
                                    unsigned num_samples, std::mt19937& rng,
                                    SHCoeffs* accumulator);

// Evaluates every punctual light (point, spot, directional) in `lights` once
// at the point, with no random sampling, and accumulates the incident radiance
// E_perp * visibility of each into `accumulator`: E_perp * Y(w) into its SH
// coefficients and E_perp * max(0, n . w) into its irradiance, where w is the
// direction to the light. Range and cone falloff apply; visibility is
// FindTransmittance() for a light that casts shadows and 1 for one that does
// not. Area lights are skipped.
void AccumulatePunctualDirect(const std::vector<Light>& lights,
                              RTCScene rtc_scene,
                              const std::vector<Material>& materials,
                              const Eigen::Vector3f& hit_point,
                              const Eigen::Vector3f& hit_point_normal,
                              SHCoeffs* accumulator);

}  // namespace sh_baker

#endif  // SH_BAKER_SRC_LIGHT_H_
