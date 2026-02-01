#include "tracer.h"

#include <cmath>
#include <iostream>

#include "light.h"
#include "material.h"
#include "occlusion.h"

namespace sh_baker {

Eigen::Vector3f SampleHemisphereUniform(std::mt19937& rng) {
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  float u1 = dist(rng);
  float u2 = dist(rng);

  float r = std::sqrt(1.0f - u1 * u1);
  float phi = 2.0f * M_PI * u2;
  return Eigen::Vector3f(r * std::cos(phi), r * std::sin(phi), u1);
}

Eigen::Vector3f Trace(const TraceConfig& config, const Eigen::Vector3f& origin,
                      const Eigen::Vector3f& dir, int depth,
                      std::mt19937& rng) {
  // Hard depth limit to prevent infinite recursion, but we rely on RR for
  // unbiased early termination.
  if (depth > config.max_depth) return Eigen::Vector3f::Zero();

  Ray visibility_ray;
  visibility_ray.origin = origin;
  visibility_ray.direction = dir;
  visibility_ray.tnear = 0.001f;

  std::optional<Occlusion> occ =
      FindOcclusion(config.rtc_scene, visibility_ray);

  if (!occ.has_value()) {
    // Sky is handled by NEE (EvaluateLights() and
    // EvaluateIncomingLightSamples()), so we excluded it here to avoid double
    // counting.
    if (depth == 0) {
      config.on_direct_hit_sky_fn();
    }
    return Eigen::Vector3f::Zero();
  }

  // Hit surface
  const Material& mat = config.scene.materials[occ->material_id];
  float alpha = GetAlpha(mat, occ->uv);

  Eigen::Vector3f color = Eigen::Vector3f::Zero();

  // If alpha < 1.0, continue ray
  if (alpha < 1.0f) {
    // Transmission
    Eigen::Vector3f hit_pos = occ->position + dir * 0.001f;
    Eigen::Vector3f transmission = Trace(config, hit_pos, dir, depth + 1, rng);
    color += (1.0f - alpha) * transmission;
    if (alpha < 0.1f) {
      // If alpha is very small, we can skip the rest of the trace.
      return color;
    }
  }

  Eigen::Vector3f hit_pos = occ->position + occ->normal * 0.005f;

  // Direct Lighting (NEE)
  // EvaluateLights returns L_e(x, x')
  Eigen::Vector3f L_direct =
      EvaluateLightSamples(config.scene, config.rtc_scene, hit_pos, occ->normal,
                           -dir, mat, occ->uv, config.num_light_samples, rng);
  color += alpha * L_direct;

  // Indirect Lighting (Recursive)
  ReflectionSample sample =
      SampleMaterial(mat, occ->uv, occ->normal, -dir, rng);
  if (sample.pdf < 1e-3f) {
    // Internal reflection.
    return color;
  }

  Eigen::Vector3f brdf =
      EvalMaterial(mat, occ->uv, occ->normal, sample.direction, -dir);
  if (depth > 2) {
    // Russian Roulette
    // We want to terminate paths with low contribution.
    // The contribution of the next bounce is roughly proportional to the
    // BRDF/pdf. (We use a simplified throughput estimation here).
    float q = std::min(0.95f, brdf.maxCoeff());

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    if (dist(rng) > q) {
      // Terminate
      return color;
    }
    // Survived, reweight
    sample.pdf *= q;
  }
  Eigen::Vector3f incoming =
      Trace(config, hit_pos, sample.direction, depth + 1, rng);
  float cosine_term = occ->normal.dot(sample.direction);
  Eigen::Vector3f L_indirect =
      incoming.cwiseProduct(brdf) * (cosine_term / sample.pdf);
  color += alpha * L_indirect;

  return color;
}

}  // namespace sh_baker
