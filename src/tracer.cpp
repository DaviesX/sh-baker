#include "tracer.h"

#include <cmath>

#include "light.h"
#include "material.h"
#include "occlusion.h"

namespace sh_baker {

Eigen::Vector3f Trace(const TraceConfig& config, const Ray& ray, int depth,
                      std::mt19937& rng) {
  // Hard depth limit to prevent infinite recursion, but we rely on RR for
  // unbiased early termination.
  if (depth > config.max_depth) return Eigen::Vector3f::Zero();

  Ray visibility_ray = ray;
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

  // A material-less hit is a pure occluder shell (a solidified thin wall): it
  // blocks transport — the ray stops here — but has no surface to shade, so it
  // contributes nothing. This both stops the light leak the shell exists to
  // prevent and avoids indexing materials[-1] below.
  if (occ->material_id < 0) return Eigen::Vector3f::Zero();

  // Hit surface
  const Material& mat = config.scene.materials[occ->material_id];
  float alpha = GetAlpha(mat, occ->uv);

  Eigen::Vector3f color = Eigen::Vector3f::Zero();

  // If alpha < 1.0, continue ray
  if (alpha < 1.0f) {
    // Transmission
    Ray transmission_ray;
    transmission_ray.tnear = 0.005f;
    transmission_ray.tfar = 1e10f;
    transmission_ray.origin = occ->position;
    transmission_ray.direction = ray.direction;
    Eigen::Vector3f transmission =
        Trace(config, transmission_ray, depth + 1, rng);

    color += (1.0f - alpha) * transmission;
    if (alpha < 0.1f) {
      // If alpha is very small, we can skip the rest of the trace.
      return color;
    }
  }

  // Direct Lighting (NEE)
  // EvaluateLights returns L_e(x, x')
  Eigen::Vector3f L_direct = EvaluateLightSamples(
      config.light_tree, config.rtc_scene, occ->position, occ->normal,
      -ray.direction, mat, occ->uv, config.num_light_samples, rng);

  color += alpha * L_direct;

  // Indirect Lighting (Recursive)
  ReflectionSample sample =
      SampleMaterial(mat, occ->uv, occ->normal, -ray.direction, rng);
  if (sample.pdf < 1e-3f) {
    // Internal reflection.
    return color;
  }
  Ray next_ray;
  next_ray.origin = occ->position;
  next_ray.tnear = 0.005f;
  next_ray.tfar = 1e10f;
  next_ray.direction = sample.direction;

  Eigen::Vector3f brdf =
      EvalMaterial(mat, occ->uv, occ->normal, sample.direction, -ray.direction);

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
  Eigen::Vector3f incoming = Trace(config, next_ray, depth + 1, rng);
  float cosine_term = occ->normal.dot(sample.direction);
  Eigen::Vector3f L_indirect =
      incoming.cwiseProduct(brdf) * (cosine_term / sample.pdf);
  color += alpha * L_indirect;

  return color;
}

}  // namespace sh_baker
