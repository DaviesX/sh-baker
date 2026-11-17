#include "tracer.h"

#include "light.h"
#include "material.h"
#include "occlusion.h"

namespace sh_baker {

Eigen::Vector3f Trace(const TraceConfig& config, const Ray& ray, int depth,
                      std::mt19937& rng, Eigen::Vector3f throughput) {
  // Hard depth limit to prevent infinite recursion, but we rely on RR for
  // unbiased early termination.
  if (depth > config.max_depth) return Eigen::Vector3f::Zero();

  Ray visibility_ray = ray;
  visibility_ray.tnear = 0.001f;
  std::optional<OcclusionWithAlpha> surface =
      FindOcclusion(config.rtc_scene, config.scene.materials, visibility_ray);

  if (!surface.has_value()) {
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
  if (surface->hit.material_id < 0) {
    return Eigen::Vector3f::Zero();
  }

  Eigen::Vector3f color = Eigen::Vector3f::Zero();

  // If alpha < 1.0, continue ray
  if (surface->alpha < 1.0f) {
    // Transmission
    Ray transmission_ray = NewPastOcclusion(ray, surface->hit);
    Eigen::Vector3f transmission =
        Trace(config, transmission_ray, depth + 1, rng,
              throughput * (1.0f - surface->alpha));

    color += (1.0f - surface->alpha) * transmission;
    if (surface->alpha < 0.1f) {
      // If alpha is very small, we can skip the rest of the trace.
      return color;
    }
  }

  // Direct Lighting (NEE)
  // EvaluateLights returns L_e(x, x')
  const Material& surface_mat =
      config.scene.materials[surface->hit.material_id];

  Eigen::Vector3f L_direct = EvaluateLightSamples(
      config.light_tree, config.rtc_scene, config.scene.materials,
      surface->hit.position, surface->hit.normal, -ray.direction, surface_mat,
      surface->hit.uv, config.num_light_samples, rng);

  color += surface->alpha * L_direct;

  // Indirect Lighting (Recursive)
  ReflectionSample sample = SampleMaterial(
      surface_mat, surface->hit.uv, surface->hit.normal, -ray.direction, rng);
  if (sample.pdf < 1e-3f) {
    // Internal reflection.
    return color;
  }
  Ray next_ray;
  next_ray.origin = surface->hit.position;
  next_ray.tnear = 0.005f;
  next_ray.tfar = 1e10f;
  next_ray.direction = sample.direction;

  Eigen::Vector3f brdf =
      EvalMaterial(surface_mat, surface->hit.uv, surface->hit.normal,
                   sample.direction, -ray.direction);

  // Accumulated path throughput after taking this bounce: the coefficient that
  // multiplies all radiance gathered deeper in the path. brdf * cos / pdf is
  // the per-bounce factor (== surface albedo under cosine sampling), so this is
  // physically meaningful — unlike the raw single-bounce BRDF used previously.
  // Clamp to 0: hemisphere sampling should keep this positive, but float error
  // or a perturbed shading normal can push it slightly negative, which would
  // flip the throughput sign and corrupt RR / radiance.
  float cosine_term = std::max(0.0f, surface->hit.normal.dot(sample.direction));
  Eigen::Vector3f next_throughput = throughput.cwiseProduct(brdf) *
                                    (surface->alpha * cosine_term / sample.pdf);

  if (depth > 2) {
    // Russian Roulette on the *accumulated throughput* (reflectance-based). The
    // old scheme rolled on brdf.maxCoeff() (~albedo/pi), which underestimated
    // survival by the pi factor and over-amplified survivors by 1/q — a direct
    // firefly source in bounce-heavy indoor scenes.
    float q = std::min(0.95f, next_throughput.maxCoeff());
    if (q <= 0.0f) return color;

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    if (dist(rng) > q) {
      // Terminate
      return color;
    }
    // Survived: reweight both the local estimator (via pdf) and the throughput
    // carried into the recursion so deeper RR sees the boosted magnitude.
    sample.pdf *= q;
    next_throughput /= q;
  }
  Eigen::Vector3f incoming =
      Trace(config, next_ray, depth + 1, rng, next_throughput);
  Eigen::Vector3f L_indirect =
      incoming.cwiseProduct(brdf) * (cosine_term / sample.pdf);

  // Firefly clamp: a safe backstop against the rare heavy-tail spike. The
  // output is low-frequency diffuse SH, so scaling down over-bright samples
  // costs negligible visible bias.
  if (config.firefly_clamp > 0.0f) {
    float m = L_indirect.maxCoeff();
    if (m > config.firefly_clamp) {
      L_indirect *= config.firefly_clamp / m;
    }
  }
  color += surface->alpha * L_indirect;

  return color;
}

}  // namespace sh_baker
