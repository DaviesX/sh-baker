#include "light.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

#include "material.h"
#include "occlusion.h"

namespace sh_baker {

namespace light_internal {

namespace {

// Returns true if visible.
bool CheckVisibility(RTCScene rtc_scene, const Eigen::Vector3f& origin,
                     const Eigen::Vector3f& target, float t_max = 1.0e10f) {
  Eigen::Vector3f dir_vec = target - origin;
  float dist = dir_vec.norm();
  if (dist < 1e-6f) return true;

  Eigen::Vector3f dir = dir_vec / dist;

  // Shadow ray
  // tnear epsilon to avoid self-intersection
  // tfar = dist - epsilon
  Ray ray;
  ray.origin = origin;
  ray.direction = dir;
  ray.tnear = 0.001f;
  ray.tfar = std::max(0.0f, dist - 0.001f);

  // IsOccluded returns true if occluded
  return !FindOcclusion(rtc_scene, ray);
}

// For directional lights (Sun/Dir), target is "infinitely far".
bool CheckVisibilityDir(RTCScene rtc_scene, const Eigen::Vector3f& origin,
                        const Eigen::Vector3f& dir) {
  Ray ray;
  ray.origin = origin;
  ray.direction = dir;
  ray.tnear = 0.001f;
  ray.tfar = 1.0e10f;  // Infinite
  return !FindOcclusion(rtc_scene, ray);
}

}  // namespace
}  // namespace light_internal

Eigen::Vector3f EvaluateLights(
    const SkyModel& sky_model,
    const std::vector<light_internal::LightSample>& light_samples,
    const Eigen::Vector3f& hit_point, const Eigen::Vector3f& hit_point_normal,
    const Eigen::Vector3f& reflected, const Material& mat,
    const Eigen::Vector2f& uv, RTCScene rtc_scene) {
  using namespace light_internal;
  Eigen::Vector3f total_radiance = Eigen::Vector3f::Zero();

  // 1. Sun (Implicitly sampled, always evaluated)
  if (sky_model.sun_intensity > 0.0f) {
    float n_dot_l = hit_point_normal.dot(sky_model.sun_direction);
    if (n_dot_l > 0.0f) {
      if (CheckVisibilityDir(rtc_scene, hit_point, sky_model.sun_direction)) {
        Eigen::Vector3f brdf = EvalMaterial(mat, uv, hit_point_normal,
                                            reflected, sky_model.sun_direction);
        total_radiance += sky_model.sun_color.cwiseProduct(brdf) *
                          sky_model.sun_intensity * n_dot_l;
      }
    }
  }

  // 2. Sampled Lights
  for (const auto& sample : light_samples) {
    if (!sample.light || sample.pdf < 1e-9f) continue;
    const Light& light = *sample.light;

    Eigen::Vector3f L;  // Vector to light
    float dist_sq;
    float dist;

    // Calc Direction and Distance
    if (light.type == Light::Type::Directional) {
      // Directional lights handled separately?
      // User prompt implies "Sample Punctual... Sample Area". Directional
      // usually Sun. But if there are other directional lights in scene list?
      // Assuming Scene::lights vector contains local lights, Scene::sky has
      // Sun. If a directional light is in Scene::lights, we handle it:
      Eigen::Vector3f dir =
          -light.direction.normalized();  // direction is emitting direction
      // Check alignment?
      L = dir * 1e5f;  // Arbitrary large distance for calc
      dist_sq = 1.0f;  // No falloff for directional
      dist = 1e5f;
    } else if (light.type == Light::Type::Point ||
               light.type == Light::Type::Spot) {
      L = light.position - hit_point;
      dist_sq = L.squaredNorm();
      dist = std::sqrt(dist_sq);
    } else if (light.type == Light::Type::Area) {
      // Approximating Area Light as Point at Center
      L = light.center - hit_point;
      dist_sq = L.squaredNorm();
      dist = std::sqrt(dist_sq);
    } else {
      continue;
    }

    if (dist < 1e-6f) continue;
    Eigen::Vector3f dir = L / dist;

    float cos_theta = hit_point_normal.dot(dir);
    if (cos_theta <= 0.0f) continue;

    // Evaluate Light Intensity/Radiance
    Eigen::Vector3f Li = Eigen::Vector3f::Zero();

    if (light.type == Light::Type::Directional) {
      if (CheckVisibilityDir(rtc_scene, hit_point, dir)) {
        Li = light.color * light.intensity;
      }
    } else if (light.type == Light::Type::Point) {
      if (CheckVisibility(rtc_scene, hit_point, light.position)) {
        Li = (light.color * light.intensity) / std::max(1e-4f, dist_sq);
      }
    } else if (light.type == Light::Type::Spot) {
      if (CheckVisibility(rtc_scene, hit_point, light.position)) {
        float spot_cos = (-dir).dot(light.direction.normalized());
        if (spot_cos > std::cos(light.outer_cone_angle)) {
          float intensity_factor = 1.0f;
          // TODO: Smooth attenuation between inner and outer
          // For now step
          Li = (light.color * light.intensity) / std::max(1e-4f, dist_sq);
        }
      }
    } else if (light.type == Light::Type::Area) {
      if (CheckVisibility(rtc_scene, hit_point, light.center)) {
        // L_e * cos_light / dist^2 * Area ?
        // User heuristic: W_i approx L_e * A * cos_l * ...
        // Actual Contribution: L_e * cos_theta_light / dist^2 * Area?
        // Or if we treat it as Punctual with defined Flux:
        // Flux = pi * A * L_e. Radiance at P = (Flux / pi) * cos_light /
        // dist^2? Actually, for Area light sampled as point: Contribution = Le
        // * cos_theta_light / dist^2 * Area_of_light (if PDF is 1/Area) But we
        // are selecting the light with Prob P_select. If we treat it as a Point
        // source with intensity I = Le * Area * cos_theta_light?

        float cos_light = (-dir).dot(light.normal);
        if (cos_light > 0.0f) {
          // Li incoming = Le * cos_light / dist^2 * Area?
          // Light.color is Le (normalized?) * Intensity.
          // Let's use Flux-based formulation or stick to simple
          // Intensity/Distance Light.flux was computed as pi * A * I. So I ~
          // Flux / (pi * A). Incoming Radiance dL = Le * cos_light * dA /
          // dist^2. We are approximating the integration over area A as: Le *
          // cos_light * A / dist^2.

          // light.color is color. light.intensity is magnitude of Le.
          Eigen::Vector3f Le = light.color * light.intensity;

          Li = Le * (light.area * cos_light / std::max(1e-4f, dist_sq));
        }
      }
    }

    if (!Li.isZero()) {
      // Contribution: Li * f_r * cos_theta / PDF
      Eigen::Vector3f brdf =
          EvalMaterial(mat, uv, hit_point_normal, reflected, dir);
      total_radiance += Li.cwiseProduct(brdf) * (cos_theta / sample.pdf);
    }
  }

  return total_radiance;
}

std::vector<light_internal::LightSample> SampleLights(const Scene& scene,
                                                      const Eigen::Vector3f& P,
                                                      const Eigen::Vector3f& N,
                                                      unsigned num_samples,
                                                      std::mt19937& rng) {
  using namespace light_internal;
  std::vector<LightSample> samples;
  if (scene.lights.empty()) return samples;

  // Build Weights
  std::vector<float> weights;
  weights.reserve(scene.lights.size());

  for (const auto& light : scene.lights) {
    float weight = 0.0f;
    float dist_sq = 1.0f;
    float cos_light = 1.0f;
    float cos_surface = 1.0f;

    // Calculate Surface Cosine and Distance
    // Note: To be strictly correct, we should use the sample point on light.
    // Here we use light position/center.
    Eigen::Vector3f L_vec;
    if (light.type == Light::Type::Area) {
      L_vec = light.center - P;
      // Cos Light
      if (L_vec.squaredNorm() > 1e-6f)
        cos_light = std::max(0.0f, (-L_vec.normalized()).dot(light.normal));
    } else if (light.type == Light::Type::Point ||
               light.type == Light::Type::Spot) {
      L_vec = light.position - P;
    } else {
      // Directional
      L_vec = -light.direction;  // Infinite
      dist_sq = 1.0f;
    }

    if (light.type != Light::Type::Directional) {
      dist_sq = std::max(1e-4f, L_vec.squaredNorm());
      cos_surface = std::max(0.0f, L_vec.normalized().dot(N));
    } else {
      cos_surface = std::max(0.0f, (-light.direction.normalized()).dot(N));
    }

    // Heuristic: W = Flux * cos_l * cos_s / dist^2
    // Use Light.flux if available, else derive from intensity
    float flux = light.flux;
    if (flux <= 0.0f) {
      // Fallback for non-area lights
      flux = light.intensity * 4.0f * (float)M_PI;        // Isotropic point
      if (light.type == Light::Type::Spot) flux *= 0.2f;  // simple scale
      if (light.type == Light::Type::Directional)
        flux = light.intensity * 10.0f;  // arbitrary
    }

    weight = (flux * cos_light * cos_surface) / (dist_sq + 0.01f);
    weights.push_back(weight);
  }

  // Create Distribution
  // If all weights zero, uniform?
  float sum_weights = std::accumulate(weights.begin(), weights.end(), 0.0f);
  if (sum_weights < 1e-9f) {
    // Uniform fallback
    std::fill(weights.begin(), weights.end(), 1.0f);
    sum_weights = weights.size();
  }

  std::discrete_distribution<int> dist(weights.begin(), weights.end());

  for (unsigned i = 0; i < num_samples; ++i) {
    int idx = dist(rng);
    LightSample sample;
    sample.light = &scene.lights[idx];
    // PMF of selecting this light = weight[idx] / sum
    sample.pdf = weights[idx] / sum_weights;
    samples.push_back(sample);
  }

  return samples;
}

}  // namespace sh_baker
