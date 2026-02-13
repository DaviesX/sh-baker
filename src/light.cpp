#include "light.h"

#include <algorithm>
#include <cmath>
#include <random>

#include "material.h"
#include "occlusion.h"
#include "sh_coeffs.h"

#define USE_UNIFORM_SAMPLING 1

namespace sh_baker {

namespace light_internal {

AreaSample SampleAreaLight(const Light& light, std::mt19937& rng) {
  if (!light.geometry || !light.material) {
    return {};
  }
  const Geometry& geo = *light.geometry;
  if (geo.indices.empty()) return {};

  size_t num_triangles = geo.indices.size() / 3;

  // 1. Select Triangle (Uniformly)
  std::uniform_int_distribution<size_t> dist(0, num_triangles - 1);
  size_t tri_idx = dist(rng);

  uint32_t i0 = geo.indices[tri_idx * 3 + 0];
  uint32_t i1 = geo.indices[tri_idx * 3 + 1];
  uint32_t i2 = geo.indices[tri_idx * 3 + 2];

  // 2. Sample Point (Uniform Barycentric)
  std::uniform_real_distribution<float> u_dist(0.0f, 1.0f);
  float u1 = u_dist(rng);
  float u2 = u_dist(rng);

  if (u1 + u2 > 1.0f) {
    u1 = 1.0f - u1;
    u2 = 1.0f - u2;
  }
  float w = 1.0f - u1 - u2;

  // 3. Interpolate Attributes
  const Eigen::Vector3f& v0 = geo.vertices[i0];
  const Eigen::Vector3f& v1 = geo.vertices[i1];
  const Eigen::Vector3f& v2 = geo.vertices[i2];

  Eigen::Vector3f p = w * v0 + u1 * v1 + u2 * v2;
  p = geo.transform * p;

  // 4. Radiance (Emission)
  Eigen::Vector2f uv = Eigen::Vector2f::Zero();
  if (!geo.texture_uvs.empty()) {
    const Eigen::Vector2f& uv0 = geo.texture_uvs[i0];
    const Eigen::Vector2f& uv1 = geo.texture_uvs[i1];
    const Eigen::Vector2f& uv2 = geo.texture_uvs[i2];
    uv = w * uv0 + u1 * uv1 + u2 * uv2;
  }

  Eigen::Vector3f emission = GetEmission(*light.material, uv);

  // 5. PDF
  // We first uniformly picked a triangle, then a point on the triangle.
  // So P(x) = P(triangle) * P(point | triangle)
  // P(triangle) = 1 / num_triangles
  // P(point | triangle) = 1 / triangle_area
  float triangle_area = (v0 - v1).cross(v0 - v2).norm() / 2.f;
  float pdf = std::max(1e-6f, 1.f / num_triangles * 1.f / triangle_area);
  return {p, emission, pdf};
}

}  // namespace light_internal

Eigen::Vector3f EvaluateLightSamples(
    const LightTree* light_tree, RTCScene rtc_scene,
    const Eigen::Vector3f& hit_point, const Eigen::Vector3f& hit_point_normal,
    const Eigen::Vector3f& reflected, const Material& mat,
    const Eigen::Vector2f& uv, unsigned num_samples, std::mt19937& rng) {
  Eigen::Vector3f result = Eigen::Vector3f::Zero();

  if (!light_tree || light_tree->Empty()) return result;

  std::uniform_real_distribution<float> u_dist(0.0f, 1.0f);

  auto brdf_fn = [&](const Eigen::Vector3f& light_dir) {
    return EvalMaterial(mat, uv, hit_point_normal, light_dir, reflected);
  };

  for (unsigned i = 0; i < num_samples; ++i) {
    float u = u_dist(rng);
#ifdef USE_UNIFORM_SAMPLING
    std::optional<SampledLight> sampled_light = light_tree->SampleUniform(u);
#else
    std::optional<SampledLight> sampled_light =
        light_tree->Sample(hit_point, hit_point_normal, u);
#endif

    if (!sampled_light || sampled_light->pdf <= 0.0f) continue;

    const Light& light = *sampled_light->light;
    Eigen::Vector3f radiance;
    Ray visibility_ray;
    float area_sample_pdf = 1.0f;

    switch (light.type) {
      case Light::Type::Directional: {
        radiance = light_internal::DirectionalLightRadiance(
            light, hit_point, hit_point_normal, brdf_fn, &visibility_ray);
        break;
      }
      case Light::Type::Point: {
        radiance = light_internal::PointLightRadiance(
            light, hit_point, hit_point_normal, brdf_fn, &visibility_ray);
        break;
      }
      case Light::Type::Spot: {
        radiance = light_internal::SpotLightRadiance(
            light, hit_point, hit_point_normal, brdf_fn, &visibility_ray);
        break;
      }
      case Light::Type::Area: {
        light_internal::AreaSample sample =
            light_internal::SampleAreaLight(light, rng);
        radiance = light_internal::AreaLightRadiance(
            sample, hit_point, hit_point_normal, brdf_fn, &visibility_ray);
        area_sample_pdf = sample.pdf;
        break;
      }
      default:
        continue;
    }

    if (radiance.isZero()) continue;

    if (FindOcclusion(rtc_scene, visibility_ray)) {
      continue;
    }

    float joint_pdf = sampled_light->pdf * area_sample_pdf;
    if (joint_pdf > 1e-8f) {
      result += radiance / joint_pdf;
    }
  }

  return result / float(num_samples);
}

void AccumulateIncomingLightSamples(const LightTree* light_tree,
                                    RTCScene rtc_scene,
                                    const Eigen::Vector3f& hit_point,
                                    const Eigen::Vector3f& hit_point_normal,
                                    unsigned num_samples, std::mt19937& rng,
                                    SHCoeffs* accumulator) {
  if (!light_tree || light_tree->Empty()) return;

  std::uniform_real_distribution<float> u_dist(0.0f, 1.0f);

  for (unsigned i = 0; i < num_samples; ++i) {
    float u = u_dist(rng);
#ifdef USE_UNIFORM_SAMPLING
    std::optional<SampledLight> sampled_light = light_tree->SampleUniform(u);
#else
    std::optional<SampledLight> sampled_light =
        light_tree->Sample(hit_point, hit_point_normal, u);
#endif

    if (!sampled_light || sampled_light->pdf <= 0.0f) continue;

    const Light& light = *sampled_light->light;
    Eigen::Vector3f radiance;
    Ray visibility_ray;
    float area_sample_pdf = 1.0f;

    switch (light.type) {
      case Light::Type::Directional: {
        light_internal::DirectionalLightIncoming incoming =
            light_internal::DirectionalLightIncomingRadiance(
                light, hit_point, hit_point_normal, &visibility_ray);
        radiance = incoming.radiance;
        break;
      }
      case Light::Type::Point: {
        radiance = light_internal::PointLightIncomingRadiance(
                       light, hit_point, hit_point_normal, &visibility_ray)
                       .radiance;
        break;
      }
      case Light::Type::Spot: {
        radiance = light_internal::SpotLightIncomingRadiance(
                       light, hit_point, hit_point_normal, &visibility_ray)
                       .radiance;
        break;
      }
      case Light::Type::Area: {
        light_internal::AreaSample sample =
            light_internal::SampleAreaLight(light, rng);
        radiance = light_internal::AreaLightIncomingRadiance(
                       sample, hit_point, hit_point_normal, &visibility_ray)
                       .radiance;
        area_sample_pdf = sample.pdf;
        break;
      }
      default:
        continue;
    }

    if (radiance.isZero()) continue;

    if (FindOcclusion(rtc_scene, visibility_ray)) {
      continue;
    }

    float joint_pdf = sampled_light->pdf * area_sample_pdf;
    if (joint_pdf > 1e-8f) {
      float Li_factor = 1.0f / (joint_pdf * float(num_samples));
      Eigen::Vector3f Li = radiance * Li_factor;
      AccumulateRadiance(Li, visibility_ray.direction, hit_point_normal,
                         accumulator);
    }
  }
}

}  // namespace sh_baker
