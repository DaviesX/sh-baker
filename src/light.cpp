#include "light.h"

#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <random>

#include "material.h"
#include "occlusion.h"
#include "sh_coeffs.h"

// #define USE_UNIFORM_SAMPLING 1

namespace sh_baker {

namespace light_internal {

AreaSample SampleAreaLightTextured(const Light& light, std::mt19937& rng) {
  CHECK(light.emission_cdf);
  CHECK(light.prim_id_map);
  CHECK(light.uv_to_world_area_ratio);
  CHECK(light.geometry);

  const Texture32F& cdf = *light.emission_cdf;
  const Texture32I& prim_map = *light.prim_id_map;
  const Texture32F& ratio_map = *light.uv_to_world_area_ratio;
  const Geometry& geo = *light.geometry;

  // CDF layout: (h+1) rows x (w+1) columns.
  // Rows [0..h-1]: conditional CDF P(u|v) for each row.
  // Row h: marginal CDF P(v).
  int tex_w = prim_map.width;   // Same as emissive texture width.
  int tex_h = prim_map.height;  // Same as emissive texture height.
  int cdf_w = tex_w + 1;

  std::uniform_real_distribution<float> u_dist(0.0f, 1.0f);

  // 1. Sample row v from marginal CDF (last row of CDF texture).
  float xi_v = u_dist(rng);
  const float* marginal = &cdf.pixel_data[tex_h * cdf_w];

  // Binary search in marginal CDF [1..h].
  int v_idx = static_cast<int>(
      std::lower_bound(marginal + 1, marginal + tex_h + 1, xi_v) -
      (marginal + 1));
  v_idx = std::clamp(v_idx, 0, tex_h - 1);

  // Marginal PDF for this row.
  float marginal_pdf = marginal[v_idx + 1] - marginal[v_idx];

  // 2. Sample column u from conditional CDF (row v_idx).
  float xi_u = u_dist(rng);
  const float* conditional = &cdf.pixel_data[v_idx * cdf_w];

  int u_idx = static_cast<int>(
      std::lower_bound(conditional + 1, conditional + tex_w + 1, xi_u) -
      (conditional + 1));
  u_idx = std::clamp(u_idx, 0, tex_w - 1);

  float conditional_pdf = conditional[u_idx + 1] - conditional[u_idx];

  // 3. Look up triangle ID.
  int prim_id = prim_map.pixel_data[v_idx * tex_w + u_idx];
  if (prim_id < 0 ||
      static_cast<size_t>(prim_id) * 3 + 2 >= geo.indices.size()) {
    // Sampled a background texel; fall through to uniform sampling.
    return {};
  }

  // 4. Map sampled texel back to 3D point using barycentrics.
  uint32_t i0 = geo.indices[prim_id * 3 + 0];
  uint32_t i1 = geo.indices[prim_id * 3 + 1];
  uint32_t i2 = geo.indices[prim_id * 3 + 2];

  // Sampled UV (center of texel).
  float su = (u_idx + 0.5f) / tex_w;
  float sv = (v_idx + 0.5f) / tex_h;

  // Triangle UVs.
  const Eigen::Vector2f& uv0 = geo.texture_uvs[i0];
  const Eigen::Vector2f& uv1 = geo.texture_uvs[i1];
  const Eigen::Vector2f& uv2 = geo.texture_uvs[i2];

  // Solve for barycentrics: P = w*uv0 + b1*uv1 + b2*uv2.
  Eigen::Vector2f d1 = uv1 - uv0;
  Eigen::Vector2f d2 = uv2 - uv0;
  Eigen::Vector2f dp = Eigen::Vector2f(su, sv) - uv0;

  float det = d1.x() * d2.y() - d1.y() * d2.x();
  if (std::abs(det) < 1e-12f) {
    return {};
  }
  float inv_det = 1.0f / det;
  float b1 = (dp.x() * d2.y() - dp.y() * d2.x()) * inv_det;
  float b2 = (d1.x() * dp.y() - d1.y() * dp.x()) * inv_det;
  float b0 = 1.0f - b1 - b2;

  // Clamp barycentrics (texel center may be slightly outside triangle).
  b0 = std::clamp(b0, 0.0f, 1.0f);
  b1 = std::clamp(b1, 0.0f, 1.0f);
  b2 = std::clamp(b2, 0.0f, 1.0f);
  float bsum = b0 + b1 + b2;
  if (bsum > 0.0f) {
    b0 /= bsum;
    b1 /= bsum;
    b2 /= bsum;
  }

  // 5. Interpolate world position.
  const Eigen::Vector3f& v0 = geo.vertices[i0];
  const Eigen::Vector3f& v1 = geo.vertices[i1];
  const Eigen::Vector3f& v2 = geo.vertices[i2];
  Eigen::Vector3f p = geo.transform * (b0 * v0 + b1 * v1 + b2 * v2);

  // 6. Emission at sampled UV.
  Eigen::Vector3f emission =
      GetEmission(*light.material, Eigen::Vector2f(su, sv));

  // 7. PDF: texture_pdf / jacobian.
  // texture_pdf = marginal_pdf * conditional_pdf * (tex_w * tex_h)
  // (the w*h factor converts from per-texel to per-unit-UV-area).
  float jacobian = ratio_map.pixel_data[v_idx * tex_w + u_idx];
  float texture_pdf = marginal_pdf * conditional_pdf * (tex_w * tex_h);
  float pdf_area = (jacobian > 1e-12f) ? (texture_pdf / jacobian) : 1e-6f;
  pdf_area = std::max(pdf_area, 1e-6f);

  return AreaSample{p, emission, pdf_area};
}

AreaSample SampleAreaLightUniform(const Light& light, std::mt19937& rng) {
  CHECK_NOTNULL(light.geometry);
  CHECK_NOTNULL(light.material);

  const Geometry& geo = *light.geometry;

  size_t num_triangles = geo.indices.size() / 3;
  std::uniform_int_distribution<size_t> dist(0, num_triangles - 1);
  size_t tri_idx = dist(rng);

  uint32_t i0 = geo.indices[tri_idx * 3 + 0];
  uint32_t i1 = geo.indices[tri_idx * 3 + 1];
  uint32_t i2 = geo.indices[tri_idx * 3 + 2];

  std::uniform_real_distribution<float> u_dist(0.0f, 1.0f);
  float u1 = u_dist(rng);
  float u2 = u_dist(rng);
  if (u1 + u2 > 1.0f) {
    u1 = 1.0f - u1;
    u2 = 1.0f - u2;
  }
  float w = 1.0f - u1 - u2;

  const Eigen::Vector3f& v0 = geo.vertices[i0];
  const Eigen::Vector3f& v1 = geo.vertices[i1];
  const Eigen::Vector3f& v2 = geo.vertices[i2];
  Eigen::Vector3f p = geo.transform * (w * v0 + u1 * v1 + u2 * v2);

  Eigen::Vector2f uv = Eigen::Vector2f::Zero();
  if (!geo.texture_uvs.empty()) {
    const Eigen::Vector2f& uv0 = geo.texture_uvs[i0];
    const Eigen::Vector2f& uv1 = geo.texture_uvs[i1];
    const Eigen::Vector2f& uv2 = geo.texture_uvs[i2];
    uv = w * uv0 + u1 * uv1 + u2 * uv2;
  }

  Eigen::Vector3f emission = GetEmission(*light.material, uv);

  float triangle_area = (v0 - v1).cross(v0 - v2).norm() / 2.f;
  float pdf = std::max(1e-6f, 1.f / num_triangles * 1.f / triangle_area);
  return {p, emission, pdf};
}

AreaSample SampleAreaLight(const Light& light, std::mt19937& rng) {
  CHECK_NOTNULL(light.geometry);
  CHECK_NOTNULL(light.material);

  if (light.material->emissive_texture &&
      !light.geometry->texture_uvs.empty()) {
    AreaSample importance_sample = SampleAreaLightTextured(light, rng);
    if (importance_sample.pdf > 0.f) {
      return importance_sample;
    }
  }

  return SampleAreaLightUniform(light, rng);
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
