#include "light.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

#include "material.h"
#include "occlusion.h"
#include "sh_coeffs.h"

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

  Eigen::Vector3f n = Eigen::Vector3f(0, 1, 0);
  if (!geo.normals.empty()) {
    const Eigen::Vector3f& n0 = geo.normals[i0];
    const Eigen::Vector3f& n1 = geo.normals[i1];
    const Eigen::Vector3f& n2 = geo.normals[i2];
    n = (w * n0 + u1 * n1 + u2 * n2).normalized();
  }

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
  return {p, n, emission, pdf};
}

}  // namespace light_internal

Eigen::Vector3f EvaluateLightSamples(
    const std::vector<Light>& lights, RTCScene rtc_scene,
    const Eigen::Vector3f& hit_point, const Eigen::Vector3f& hit_point_normal,
    const Eigen::Vector3f& reflected, const Material& mat,
    const Eigen::Vector2f& uv, unsigned num_samples, std::mt19937& rng) {
  // Build sampling distribution using the cheap heuristic:
  // score = L(sample) * brdf * \cos \theta / dist^2.
  // By omitting the visibility term, we can sample the distribution extremely
  // efficiently. Given the lights set we have here is potentially visible, our
  // probability distribution is very close to the actual radiance function,
  // yielding low variance.
  std::vector<Eigen::Vector3f> radiances_without_visibility;
  std::vector<Ray> visibility_rays;
  std::vector<float> area_sample_pdfs;
  std::vector<float> weights;
  radiances_without_visibility.reserve(lights.size());
  visibility_rays.reserve(lights.size());
  area_sample_pdfs.reserve(lights.size());
  weights.reserve(lights.size());

  auto brdf_fn = [&](const Eigen::Vector3f& light_dir) {
    // EvalMaterial expects (..., incident, reflected).
    // incident = Surface->Light = light_dir.
    // reflected = Surface->Eye = reflected (This variable passed to
    // EvaluateLightSamples is wo).
    return EvalMaterial(mat, uv, hit_point_normal, light_dir, reflected);
  };

  for (const auto& light : lights) {
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
      default: {
        radiance = Eigen::Vector3f::Zero();
        break;
      }
    }

    radiances_without_visibility.push_back(radiance);
    visibility_rays.push_back(visibility_ray);
    area_sample_pdfs.push_back(area_sample_pdf);
    weights.push_back(radiance.maxCoeff());
  }

  // Create Distribution
  float sum_weights = std::accumulate(weights.begin(), weights.end(), 0.0f);
  if (sum_weights < 1e-6f) {
    // All lights are almost invisible.
    return Eigen::Vector3f::Zero();
  }

  // Sample from the distribution and accumulate the result.
  Eigen::Vector3f result = Eigen::Vector3f::Zero();
  std::discrete_distribution<int> dist(weights.begin(), weights.end());
  for (unsigned i = 0; i < num_samples; ++i) {
    int idx = dist(rng);

    const Eigen::Vector3f& radiance_without_visibility =
        radiances_without_visibility[idx];
    const Ray& visibility_ray = visibility_rays[idx];

    if (FindOcclusion(rtc_scene, visibility_ray)) {
      // Visibility term is 0.
      continue;
    }

    float pdf = weights[idx] / sum_weights;
    float area_sample_pdf = area_sample_pdfs[idx];
    float inverse_joint_pdf = 1.f / (pdf * area_sample_pdf);

    result += inverse_joint_pdf * radiance_without_visibility;
  }

  return result / num_samples;
}

void AccumulateIncomingLightSamples(const std::vector<Light>& lights,
                                    RTCScene rtc_scene,
                                    const Eigen::Vector3f& hit_point,
                                    const Eigen::Vector3f& hit_point_normal,
                                    unsigned num_samples, std::mt19937& rng,
                                    SHCoeffs* accumulator) {
  // Build sampling distribution using the cheap heuristic:
  // score = L(sample)* G(hit_point_normal, sample) / dist^2.
  // By omitting the visibility term, we can sample the distribution extremely
  // efficiently. Given the lights set we have here is potentially visible, our
  // probability distribution is very close to the actual incoming radiance
  // function, yielding low variance.
  std::vector<Eigen::Vector3f> radiances_without_visibility;
  std::vector<Ray> visibility_rays;
  std::vector<float> area_sample_pdfs;
  std::vector<float> weights;
  radiances_without_visibility.reserve(lights.size());
  visibility_rays.reserve(lights.size());
  area_sample_pdfs.reserve(lights.size());
  weights.reserve(lights.size());

  for (const auto& light : lights) {
    Eigen::Vector3f radiance;
    Ray visibility_ray;
    float area_sample_pdf = 1.0f;
    switch (light.type) {
      case Light::Type::Directional: {
        radiance = light_internal::DirectionalLightIncomingRadiance(
                       light, hit_point, hit_point_normal, &visibility_ray)
                       .radiance;
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
      default: {
        radiance = Eigen::Vector3f::Zero();
        break;
      }
    }

    radiances_without_visibility.push_back(radiance);
    visibility_rays.push_back(visibility_ray);
    area_sample_pdfs.push_back(area_sample_pdf);
    weights.push_back(radiance.maxCoeff());
  }

  // Create Distribution
  float sum_weights = std::accumulate(weights.begin(), weights.end(), 0.0f);
  if (sum_weights < 1e-6f) {
    // All lights are almost invisible.
    return;
  }

  // Sample from the distribution and accumulate the result.
  std::discrete_distribution<int> dist(weights.begin(), weights.end());
  float inv_num_samples = 1.0f / num_samples;

  for (unsigned i = 0; i < num_samples; ++i) {
    int idx = dist(rng);

    const Eigen::Vector3f& radiance_without_visibility =
        radiances_without_visibility[idx];
    const Ray& visibility_ray = visibility_rays[idx];

    if (FindOcclusion(rtc_scene, visibility_ray)) {
      // Visibility term is 0.
      continue;
    }

    float pdf = weights[idx] / sum_weights;
    float area_sample_pdf = area_sample_pdfs[idx];
    float inverse_joint_pdf = 1.f / (pdf * area_sample_pdf);

    Eigen::Vector3f Li =
        (inverse_joint_pdf * inv_num_samples) * radiance_without_visibility;

    // Accumulate into SH (using direction TO the light).
    AccumulateRadiance(Li, visibility_ray.direction, accumulator);
  }
}

}  // namespace sh_baker
