#include "light.h"

#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

#include "material.h"
#include "occlusion.h"
#include "sh_coeffs.h"

// #define USE_UNIFORM_SAMPLING 1

namespace sh_baker {
namespace {

constexpr float kPi = 3.14159265359f;
constexpr float kTwoPi = 6.28318530718f;

// Coefficients for a standard "Clear Sky" (Turbidity ~2.5)
const float perez_A = -1.0f;   // Darkening of horizon
const float perez_B = -0.32f;  // Luminance gradient near horizon
const float perez_C = 10.0f;   // Relative intensity of circumsolar region
const float perez_D = -3.0f;   // Width of circumsolar region
const float perez_E = 0.45f;   // Relative backscatter

// World-space surface normal at a barycentric point on a triangle. Uses the
// interpolated vertex normals when present (matching TransformedNormals'
// rotation-only convention), else the geometric face normal.
Eigen::Vector3f InterpolatedWorldNormal(const Geometry& geo, uint32_t i0,
                                        uint32_t i1, uint32_t i2, float b0,
                                        float b1, float b2) {
  Eigen::Vector3f n;
  if (!geo.normals.empty()) {
    n = geo.transform.rotation() *
        (b0 * geo.normals[i0] + b1 * geo.normals[i1] + b2 * geo.normals[i2]);
  } else {
    // Transform vertices to world space *before* the cross product: for a face
    // normal, R * (edge x edge) is wrong under non-uniform scale (the correct
    // map is the inverse-transpose, which (M*e1) x (M*e2) yields up to a
    // positive scalar). Crossing world-space edges is exact and simpler.
    const Eigen::Vector3f v0 = geo.transform * geo.vertices[i0];
    const Eigen::Vector3f v1 = geo.transform * geo.vertices[i1];
    const Eigen::Vector3f v2 = geo.transform * geo.vertices[i2];
    n = (v1 - v0).cross(v2 - v0);
  }
  float len = n.norm();
  return (len > 1e-12f) ? Eigen::Vector3f(n / len) : Eigen::Vector3f::UnitZ();
}

Eigen::Vector3f GetPixel(const Texture32F& tex, int x, int y) {
  int idx = (y * tex.width + x) * tex.channels;
  float r = tex.pixel_data[idx + 0];
  float g = tex.pixel_data[idx + 1];
  float b = tex.pixel_data[idx + 2];
  return Eigen::Vector3f(r, g, b);
}

// Projects an Equirectangular (LatLong) map to SH
SHCoeffs ProjectLatLongMap(const Texture32F& tex) {
  SHCoeffs coeffs;
  int w = tex.width;
  int h = tex.height;

  // Solid angle of a pixel (approx constant per row y)
  // d_omega = (2pi / W) * (pi / H) * sin(theta)
  float d_phi = kTwoPi / w;
  float d_theta = kPi / h;

  Eigen::Vector3f up(0.0f, 1.0f, 0.0f);

  for (int y = 0; y < h; ++y) {
    // Theta from 0 (top) to pi (bottom)
    float theta = (y + 0.5f) * d_theta;
    float sin_theta = std::sin(theta);
    float d_omega = d_phi * d_theta * sin_theta;

    // Optimization: if sin_theta is near 0 (poles), weight is near 0.
    if (sin_theta < 1e-4f) continue;

    for (int x = 0; x < w; ++x) {
      // Phi from 0 to 2pi
      // Standard Equirectangular: Center (0.5) is usually +Z or -Z depending on
      // convention. Here assuming u=0 -> +Z (phi=0), u=0.25 -> +X. Note: glTF
      // usually assumes +Z is front.
      float phi = (x + 0.5f) * d_phi;

      // Convert Spherical to Cartesian (Y-up)
      // y = cos(theta)
      // x = sin(theta) * sin(phi)
      // z = sin(theta) * cos(phi)
      Eigen::Vector3f dir(std::sin(theta) * std::sin(phi), std::cos(theta),
                          std::sin(theta) * std::cos(phi));

      Eigen::Vector3f radiance = GetPixel(tex, x, y);

      AccumulateRadiance(radiance * d_omega, dir, up, &coeffs);
    }
  }
  return coeffs;
}

// Corrected Preetham / Perez Sky Evaluation
// Evaluates the Perez function for a given view direction
float EvaluatePerez(const Eigen::Vector3f& dir, const Eigen::Vector3f& sunDir) {
  // 1. Zenith Angle (gamma): Angle between View and Up (Y-axis)
  float cosThetaZenith = std::max(0.001f, dir.y());  // Clamp to horizon

  // 2. Sun Angle (zeta): Angle between View and Sun
  float cosThetaSun = dir.dot(sunDir);
  // Clamp for acos safety
  float gamma = std::acos(std::clamp(cosThetaSun, -1.0f, 1.0f));

  // Perez Function: F(theta, gamma) = (1 + A*exp(B/cos_theta)) * (1 +
  // C*exp(D*gamma) + E*cos^2(gamma))
  float f_zenith = (1.0f + perez_A * std::exp(perez_B / cosThetaZenith));
  float f_sun = (1.0f + perez_C * std::exp(perez_D * gamma) +
                 perez_E * cosThetaSun * cosThetaSun);

  return f_zenith * f_sun;
}

// The zenith Perez value the Preetham radiance is normalised by.
float PreethamZenith(const Eigen::Vector3f& sun_dir) {
  float zenith_val = EvaluatePerez(Eigen::Vector3f(0.0f, 1.0f, 0.0f), sun_dir);
  if (zenith_val <= 1e-6f) zenith_val = 1.0f;
  return zenith_val;
}

// The Preetham sky's radiance in Y-up direction `dir` (above the horizon; the
// caller handles y < 0): the Perez value capped at 30x the zenith, coloured as
// a blue sky scaled by luminance / zenith.
Eigen::Vector3f PreethamRadiance(const Eigen::Vector3f& dir,
                                 const Eigen::Vector3f& sun_dir,
                                 float zenith_val) {
  float raw_luminance = EvaluatePerez(dir, sun_dir);

  // Apply the cap (e.g., 30x zenith) to prevent ringing from sun disk
  float cap = zenith_val * 30.0f;
  float luminance = std::min(raw_luminance, cap);

  // Convert luminance to RGB (Simplified: Blue sky color scaled by
  // luminance)
  return Eigen::Vector3f(0.2f, 0.5f, 0.9f) * (luminance / zenith_val);
}

SHCoeffs ProjectPreethamToSH(const Environment& env) {
  SHCoeffs coeffs;

  // Integration resolution (higher = more accurate)
  const int steps_theta = 64;  // Latitude (0 to PI)
  const int steps_phi = 128;   // Longitude (0 to 2PI)

  float d_theta = kPi / steps_theta;
  float d_phi = kTwoPi / steps_phi;

  Eigen::Vector3f sun_dir = env.sun_direction.normalized();
  Eigen::Vector3f up(0.0f, 1.0f, 0.0f);

  // Zenith luminance for normalization
  float zenith_val = PreethamZenith(sun_dir);

  for (int t = 0; t < steps_theta; ++t) {
    float theta = (t + 0.5f) * d_theta;  // 0 (North Pole/Up) to PI
    float sin_theta = std::sin(theta);

    for (int p = 0; p < steps_phi; ++p) {
      float phi = (p + 0.5f) * d_phi;

      // 1. Convert Spherical to Cartesian (Y-up)
      // Note: ProjectLatLongMap convention was:
      // y = cos(theta)
      // x = sin(theta) * sin(phi)
      // z = sin(theta) * cos(phi)
      // This makes theta=0 -> y=1 (Up). Consistent with Preetham.
      float y = std::cos(theta);
      float x = sin_theta * std::sin(phi);
      float z = sin_theta * std::cos(phi);
      Eigen::Vector3f dir(x, y, z);

      // Handle Horizon clipping (ground is black)
      if (y < 0.0f) continue;

      // 2. Evaluate Radiance using Preetham function
      Eigen::Vector3f radiance = PreethamRadiance(dir, sun_dir, zenith_val);

      // 3. Differential solid angle (dA = sin(theta) * dTheta * dPhi)
      float differential_solid_angle = sin_theta * d_theta * d_phi;

      // 4. Accumulate
      AccumulateRadiance(radiance * differential_solid_angle, dir, up, &coeffs);
    }
  }

  return coeffs;
}

}  // namespace

SHCoeffs ProjectEnvironmentToSH(const Environment& env) {
  if (env.type == Environment::Type::Texture) {
    return ProjectLatLongMap(env.texture);
  } else {
    return ProjectPreethamToSH(env);
  }
}

Eigen::Vector3f EvaluateEnvironment(const Environment& env,
                                    const Eigen::Vector3f& world_dir) {
  Eigen::Vector3f dir = env.to_environment * world_dir;
  float norm = dir.norm();
  if (!(norm > 0.0f)) return Eigen::Vector3f::Zero();
  dir /= norm;

  if (env.type == Environment::Type::Preetham) {
    // Ground is black, as in ProjectPreethamToSH().
    if (dir.y() < 0.0f) return Eigen::Vector3f::Zero();
    Eigen::Vector3f sun_dir = env.sun_direction.normalized();
    return PreethamRadiance(dir, sun_dir, PreethamZenith(sun_dir));
  }

  // Inverse of ProjectLatLongMap()'s mapping: texel (x, y) holds
  // theta = (y + 0.5) pi / H from +Y and phi = (x + 0.5) 2 pi / W, with
  // dir = (sin theta sin phi, cos theta, sin theta cos phi).
  const Texture32F& tex = env.texture;
  const int w = static_cast<int>(tex.width);
  const int h = static_cast<int>(tex.height);
  if (w <= 0 || h <= 0 || tex.channels < 3) return Eigen::Vector3f::Zero();

  float theta = std::acos(std::clamp(dir.y(), -1.0f, 1.0f));
  float phi = std::atan2(dir.x(), dir.z());
  if (phi < 0.0f) phi += kTwoPi;

  float u = phi / kTwoPi * w - 0.5f;
  float v = theta / kPi * h - 0.5f;
  float u_floor = std::floor(u);
  float v_floor = std::floor(v);
  float fu = u - u_floor;
  float fv = v - v_floor;

  // Wrap in longitude.
  int x0 = static_cast<int>(u_floor) % w;
  if (x0 < 0) x0 += w;
  int x1 = (x0 + 1) % w;
  // Clamp in latitude.
  int y0 = std::clamp(static_cast<int>(v_floor), 0, h - 1);
  int y1 = std::clamp(static_cast<int>(v_floor) + 1, 0, h - 1);

  Eigen::Vector3f top =
      (1.0f - fu) * GetPixel(tex, x0, y0) + fu * GetPixel(tex, x1, y0);
  Eigen::Vector3f bottom =
      (1.0f - fu) * GetPixel(tex, x0, y1) + fu * GetPixel(tex, x1, y1);
  return (1.0f - fv) * top + fv * bottom;
}

namespace light_internal {

AreaSample SampleAreaLightTextured(const Light& light, std::mt19937& rng) {
  CHECK(light.emission_cdf);
  CHECK(light.prim_id_map);
  CHECK(light.uv_to_world_area_ratio);
  CHECK(light.geometry);

  // 1. Sample UV from the CDF.
  const EmissionCDF& cdf = *light.emission_cdf;
  std::uniform_real_distribution<float> u_dist(0.0f, 1.0f);
  auto [uv, pdf_uv] = SampleEmissionCDF(cdf, u_dist(rng), u_dist(rng));

  if (pdf_uv <= 0.0f) {
    return {};
  }

  const Texture32I& prim_map = *light.prim_id_map;
  const Texture32F& ratio_map = *light.uv_to_world_area_ratio;
  const Geometry& geo = *light.geometry;

  int tex_w = prim_map.width;
  int tex_h = prim_map.height;

  // 2. Identify the texel (u_idx, v_idx).
  int u_idx = std::clamp(static_cast<int>(uv.x() * tex_w), 0, tex_w - 1);
  int v_idx = std::clamp(static_cast<int>(uv.y() * tex_h), 0, tex_h - 1);

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

  // Triangle UVs.
  const Eigen::Vector2f& uv0 = geo.texture_uvs[i0];
  const Eigen::Vector2f& uv1 = geo.texture_uvs[i1];
  const Eigen::Vector2f& uv2 = geo.texture_uvs[i2];

  // Solve for barycentrics: P = w*uv0 + b1*uv1 + b2*uv2.
  Eigen::Vector2f d1 = uv1 - uv0;
  Eigen::Vector2f d2 = uv2 - uv0;
  Eigen::Vector2f dp = uv - uv0;

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
  Eigen::Vector3f normal = InterpolatedWorldNormal(geo, i0, i1, i2, b0, b1, b2);

  // 6. Emission at sampled UV.
  Eigen::Vector3f emission = GetEmission(*light.material, uv);

  // 7. PDF: pdf_uv / jacobian. A degenerate jacobian means this texel has no
  // real world-space footprint; reject the sample rather than flooring the pdf
  // to a tiny constant, which would turn radiance/pdf into a huge firefly.
  float jacobian = ratio_map.pixel_data[v_idx * tex_w + u_idx];
  if (jacobian <= 1e-12f) {
    return {};
  }
  float pdf_area = pdf_uv / jacobian;

  return AreaSample{p, emission, normal, pdf_area};
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
  Eigen::Vector3f normal = InterpolatedWorldNormal(geo, i0, i1, i2, w, u1, u2);

  float triangle_area = (v0 - v1).cross(v0 - v2).norm() / 2.f;
  if (triangle_area <= 1e-12f) {
    // Degenerate (zero-area) triangle: no valid area-measure pdf. Reject rather
    // than floor the pdf, which would produce a firefly.
    return {};
  }
  float pdf = 1.f / (num_triangles * triangle_area);
  return {p, emission, normal, pdf};
}

AreaSample SampleAreaLight(const Light& light, std::mt19937& rng) {
  CHECK_NOTNULL(light.geometry);
  CHECK_NOTNULL(light.material);

  if (light.emission_cdf && light.material->emissive_texture &&
      !light.geometry->texture_uvs.empty()) {
    AreaSample importance_sample = SampleAreaLightTextured(light, rng);
    if (importance_sample.pdf > 0.f) {
      return importance_sample;
    }
  }

  return SampleAreaLightUniform(light, rng);
}

float FindTransmittance(RTCScene scene, const std::vector<Material>& materials,
                        const Ray& ray) {
  Ray walk = ray;
  float result = 1.0f;

  for (int crossings = 0;; ++crossings) {
    std::optional<OcclusionWithAlpha> occ =
        FindOcclusion(scene, materials, walk);
    if (!occ) {
      return result;
    }
    if (occ->alpha >= 1.0f) {
      return 0.0f;
    }
    if (crossings >= kMaxTransmittanceCrossings) {
      return 0.0f;
    }

    result *= 1.0f - occ->alpha;

    walk = ContinuePastOcclusion(walk, occ->hit);
    if (walk.tnear >= walk.tfar) {
      return result;
    }
  }
}

}  // namespace light_internal

Eigen::Vector3f EvaluateLightSamples(
    const LightTree* light_tree, RTCScene rtc_scene,
    const std::vector<Material>& materials, const Eigen::Vector3f& hit_point,
    const Eigen::Vector3f& hit_point_normal, const Eigen::Vector3f& reflected,
    const Material& mat, const Eigen::Vector2f& uv, unsigned num_samples,
    std::mt19937& rng) {
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

    // Opaque content gives exactly 0 or 1, so an unoccluded sample is scaled by
    // 1.0f exactly and its result is unchanged.
    float visibility =
        light_internal::FindTransmittance(rtc_scene, materials, visibility_ray);
    if (visibility <= 0.0f) continue;
    radiance *= visibility;

    float joint_pdf = sampled_light->pdf * area_sample_pdf;
    if (joint_pdf > 1e-8f) {
      result += radiance / joint_pdf;
    }
  }

  return result / float(num_samples);
}

void AccumulateIncomingLightSamples(const LightTree* light_tree,
                                    RTCScene rtc_scene,
                                    const std::vector<Material>& materials,
                                    const Eigen::Vector3f& hit_point,
                                    const Eigen::Vector3f& hit_point_normal,
                                    unsigned num_samples, std::mt19937& rng,
                                    SHCoeffs* accumulator, bool full_sphere) {
  if (!light_tree || light_tree->Empty()) return;

  std::uniform_real_distribution<float> u_dist(0.0f, 1.0f);

  // The normal light selection weighs by; zero (no incident-angle term) for a
  // full-sphere point.
  const Eigen::Vector3f selection_normal =
      full_sphere ? Eigen::Vector3f::Zero() : hit_point_normal;

  for (unsigned i = 0; i < num_samples; ++i) {
    float u = u_dist(rng);
#ifdef USE_UNIFORM_SAMPLING
    std::optional<SampledLight> sampled_light = light_tree->SampleUniform(u);
#else
    std::optional<SampledLight> sampled_light =
        light_tree->Sample(hit_point, selection_normal, u);
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
                light, hit_point, hit_point_normal, &visibility_ray,
                full_sphere);
        radiance = incoming.radiance;
        break;
      }
      case Light::Type::Point: {
        radiance = light_internal::PointLightIncomingRadiance(
                       light, hit_point, hit_point_normal, &visibility_ray,
                       full_sphere)
                       .radiance;
        break;
      }
      case Light::Type::Spot: {
        radiance = light_internal::SpotLightIncomingRadiance(
                       light, hit_point, hit_point_normal, &visibility_ray,
                       full_sphere)
                       .radiance;
        break;
      }
      case Light::Type::Area: {
        light_internal::AreaSample sample =
            light_internal::SampleAreaLight(light, rng);
        radiance = light_internal::AreaLightIncomingRadiance(
                       sample, hit_point, hit_point_normal, &visibility_ray,
                       full_sphere)
                       .radiance;
        area_sample_pdf = sample.pdf;
        break;
      }
      default:
        continue;
    }

    if (radiance.isZero()) continue;

    // Opaque content gives exactly 0 or 1, so an unoccluded sample is scaled by
    // 1.0f exactly and its result is unchanged.
    float visibility =
        light_internal::FindTransmittance(rtc_scene, materials, visibility_ray);
    if (visibility <= 0.0f) continue;
    radiance *= visibility;

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
