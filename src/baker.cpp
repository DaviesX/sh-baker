#include "baker.h"

#include <embree4/rtcore.h>
#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <random>

#include "material.h"
#include "occlusion.h"
#include "rasterizer.h"

namespace sh_baker {

namespace {

Eigen::Vector3f SampleHemisphereUniform(std::mt19937& rng) {
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  float u1 = dist(rng);
  float u2 = dist(rng);

  float r = std::sqrt(1.0f - u1 * u1);
  float phi = 2.0f * M_PI * u2;
  return Eigen::Vector3f(r * std::cos(phi), r * std::sin(phi), u1);
}

// Trace path using Occlusion helper
Eigen::Vector3f Trace(RTCScene rtc_scene, const Scene& scene,
                      const Eigen::Vector3f& origin, const Eigen::Vector3f& dir,
                      int depth, int max_depth, std::mt19937& rng) {
  if (depth > max_depth) return Eigen::Vector3f::Zero();

  Ray ray;
  ray.origin = origin;
  ray.direction = dir;
  ray.tnear = 0.001f;

  std::optional<Occlusion> occ = FindOcclusion(rtc_scene, ray);

  if (!occ.has_value()) {
    // Sky
    float sun = std::max(0.0f, dir.dot(scene.sky.sun_direction));
    // Blazing sun + ambient
    return scene.sky.sun_color * scene.sky.sun_intensity * sun +
           Eigen::Vector3f(0.05f, 0.05f, 0.05f);  // ambient
  }

  // Hit surface
  const Material& mat = scene.materials[occ->material_id];
  float alpha = GetAlpha(mat, occ->uv);

  Eigen::Vector3f color = Eigen::Vector3f::Zero();

  // If alpha < 1.0, continue ray
  if (alpha < 1.0f) {
    // Transmission
    // New origin is hit position + epsilon * dir?
    // FindOcclusion returns exact hit position.
    Eigen::Vector3f hit_pos = occ->position + dir * 0.001f;
    Eigen::Vector3f transmission =
        Trace(rtc_scene, scene, hit_pos, dir, depth, max_depth, rng);

    color += (1.0f - alpha) * transmission;
  }

  if (alpha == 0.0f) {
    // Short circuit. We don't need to evaluate surface.
    return color;
  }

  // Evaluate surface
  // Emission
  Eigen::Vector3f emission = GetEmission(mat, occ->uv);
  if (!emission.isZero()) {
    color += alpha * emission;
  } else {
    // Reflection
    // Normal from occlusion
    Eigen::Vector3f hit_normal = occ->normal;
    Eigen::Vector3f hit_pos = occ->position + hit_normal * 0.001f;

    // Sample Material
    ReflectionSample sample =
        SampleMaterial(mat, occ->uv, hit_normal, dir, rng);

    Eigen::Vector3f incoming = Trace(
        rtc_scene, scene, hit_pos, sample.direction, depth + 1, max_depth, rng);

    Eigen::Vector3f brdf =
        EvalMaterial(mat, occ->uv, hit_normal, dir, sample.direction);

    float cosine_term = std::max(0.0f, hit_normal.dot(sample.direction));

    // Estimator: L_i * f_r * cos / pdf
    if (sample.pdf > 1e-6f) {
      color += alpha * incoming.cwiseProduct(brdf) * (cosine_term / sample.pdf);
    }
  }

  return color;
}

}  // namespace

SHTexture BakeSHLightMap(const Scene& scene,
                         const std::vector<SurfacePoint>& surface_points,
                         const RasterConfig& raster_config,
                         const BakeConfig& config) {
  SHTexture output;
  output.width = raster_config.width;
  output.height = raster_config.height;
  output.pixels.resize(raster_config.width * raster_config.height);

  RTCDevice device = rtcNewDevice(nullptr);
  RTCScene rtc_scene = BuildBVH(scene, device);

  // Bake
  std::mt19937 rng(12345);

  float inv_pdf_uniform = 2.0f * M_PI;

  for (int idx = 0; idx < surface_points.size(); ++idx) {
    const SurfacePoint& sp = surface_points[idx];
    if (!sp.valid) continue;

    SHCoeffs sh_accum;

    // Offset position to avoid self-intersection
    Eigen::Vector3f origin = sp.position + sp.normal * 0.001f;

    for (int s = 0; s < config.samples; ++s) {
      Eigen::Vector3f dir_local = SampleHemisphereUniform(rng);  // Z is up

      // Transform to World
      Eigen::Vector3f dir_world = sp.tangent * dir_local.x() +
                                  sp.bitangent * dir_local.y() +
                                  sp.normal * dir_local.z();

      Eigen::Vector3f Li =
          Trace(rtc_scene, scene, origin, dir_world, 0, config.bounces, rng);

      AccumulateRadiance(Li * inv_pdf_uniform, dir_world, &sh_accum);
    }

    // Average
    output.pixels[idx] = sh_accum * (1.0f / config.samples);
  }

  // Clean up
  rtcReleaseScene(rtc_scene);
  rtcReleaseDevice(device);

  return output;
}

}  // namespace sh_baker
