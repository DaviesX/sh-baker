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

// Uniform sample on hemisphere
// Moved to material.h/cpp

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

  if (alpha > 0.0f) {
    // Evaluate surface
    // Emission
    if (mat.emission_intensity > 0.0f) {
      color += alpha * EvalMaterial(mat, occ->uv);
    } else {
      // Reflection (Lambertian)
      // Normal from occlusion
      Eigen::Vector3f hit_normal = occ->normal;

      // Cosine weighted sample for Lambertian
      std::uniform_real_distribution<float> dist(0.0f, 1.0f);
      float r1 = dist(rng);
      float r2 = dist(rng);

      // Cosine sample hemisphere
      float phi = 2.0f * M_PI * r1;
      float theta = std::acos(std::sqrt(r2));
      float sin_theta = std::sin(theta);
      Eigen::Vector3f bounce_dir_local(std::cos(phi) * sin_theta,
                                       std::sin(phi) * sin_theta,
                                       std::cos(theta));  // Z-up local

      // Basis
      Eigen::Vector3f t, b;
      if (std::abs(hit_normal.x()) > std::abs(hit_normal.z())) {
        t = Eigen::Vector3f(-hit_normal.y(), hit_normal.x(), 0.0f);
      } else {
        t = Eigen::Vector3f(0.0f, -hit_normal.z(), hit_normal.y());
      }
      t.normalize();
      b = hit_normal.cross(t);

      Eigen::Vector3f bounce_dir = t * bounce_dir_local.x() +
                                   b * bounce_dir_local.y() +
                                   hit_normal * bounce_dir_local.z();

      Eigen::Vector3f hit_pos = occ->position + hit_normal * 0.001f;

      Eigen::Vector3f incoming = Trace(rtc_scene, scene, hit_pos, bounce_dir,
                                       depth + 1, max_depth, rng);

      Eigen::Vector3f albedo = EvalMaterial(mat, occ->uv);
      color += alpha * albedo.cwiseProduct(incoming);
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
      std::uniform_real_distribution<float> dist(0.0f, 1.0f);
      float u1 = dist(rng);
      float u2 = dist(rng);

      Eigen::Vector3f dir_local = SampleHemisphereUniform(u1, u2);  // Z is up

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
