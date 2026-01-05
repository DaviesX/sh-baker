#include "baker.h"

#include <embree4/rtcore.h>
#include <glog/logging.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <algorithm>
#include <cmath>
#include <random>

#include "light.h"
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

// Computes a Monte Carlo path and return a radiance sample.
// Rendering equation:
// L_o(x, \omega_o) = L_e(x, \omega_o) + \int_{\Omega} f_r(x, \omega_i,
// \omega_o) * L_i(x, \omega_i) * cos(\omega_i) d\omega_i
//
// where L_o is the radiance at the camera, f_r is the BRDF, L_i is the
// radiance from the light, and \omega_i is the direction from the light to the
// surface.
//
// To drastically reduce variance, we partition the paths into 2 disjoint sets:
// 1. Primary rays: see the sky/sun directly
// 2. Secondary rays: bounce off a surface
//
// Formally,
// L_o(x, \omega_o) = L_e(x, \omega_o) + \int_{A_e} ...Le(x, x')...dA_e(x') +
// \int_{\Omega \setminus A_e} ...L_i(x, \omega_i)...d\omega_i
//
// where Le(x, x') is the radiance from the light, L_i(x, \omega_i) is the
// radiance from the environment, and \omega_i is the direction from the light
// to the surface.
//
// This is also known as a technique called next event estimation (NEE).
Eigen::Vector3f Trace(RTCScene rtc_scene, const Scene& scene,
                      const Eigen::Vector3f& origin, const Eigen::Vector3f& dir,
                      int depth, int max_depth, int num_light_samples,
                      std::mt19937& rng) {
  if (depth > max_depth) return Eigen::Vector3f::Zero();

  Ray visibility_ray;
  visibility_ray.origin = origin;
  visibility_ray.direction = dir;
  visibility_ray.tnear = 0.001f;

  std::optional<Occlusion> occ = FindOcclusion(rtc_scene, visibility_ray);

  if (!occ.has_value()) {
    // Sky
    float sun = std::max(0.0f, dir.dot(scene.sky.sun_direction));
    Eigen::Vector3f sky_radiance =
        scene.sky.sun_color * scene.sky.sun_intensity * sun +
        Eigen::Vector3f(0.05f, 0.05f, 0.05f);  // ambient

    if (depth == 0) {
      // Primary rays see the sky/sun directly
      return sky_radiance;
    } else {
      // Secondary rays (Indirect):
      // Sun is handled by NEE (EvaluateLights), so we excluded it here to avoid
      // double counting. We only return the ambient component.
      return Eigen::Vector3f(0.05f, 0.05f, 0.05f);
    }
  }

  // Hit surface
  const Material& mat = scene.materials[occ->material_id];
  float alpha = GetAlpha(mat, occ->uv);

  Eigen::Vector3f color = Eigen::Vector3f::Zero();

  // If alpha < 1.0, continue ray
  if (alpha < 1.0f) {
    // Transmission
    Eigen::Vector3f hit_pos = occ->position + dir * 0.001f;
    Eigen::Vector3f transmission = Trace(rtc_scene, scene, hit_pos, dir, depth,
                                         max_depth, num_light_samples, rng);

    color += (1.0f - alpha) * transmission;
  }

  if (alpha == 0.0f) {
    return color;
  }

  // Check if material is emissive
  Eigen::Vector3f emission = GetEmission(mat, occ->uv);
  if (!emission.isZero()) {
    // If depth == 0, we see the emissive surface directly (Le).
    // If depth > 0, we are bouncing. In NEE, we sample lights explicitly,
    // so we ignore implicit hits on emissive surfaces to avoid double counting.
    if (depth == 0) {
      color += alpha * emission;
    }
  }

  // Direct Lighting (NEE)
  {
    // Compute Shading Normal
    Eigen::Vector3f shading_normal = GetNormalFromMap(
        mat, occ->uv, occ->normal, occ->tangent, occ->bitangent);

    Eigen::Vector3f hit_pos = occ->position + occ->normal * 0.001f;
    // View direction is -dir
    Eigen::Vector3f wo = -dir;

    // Evaluate Direct Light (NEE)
    // EvaluateLights returns L_e(x, x')
    Eigen::Vector3f L_direct = EvaluateLightSamples(
        scene.sky, scene.lights, rtc_scene, hit_pos, shading_normal, wo, mat,
        occ->uv, num_light_samples, rng);

    color += alpha * L_direct;

    // Note: We might want to pass shading_normal to EvaluateLightSamples
    // but we also need to be careful about geometric visibility (hit_normal).
    // EvaluateLightSamples generally uses normal for Cosine term and BRDF.
    // PBR: use shading_normal.
  }

  // Indirect Lighting (Recursive)
  {
    Eigen::Vector3f shading_normal = GetNormalFromMap(
        mat, occ->uv, occ->normal, occ->tangent, occ->bitangent);

    Eigen::Vector3f hit_pos = occ->position + occ->normal * 0.001f;
    ReflectionSample sample =
        SampleMaterial(mat, occ->uv, shading_normal, dir, rng);

    Eigen::Vector3f incoming =
        Trace(rtc_scene, scene, hit_pos, sample.direction, depth + 1, max_depth,
              num_light_samples, rng);

    Eigen::Vector3f brdf =
        EvalMaterial(mat, occ->uv, shading_normal, dir, sample.direction);

    // Cosine term is based on shading normal for the integral over hemisphere
    // defined by shading normal?
    // The rendering equation integrates over the hemisphere around the MACRO
    // normal? Actually, usually we integrate over the hemisphere around the
    // SHADING normal.
    float cosine_term = std::max(0.0f, shading_normal.dot(sample.direction));

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
  // If supersampling is used, the rasterized points size might be larger than
  // config.width * config.height implies if config was the original size.
  // Actually, RasterConfig passed here should match surface_points.
  // We assume raster_config describes the dimensions of surface_points.
  // We compute the output size based on surface_points and config.width/height
  // is just metadata? Let's assume raster_config.width/height IS the resolution
  // of surface_points.
  int width = raster_config.width * raster_config.supersample_scale;
  int height = raster_config.height * raster_config.supersample_scale;

  // Sanity check
  if (surface_points.size() != width * height) {
    // If mismatch, assume width/height in config are already scaled or don't
    // use scale here? Let's rely on the fact that surface_points size is
    // authoritative for the loop. But we need width/height for structure. Let's
    // adhere to: raster_config.width/height are BASE dimensions. And
    // supersample_scale tells us the factor.
  }

  output.width = width;
  output.height = height;
  output.pixels.resize(width * height);

  RTCDevice device = rtcNewDevice(nullptr);
  RTCScene rtc_scene = BuildBVH(scene, device);

  float inv_pdf_uniform = 2.0f * M_PI;

  tbb::parallel_for(
      tbb::blocked_range<size_t>(0, surface_points.size()),
      [&](const tbb::blocked_range<size_t>& r) {
        // Each thread needs its own RNG? Or use thread-local?
        // Simple way: Seed based on index.
        for (size_t idx = r.begin(); idx != r.end(); ++idx) {
          const SurfacePoint& sp = surface_points[idx];
          if (!sp.valid) continue;

          SHCoeffs sh_accum;

          // Seeding RNG with index to make it deterministic but different per
          // pixel
          std::mt19937 rng(12345 + idx);

          // Offset position to avoid self-intersection
          Eigen::Vector3f origin = sp.position + sp.normal * 0.001f;

          for (int s = 0; s < config.samples; ++s) {
            Eigen::Vector3f dir_local =
                SampleHemisphereUniform(rng);  // Z is up

            // Transform to World
            Eigen::Vector3f dir_world = sp.tangent * dir_local.x() +
                                        sp.bitangent * dir_local.y() +
                                        sp.normal * dir_local.z();

            Eigen::Vector3f Li =
                Trace(rtc_scene, scene, origin, dir_world, 0, config.bounces,
                      config.num_light_samples, rng);

            AccumulateRadiance(Li * inv_pdf_uniform, dir_world, &sh_accum);
          }

          // Average
          output.pixels[idx] = sh_accum * (1.0f / config.samples);
        }
      });

  // Clean up
  rtcReleaseScene(rtc_scene);
  rtcReleaseDevice(device);

  return output;
}

SHTexture DownsampleSHTexture(const SHTexture& input, int scale) {
  if (scale <= 1) return input;

  SHTexture output;
  output.width = input.width / scale;
  output.height = input.height / scale;
  output.pixels.resize(output.width * output.height);

  for (int y = 0; y < output.height; ++y) {
    for (int x = 0; x < output.width; ++x) {
      SHCoeffs avg;
      int count = 0;
      for (int dy = 0; dy < scale; ++dy) {
        for (int dx = 0; dx < scale; ++dx) {
          int sx = x * scale + dx;
          int sy = y * scale + dy;
          if (sx < input.width && sy < input.height) {
            avg += input.pixels[sy * input.width + sx];
            count++;
          }
        }
      }
      if (count > 0) {
        output.pixels[y * output.width + x] = avg * (1.0f / count);
      }
    }
  }
  return output;
}

}  // namespace sh_baker
