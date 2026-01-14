#include "baker.h"

#include <embree4/rtcore.h>
#include <glog/logging.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <atomic>
#include <cmath>
#include <functional>
#include <iostream>
#include <random>

#include "light.h"
#include "material.h"
#include "occlusion.h"
#include "rasterizer.h"

namespace sh_baker {
namespace {

struct TraceConfig {
  TraceConfig(const RTCScene rtc_scene, const Scene& scene, int max_depth,
              int num_light_samples, std::function<void()> on_direct_hit_sky_fn)
      : rtc_scene(rtc_scene),
        scene(scene),
        max_depth(max_depth),
        num_light_samples(num_light_samples),
        on_direct_hit_sky_fn(on_direct_hit_sky_fn) {}

  const RTCScene rtc_scene;
  const Scene& scene;
  const int max_depth;
  const int num_light_samples;
  const std::function<void()> on_direct_hit_sky_fn;
};

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
Eigen::Vector3f Trace(const TraceConfig& config, const Eigen::Vector3f& origin,
                      const Eigen::Vector3f& dir, int depth,
                      std::mt19937& rng) {
  if (depth > config.max_depth) return Eigen::Vector3f::Zero();

  Ray visibility_ray;
  visibility_ray.origin = origin;
  visibility_ray.direction = dir;
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

  // Hit surface
  const Material& mat = config.scene.materials[occ->material_id];
  float alpha = GetAlpha(mat, occ->uv);

  Eigen::Vector3f color = Eigen::Vector3f::Zero();

  // If alpha < 1.0, continue ray
  if (alpha < 1.0f) {
    // Transmission
    Eigen::Vector3f hit_pos = occ->position + dir * 0.001f;
    Eigen::Vector3f transmission = Trace(config, hit_pos, dir, depth + 1, rng);
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

  Eigen::Vector3f hit_pos = occ->position + occ->normal * 0.001f;

  // Direct Lighting (NEE)
  // EvaluateLights returns L_e(x, x')
  Eigen::Vector3f L_direct =
      EvaluateLightSamples(config.scene, config.rtc_scene, hit_pos, occ->normal,
                           -dir, mat, occ->uv, config.num_light_samples, rng);
  color += alpha * L_direct;

  // Indirect Lighting (Recursive)
  ReflectionSample sample =
      SampleMaterial(mat, occ->uv, occ->normal, -dir, rng);
  if (sample.pdf < 1e-3f) {
    // Internal reflection.
    return color;
  }

  Eigen::Vector3f incoming =
      Trace(config, hit_pos, sample.direction, depth + 1, rng);
  Eigen::Vector3f brdf =
      EvalMaterial(mat, occ->uv, occ->normal, sample.direction, -dir);
  float cosine_term = occ->normal.dot(sample.direction);
  Eigen::Vector3f L_indirect =
      incoming.cwiseProduct(brdf) * (cosine_term / sample.pdf);
  color += alpha * L_indirect;

  return color;
}

template <typename T>
std::vector<T> DownsampleTexture(const std::vector<T>& input, int input_width,
                                 int input_height, int scale) {
  if (scale <= 1) return input;

  const int output_width = input_width / scale;
  const int output_height = input_height / scale;

  std::vector<T> output(output_width * output_height);

  for (int y = 0; y < output_height; ++y) {
    for (int x = 0; x < output_width; ++x) {
      T avg = T();
      int count = 0;
      for (int dy = 0; dy < scale; ++dy) {
        for (int dx = 0; dx < scale; ++dx) {
          int sx = x * scale + dx;
          int sy = y * scale + dy;
          if (sx < input_width && sy < input_height) {
            avg += input[sy * input_width + sx];
            count++;
          }
        }
      }
      if (count > 0) {
        output[y * output_width + x] = avg * (1.f / float(count));
      }
    }
  }

  return output;
}

}  // namespace

BakeResult BakeSHLightMap(const Scene& scene,
                          const std::vector<SurfacePoint>& surface_points,
                          const RasterConfig& raster_config,
                          const BakeConfig& config) {
  BakeResult result;
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

  result.sh_texture.width = width;
  result.sh_texture.height = height;
  result.sh_texture.pixels.resize(width * height);

  result.environment_visibility_texture.width = width;
  result.environment_visibility_texture.height = height;
  result.environment_visibility_texture.pixel_data.resize(width * height);

  RTCDevice device = rtcNewDevice(nullptr);
  RTCScene rtc_scene = BuildBVH(scene, device);

  float inv_pdf_uniform = 2.0f * M_PI;

  size_t total_pixels = surface_points.size();
  std::atomic<size_t> processed_count{0};
  std::atomic<int> last_progress{-1};

  tbb::parallel_for(
      tbb::blocked_range<size_t>(0, total_pixels),
      [&](const tbb::blocked_range<size_t>& r) {
        // Each thread needs its own RNG? Or use thread-local?
        // Simple way: Seed based on index.
        for (size_t idx = r.begin(); idx != r.end(); ++idx) {
          // Update progress.
          size_t done =
              processed_count.fetch_add(1, std::memory_order_relaxed) + 1;
          int percent = static_cast<int>((done * 100) / total_pixels);
          int expected = last_progress.load(std::memory_order_relaxed);
          if (percent > expected) {
            if (last_progress.compare_exchange_strong(expected, percent)) {
              std::cout << "\r[Baking] " << percent << "%" << std::flush;
            }
          }

          // Process lightmap texel.
          const SurfacePoint& sp = surface_points[idx];
          if (sp.material_id < 0) {
            // Invalid lightmap texel.
            continue;
          }

          // Accumulate SH coefficients for the specified number of samples.
          SHCoeffs sh_accum;

          std::mt19937 rng(12345 +
                           idx);  // Seeding RNG with index to make it
                                  // deterministic but different per pixel
          Eigen::Vector3f origin =
              sp.position +
              sp.normal * 0.001f;  // Offset position to avoid self-intersection

          for (int s = 0; s < config.samples; ++s) {
            Eigen::Vector3f dir_local =
                SampleHemisphereUniform(rng);  // Z is up

            // Transform to World
            // Calculate bitangent (using w for handedness)
            Eigen::Vector3f bitangent =
                sp.normal.cross(sp.tangent.head<3>()) * sp.tangent.w();
            Eigen::Vector3f dir_world = sp.tangent.head<3>() * dir_local.x() +
                                        bitangent * dir_local.y() +
                                        sp.normal * dir_local.z();

            // Direct lighting (NEE).
            AccumulateIncomingLightSamples(scene, rtc_scene, sp.position,
                                           sp.normal, config.num_light_samples,
                                           rng, &sh_accum);

            // Indirect lighting.
            TraceConfig trace_config(
                rtc_scene, scene, config.bounces, config.num_light_samples,
                /*on_direct_hit_sky_fn=*/[&result, idx]() {
                  result.environment_visibility_texture.pixel_data[idx] += 1.0f;
                });
            Eigen::Vector3f Li_indirect =
                Trace(trace_config, origin, dir_world, /*depth=*/0, rng) *
                inv_pdf_uniform;

            AccumulateRadiance(Li_indirect, dir_world, &sh_accum);
          }

          // Average
          result.sh_texture.pixels[idx] = sh_accum * (1.0f / config.samples);
          result.environment_visibility_texture.pixel_data[idx] *=
              (1.0f / config.samples);
        }
      });
  std::cout << std::endl;

  // Clean up
  ReleaseBVH(rtc_scene);
  rtcReleaseDevice(device);

  return result;
}

SHTexture DownsampleSHTexture(const SHTexture& input, int scale) {
  SHTexture output;
  output.width = input.width / scale;
  output.height = input.height / scale;
  output.pixels =
      DownsampleTexture(input.pixels, input.width, input.height, scale);
  return output;
}

Texture32F DownsampleEnvironmentVisibilityTexture(const Texture32F& input,
                                                  int scale) {
  Texture32F output;
  output.width = input.width / scale;
  output.height = input.height / scale;
  output.pixel_data =
      DownsampleTexture(input.pixel_data, input.width, input.height, scale);
  return output;
}

}  // namespace sh_baker
