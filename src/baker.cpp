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
#include "tracer.h"

namespace sh_baker {

template <typename T, typename ValidateFn>
std::vector<T> DownsampleTexture(const std::vector<T>& input, int input_width,
                                 int input_height, int scale,
                                 ValidateFn validate_fn) {
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
          int index = sy * input_width + sx;
          if (sx < input_width && sy < input_height &&
              validate_fn(input[index])) {
            avg += input[index];
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
  result.sh_texture.pixels.resize(width * height, SHCoeffs(-1));

  result.environment_visibility_texture.width = width;
  result.environment_visibility_texture.height = height;
  result.environment_visibility_texture.pixel_data.resize(width * height, -1);

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

          // Accumulate SH coefficients and environment visibility factor for
          // the specified number of samples.
          SHCoeffs sh_accum;
          float visibility_accum = 0.0f;

          std::mt19937 rng(12345 +
                           idx);  // Seeding RNG with index to make it
                                  // deterministic but different per pixel
          Eigen::Vector3f origin =
              sp.position +
              sp.normal * 0.005f;  // Offset position to avoid self-intersection

          int actual_samples = 0;

          // Online variance tracking (Welford's algorithm)
          // We track the L0 luminance (DC component brightness).
          float mean_lum = 0.0f;
          float m2_lum = 0.0f;

          // Hardcoded adaptive sampling parameters
          constexpr int kMinSamples = 16;
          // Coefficient of Variation of the Mean (Standard Error / Mean)
          // threshold
          constexpr float kCVThreshold = 0.01f;

          for (int s = 0; s < config.samples; ++s) {
            actual_samples++;
            Eigen::Vector3f dir_local =
                SampleHemisphereUniform(rng);  // Z is up

            // Transform to World
            // Calculate bitangent (using w for handedness)
            Eigen::Vector3f bitangent =
                (sp.normal.cross(sp.tangent.head<3>()) * sp.tangent.w())
                    .normalized();
            if (sp.tangent.w() > 0) {
              CHECK_NEAR(sp.tangent.w(), 1.0f, 1e-2f);
            } else {
              CHECK_NEAR(sp.tangent.w(), -1.0f, 1e-2f);
            }

            Eigen::Vector3f dir_world = sp.tangent.head<3>() * dir_local.x() +
                                        bitangent * dir_local.y() +
                                        sp.normal * dir_local.z();

            // Direct lighting (NEE).
            SHCoeffs sample_sh_accum;  // Accumulate for this sample only
            AccumulateIncomingLightSamples(scene, rtc_scene, sp.position,
                                           sp.normal, config.num_light_samples,
                                           rng, &sample_sh_accum);

            // Indirect lighting.
            TraceConfig trace_config(
                rtc_scene, scene, config.bounces, config.num_light_samples,
                /*on_direct_hit_sky_fn=*/[&visibility_accum]() {
                  visibility_accum += 1.0f;
                });
            Eigen::Vector3f Li_indirect =
                Trace(trace_config, origin, dir_world, /*depth=*/0, rng) *
                inv_pdf_uniform;

            AccumulateRadiance(Li_indirect, dir_world, &sample_sh_accum);

            // Add to total
            sh_accum += sample_sh_accum;

            // Adaptive Sampling Update
            // Estimate luminance of L0 from this sample
            float lum = sample_sh_accum.coeffs[0].norm();

            float delta = lum - mean_lum;
            mean_lum += delta / actual_samples;
            float delta2 = lum - mean_lum;
            m2_lum += delta * delta2;

            if (actual_samples >= kMinSamples) {
              // Calculate Standard Error of the Mean
              if (m2_lum > 0.0f && mean_lum > 1e-3f) {
                float variance = m2_lum / (actual_samples - 1);
                float std_dev = std::sqrt(variance);
                float sem = std_dev / std::sqrt((float)actual_samples);

                // Coefficient of Variation of the Mean = SEM / Mean
                if (sem < kCVThreshold * mean_lum) {
                  break;
                }
              } else if (mean_lum <= 1e-3f) {
                // If it's pitch black, we can stop early too.
                break;
              }
            }
          }

          // Average
          float inv_samples = 1.0f / actual_samples;
          result.sh_texture.pixels[idx] = sh_accum * inv_samples;
          result.environment_visibility_texture.pixel_data[idx] =
              visibility_accum * inv_samples;
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
      DownsampleTexture(input.pixels, input.width, input.height, scale,
                        /*validate_fn=*/[](const SHCoeffs& sh) {
                          return sh.coeffs[0].x() != -1;
                        });
  return output;
}

Texture32F DownsampleEnvironmentVisibilityTexture(const Texture32F& input,
                                                  int scale) {
  Texture32F output;
  output.width = input.width / scale;
  output.height = input.height / scale;
  output.pixel_data = DownsampleTexture(
      input.pixel_data, input.width, input.height, scale,
      /*validate_fn=*/[](float visibility) { return visibility != -1; });
  return output;
}

}  // namespace sh_baker
