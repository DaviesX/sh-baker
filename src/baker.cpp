#include "baker.h"

#include <embree4/rtcore.h>
#include <glog/logging.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <algorithm>
#include <atomic>
#include <functional>
#include <mutex>
#include <random>

#include "light.h"
#include "light_tree.h"
#include "rasterizer.h"
#include "sensor.h"
#include "tracer.h"

#define USE_TBB 1

namespace sh_baker {
namespace {

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

class BakeObserver {
 public:
  BakeObserver(const BakeConfig& config, size_t total_tasks)
      : config_(config), total_tasks_(total_tasks) {}

  bool Cancelled() {
    if (!config_.cancel_fn) {
      return false;
    }

    if (cancelled_.load(std::memory_order_acquire)) {
      return true;
    }

    std::lock_guard<std::mutex> lock(cancel_mutex_);
    if (!cancelled_.load(std::memory_order_relaxed) && config_.cancel_fn()) {
      cancelled_.store(true, std::memory_order_release);
    }
    return cancelled_.load(std::memory_order_relaxed);
  }

  void FinishOneTask() {
    if (!config_.progress_fn) {
      return;
    }

    const size_t done_tasks =
        finished_tasks_.fetch_add(1, std::memory_order_relaxed) + 1;
    const int percent = static_cast<int>((done_tasks * 100) / total_tasks_);
    if (percent <= last_percent_.load(std::memory_order_relaxed)) {
      return;
    }

    std::lock_guard<std::mutex> lock(progress_mutex_);
    if (percent <= last_percent_.load(std::memory_order_relaxed)) {
      return;
    }
    last_percent_.store(percent, std::memory_order_relaxed);

    config_.progress_fn(done_tasks, total_tasks_);
  }

  bool WasCancelled() const {
    return cancelled_.load(std::memory_order_acquire);
  }

  void Done() {
    if (config_.done_fn) {
      config_.done_fn();
    }
  }

 private:
  const BakeConfig& config_;
  const size_t total_tasks_;
  std::atomic<bool> cancelled_{false};
  std::mutex cancel_mutex_;
  std::atomic<size_t> finished_tasks_{0};
  std::atomic<int> last_percent_{-1};
  std::mutex progress_mutex_;
};

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
  result.sh_texture.pixels.resize(width * height, SHCoeffs(-1));

  result.environment_visibility_texture.width = width;
  result.environment_visibility_texture.height = height;
  result.environment_visibility_texture.pixel_data.resize(width * height, -1);

  RTCDevice device = rtcNewDevice(nullptr);
  RTCScene rtc_scene = BuildBVH(scene, device);

  // Full light tree (punctual + area).
  LightTree light_tree;
  light_tree.Build(scene.lights);

  const size_t total_pixels = surface_points.size();
  BakeObserver bake_observer(config, total_pixels);

#if USE_TBB
  tbb::parallel_for(
      tbb::blocked_range<size_t>(0, total_pixels),
      [&](const tbb::blocked_range<size_t>& r) {
        for (size_t idx = r.begin(); idx != r.end(); ++idx) {
#else
  for (size_t idx = 0; idx < total_pixels; ++idx) {
#endif
          // Process lightmap texel.
          // Polled before the validity check: once cancelled, no further
          // point is baked or counted as done.
          if (bake_observer.Cancelled()) break;

          const SurfacePoint& sp = surface_points[idx];
          if (sp.material_id < 0) {
            // Invalid lightmap texel.
            bake_observer.FinishOneTask();
            continue;
          }

          std::mt19937 rng(12345 +
                           idx);  // Seeding RNG with index to make it
                                  // deterministic but different per pixel

          Sensor sensor(sp, config.samples, config.confidence_threshold);
          // 2 pi for a surface point, 4 pi for a full-sphere point.
          const float inv_pdf_uniform = SampleRayInvPdf(sensor);

          // Environment visibility. A surface point's is cosine-weighted,
          // sum(escaped_i cos_i) / sum(cos_i) over its first rays, the weight
          // irradiance needs (a ratio estimator: exactly 1 fully open, 0
          // closed, always in [0, 1]). A full-sphere point has no normal, so
          // its visibility is the uniform fraction of first rays that escape.
          bool escaped = false;
          float escapes = 0.0f;
          float vis_num = 0.0f;
          float vis_den = 0.0f;

          // Loop-invariant for this texel: build once, not per sample (the
          // std::function makes per-iteration construction needlessly costly).
          TraceConfig trace_config(
              rtc_scene, scene, &light_tree, config.bounces,
              config.num_light_samples,
              /*on_direct_hit_sky_fn=*/
              [&escaped]() { escaped = true; }, config.firefly_clamp,
              config.indirect_only);

          while (true) {
            std::optional<Ray> ray = SampleRay(sensor, rng);
            if (!ray.has_value()) {
              break;
            }

            SHCoeffs sample_sh_accum;  // Accumulate for this sample only
            if (!config.indirect_only) {
              AccumulateIncomingLightSamples(
                  &light_tree, rtc_scene, scene.materials, sp.position,
                  sp.normal, config.num_light_samples, rng, &sample_sh_accum,
                  sp.full_sphere);
            }

            // Trace incident radiance, including direct sky in a full bake.
            escaped = false;
            Eigen::Vector3f Li =
                Trace(trace_config, *ray, /*depth=*/0, rng) * inv_pdf_uniform;
            if (sp.full_sphere) {
              if (escaped) escapes += 1.0f;
            } else {
              const float cos_theta =
                  std::max(0.0f, ray->direction.dot(sp.normal));
              vis_den += cos_theta;
              if (escaped) vis_num += cos_theta;
            }
            AccumulateRadiance(Li, ray->direction, sp.normal,
                               &sample_sh_accum);

            AddSample(sample_sh_accum, &sensor);
          }

          result.sh_texture.pixels[idx] = GetEstimation(sensor);
          float visibility;
          if (sp.full_sphere) {
            visibility = sensor.sample_count > 0
                             ? escapes * (1.0f / sensor.sample_count)
                             : 0.0f;
          } else {
            visibility = vis_den > 0.0f ? vis_num / vis_den : 0.0f;
          }
          result.environment_visibility_texture.pixel_data[idx] = visibility;

          bake_observer.FinishOneTask();
        }
#if USE_TBB
      });
#endif

  bake_observer.Done();

  // Clean up
  ReleaseBVH(rtc_scene);
  rtcReleaseDevice(device);

  result.cancelled = bake_observer.WasCancelled();
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
