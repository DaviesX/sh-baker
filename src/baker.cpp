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

  const bool split = config.split_punctual_direct;
  if (split) {
    result.punctual_direct_sh_texture.width = width;
    result.punctual_direct_sh_texture.height = height;
    result.punctual_direct_sh_texture.pixels.resize(width * height,
                                                    SHCoeffs(-1));
    result.punctual_direct_irradiance.width = width;
    result.punctual_direct_irradiance.height = height;
    result.punctual_direct_irradiance.channels = 3;
    result.punctual_direct_irradiance.pixel_data.resize(width * height * 3, -1);
  }

  RTCDevice device = rtcNewDevice(nullptr);
  RTCScene rtc_scene = BuildBVH(scene, device);

  // Full light tree (punctual + area).
  LightTree light_tree;
  light_tree.Build(scene.lights);

  // Area lights only tree.
  std::vector<Light> area_lights;
  area_lights.reserve(scene.lights.size());
  std::copy_if(
      scene.lights.begin(), scene.lights.end(), std::back_inserter(area_lights),
      [](const Light& light) { return light.type == Light::Type::Area; });
  LightTree area_light_tree;
  area_light_tree.Build(area_lights);

  // Determine which light tree to use for direct lighting contribution based on
  // the bake config. Splitting the punctual direct light off drops it from the
  // main result exactly as indirect_only does.
  const LightTree& direct_light_tree =
      (config.indirect_only || split) ? area_light_tree : light_tree;

  float inv_pdf_uniform = 2.0f * M_PI;

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

          float visibility_accum = 0.0f;

          // Loop-invariant for this texel: build once, not per sample (the
          // std::function makes per-iteration construction needlessly costly).
          TraceConfig trace_config(
              rtc_scene, scene, &light_tree, config.bounces,
              config.num_light_samples,
              /*on_direct_hit_sky_fn=*/
              [&visibility_accum]() { visibility_accum += 1.0f; },
              config.firefly_clamp);

          while (true) {
            std::optional<Ray> ray = SampleRay(sensor, rng);
            if (!ray.has_value()) {
              break;
            }

            // Direct lighting (NEE). `direct_light_tree` is the full light set
            // for a normal bake, or area-lights-only under indirect_only (their
            // direct term is always baked; punctual direct is real-time). Empty
            // area set under indirect_only -> AccumulateIncomingLightSamples
            // returns immediately.
            SHCoeffs sample_sh_accum;  // Accumulate for this sample only
            AccumulateIncomingLightSamples(
                &direct_light_tree, rtc_scene, scene.materials, sp.position,
                sp.normal, config.num_light_samples, rng, &sample_sh_accum);

            // Indirect lighting.
            Eigen::Vector3f Li_indirect =
                Trace(trace_config, *ray, /*depth=*/0, rng) * inv_pdf_uniform;
            AccumulateRadiance(Li_indirect, ray->direction, sp.normal,
                               &sample_sh_accum);

            AddSample(sample_sh_accum, &sensor);
          }

          result.sh_texture.pixels[idx] = GetEstimation(sensor);
          float inv_samples =
              sensor.sample_count > 0 ? (1.0f / sensor.sample_count) : 0.0f;
          result.environment_visibility_texture.pixel_data[idx] =
              visibility_accum * inv_samples;

          if (split) {
            // Exact punctual direct light: every light once, no sampling.
            SHCoeffs punctual;
            AccumulatePunctualDirect(scene.lights, rtc_scene, scene.materials,
                                     sp.position, sp.normal, &punctual);
            result.punctual_direct_irradiance.pixel_data[idx * 3 + 0] =
                punctual.irradiance.x();
            result.punctual_direct_irradiance.pixel_data[idx * 3 + 1] =
                punctual.irradiance.y();
            result.punctual_direct_irradiance.pixel_data[idx * 3 + 2] =
                punctual.irradiance.z();
            result.punctual_direct_sh_texture.pixels[idx] = punctual;
          }

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
