#ifndef SH_BAKER_SRC_BAKER_H_
#define SH_BAKER_SRC_BAKER_H_

#include <cstddef>
#include <functional>

#include "rasterizer.h"
#include "saver.h"
#include "scene.h"

namespace sh_baker {

// BakeConfig and BakeResult are shared with callers by layout, with no ABI
// versioning: callers (q3map2) must be rebuilt against these headers whenever
// libsh_baker.so moves to a new commit.
struct BakeConfig {
  int samples = 128;          // Rays per texel
  int bounces = 5;            // Max path depth
  int num_light_samples = 1;  // Number of light samples for NEE
  bool indirect_only = false;  // Skip direct lighting at baked points.
  // Max per-sample indirect luminance; over-bright samples are scaled down to
  // suppress fireflies. <= 0 disables (unbiased).
  float firefly_clamp = 0.0f;
  // Adaptive-sampling stop tolerance: stop a texel once the 3-sigma SEM of its
  // luminance falls below confidence_threshold * mean.
  float confidence_threshold = 0.01f;

  // Optional progress report: called with (points finished, total points),
  // where finished counts points baked or skipped as invalid. Calls come from
  // bake worker threads but never overlap; `done` strictly increases, at most
  // once per whole percent of done / total. A bake of at least one point that
  // is not cancelled ends with (total, total). Not called for zero points.
  std::function<void(size_t done, size_t total)> progress_fn;

  // Optional cancel poll: called before each point in the bake loop. Polls never overlap, but may come from
  // different worker threads, so the callback need not be thread-safe. Once a
  // poll returns true there are no more: no further point starts, points in
  // progress finish, and the bake returns with BakeResult::cancelled set.
  std::function<bool()> cancel_fn;

  // Called once on the calling thread after the bake finishes, including for
  // empty or cancelled bakes, before BakeSHLightMap returns.
  std::function<void()> done_fn;

  // The progress and cancel callbacks must be short, and must not call back into sh-baker, run
  // TBB work, or block waiting on the bake: they run on bake worker threads,
  // and the progress callback runs under the bake's reporting lock. Either may
  // be left empty. done_fn may also be left empty.
};

struct BakeResult {
  SHTexture sh_texture;
  Texture32F environment_visibility_texture;
  // True when cancel_fn returned true, even if every point had already
  // started. Points that were not baked keep the not-baked marker, as invalid
  // points do; every baked point holds the result a full bake gives it.
  bool cancelled = false;
};

// Bakes SH lighting at the given surface points, one result per point.
// `surface_points` holds width * height points, where width and height are
// raster_config's times its supersample_scale; RasterizeScene() makes such a
// buffer from the scene's lightmap UVs. Any list of points bakes the same way,
// with no lightmap UVs and no rasterization: lay N points out as N x 1 (width
// N, height 1, supersample_scale 1). Each point supplies its position, a unit
// normal, a unit tangent perpendicular to the normal with w of +1 or -1 (the
// tangent only orients the sampling hemisphere, and w = 0 would flatten it),
// and material_id >= 0; a point with material_id < 0 is skipped. Result i
// belongs to point i, and a skipped point keeps the not-baked marker
// (SHCoeffs(-1), environment visibility -1). One call builds the BVH and the
// light trees once for all its points.
BakeResult BakeSHLightMap(const Scene& scene,
                          const std::vector<SurfacePoint>& surface_points,
                          const RasterConfig& raster_config,
                          const BakeConfig& config);

// Downsamples an SH texture by averaging block of scale x scale pixels.
SHTexture DownsampleSHTexture(const SHTexture& input, int scale);

// Downsamples an environment visibility texture by averaging block of scale x
// scale pixels.
Texture32F DownsampleEnvironmentVisibilityTexture(const Texture32F& input,
                                                  int scale);

}  // namespace sh_baker

#endif  // SH_BAKER_SRC_BAKER_H_
