#ifndef SH_BAKER_SRC_BAKER_H_
#define SH_BAKER_SRC_BAKER_H_

#include "rasterizer.h"
#include "saver.h"
#include "scene.h"

namespace sh_baker {

struct BakeConfig {
  int samples = 128;          // Rays per texel
  int bounces = 5;            // Max path depth
  int num_light_samples = 1;  // Number of light samples for NEE
  bool indirect_only = false;
  // Max per-sample indirect luminance; over-bright samples are scaled down to
  // suppress fireflies. <= 0 disables (unbiased).
  float firefly_clamp = 0.0f;
  // Adaptive-sampling stop tolerance: stop a texel once the 3-sigma SEM of its
  // luminance falls below confidence_threshold * mean.
  float confidence_threshold = 0.01f;
};

struct BakeResult {
  SHTexture sh_texture;
  Texture32F environment_visibility_texture;
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
