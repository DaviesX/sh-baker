#ifndef SH_BAKER_SRC_BAKER_H_
#define SH_BAKER_SRC_BAKER_H_

#include "rasterizer.h"
#include "saver.h"
#include "scene.h"

namespace sh_baker {

struct BakeConfig {
  int samples = 128;          // Rays per texel
  int bounces = 3;            // Max path depth
  int num_light_samples = 1;  // Number of light samples for NEE
};

struct BakeResult {
  SHTexture sh_texture;
  Texture32F environment_visibility_texture;
};

// Bakes the SH Lightmap for the given scene.
// Returns an SHTexture containing the baked coefficients.
// Bakes SH lighting for the given surface points (rasterized geometry).
// Returns a texture of size width * height (from RasterConfig).
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
