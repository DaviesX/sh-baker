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

// Bakes the SH Lightmap for the given scene.
// Returns an SHTexture containing the baked coefficients.
SHTexture BakeSHLightMap(const Scene& scene,
                         const std::vector<SurfacePoint>& surface_points,
                         const RasterConfig& raster_config,
                         const BakeConfig& config);

}  // namespace sh_baker

#endif  // SH_BAKER_SRC_BAKER_H_
