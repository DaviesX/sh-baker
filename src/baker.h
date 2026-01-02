#ifndef SH_BAKER_SRC_BAKER_H_
#define SH_BAKER_SRC_BAKER_H_

#include "saver.h"
#include "scene.h"

namespace sh_baker {

struct BakeConfig {
  int width = 1024;
  int height = 1024;
  int samples = 128;  // Rays per texel
  int bounces = 3;    // Max path depth
};

// Bakes the SH Lightmap for the given scene.
// Returns an SHTexture containing the baked coefficients.
SHTexture BakeSHLightMap(const Scene& scene, const BakeConfig& config);

}  // namespace sh_baker

#endif  // SH_BAKER_SRC_BAKER_H_
