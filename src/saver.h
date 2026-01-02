#ifndef SH_BAKER_SRC_SAVER_H_
#define SH_BAKER_SRC_SAVER_H_

#include <filesystem>
#include <string>
#include <vector>

#include "sh_coeffs.h"

namespace sh_baker {

struct SHTexture {
  int width = 0;
  int height = 0;
  std::vector<SHCoeffs> pixels;
};

// Saves the SH Lightmap to a multi-channel OpenEXR file.
// Channels will be named "L0_R", "L0_G", "L0_B", "L1m1_R", etc.
// Returns true on success.
bool SaveSHLightMap(const SHTexture& sh_texture,
                    const std::filesystem::path& path);

}  // namespace sh_baker

#endif  // SH_BAKER_SRC_SAVER_H_
