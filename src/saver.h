#ifndef SH_BAKER_SRC_SAVER_H_
#define SH_BAKER_SRC_SAVER_H_

#include <filesystem>
#include <string>
#include <vector>

#include "scene.h"
#include "sh_coeffs.h"

namespace sh_baker {

struct SHTexture {
  int width = 0;
  int height = 0;
  std::vector<SHCoeffs> pixels;
};

// Mode for saving SH Lightmap
enum class SaveMode {
  kCombined,        // One 27-channel EXR
  kSplitChannels,   // 9 separate 3-channel EXRs (RGB per coeff)
  kLuminancePacked  // 3 RGBA textures, L1/L2 luminance only
};

// Saves the SH Lightmap to OpenEXR file(s).
//
// In kCombined mode:
//   Saves a single file to 'path'.
//   Channels will be named "L0_R", "L0_G", "L0_B", "L1m1_R", etc.
//
// In kSplitChannels mode:
//   Saves 9 separate files.
//   'path' is treated as a base name. If 'path' is "output/lightmap.exr",
//   it generates "output/lightmap_L0.exr", "output/lightmap_L1m1.exr", etc.
//
// Returns true on success.
bool SaveSHLightMap(const SHTexture& sh_texture,
                    const std::filesystem::path& path,
                    SaveMode mode = SaveMode::kCombined);

// Saves the Scene to a glTF file.
// Serializes geometry with both texture_uvs (TEXCOORD_0) and lightmap_uvs
// (TEXCOORD_1).
bool SaveScene(const Scene& scene, const std::filesystem::path& path);

}  // namespace sh_baker

#endif  // SH_BAKER_SRC_SAVER_H_
