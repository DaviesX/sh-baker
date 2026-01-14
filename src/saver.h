#ifndef SH_BAKER_SRC_SAVER_H_
#define SH_BAKER_SRC_SAVER_H_

#include <filesystem>
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
//   Saves 10 separate files.
//   'path' is treated as a base name. If 'path' is "output/lightmap.exr",
//   it generates "output/lightmap_L0.exr", "output/lightmap_L1m1.exr", etc.
//   The 10th file is "output/lightmap_EnvVisibility.exr".
//
// In kLuminancePacked mode:
//   Saves 3 RGBA textures.
//   'path' is treated as a base name. If 'path' is "output/lightmap.exr",
//   it generates "output/lightmap_0.exr", "output/lightmap_1.exr", etc.
//   The 3rd file is "output/lightmap_2.exr".
//
// Returns true on success.
bool SaveSHLightMap(const SHTexture& sh_texture,
                    const Texture32F& environment_visibility_texture,
                    const std::filesystem::path& path,
                    SaveMode mode = SaveMode::kCombined);

// Saves the Scene to a glTF file.
// Serializes geometry with both texture_uvs (TEXCOORD_0) and lightmap_uvs
// (TEXCOORD_1).
bool SaveScene(const Scene& scene, const std::filesystem::path& path);

// Saves a standard Texture to a PNG file.
bool SaveTexture(const Texture& texture, const std::filesystem::path& path);

}  // namespace sh_baker

#endif  // SH_BAKER_SRC_SAVER_H_
