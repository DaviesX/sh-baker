#ifndef SH_BAKER_SRC_TEXTURE_H_
#define SH_BAKER_SRC_TEXTURE_H_

#include <cstdint>
#include <filesystem>
#include <optional>
#include <vector>

namespace sh_baker {

// --- Texture ---
struct Texture {
  // If set, the texture is loaded from a file. This denotes the provenance of
  // the texture.
  std::optional<std::filesystem::path> file_path;

  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t channels = 0;
  std::vector<uint8_t> pixel_data;
};

// --- Texture32F ---
struct Texture32F {
  // If set, the texture is loaded from a file. This denotes the provenance of
  // the texture.
  std::optional<std::filesystem::path> file_path;

  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t channels = 0;
  std::vector<float> pixel_data;
};

// --- Texture32I ---
struct Texture32I {
  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t channels = 0;
  std::vector<int32_t> pixel_data;
};

}  // namespace sh_baker

#endif  // SH_BAKER_SRC_TEXTURE_H_
