#ifndef SH_BAKER_SRC_LOADER_H_
#define SH_BAKER_SRC_LOADER_H_

#include <filesystem>
#include <optional>

#include "scene.h"

namespace sh_baker {

// Loads a glTF file and returns a Scene object.
// Returns std::nullopt if loading fails.
std::optional<Scene> LoadScene(const std::filesystem::path& gltf_file);

}  // namespace sh_baker

#endif  // SH_BAKER_SRC_LOADER_H_
