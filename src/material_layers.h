#ifndef SH_BAKER_SRC_MATERIAL_LAYERS_H_
#define SH_BAKER_SRC_MATERIAL_LAYERS_H_

#include <filesystem>
#include <unordered_map>

#include "tiny_gltf.h"

namespace sh_baker {

// Verbatim carrier for a material's `SH_material_layers` glTF extension so the
// baker can pass it through to its output unchanged. The baker composites the
// stack into mat.albedo for its own bake, but that parse is lossy on exactly the
// data the renderer needs (animMap frames, scroll/rotate/turb tcMod), so the
// pass-through keeps the original JSON and only remaps texture indices at save.
struct MaterialLayers {
  // The original extension Value. Its `layers[].texture.index` and
  // `layers[].animFrames[]` refer to the INPUT model's textures.
  tinygltf::Value extension;
  // Resolves each input texture index referenced by `extension` to its absolute
  // source file path, so the saver can copy the image and assign a fresh index.
  std::unordered_map<int, std::filesystem::path> texture_paths;
};

}  // namespace sh_baker

#endif  // SH_BAKER_SRC_MATERIAL_LAYERS_H_
