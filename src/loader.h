#ifndef SH_BAKER_SRC_LOADER_H_
#define SH_BAKER_SRC_LOADER_H_

#include <filesystem>
#include <optional>

#include "scene.h"

namespace sh_baker {

// Loads a glTF file and returns a Scene object. Punctual/directional/environment
// lights are populated; AREA lights are NOT (they depend on the final geometry
// -- the atlas renumbers/drops triangles, which their emission CDFs index). Call
// CreateAreaLights once the geometry pipeline is finalized.
// Returns std::nullopt if loading fails.
std::optional<Scene> LoadScene(const std::filesystem::path& gltf_file);

// Appends area lights for the scene's emissive materials, built from the CURRENT
// scene.geometries (emitter geometry + emission CDF / prim_id_map). Call after
// the geometry is final so the CDFs match the geometry the bake samples.
void CreateAreaLights(Scene& scene);

}  // namespace sh_baker

#endif  // SH_BAKER_SRC_LOADER_H_
