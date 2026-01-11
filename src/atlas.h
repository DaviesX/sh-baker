#ifndef SH_BAKER_SRC_ATLAS_H_
#define SH_BAKER_SRC_ATLAS_H_

#include <vector>

#include "scene.h"

namespace sh_baker {
namespace atlas_internal {

// Calculates the scaling factor for each geometry based on its material's
// albedo texture resolution and UV tiling.
//
// The heuristic is:
//   TargetScale = density_multiplier * sqrt(AlbedoWidth * AlbedoHeight)
//   EffectiveScale = TargetScale * sqrt(TileCount)
//
// Where TileCount is estimated from the UV bounding box range.
// The function also applies outlier clamping to ensure no single mesh exceeds
// 5x the median scale.
std::vector<float> CalculateGeometryScales(
    const std::vector<Geometry>& geometries,
    const std::vector<Material>& materials, float density_multiplier);

}  // namespace atlas_internal

struct AtlasResult {
  // Atlas geometries.
  std::vector<Geometry> geometries;

  // Required atlas width.
  int width;

  // Required atlas height.
  int height;
};

// Generates a new set of geometries with non-overlapping lightmap UVs using
// xatlas. This function will re-index the geometry (splitting vertices at
// seams) and populate the lightmap_uvs field.
//
// Arguments:
//   scene:      The scene containing geometries and materials.
//   target_resolution: The target resolution for the atlas (width and height).
//   padding:    The minimum padding between charts in pixels.
//   density_multiplier: Global multiplier for the target density.
std::optional<AtlasResult> CreateAtlasGeometries(
    const Scene& scene, unsigned target_resolution, unsigned padding,
    float density_multiplier = 1.0f);

}  // namespace sh_baker

#endif  // SH_BAKER_SRC_ATLAS_H_
