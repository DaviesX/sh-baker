#ifndef SH_BAKER_SRC_ATLAS_H_
#define SH_BAKER_SRC_ATLAS_H_

#include <vector>

#include "scene.h"

namespace sh_baker {

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
//   geometries: The input geometries.
//   target_resolution: The target resolution for the atlas (width and height).
//   padding:    The minimum padding between charts in pixels.
std::optional<AtlasResult> CreateAtlasGeometries(
    const std::vector<Geometry>& geometries, unsigned target_resolution,
    unsigned padding);

}  // namespace sh_baker

#endif  // SH_BAKER_SRC_ATLAS_H_
