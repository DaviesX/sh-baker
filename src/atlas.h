#ifndef SH_BAKER_SRC_ATLAS_H_
#define SH_BAKER_SRC_ATLAS_H_

#include <vector>

#include "scene.h"

namespace sh_baker {

// Generates a new set of geometries with non-overlapping lightmap UVs using
// xatlas. This function will re-index the geometry (splitting vertices at
// seams) and populate the lightmap_uvs field. Texture UVs, normals, and
// positions are preserved (though re-mapped/duplicated as needed).
std::vector<Geometry> CreateAtlasGeometries(
    const std::vector<Geometry>& geometries);

}  // namespace sh_baker

#endif  // SH_BAKER_SRC_ATLAS_H_
