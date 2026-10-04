#ifndef SH_BAKER_SRC_LOADER_Q3MAP2_H_
#define SH_BAKER_SRC_LOADER_Q3MAP2_H_

#include <Eigen/Dense>
#include <cstdint>
#include <vector>

#include "scene.h"

namespace sh_baker {

// One q3map2 draw surface, as q3map2 hands it to sh-baker for tracing.
struct Q3Map2Surface {
  std::vector<Eigen::Vector3f> positions;    // world space
  std::vector<Eigen::Vector3f> normals;      // one per vertex
  std::vector<Eigen::Vector2f> texture_uvs;  // may be empty for occluders
  std::vector<uint32_t> indices;  // q3map2 winding: front is clockwise
  int material_id = -1;           // < 0: pure occluder
};

// Appends one Geometry for tracing to `scene`, with no lightmap UVs and an
// identity transform. Nothing else in the scene changes.
//
// Input only a caller bug or corrupt data can produce fails a CHECK, whose
// message names the field and value:
//   - no positions;
//   - normals not one per vertex;
//   - texture UVs not one per vertex, when the surface has a material or
//     gives any;
//   - an index count that is not a multiple of 3, or an index not below the
//     vertex count;
//   - a material index not below the scene's material count;
//   - a non-finite position, normal or texture UV.
// Texture UVs outside [0, 1] tile and are accepted.
//
// What real content produces is repaired:
//   - each triangle's second and third index are swapped, from q3map2's
//     clockwise front faces to sh-baker's counter-clockwise ones;
//   - every normal is normalized; one shorter than 1e-6 (degenerate patches
//     and models make them) becomes the normalized sum of its triangles'
//     front-face normals, or (0, 0, 1) if that sum is degenerate too;
//   - an occluder without texture UVs gets (0, 0) for each vertex;
//   - triangles that repeat an index are dropped.
// Tangents come from MikkTSpace when texture UVs are given, and from the
// fallback basis otherwise, as in the glTF loader. MikkTSpace writes one
// tangent per vertex, so a vertex shared across a UV seam gets the tangent of
// whichever face wrote it last; q3map2 splits its vertices where `st` differs.
//
// Assembly order: add every surface first, then create the lights, then build
// the BVH. Lights and the BVH hold raw pointers into scene->geometries, which
// an append can move. Adding a surface to a scene that already holds a light
// with a geometry pointer (an area light) fails a CHECK.
void AddQ3Map2Surface(const Q3Map2Surface& surface, Scene* scene);

}  // namespace sh_baker

#endif  // SH_BAKER_SRC_LOADER_Q3MAP2_H_
