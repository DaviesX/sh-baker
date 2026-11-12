#ifndef SH_BAKER_SRC_TANGENTS_H_
#define SH_BAKER_SRC_TANGENTS_H_

#include "scene.h"

namespace sh_baker {

// Fills geo->tangents (one per vertex) if empty: MikkTSpace when
// use_texture_uvs, else the fallback basis. Shared by every loader, so all
// producers generate tangents the same way.
void GenerateTangents(Geometry* geo, bool use_texture_uvs);

}  // namespace sh_baker

#endif  // SH_BAKER_SRC_TANGENTS_H_
