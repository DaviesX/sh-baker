#include "atlas.h"

#include <glog/logging.h>
#include <xatlas.h>

#include <vector>

namespace sh_baker {

std::vector<Geometry> CreateAtlasGeometries(
    const std::vector<Geometry>& geometries) {
  if (geometries.empty()) {
    return {};
  }

  // 1. Create Atlas
  xatlas::Atlas* atlas = xatlas::Create();

  // 2. Add Meshes
  // We need to keep track of which geometry corresponds to which mesh added to
  // xatlas In this simple case, the order is preserved.
  for (size_t i = 0; i < geometries.size(); ++i) {
    const auto& geo = geometries[i];
    xatlas::MeshDecl mesh_decl;
    mesh_decl.vertexCount = (uint32_t)geo.vertices.size();
    mesh_decl.vertexPositionData = geo.vertices.data();
    mesh_decl.vertexPositionStride = sizeof(Eigen::Vector3f);

    if (!geo.normals.empty()) {
      mesh_decl.vertexNormalData = geo.normals.data();
      mesh_decl.vertexNormalStride = sizeof(Eigen::Vector3f);
    }

    if (!geo.texture_uvs.empty()) {
      mesh_decl.vertexUvData = geo.texture_uvs.data();
      mesh_decl.vertexUvStride = sizeof(Eigen::Vector2f);
    }

    mesh_decl.indexCount = (uint32_t)geo.indices.size();
    mesh_decl.indexData = geo.indices.data();
    mesh_decl.indexFormat = xatlas::IndexFormat::UInt32;

    xatlas::AddMeshError error = xatlas::AddMesh(atlas, mesh_decl);
    if (error != xatlas::AddMeshError::Success) {
      LOG(ERROR) << "Error adding mesh " << i
                 << " to xatlas: " << xatlas::StringForEnum(error);
      xatlas::Destroy(atlas);
      return geometries;  // Fallback? Or empty? usage suggests we return valid
                          // geometries.
      // If we fail, returning original geometries means lightmap_uvs is empty,
      // likely causing crash or no bake.
    }
  }

  // 3. Generate Atlas
  xatlas::Generate(atlas);

  // 4. Reconstruct Geometries
  std::vector<Geometry> result_geometries;
  result_geometries.reserve(geometries.size());

  for (size_t i = 0; i < geometries.size(); ++i) {
    const auto& src_geo = geometries[i];
    const xatlas::Mesh& atlas_mesh = atlas->meshes[i];

    Geometry new_geo;
    new_geo.material_id = src_geo.material_id;
    new_geo.transform = src_geo.transform;

    uint32_t new_vertex_count = atlas_mesh.vertexCount;
    new_geo.vertices.resize(new_vertex_count);
    new_geo.normals.resize(new_vertex_count);
    new_geo.texture_uvs.resize(new_vertex_count);
    new_geo.lightmap_uvs.resize(new_vertex_count);
    new_geo.tangents.resize(new_vertex_count);

    // Fill vertices
    for (uint32_t v = 0; v < new_vertex_count; ++v) {
      const auto& vertex_ref = atlas_mesh.vertexArray[v];
      uint32_t original_index = vertex_ref.xref;

      // Copy attributes from original mesh
      if (original_index < src_geo.vertices.size()) {
        // Enforce invariants
        CHECK(!src_geo.normals.empty()) << "Source geometry must have normals";
        CHECK(!src_geo.tangents.empty())
            << "Source geometry must have tangents";

        new_geo.vertices[v] = src_geo.vertices[original_index];
        new_geo.normals[v] = src_geo.normals[original_index];
        new_geo.tangents[v] = src_geo.tangents[original_index];

        if (!src_geo.texture_uvs.empty())
          new_geo.texture_uvs[v] = src_geo.texture_uvs[original_index];
        else
          new_geo.texture_uvs[v] = Eigen::Vector2f(0, 0);

      } else {
        // Should not happen if xref is valid
        LOG(ERROR) << "xatlas xref out of bounds!";
      }

      // Set new lightmap UV
      new_geo.lightmap_uvs[v] =
          Eigen::Vector2f(vertex_ref.uv[0] / (float)atlas->width,
                          vertex_ref.uv[1] / (float)atlas->height);
    }

    // Fill indices
    new_geo.indices.resize(atlas_mesh.indexCount);
    for (uint32_t idx = 0; idx < atlas_mesh.indexCount; ++idx) {
      new_geo.indices[idx] = atlas_mesh.indexArray[idx];
    }

    result_geometries.push_back(std::move(new_geo));
  }

  xatlas::Destroy(atlas);
  return result_geometries;
}

}  // namespace sh_baker
