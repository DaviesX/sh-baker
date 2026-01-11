#include "atlas.h"

#include <glog/logging.h>
#include <xatlas.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "scene.h"

namespace sh_baker {
namespace {

const float kMinScale = 0.001f;
const float kMaxScaleFactor = 5.0f;

}  // namespace

namespace atlas_internal {

std::vector<float> CalculateGeometryScales(
    const std::vector<Geometry>& geometries,
    const std::vector<Material>& materials, float density_multiplier) {
  // 1. Calculate Scaling Factors
  // Heuristic:
  // TargetScale = density_multiplier * sqrt(Area_tex)
  // If Tiling: EffectiveScale = TargetScale * sqrt(TileCount)
  std::vector<float> mesh_scales;
  mesh_scales.reserve(geometries.size());

  for (const auto& geo : geometries) {
    const Material& mat = materials[geo.material_id];

    // Step 1: Albedo-Relative Scaling
    // Use albedo dimensions as base area.
    float width = static_cast<float>(mat.albedo.width);
    float height = static_cast<float>(mat.albedo.height);

    float tex_area_sqrt = std::sqrt(width * height);
    float target_scale = static_cast<float>(density_multiplier) * tex_area_sqrt;

    // Step 2: Tiling Estimation
    float u_min = std::numeric_limits<float>::max();
    float u_max = std::numeric_limits<float>::lowest();
    float v_min = std::numeric_limits<float>::max();
    float v_max = std::numeric_limits<float>::lowest();

    CHECK(!geo.texture_uvs.empty());
    for (const auto& uv : geo.texture_uvs) {
      u_min = std::min(u_min, uv.x());
      u_max = std::max(u_max, uv.x());
      v_min = std::min(v_min, uv.y());
      v_max = std::max(v_max, uv.y());
    }

    float u_range = u_max - u_min;
    float v_range = v_max - v_min;

    // Avoid zero range
    u_range = std::max(u_range, 0.001f);
    v_range = std::max(v_range, 0.001f);

    float tile_count = u_range * v_range;
    float effective_scale = target_scale * std::sqrt(tile_count);

    DLOG(INFO) << "Mesh " << &geo - geometries.data() << ": Albedo "
               << mat.albedo.width << "x" << mat.albedo.height << ", Tiling "
               << tile_count << " -> Scale " << effective_scale;

    mesh_scales.push_back(effective_scale);
  }

  // Step 3: Constraints (Median Filtering)
  if (!mesh_scales.empty()) {
    std::vector<float> sorted_scales = mesh_scales;
    std::sort(sorted_scales.begin(), sorted_scales.end());
    float median_scale = sorted_scales[sorted_scales.size() / 2];

    // Avoid degenerate median
    median_scale = std::max(median_scale, 0.001f);

    float max_allowed_scale = kMaxScaleFactor * median_scale;
    for (size_t i = 0; i < mesh_scales.size(); ++i) {
      if (mesh_scales[i] > max_allowed_scale) {
        DLOG(INFO) << "Clamping mesh " << i << " scale " << mesh_scales[i]
                   << " to " << max_allowed_scale;
        mesh_scales[i] = max_allowed_scale;
      }
    }
  }

  return mesh_scales;
}

}  // namespace atlas_internal

std::optional<AtlasResult> CreateAtlasGeometries(const Scene& scene,
                                                 unsigned target_resolution,
                                                 unsigned padding,
                                                 float density_multiplier) {
  const auto& geometries = scene.geometries;
  if (geometries.empty()) {
    return std::nullopt;
  }

  std::vector<float> geometry_scales = atlas_internal::CalculateGeometryScales(
      geometries, scene.materials, density_multiplier);

  // 1. Create Atlas
  xatlas::Atlas* atlas = xatlas::Create();

  // 2. Add Meshes
  for (size_t i = 0; i < geometries.size(); ++i) {
    const auto& geo = geometries[i];
    auto vertices = TransformedVertices(geo);
    auto normals = TransformedNormals(geo);

    // Apply calculated scale to vertices (Emulating meshRelativeScaling)
    float scale = geometry_scales[i];
    for (auto& v : vertices) {
      v *= scale;
    }

    xatlas::MeshDecl mesh_decl;
    mesh_decl.vertexCount = static_cast<uint32_t>(vertices.size());
    mesh_decl.vertexPositionData = vertices.data();
    mesh_decl.vertexPositionStride = sizeof(Eigen::Vector3f);

    if (!normals.empty()) {
      mesh_decl.vertexNormalData = normals.data();
      mesh_decl.vertexNormalStride = sizeof(Eigen::Vector3f);
    }

    if (!geo.texture_uvs.empty()) {
      mesh_decl.vertexUvData = geo.texture_uvs.data();
      mesh_decl.vertexUvStride = sizeof(Eigen::Vector2f);
    }

    mesh_decl.indexCount = static_cast<uint32_t>(geo.indices.size());
    mesh_decl.indexData = geo.indices.data();
    mesh_decl.indexFormat = xatlas::IndexFormat::UInt32;

    xatlas::AddMeshError error = xatlas::AddMesh(atlas, mesh_decl);
    if (error != xatlas::AddMeshError::Success) {
      LOG(ERROR) << "Error adding mesh " << i
                 << " to xatlas: " << xatlas::StringForEnum(error);
      xatlas::Destroy(atlas);
      return std::nullopt;
    }
  }

  // 3. Generate Atlas
  xatlas::PackOptions pack_options;
  pack_options.resolution = target_resolution;
  pack_options.padding = padding;

  // Use higher quality packing (brute force) if desired, but defaults are
  // usually fine. pack_options.bruteForce = true;

  xatlas::Generate(atlas, xatlas::ChartOptions(), pack_options);

  if (atlas->width == 0 || atlas->height == 0) {
    LOG(ERROR) << "xatlas failed to generate any content.";
    xatlas::Destroy(atlas);
    return std::nullopt;
  }

  if (atlas->atlasCount > 1) {
    LOG(ERROR) << "xatlas failed to fit geometries into a single "
               << target_resolution << "x" << target_resolution
               << " atlas with padding " << padding << "."
               << " Requested " << atlas->atlasCount << " atlases.";
    // We strictly require a single atlas page.
    xatlas::Destroy(atlas);
    return std::nullopt;
  }

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

      if (vertex_ref.atlasIndex != 0) {
        // Discard vertices that are not in the first atlas.
        continue;
      }

      // Set new lightmap UV
      new_geo.lightmap_uvs[v] =
          Eigen::Vector2f(vertex_ref.uv[0] / (float)atlas->width,
                          vertex_ref.uv[1] / (float)atlas->height);

      // Copy attributes from original mesh
      if (original_index >= src_geo.vertices.size()) {
        // Should not happen if xref is valid.
        LOG(ERROR) << "xatlas xref out of bounds!";
        continue;
      }

      // Enforce invariants
      CHECK(!src_geo.normals.empty()) << "Source geometry must have normals";
      CHECK(!src_geo.tangents.empty()) << "Source geometry must have tangents";

      new_geo.vertices[v] = src_geo.vertices[original_index];
      new_geo.normals[v] = src_geo.normals[original_index];
      new_geo.tangents[v] = src_geo.tangents[original_index];

      if (!src_geo.texture_uvs.empty())
        new_geo.texture_uvs[v] = src_geo.texture_uvs[original_index];
      else
        new_geo.texture_uvs[v] = Eigen::Vector2f(0, 0);
    }

    // Fill indices
    new_geo.indices.resize(atlas_mesh.indexCount);
    for (uint32_t idx = 0; idx < atlas_mesh.indexCount; ++idx) {
      new_geo.indices[idx] = atlas_mesh.indexArray[idx];
    }

    result_geometries.push_back(std::move(new_geo));
  }

  AtlasResult result;
  result.geometries = std::move(result_geometries);
  result.width = atlas->width;
  result.height = atlas->height;

  xatlas::Destroy(atlas);
  return result;
}

}  // namespace sh_baker
