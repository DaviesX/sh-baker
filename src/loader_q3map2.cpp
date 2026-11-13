#include "loader_q3map2.h"

#include <glog/logging.h>

#include <cstddef>
#include <utility>

namespace sh_baker {
namespace {

// Normals shorter than this are degenerate and repaired.
constexpr float kMinNormalLength = 1e-6f;

std::vector<Eigen::Vector2f> GenerateOccluderTextureUvs(const Geometry& geo) {
  std::vector<Eigen::Vector2f> result;
  result.reserve(geo.vertices.size());
  for (const Eigen::Vector3f& position : geo.vertices) {
    result.emplace_back(position.x(), position.y());
  }
  return result;
}

}  // namespace

namespace loader_q3map2_internal {

// CHECKs every case of malformed input; see the header.
void CheckSurface(const Q3Map2Surface& surface, const Scene& scene) {
  for (size_t l = 0; l < scene.lights.size(); ++l) {
    CHECK(scene.lights[l].geometry == nullptr)
        << "AddQ3Map2Surface: scene assembly order violated: light " << l
        << " already points at a geometry. Add every surface before creating "
           "area lights and building the BVH.";
  }

  const size_t vertex_count = surface.positions.size();
  CHECK_GT(vertex_count, 0u) << "Q3Map2Surface positions: empty";
  CHECK_EQ(surface.normals.size(), vertex_count)
      << "Q3Map2Surface normals: need one per vertex";
  if (surface.material_id >= 0 || !surface.texture_uvs.empty()) {
    CHECK_EQ(surface.texture_uvs.size(), vertex_count)
        << "Q3Map2Surface texture_uvs: need one per vertex (material_id "
        << surface.material_id << ")";
  }
  CHECK_EQ(surface.indices.size() % 3, 0u)
      << "Q3Map2Surface indices: count " << surface.indices.size()
      << " is not a multiple of 3";
  for (size_t i = 0; i < surface.indices.size(); ++i) {
    CHECK_LT(surface.indices[i], vertex_count)
        << "Q3Map2Surface indices[" << i << "]: index " << surface.indices[i]
        << " is not below the vertex count " << vertex_count;
  }
  CHECK_LT(surface.material_id, static_cast<int>(scene.materials.size()))
      << "Q3Map2Surface material_id: " << surface.material_id
      << " is not below the scene's material count " << scene.materials.size();

  for (size_t v = 0; v < vertex_count; ++v) {
    CHECK(surface.positions[v].allFinite())
        << "Q3Map2Surface positions[" << v
        << "]: not finite: " << surface.positions[v].transpose();
    CHECK(surface.normals[v].allFinite())
        << "Q3Map2Surface normals[" << v
        << "]: not finite: " << surface.normals[v].transpose();
    if (!surface.texture_uvs.empty()) {
      CHECK(surface.texture_uvs[v].allFinite())
          << "Q3Map2Surface texture_uvs[" << v
          << "]: not finite: " << surface.texture_uvs[v].transpose();
    }
  }
}

std::vector<Eigen::Vector3f> NormalizeAndRepairNormals(const Geometry& geo) {
  // Find the general direction of every vertex as fallback.
  std::vector<Eigen::Vector3f> vertex_outward_sum(geo.normals.size(),
                                                  Eigen::Vector3f::Zero());
  for (size_t t = 0; t + 2 < geo.indices.size(); t += 3) {
    const uint32_t i0 = geo.indices[t + 0];
    const uint32_t i1 = geo.indices[t + 1];
    const uint32_t i2 = geo.indices[t + 2];
    const Eigen::Vector3f outward =
        (geo.vertices[i1] - geo.vertices[i0])
            .cross(geo.vertices[i2] - geo.vertices[i0]);
    vertex_outward_sum[i0] += outward;
    vertex_outward_sum[i1] += outward;
    vertex_outward_sum[i2] += outward;
  }

  // Normalize or fix degeneracy.
  std::vector<Eigen::Vector3f> result;
  result.reserve(geo.vertices.size());
  int num_normal_repaired = 0;
  for (size_t i = 0; i < geo.normals.size(); ++i) {
    if (geo.normals[i].norm() >= kMinNormalLength) {
      result.emplace_back(geo.normals[i].normalized());
    } else if (vertex_outward_sum[i].norm() >= kMinNormalLength) {
      result.emplace_back(vertex_outward_sum[i].normalized());
      ++num_normal_repaired;
    } else {
      result.emplace_back(0.f, 0.f, 1.f);
      ++num_normal_repaired;
    }
  }

  VLOG(1) << "Repaired " << num_normal_repaired << " degenerate normals.";
  return result;
}

std::vector<uint32_t> DropDegenerateTriangles(const Geometry& geo) {
  std::vector<uint32_t> result;
  result.reserve(geo.indices.size());

  for (size_t t = 0; t + 2 < geo.indices.size(); t += 3) {
    const uint32_t i0 = geo.indices[t + 0];
    const uint32_t i1 = geo.indices[t + 1];
    const uint32_t i2 = geo.indices[t + 2];
    if (i0 == i1 || i0 == i2 || i1 == i2) {
      continue;
    }
    result.push_back(i0);
    result.push_back(i1);
    result.push_back(i2);
  }

  if (result.size() != geo.indices.size()) {
    VLOG(1) << "Removed " << (geo.indices.size() - result.size()) / 3
            << " degenerate triangles from a q3map2 surface.";
  }

  return result;
}

}  // namespace loader_q3map2_internal

void AddQ3Map2Surface(const Q3Map2Surface& surface, Scene* scene) {
  loader_q3map2_internal::CheckSurface(surface, *scene);

  Geometry geo;
  geo.material_id = surface.material_id;
  geo.vertices = surface.positions;
  geo.normals = surface.normals;
  geo.texture_uvs = surface.texture_uvs;

  // q3map2's front faces are clockwise; sh-baker's are counter-clockwise.
  geo.indices = surface.indices;
  for (size_t t = 0; t + 2 < geo.indices.size(); t += 3) {
    std::swap(geo.indices[t + 1], geo.indices[t + 2]);
  }

  geo.normals = loader_q3map2_internal::NormalizeAndRepairNormals(geo);
  geo.indices = loader_q3map2_internal::DropDegenerateTriangles(geo);
  // Tangents come first: an occluder without texture UVs takes the fallback
  // basis, since its generated XY UVs have no area on walls facing +-X.
  geo.tangents = ComputeTangents(geo);
  if (geo.texture_uvs.empty()) {
    geo.texture_uvs = GenerateOccluderTextureUvs(geo);
  }

  scene->geometries.emplace_back(std::move(geo));
}

}  // namespace sh_baker
