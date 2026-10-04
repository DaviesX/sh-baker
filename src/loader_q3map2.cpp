#include "loader_q3map2.h"

#include <glog/logging.h>

#include <cstddef>
#include <utility>

#include "tangents.h"

namespace sh_baker {
namespace {

// Normals shorter than this are degenerate and repaired.
constexpr float kMinNormalLength = 1e-6f;

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

// Normalizes every normal, replacing a degenerate one with the normalized sum
// of the unnormalized front-face normals of the (counter-clockwise) triangles
// that use its vertex, or with (0, 0, 1) if that sum is degenerate too.
void NormalizeAndRepairNormals(Geometry* geo) {
  std::vector<bool> degenerate(geo->normals.size(), false);
  bool any_degenerate = false;
  for (size_t v = 0; v < geo->normals.size(); ++v) {
    Eigen::Vector3f& n = geo->normals[v];
    if (n.norm() >= kMinNormalLength) {
      n.normalize();
    } else {
      degenerate[v] = true;
      any_degenerate = true;
    }
  }
  if (!any_degenerate) return;

  std::vector<Eigen::Vector3f> face_sum(geo->normals.size(),
                                        Eigen::Vector3f::Zero());
  for (size_t t = 0; t + 2 < geo->indices.size(); t += 3) {
    const uint32_t i0 = geo->indices[t];
    const uint32_t i1 = geo->indices[t + 1];
    const uint32_t i2 = geo->indices[t + 2];
    const Eigen::Vector3f face =
        (geo->vertices[i1] - geo->vertices[i0])
            .cross(geo->vertices[i2] - geo->vertices[i0]);
    face_sum[i0] += face;
    face_sum[i1] += face;
    face_sum[i2] += face;
  }
  int repaired = 0;
  for (size_t v = 0; v < geo->normals.size(); ++v) {
    if (!degenerate[v]) continue;
    // A vertex only on zero-area triangles is never hit by a ray.
    geo->normals[v] = face_sum[v].norm() >= kMinNormalLength
                          ? face_sum[v].normalized()
                          : Eigen::Vector3f(0, 0, 1);
    ++repaired;
  }
  VLOG(1) << "Repaired " << repaired << " degenerate normals.";
}

// Drops triangles that repeat a vertex index, as the glTF loader does.
void DropDegenerateTriangles(Geometry* geo) {
  std::vector<uint32_t> valid;
  valid.reserve(geo->indices.size());
  for (size_t t = 0; t + 2 < geo->indices.size(); t += 3) {
    const uint32_t i0 = geo->indices[t];
    const uint32_t i1 = geo->indices[t + 1];
    const uint32_t i2 = geo->indices[t + 2];
    if (i0 == i1 || i0 == i2 || i1 == i2) continue;
    valid.push_back(i0);
    valid.push_back(i1);
    valid.push_back(i2);
  }
  if (valid.size() != geo->indices.size()) {
    VLOG(1) << "Removed " << (geo->indices.size() - valid.size()) / 3
            << " degenerate triangles from a q3map2 surface.";
    geo->indices = std::move(valid);
  }
}

}  // namespace

void AddQ3Map2Surface(const Q3Map2Surface& surface, Scene* scene) {
  CheckSurface(surface, *scene);

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

  NormalizeAndRepairNormals(&geo);
  if (geo.texture_uvs.empty()) {
    // An occluder without texture UVs, as the glTF loader fills them.
    geo.texture_uvs.assign(geo.vertices.size(), Eigen::Vector2f::Zero());
  }
  DropDegenerateTriangles(&geo);
  GenerateTangents(&geo, !surface.texture_uvs.empty());

  scene->geometries.push_back(std::move(geo));
}

}  // namespace sh_baker
