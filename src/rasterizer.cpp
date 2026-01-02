#include "rasterizer.h"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace sh_baker {

namespace {

// Helper to compute Barycentric coordinates
// Returns true if inside triangle.
bool Barycentric(const Eigen::Vector2f& p, const Eigen::Vector2f& a,
                 const Eigen::Vector2f& b, const Eigen::Vector2f& c, float& u,
                 float& v, float& w) {
  Eigen::Vector2f v0 = b - a, v1 = c - a, v2 = p - a;
  float d00 = v0.dot(v0);
  float d01 = v0.dot(v1);
  float d11 = v1.dot(v1);
  float d20 = v2.dot(v0);
  float d21 = v2.dot(v1);
  float denom = d00 * d11 - d01 * d01;
  if (std::abs(denom) < 1e-8f) return false;
  v = (d11 * d20 - d01 * d21) / denom;
  w = (d00 * d21 - d01 * d20) / denom;
  u = 1.0f - v - w;
  return (v >= 0.0f) && (w >= 0.0f) && (u >= 0.0f);
}

// Orthonormal basis from normal
void BuildBasis(const Eigen::Vector3f& n, Eigen::Vector3f& t,
                Eigen::Vector3f& b) {
  if (std::abs(n.x()) > std::abs(n.z())) {
    t = Eigen::Vector3f(-n.y(), n.x(), 0.0f);
  } else {
    t = Eigen::Vector3f(0.0f, -n.z(), n.y());
  }
  t.normalize();
  b = n.cross(t);
}

}  // namespace

std::vector<SurfacePoint> RasterizeScene(const Scene& scene,
                                         const RasterConfig& config) {
  std::vector<SurfacePoint> surface_map(config.width * config.height);

  for (size_t geom_idx = 0; geom_idx < scene.geometries.size(); ++geom_idx) {
    const auto& geo = scene.geometries[geom_idx];
    size_t tri_count = geo.indices.size() / 3;

    for (size_t i = 0; i < tri_count; ++i) {
      uint32_t idx0 = geo.indices[i * 3 + 0];
      uint32_t idx1 = geo.indices[i * 3 + 1];
      uint32_t idx2 = geo.indices[i * 3 + 2];

      Eigen::Vector2f uv0 = geo.uvs[idx0];
      Eigen::Vector2f uv1 = geo.uvs[idx1];
      Eigen::Vector2f uv2 = geo.uvs[idx2];

      // Bounding box in UV space
      float min_u = std::min({uv0.x(), uv1.x(), uv2.x()});
      float max_u = std::max({uv0.x(), uv1.x(), uv2.x()});
      float min_v = std::min({uv0.y(), uv1.y(), uv2.y()});
      float max_v = std::max({uv0.y(), uv1.y(), uv2.y()});

      int min_x = std::max(0, (int)(min_u * config.width));
      int max_x =
          std::min(config.width - 1, (int)(std::ceil(max_u * config.width)));
      int min_y = std::max(0, (int)(min_v * config.height));
      int max_y =
          std::min(config.height - 1, (int)(std::ceil(max_v * config.height)));

      for (int y = min_y; y <= max_y; ++y) {
        for (int x = min_x; x <= max_x; ++x) {
          Eigen::Vector2f p((x + 0.5f) / config.width,
                            (y + 0.5f) / config.height);
          float u, v, w;
          if (Barycentric(p, uv0, uv1, uv2, u, v, w)) {
            int pixel_idx = y * config.width + x;
            // Store surface point
            SurfacePoint& sp = surface_map[pixel_idx];
            sp.valid = true;
            sp.material_id = geo.material_id;

            Eigen::Vector3f pos0 = geo.vertices[idx0];
            Eigen::Vector3f pos1 = geo.vertices[idx1];
            Eigen::Vector3f pos2 = geo.vertices[idx2];
            sp.position = pos0 * u + pos1 * v + pos2 * w;

            Eigen::Vector3f n0 = geo.normals[idx0];
            Eigen::Vector3f n1 = geo.normals[idx1];
            Eigen::Vector3f n2 = geo.normals[idx2];
            sp.normal = (n0 * u + n1 * v + n2 * w).normalized();

            BuildBasis(sp.normal, sp.tangent, sp.bitangent);
          }
        }
      }
    }
  }
  return surface_map;
}

std::vector<uint8_t> CreateValidityMask(
    const std::vector<SurfacePoint>& points) {
  std::vector<uint8_t> mask(points.size());
  for (size_t i = 0; i < points.size(); ++i) {
    mask[i] = points[i].valid ? 1 : 0;
  }
  return mask;
}

}  // namespace sh_baker
