#include "rasterizer.h"

#include <glog/logging.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

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

bool AnyValidSubSamples(int x, int y, int width, int scale,
                        const std::vector<uint8_t>& high_res_mask) {
  const int stride = width * scale;
  for (int src_y = y * scale; src_y < (y + 1) * scale; ++src_y) {
    int line_idx = src_y * stride;
    for (int src_x = x * scale; src_x < (x + 1) * scale; ++src_x) {
      int src_idx = line_idx + src_x;
      if (high_res_mask[src_idx]) {
        return true;
      }
    }
  }
  return false;
}

}  // namespace

std::vector<SurfacePoint> RasterizeScene(const Scene& scene,
                                         const RasterConfig& config) {
  int scaled_width = config.width * config.supersample_scale;
  int scaled_height = config.height * config.supersample_scale;
  std::vector<SurfacePoint> surface_map(scaled_width * scaled_height);

  // We parallelize over geometries. TBB's work stealing should balance it?
  // If a geometry is huge, we should parallelize inside it.

  tbb::parallel_for(
      tbb::blocked_range<size_t>(0, scene.geometries.size()),
      [&](const tbb::blocked_range<size_t>& r_geom) {
        for (size_t geom_idx = r_geom.begin(); geom_idx != r_geom.end();
             ++geom_idx) {
          const auto& geo = scene.geometries[geom_idx];
          size_t tri_count = geo.indices.size() / 3;

          for (size_t i = 0; i < tri_count; ++i) {
            uint32_t idx0 = geo.indices[i * 3 + 0];
            uint32_t idx1 = geo.indices[i * 3 + 1];
            uint32_t idx2 = geo.indices[i * 3 + 2];

            Eigen::Vector2f uv0 = geo.lightmap_uvs[idx0];
            Eigen::Vector2f uv1 = geo.lightmap_uvs[idx1];
            Eigen::Vector2f uv2 = geo.lightmap_uvs[idx2];

            // Bounding box in UV space
            float min_u = std::min({uv0.x(), uv1.x(), uv2.x()});
            float max_u = std::max({uv0.x(), uv1.x(), uv2.x()});
            float min_v = std::min({uv0.y(), uv1.y(), uv2.y()});
            float max_v = std::max({uv0.y(), uv1.y(), uv2.y()});

            int min_x = std::max(0, (int)(min_u * scaled_width));
            int max_x = std::min(scaled_width - 1,
                                 (int)(std::ceil(max_u * scaled_width)));
            int min_y = std::max(0, (int)(min_v * scaled_height));
            int max_y = std::min(scaled_height - 1,
                                 (int)(std::ceil(max_v * scaled_height)));

            for (int y = min_y; y <= max_y; ++y) {
              for (int x = min_x; x <= max_x; ++x) {
                Eigen::Vector2f p((x + 0.5f) / scaled_width,
                                  (y + 0.5f) / scaled_height);
                float u, v, w;
                if (!Barycentric(p, uv0, uv1, uv2, u, v, w)) {
                  // Outside of the triangle.
                  continue;
                }

                int pixel_idx = y * scaled_width + x;
                // Store surface point.
                // Note: This is a read-modify-write if we had blending,
                // but here we just overwrite.
                // Potential race with shared edges, but deterministic
                // enough for now.
                SurfacePoint sp;
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

                CHECK(!geo.tangents.empty());
                Eigen::Vector4f t0 = geo.tangents[idx0];
                Eigen::Vector4f t1 = geo.tangents[idx1];
                Eigen::Vector4f t2 = geo.tangents[idx2];
                Eigen::Vector4f interpolated_tan = t0 * u + t1 * v + t2 * w;

                Eigen::Vector3f t = interpolated_tan.head<3>();
                // Gram-Schmidt orthogonalization
                sp.tangent = (t - sp.normal * sp.normal.dot(t)).normalized();
                // Calculate bitangent (using w for handedness)
                sp.bitangent =
                    sp.normal.cross(sp.tangent) * interpolated_tan.w();

                surface_map[pixel_idx] = sp;
              }
            }
          }
        }
      });
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

std::vector<uint8_t> DownsampleValidityMask(
    const std::vector<uint8_t>& high_res_mask, int width, int height,
    int scale) {
  std::vector<uint8_t> mask(width * height, 0);

  // Parallelize over output pixels
  tbb::parallel_for(tbb::blocked_range<int>(0, height),
                    [&](const tbb::blocked_range<int>& r) {
                      for (int y = r.begin(); y != r.end(); ++y) {
                        for (int x = 0; x < width; ++x) {
                          mask[y * width + x] = AnyValidSubSamples(
                              x, y, width, scale, high_res_mask);
                        }
                      }
                    });
  return mask;
}

Texture CreateMaterialMap(const std::vector<SurfacePoint>& surface_points,
                          int width, int height) {
  Texture texture;
  texture.width = width;
  texture.height = height;
  texture.channels = 3;
  texture.pixel_data.resize(width * height * 3);

  // Parallelize for speed
  tbb::parallel_for(tbb::blocked_range<int>(0, height),
                    [&](const tbb::blocked_range<int>& r) {
                      for (int y = r.begin(); y != r.end(); ++y) {
                        for (int x = 0; x < width; ++x) {
                          int idx = y * width + x;
                          const auto& sp = surface_points[idx];
                          uint8_t r = 0, g = 0, b = 0;

                          if (sp.valid) {
                            // Generate arbitrary color from material_id
                            // Use a simple hash to get deterministic colors
                            uint32_t id = sp.material_id;
                            // Gold Noise / Hash
                            id = ((id >> 16) ^ id) * 0x45d9f3b;
                            id = ((id >> 16) ^ id) * 0x45d9f3b;
                            id = (id >> 16) ^ id;

                            r = (id & 0xFF);
                            g = ((id >> 8) & 0xFF);
                            b = ((id >> 16) & 0xFF);

                            // Ensure it's not too dark if valid
                            if (r < 50 && g < 50 && b < 50) {
                              r += 50;
                              g += 50;
                              b += 50;
                            }
                          }

                          texture.pixel_data[idx * 3 + 0] = r;
                          texture.pixel_data[idx * 3 + 1] = g;
                          texture.pixel_data[idx * 3 + 2] = b;
                        }
                      }
                    });

  return texture;
}

}  // namespace sh_baker
