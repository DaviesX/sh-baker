#include "rasterizer.h"

#include <Eigen/src/Core/Matrix.h>
#include <glog/logging.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "scene.h"

namespace sh_baker {
namespace {

struct MaterialIdVertex {
  MaterialIdVertex() : r(0), g(0), b(0) {}

  MaterialIdVertex(uint32_t id) {
    // Gold Noise / Hash
    id = ((id >> 16) ^ id) * 0x45d9f3b;
    id = ((id >> 16) ^ id) * 0x45d9f3b;
    id = (id >> 16) ^ id;

    r = (id & 0xFF);
    g = ((id >> 8) & 0xFF);
    b = ((id >> 16) & 0xFF);

    if (r < 50 && g < 50 && b < 50) {
      r += 50;
      g += 50;
      b += 50;
    }
  }

  // Does not support interpolation.
  MaterialIdVertex operator+(const MaterialIdVertex& b) { return *this; }
  MaterialIdVertex operator-(const MaterialIdVertex& b) { return *this; }
  MaterialIdVertex operator*(float s) const { return *this; }
  MaterialIdVertex operator/(float s) const { return *this; }

  uint8_t r;
  uint8_t g;
  uint8_t b;
};

struct SurfaceVertex {
  SurfaceVertex() = default;
  SurfaceVertex(const Eigen::Vector3f& p, const Eigen::Vector3f& n,
                const Eigen::Vector4f& t)
      : position(p), normal(n), tangent(t) {}

  SurfaceVertex operator+(const SurfaceVertex& other) const {
    return SurfaceVertex(position + other.position, normal + other.normal,
                         tangent + other.tangent);
  }
  SurfaceVertex operator-(const SurfaceVertex& other) const {
    return SurfaceVertex(position - other.position, normal - other.normal,
                         tangent - other.tangent);
  }
  SurfaceVertex operator*(float s) const {
    return SurfaceVertex(position * s, normal * s, tangent * s);
  }
  SurfaceVertex operator/(float s) const {
    return SurfaceVertex(position / s, normal / s, tangent / s);
  }

  Eigen::Vector3f position;
  Eigen::Vector3f normal;
  Eigen::Vector4f tangent;
};

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

bool AnyValidSubSamples(int x, int y, int stride, int scale,
                        const std::vector<uint8_t>& high_res_mask) {
  const int start_x = x * scale;
  const int start_y = y * scale;
  const int end_x = (x + 1) * scale;
  const int end_y = (y + 1) * scale;

  for (int src_y = start_y; src_y < end_y; ++src_y) {
    int line_idx = src_y * stride;
    for (int src_x = start_x; src_x < end_x; ++src_x) {
      int src_idx = line_idx + src_x;
      if (high_res_mask[src_idx]) {
        return true;
      }
    }
  }
  return false;
}

template <typename VertexType, typename DrawFn>
void RasterizeTriangle(Eigen::Vector2i t0, Eigen::Vector2i t1,
                       Eigen::Vector2i t2, VertexType v0, VertexType v1,
                       VertexType v2, const DrawFn& draw_fn) {
  // Sort vertices by y-coordinate.
  if (t0.y() > t1.y()) {
    std::swap(t0, t1);
    std::swap(v0, v1);
  }
  if (t0.y() > t2.y()) {
    std::swap(t0, t2);
    std::swap(v0, v2);
  }
  if (t1.y() > t2.y()) {
    std::swap(t1, t2);
    std::swap(v1, v2);
  }

  // Align to pixel centers: start at first half-integer >= t0.y()
  int total_height = t2.y() - t0.y();
  if (total_height == 0) {
    // Let's sort by x.
    if (t0.x() > t1.x()) {
      std::swap(t0, t1);
      std::swap(v0, v1);
    }
    if (t0.x() > t2.x()) {
      std::swap(t0, t2);
      std::swap(v0, v2);
    }
    if (t1.x() > t2.x()) {
      std::swap(t1, t2);
      std::swap(v1, v2);
    }

    // Draw a line.
    int total_width = t2.x() - t0.x();
    if (total_width == 0) {
      // Draw a point.
      draw_fn(t0, v0);
      return;
    }

    CHECK_GE(t0.x(), 0);

    for (int w = 0; w < total_width; ++w) {
      float alpha = float(w + 0.5f) / total_width;
      VertexType vc = v0 + (v2 - v0) * alpha;
      draw_fn(Eigen::Vector2i(t0.x() + w, t0.y()), vc);
    }
    return;
  }

  for (int h = 0; h < total_height; ++h) {
    bool second_half = (h + 0.5f) > (t1.y() - t0.y()) || t1.y() == t0.y();

    int segment_height;
    if (second_half) {
      segment_height = t2.y() - t1.y();
    } else {
      segment_height = t1.y() - t0.y();
    }

    CHECK_GT(segment_height, 0);

    float alpha = float(h + 0.5f) / total_height;
    float beta;
    if (second_half) {
      beta = float(h + 0.5f - (t1.y() - t0.y())) / segment_height;
    } else {
      beta = float(h + 0.5f) / segment_height;
    }

    Eigen::Vector2f ta_f = t0.cast<float>() + (t2 - t0).cast<float>() * alpha;
    Eigen::Vector2f tb_f;
    VertexType va = v0 + (v2 - v0) * alpha;
    VertexType vb;

    if (second_half) {
      tb_f = t1.cast<float>() + (t2 - t1).cast<float>() * beta;
      vb = v1 + (v2 - v1) * beta;
    } else {
      tb_f = t0.cast<float>() + (t1 - t0).cast<float>() * beta;
      vb = v0 + (v1 - v0) * beta;
    }

    if (ta_f.x() > tb_f.x()) {
      std::swap(ta_f, tb_f);
      std::swap(va, vb);
    }

    // Pixel coverage: center (x+0.5) must be within [ta_f.x, tb_f.x]
    int x_start = std::ceil(ta_f.x() - 0.5f);
    int x_end = std::floor(tb_f.x() - 0.5f);

    if (x_start >= x_end) {
      // Draw a point.
      draw_fn(Eigen::Vector2i(x_start, t0.y() + h), va);
      continue;
    }

    VertexType grad = (vb - va) / (tb_f.x() - ta_f.x());
    for (int x = x_start; x <= x_end; ++x) {
      float dist = (x + 0.5f) - ta_f.x();
      VertexType vc = va + grad * dist;
      draw_fn(Eigen::Vector2i(x, t0.y() + h), vc);
    }
  }
}

}  // namespace

std::vector<SurfacePoint> RasterizeScene(const Scene& scene,
                                         const RasterConfig& config) {
  int scaled_width = config.width * config.supersample_scale;
  int scaled_height = config.height * config.supersample_scale;
  std::vector<SurfacePoint> surface_map(scaled_width * scaled_height);

  // Serial execution as requested for correctness priority
  for (const auto& geo : scene.geometries) {
    // Occluder shells have no lightmap chart (empty lightmap_uvs); they occlude
    // in the path tracer but produce no bake sensors.
    if (geo.material_id < 0) continue;

    auto vertices = TransformedVertices(geo);
    auto normals = TransformedNormals(geo);
    auto tangents = TransformedTangents(geo);

    size_t tri_count = geo.indices.size() / 3;
    for (size_t i = 0; i < tri_count; ++i) {
      uint32_t idx0 = geo.indices[i * 3 + 0];
      uint32_t idx1 = geo.indices[i * 3 + 1];
      uint32_t idx2 = geo.indices[i * 3 + 2];

      Eigen::Vector2f uv0 = geo.lightmap_uvs[idx0];
      Eigen::Vector2f uv1 = geo.lightmap_uvs[idx1];
      Eigen::Vector2f uv2 = geo.lightmap_uvs[idx2];

      // Convert UV to raster coordinates
      Eigen::Vector2i t0(int(uv0.x() * scaled_width),
                         int(uv0.y() * scaled_height));
      Eigen::Vector2i t1(int(uv1.x() * scaled_width),
                         int(uv1.y() * scaled_height));
      Eigen::Vector2i t2(int(uv2.x() * scaled_width),
                         int(uv2.y() * scaled_height));

      SurfaceVertex v0(vertices[idx0], normals[idx0], tangents[idx0]);
      SurfaceVertex v1(vertices[idx1], normals[idx1], tangents[idx1]);
      SurfaceVertex v2(vertices[idx2], normals[idx2], tangents[idx2]);

      RasterizeTriangle(
          t0, t1, t2, v0, v1, v2,
          [&](const Eigen::Vector2i& p, const SurfaceVertex& v) {
            int x = p.x();
            int y = p.y();

            // Boundary check
            CHECK_GE(x, 0);
            CHECK_GE(y, 0);
            CHECK_LT(x, scaled_width);
            CHECK_LT(y, scaled_height);

            int pixel_idx = y * scaled_width + x;

            SurfacePoint sp;
            sp.material_id = geo.material_id;
            sp.position = v.position;
            sp.normal = v.normal.normalized();

            Eigen::Vector3f tangent3 = v.tangent.head<3>();
            tangent3 = (tangent3 - sp.normal * sp.normal.dot(tangent3))
                           .normalized();  // Gram-Schmidt orthogonalization
            sp.tangent =
                Eigen::Vector4f(tangent3.x(), tangent3.y(), tangent3.z(),
                                v.tangent.w() > 0 ? 1.0f : -1.0f);

            surface_map[pixel_idx] = sp;
          });
    }
  }

  return surface_map;
}

GeometryUVMaps RasterizeGeometryUVMaps(const Geometry& geo,
                                       const RasterConfig& config) {
  GeometryUVMaps result;

  // Initialize the area ratio texture (1 channel float).
  result.uv_to_world_area_ratio.width = config.width;
  result.uv_to_world_area_ratio.height = config.height;
  result.uv_to_world_area_ratio.channels = 1;
  result.uv_to_world_area_ratio.pixel_data.resize(config.width * config.height,
                                                  0.0f);

  // Initialize the primitive ID map (1 channel int, -1 = no triangle).
  result.prim_id_map.width = config.width;
  result.prim_id_map.height = config.height;
  result.prim_id_map.channels = 1;
  result.prim_id_map.pixel_data.resize(config.width * config.height, -1);

  if (geo.indices.empty() || geo.texture_uvs.empty()) {
    return result;
  }

  auto vertices = TransformedVertices(geo);
  size_t tri_count = geo.indices.size() / 3;

  // A simple vertex type that carries the area ratio and triangle index.
  struct RatioVertex {
    float ratio;
    int32_t tri_id;

    RatioVertex() : ratio(0.0f), tri_id(-1) {}
    RatioVertex(float r, int32_t t) : ratio(r), tri_id(t) {}

    // These operators exist to satisfy the RasterizeTriangle template, but
    // the ratio and tri_id are constant over a triangle so interpolation
    // is a no-op effectively.
    RatioVertex operator+(const RatioVertex& b) const { return *this; }
    RatioVertex operator-(const RatioVertex& b) const { return *this; }
    RatioVertex operator*(float s) const { return *this; }
    RatioVertex operator/(float s) const { return *this; }
  };

  for (size_t i = 0; i < tri_count; ++i) {
    uint32_t idx0 = geo.indices[i * 3 + 0];
    uint32_t idx1 = geo.indices[i * 3 + 1];
    uint32_t idx2 = geo.indices[i * 3 + 2];

    // World-space triangle area.
    const Eigen::Vector3f& v0 = vertices[idx0];
    const Eigen::Vector3f& v1 = vertices[idx1];
    const Eigen::Vector3f& v2 = vertices[idx2];
    float world_area = 0.5f * (v1 - v0).cross(v2 - v0).norm();

    // UV-space triangle area.
    Eigen::Vector2f uv_scale(config.width, config.height);
    const Eigen::Vector2f uv0 = geo.texture_uvs[idx0].cwiseProduct(uv_scale);
    const Eigen::Vector2f uv1 = geo.texture_uvs[idx1].cwiseProduct(uv_scale);
    const Eigen::Vector2f uv2 = geo.texture_uvs[idx2].cwiseProduct(uv_scale);
    float uv_area = 0.5f * std::abs((uv1.x() - uv0.x()) * (uv2.y() - uv0.y()) -
                                    (uv2.x() - uv0.x()) * (uv1.y() - uv0.y()));

    float ratio = (uv_area > 1e-12f) ? (world_area / uv_area) : 0.0f;

    RatioVertex vertex(ratio, static_cast<int32_t>(i));

    // Convert UV to raster coordinates.
    Eigen::Vector2i t0(int(uv0.x()), int(uv0.y()));
    Eigen::Vector2i t1(int(uv1.x()), int(uv1.y()));
    Eigen::Vector2i t2(int(uv2.x()), int(uv2.y()));

    RasterizeTriangle(t0, t1, t2, vertex, vertex, vertex,
                      [&](const Eigen::Vector2i& p, const RatioVertex& v) {
                        int x = p.x();
                        int y = p.y();
                        // Wrap coordinates
                        x = x % config.width;
                        y = y % config.height;
                        if (x < 0) x += config.width;
                        if (y < 0) y += config.height;

                        int pixel_idx = y * config.width + x;
                        result.uv_to_world_area_ratio.pixel_data[pixel_idx] =
                            v.ratio;
                        result.prim_id_map.pixel_data[pixel_idx] = v.tri_id;
                      });
  }

  return result;
}

Texture RasterizeSceneMaterial(const Scene& scene, const RasterConfig& config) {
  Texture texture;
  texture.width = config.width;
  texture.height = config.height;
  texture.channels = 3;
  texture.pixel_data.resize(config.width * config.height * 3, 0);

  for (const auto& geo : scene.geometries) {
    if (geo.material_id < 0) continue;  // occluder shells have no chart
    uint32_t id = geo.material_id;
    MaterialIdVertex vertex(id);

    size_t tri_count = geo.indices.size() / 3;
    for (size_t i = 0; i < tri_count; ++i) {
      uint32_t idx0 = geo.indices[i * 3 + 0];
      uint32_t idx1 = geo.indices[i * 3 + 1];
      uint32_t idx2 = geo.indices[i * 3 + 2];

      Eigen::Vector2f uv0 = geo.lightmap_uvs[idx0];
      Eigen::Vector2f uv1 = geo.lightmap_uvs[idx1];
      Eigen::Vector2f uv2 = geo.lightmap_uvs[idx2];

      Eigen::Vector2i t0(int(uv0.x() * config.width),
                         int(uv0.y() * config.height));
      Eigen::Vector2i t1(int(uv1.x() * config.width),
                         int(uv1.y() * config.height));
      Eigen::Vector2i t2(int(uv2.x() * config.width),
                         int(uv2.y() * config.height));

      RasterizeTriangle(
          t0, t1, t2, vertex, vertex, vertex,
          [&texture, width = config.width, height = config.height](
              const Eigen::Vector2i& t, const MaterialIdVertex& v) {
            int x = t.x();
            int y = t.y();
            CHECK_GE(x, 0);
            CHECK_GE(y, 0);
            CHECK_LT(x, width);
            CHECK_LT(y, height);
            int index = (y * width + x) * sizeof(MaterialIdVertex);
            texture.pixel_data[index + 0] = v.r;
            texture.pixel_data[index + 1] = v.g;
            texture.pixel_data[index + 2] = v.b;
          });
    }
  }

  return texture;
}

std::vector<uint8_t> CreateValidityMask(
    const std::vector<SurfacePoint>& points) {
  std::vector<uint8_t> mask(points.size());
  for (size_t i = 0; i < points.size(); ++i) {
    mask[i] = points[i].material_id >= 0;
  }
  return mask;
}

std::vector<uint8_t> DownsampleValidityMask(
    const std::vector<uint8_t>& high_res_mask, int width, int height,
    int scale) {
  CHECK_EQ(high_res_mask.size(), width * height * scale * scale);

  std::vector<uint8_t> mask(width * height, 0);

  // Parallelize over output pixels
  const int high_res_stride = width * scale;
  tbb::parallel_for(tbb::blocked_range<int>(0, height),
                    [&](const tbb::blocked_range<int>& r) {
                      for (int y = r.begin(); y < r.end(); ++y) {
                        for (int x = 0; x < width; ++x) {
                          mask[y * width + x] = AnyValidSubSamples(
                              x, y, high_res_stride, scale, high_res_mask);
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

                          if (sp.material_id >= 0) {
                            // Generate arbitrary color from material_id
                            // Use a simple hash to get deterministic colors
                            uint32_t id = sp.material_id;
                            MaterialIdVertex v(id);

                            texture.pixel_data[idx * 3 + 0] = v.r;
                            texture.pixel_data[idx * 3 + 1] = v.g;
                            texture.pixel_data[idx * 3 + 2] = v.b;
                          }
                        }
                      }
                    });

  return texture;
}

}  // namespace sh_baker
