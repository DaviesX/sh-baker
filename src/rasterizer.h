#ifndef SH_BAKER_SRC_RASTERIZER_H_
#define SH_BAKER_SRC_RASTERIZER_H_

#include <Eigen/Dense>
#include <cstdint>
#include <vector>

#include "scene.h"

namespace sh_baker {

struct RasterConfig {
  int width = 1024;
  int height = 1024;
  int supersample_scale = 1;
};

struct SurfacePoint {
  Eigen::Vector3f position = Eigen::Vector3f::Zero();
  Eigen::Vector3f normal = Eigen::Vector3f::Zero();
  Eigen::Vector4f tangent = Eigen::Vector4f::Zero();
  int32_t material_id = -1;  // Invalid if < 0.
};

// Rasterizes the scene UVs into a buffer of SurfacePoints.
// The buffer size will be config.width * config.supersample_scale *
// config.height * config.supersample_scale.
std::vector<SurfacePoint> RasterizeScene(const Scene& scene,
                                         const RasterConfig& config);

// Result of rasterizing geometry UV maps: the area ratio and primitive ID map.
struct GeometryUVMaps {
  Texture32F uv_to_world_area_ratio;
  Texture32I prim_id_map;
};

// Rasterizes the area scale of the texel in the UV. The area scale is the
// ratio: world triangle area / UV triangle area. It would be constant over a
// triangle. Also rasterizes the primitive (triangle) index for each texel.
GeometryUVMaps RasterizeGeometryUVMaps(const Geometry& geo,
                                       const RasterConfig& config);

// Rasterizes the scene UVs into a buffer of color coded material IDs.
// The buffer size will be config.width * config.height (supersample scale
// is not applied).
Texture RasterizeSceneMaterial(const Scene& scene, const RasterConfig& config);

// Extracts a validity mask (1 for valid, 0 for invalid) from surface points.
std::vector<uint8_t> CreateValidityMask(
    const std::vector<SurfacePoint>& points);

// Downsamples the validity mask from a supersampled buffer.
// Returns a mask of size width * height.
// A pixel is valid if ANY of its subpixels (scale*scale block) are valid.
std::vector<uint8_t> DownsampleValidityMask(
    const std::vector<uint8_t>& high_res_mask, int width, int height,
    int scale);

// Creates a material map for debugging.
// Assigns arbitrary RGB color for the material ID.
// Fills black color to invalid pixel locations.
// width and height must match the dimension of the surface points.
Texture CreateMaterialMap(const std::vector<SurfacePoint>& surface_points,
                          int width, int height);

}  // namespace sh_baker

#endif  // SH_BAKER_SRC_RASTERIZER_H_
