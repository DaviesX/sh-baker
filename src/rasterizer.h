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

// A point to bake. By default it is a point on a surface: BakeSHLightMap
// samples the hemisphere around `normal`, oriented by `tangent`.
struct SurfacePoint {
  Eigen::Vector3f position = Eigen::Vector3f::Zero();
  Eigen::Vector3f normal = Eigen::Vector3f::Zero();
  Eigen::Vector4f tangent = Eigen::Vector4f::Zero();
  int32_t material_id = -1;  // Invalid if < 0.
  // A probe in open space instead of a surface point, such as one of q3map2's
  // light-grid points. Its SH is the projection of the incident radiance over
  // the whole sphere, the integral of L_i(w) Y(w) dw, with no cosine:
  //   - its sample rays are uniform over the sphere in world space, each
  //     weighted 4 pi instead of 2 pi;
  //   - next-event estimation passes the light tree a zero normal and counts
  //     light from every direction;
  //   - `normal` and `tangent` orient nothing, so a zero tangent is fine;
  //   - the bounces of its paths shade surfaces exactly as for any point.
  // Its SHCoeffs::irradiance still accumulates max(0, normal . w) and has no
  // meaning; callers must not use it. Its environment visibility is the
  // fraction of first rays, over the whole sphere, that escape the scene.
  // material_id must still be >= 0 for the point to be baked.
  bool full_sphere = false;
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
