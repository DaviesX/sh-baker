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
};

struct SurfacePoint {
  bool valid = false;
  Eigen::Vector3f position = Eigen::Vector3f::Zero();
  Eigen::Vector3f normal = Eigen::Vector3f::Zero();
  uint32_t material_id = 0;
  // Tangent frame for sampling
  Eigen::Vector3f tangent = Eigen::Vector3f::Zero();
  Eigen::Vector3f bitangent = Eigen::Vector3f::Zero();
};

// Rasterizes the scene UVs into a buffer of SurfacePoints.
// The buffer size will be config.width * config.height.
std::vector<SurfacePoint> RasterizeScene(const Scene& scene,
                                         const RasterConfig& config);

// Extracts a validity mask (1 for valid, 0 for invalid) from surface points.
std::vector<uint8_t> CreateValidityMask(
    const std::vector<SurfacePoint>& points);

}  // namespace sh_baker

#endif  // SH_BAKER_SRC_RASTERIZER_H_
