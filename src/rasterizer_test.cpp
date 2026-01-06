#include "rasterizer.h"

#include <gtest/gtest.h>

namespace sh_baker {

TEST(RasterizerTest, RasterizeQuad) {
  Scene scene;
  Geometry quad;
  // Full 0-1 UV quad
  quad.vertices = {{-1, -1, 0}, {1, -1, 0}, {1, 1, 0}, {-1, 1, 0}};
  quad.normals = {{0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}};
  quad.texture_uvs = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
  quad.lightmap_uvs = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
  quad.tangents = {{1, 0, 0, 1}, {1, 0, 0, 1}, {1, 0, 0, 1}, {1, 0, 0, 1}};
  quad.indices = {0, 1, 2, 0, 2, 3};
  scene.geometries.push_back(quad);

  RasterConfig config;
  config.width = 4;
  config.height = 4;

  std::vector<SurfacePoint> result = RasterizeScene(scene, config);

  EXPECT_EQ(result.size(), 16);

  // All should be valid
  for (int i = 0; i < 16; ++i) {
    EXPECT_TRUE(result[i].valid) << "Pixel " << i << " should be covered.";
  }

  // Check center (approx)
  // Pixel 5 (1,1) -> UV (0.375, 0.375)
  // Vertex positions interpolated.
  // bottom-left is -1,-1. Top-right is 1,1.
  // UV 0,0 -> -1,-1. UV 1,1 -> 1,1.
  // UV 0.375 -> -1 + 0.375*2 = -0.25.
  EXPECT_NEAR(result[5].position.x(), -0.25f, 0.001f);
  EXPECT_NEAR(result[5].position.y(), -0.25f, 0.001f);
  EXPECT_NEAR(result[5].normal.z(), 1.0f, 0.001f);
}

TEST(RasterizerTest, RasterizeQuadSupersampled) {
  Scene scene;
  Geometry quad;
  // Full 0-1 UV quad
  quad.vertices = {{-1, -1, 0}, {1, -1, 0}, {1, 1, 0}, {-1, 1, 0}};
  quad.normals = {{0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}};
  quad.texture_uvs = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
  quad.lightmap_uvs = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
  quad.tangents = {{1, 0, 0, 1}, {1, 0, 0, 1}, {1, 0, 0, 1}, {1, 0, 0, 1}};
  quad.indices = {0, 1, 2, 0, 2, 3};
  scene.geometries.push_back(quad);

  RasterConfig config;
  config.width = 4;
  config.height = 4;
  config.supersample_scale = 2;  // 8x8 result

  std::vector<SurfacePoint> result = RasterizeScene(scene, config);

  EXPECT_EQ(result.size(), 64);  // 8x8

  // All should be valid
  for (int i = 0; i < 64; ++i) {
    EXPECT_TRUE(result[i].valid) << "Pixel " << i << " should be covered.";
  }
}

TEST(RasterizerTest, ValidityMask) {
  // Create dummy points
  std::vector<SurfacePoint> points(3);
  points[0].valid = true;
  points[1].valid = false;
  points[2].valid = true;

  std::vector<uint8_t> mask = CreateValidityMask(points);

  EXPECT_EQ(mask.size(), 3);
  EXPECT_EQ(mask[0], 1);
  EXPECT_EQ(mask[1], 0);
  EXPECT_EQ(mask[2], 1);
}

TEST(RasterizerTest, DownsampleValidityMask) {
  int width = 2;
  int height = 2;
  int scale = 2;
  // High res 4x4 = 16 pixels
  std::vector<uint8_t> points(16, false);

  // Case 1: Output (0,0) -> Input block [(0,0), (1,0), (0,1), (1,1)]
  // indices: 0, 1, 4, 5. Keep all invalid.
  // Expectation: Invalid (0)

  // Case 2: Output (1,0) -> Input block [(2,0), (3,0), (2,1), (3,1)]
  // indices: 2, 3, 6, 7. Set one valid.
  points[2] = true;
  // Expectation: Valid (1)

  // Case 3: Output (0,1) -> Input block [(0,2), (1,2), (0,3), (1,3)]
  // indices: 8, 9, 12, 13. Set all valid.
  points[8] = true;
  points[9] = true;
  points[12] = true;
  points[13] = true;
  // Expectation: Valid (1)

  // Case 4: Output (1,1) -> Input block [(2,2), (3,2), (2,3), (3,3)]
  // indices: 10, 11, 14, 15. Set one valid (last one).
  points[15] = true;
  // Expectation: Valid (1)

  std::vector<uint8_t> mask =
      DownsampleValidityMask(points, width, height, scale);

  ASSERT_EQ(mask.size(), 4);
  EXPECT_EQ(mask[0], 0);  // (0,0)
  EXPECT_EQ(mask[1], 1);  // (1,0)
  EXPECT_EQ(mask[2], 1);  // (0,1)
  EXPECT_EQ(mask[3], 1);  // (1,1)
}

}  // namespace sh_baker
