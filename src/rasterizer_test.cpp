#include "rasterizer.h"

#include <gtest/gtest.h>

namespace sh_baker {

TEST(RasterizerTest, RasterizeQuad) {
  Scene scene;
  Geometry quad;
  // Full 0-1 UV quad
  quad.vertices = {{-1, -1, 0}, {1, -1, 0}, {1, 1, 0}, {-1, 1, 0}};
  quad.normals = {{0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}};
  quad.uvs = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
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

}  // namespace sh_baker
