#include "rasterizer.h"

#include <gtest/gtest.h>

namespace sh_baker {

TEST(RasterizerTest, RasterizeQuad) {
  Scene scene;
  Geometry quad;
  // Full 0-1 UV quad
  quad.material_id = 0;
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
    EXPECT_GE(result[i].material_id, 0)
        << "Pixel " << i << " should be covered.";
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
  quad.material_id = 0;
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
    EXPECT_GE(result[i].material_id, 0)
        << "Pixel " << i << " should be covered.";
  }
}

TEST(RasterizerTest, ValidityMask) {
  // Create dummy points
  std::vector<SurfacePoint> points(3);
  points[0].material_id = 0;
  points[1].material_id = -1;
  points[2].material_id = 0;

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

TEST(RasterizerTest, RasterizeSceneMaterial) {
  Scene scene;
  Geometry quad;
  // Full 0-1 UV quad
  quad.material_id = 123;
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

  Texture result = RasterizeSceneMaterial(scene, config);

  EXPECT_EQ(result.width, 4);
  EXPECT_EQ(result.height, 4);
  EXPECT_EQ(result.channels, 3);
  EXPECT_EQ(result.pixel_data.size(), 4 * 4 * 3);

  // All pixels should be colored with the material color
  for (int i = 0; i < 16; ++i) {
    uint8_t r = result.pixel_data[i * 3 + 0];
    uint8_t g = result.pixel_data[i * 3 + 1];
    uint8_t b = result.pixel_data[i * 3 + 2];
    // Color should NOT be black
    EXPECT_TRUE(r != 0 || g != 0 || b != 0)
        << "Pixel " << i << " should be colored";
  }
}

TEST(RasterizerTest, RasterizeQuadScanline) {
  Scene scene;
  Geometry quad;
  // Full 0-1 UV quad
  quad.material_id = 0;
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
    EXPECT_GE(result[i].material_id, 0)
        << "Pixel " << i << " should be covered.";
  }

  // Check center (approx)
  // Pixel 5 (1,1) -> UV (0.375, 0.375)
  // (-1) + (1 - (-1)) * 0.375 = -1 + 2 * 0.375 = -1 + 0.75 = -0.25
  EXPECT_NEAR(result[5].position.x(), -0.25f, 0.001f);
  EXPECT_NEAR(result[5].position.y(), -0.25f, 0.001f);
  EXPECT_NEAR(result[5].normal.z(), 1.0f, 0.001f);

  // Check Pixel 6 (2,1) -> UV (0.625, 0.375)
  // (-1) + (1 - (-1)) * 0.625 = -1 + 2 * 0.625 = 0.25
  // (-1) + (1 - (-1)) * 0.375 = -1 + 2 * 0.375 = -0.25
  EXPECT_NEAR(result[6].position.x(), 0.25f, 0.001f);
  EXPECT_NEAR(result[6].position.y(), -0.25f, 0.001f);
  EXPECT_NEAR(result[6].normal.z(), 1.0f, 0.001f);
}

TEST(RasterizerTest, RasterizeGeometryUVMapsWrapping) {
  Scene scene;
  Geometry quad;
  // Quad with UVs strictly outside [0, 1]
  // UVs: (1.0, 1.0) to (2.0, 2.0)
  quad.vertices = {{-1, -1, 0}, {1, -1, 0}, {1, 1, 0}, {-1, 1, 0}};
  quad.normals = {{0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}};
  quad.texture_uvs = {{1.0, 1.0}, {2.0, 1.0}, {2.0, 2.0}, {1.0, 2.0}};
  quad.lightmap_uvs = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};  // Unused here
  quad.indices = {0, 1, 2, 0, 2, 3};

  // Calculate area ratio:
  // World Area: 2x2 = 4.
  // UV Area: 2x2 = 4.
  // Ratio: 1.0.

  RasterConfig config;
  config.width = 4;
  config.height = 4;

  GeometryUVMaps result = RasterizeGeometryUVMaps(quad, config);

  // With wrapping, the entire 4x4 texture should be filled because the UVs
  // cover a 2x2 area, meaning it covers the [0,1] range 4 times. Actually,
  // wait. The function rasterizes *into* the texture based on UVs. The input
  // vertices have UVs from -0.5 to 1.5. The triangles in UV space cover
  // [-0.5, 1.5] x [-0.5, 1.5]. The rasterizer iterates over pixels in the
  // texture (0..width, 0..height). Wait, `RasterizeTriangle` takes vertex
  // coordinates in pixels. t0 = (-0.5 * 4, -0.5 * 4) = (-2, -2) t1 = (1.5 * 4,
  // -0.5 * 4) = (6, -2) t2 = (1.5 * 4, 1.5 * 4) = (6, 6) The rasterizer
  // iterates from min_y to max_y of the triangle. So it will iterate from -2
  // to 6. The lambda checks `if (x < 0 || y < 0 ...)` and returns. So
  // currently, only pixels within [0, 3] x [0, 3] that fall inside the triangle
  // are drawn. Since the triangle covers the entire [0, 1] x [0, 1] range (and
  // more), ALL pixels in the 4x4 image should be covered.

  // IF the rasterizer clips correctly or iterates correctly.
  // `RasterizeTriangle` calculates bounding box of the triangle.
  // It iterates y from t0.y to t2.y.
  // In this case -2 to 6.
  // Inside the loop, it checks x bounds?
  // No, `RasterizeTriangle` does `draw_fn(Eigen::Vector2i(x, t0.y() + h),
  // vc);`. It assumes the draw_fn handles clipping.

  // In `RasterizeGeometryUVMaps`:
  /*
          if (x < 0 || y < 0 || x >= config.width || y >= config.height) {
            return;
          }
  */
  // This clips to the texture bounds.
  // But wait. "Warp the texture access" usually means when sampling FROM a
  // texture. Here we are rasterizing INTO a texture (UV map).
  // `RasterizeGeometryUVMaps` produces `uv_to_world_area_ratio` and
  // `prim_id_map`. These maps are used to "inverse sample" later? Or are they
  // used to lookup geometry info from UV? If I have a lightmap texel at UV (u,
  // v), I want to know corresponding world position/area. The lightmap
  // parameterization is usually unique and in [0, 1]. BUT the user said:
  // "Unlike light map UVs, texture UVs may not be normalized." And pointed to
  // `RasterizeGeometryUVMaps`.

  // Let's re-read the code.
  // `geometry.texture_uvs` are the material texture UVs.
  // `geometry.lightmap_uvs` are the unique parameterization.

  // `RasterizeGeometryUVMaps` uses `geo.texture_uvs`.
  // Wait. Why rasterize using `texture_uvs`?
  // "Rasterizes the area scale of the texel in the UV. The area scale is the
  // ratio: world triangle area / UV triangle area." This sounds like it's used
  // for resolution analysis? If the texture UVs correspond to a tiling texture,
  // then a single triangle might cover UV range [0, 10]. If we rasterize this
  // into a 1024x1024 map, we are effectively splatting the triangle onto the
  // texture map. If the UVs are [0, 10], and we wrap, then the triangle is
  // drawn 100 times? No, `RasterizeTriangle` rasterizes the single triangle
  // t0-t1-t2. If t0=(0,0) and t1=(10240, 0), it will synthesize pixels from 0
  // to 10240. The lambda receives `p`. We need to map `p` to [0, width) by
  // wrapping.

  // Example: Triangle [0, 0] to [2, 0] to [0, 2] in UV space.
  // Mapped to 4x4 texture.
  // t0=(0,0), t1=(8,0), t2=(0,8).
  // Pixel at (0,0) is covered.
  // Pixel at (4,0) (which is wrapped to 0,0) should also be covered?
  // No, "wrapping" means if I write to (4, 0), it actually writes to (0, 0).
  // Yes.

  // So checking my test case:
  // UVs [-0.5, 1.5].
  // Rasterizer will generate fragments for x in [-2, 6], y in [-2, 6].
  // Current code returns coverage for [0, 0] to [3, 3] implicitly because it
  // clips. BUT, fragments at (-1, -1) should wrap to (3, 3). Fragments at (4,
  // 4) should wrap to (0, 0). Because the triangle covers the whole 4x4 area
  // "logically" multiple times? Actually, with [-0.5, 1.5], the center [0, 1]
  // is fully covered. The regions [-0.5, 0] and [1, 1.5] should wrap around.

  // Wait, if I have indices covering the whole image, I expect `prim_id_map` to
  // be filled. Currently, `RasterizeTriangle` iterates over the large triangle.
  // `draw_fn` checks bounds.
  // If I change `draw_fn` to wrap, then `prim_id_map` should be filled.
  // Wait, currently checks bounds:
  // if (x < 0 || y < 0 ...) return;
  // So pixels at (0,0) are drawn (from the [0,1] part of the triangle).
  // Pixels at (-1, -1) are discarded.
  // If we wrap, (-1, -1) becomes (3, 3).

  // Check the test case again.
  // Vertices cover [-0.5, 1.5].
  // Center part [0, 1] fills the 4x4 texture completely.
  // So `prim_id_map` should ALREADY be full of `0`s even with clipping.
  // Because the triangle is a superset of the [0, 1] square.

  // I need a test case where the triangle is OUTSIDE [0, 1] but wraps into it.
  // Example: Triangle at [1.0, 1.0] to [2.0, 1.0] to [1.0, 2.0].
  // UVs: {1, 1}, {2, 1}, {1, 2}.
  // Raster coords (4x4): (4, 4), (8, 4), (4, 8).
  // Rasterizer will generate pixels x >= 4, y >= 4.
  // Current code clips -> draws nothing.
  // Wrapping code -> (4,4) becomes (0,0) -> draws.

  quad.vertices = {{-1, -1, 0}, {1, -1, 0}, {-1, 1, 0}};  // Just a triangle
  quad.indices = {0, 1, 2};
  quad.texture_uvs = {{1.0f, 1.0f}, {2.0f, 1.0f}, {1.0f, 2.0f}};
  // This covers the area equivalent to {0,0}, {1,0}, {0,1}.
  // Should fill roughly half the texture.

  int filled_count = 0;
  for (int i = 0; i < 16; ++i) {
    if (result.prim_id_map.pixel_data[i] != -1) filled_count++;
  }
  EXPECT_GT(filled_count, 0);
}

}  // namespace sh_baker
