#include "atlas.h"

#include <gtest/gtest.h>

namespace sh_baker {

TEST(AtlasTest, SimpleQuad) {
  // Create a Quad (2 triangles) sharing an edge
  // Vertices:
  // 0: -1, -1, 0 (Bottom Left)
  // 1:  1, -1, 0 (Bottom Right)
  // 2: -1,  1, 0 (Top Left)
  // 3:  1,  1, 0 (Top Right)
  //
  // Tri 1: 0, 1, 2
  // Tri 2: 2, 1, 3

  Geometry input_geo;
  input_geo.vertices = {{-1, -1, 0}, {1, -1, 0}, {-1, 1, 0}, {1, 1, 0}};
  input_geo.normals = {{0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}};
  input_geo.tangents = {{1, 0, 0, 1}, {1, 0, 0, 1}, {1, 0, 0, 1}, {1, 0, 0, 1}};
  input_geo.texture_uvs = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};
  input_geo.indices = {0, 1, 2, 2, 1, 3};

  std::vector<Geometry> input_geometries = {input_geo};
  Scene scene;
  scene.geometries = std::move(input_geometries);

  // Add a dummy material so atlas.cpp doesn't crash
  Material mat;
  // Default 1x1 albedo
  mat.albedo.width = 256;
  mat.albedo.height = 256;
  scene.materials.push_back(mat);
  scene.geometries[0].material_id = 0;

  std::optional<AtlasResult> atlas_result =
      CreateAtlasGeometries(scene, 256, 2);
  ASSERT_TRUE(atlas_result.has_value());

  EXPECT_GE(atlas_result->width, 0);
  EXPECT_GE(atlas_result->height, 0);

  const auto& output_geometries = atlas_result->geometries;
  ASSERT_EQ(output_geometries.size(), 1);
  const auto& out_geo = output_geometries[0];

  // Check that lightmap UVs are populated
  ASSERT_EQ(out_geo.lightmap_uvs.size(), out_geo.vertices.size());
  for (const auto& uv : out_geo.lightmap_uvs) {
    EXPECT_GE(uv.x(), 0.0f);
    EXPECT_LE(uv.x(), 1.0f);
    EXPECT_GE(uv.y(), 0.0f);
    EXPECT_LE(uv.y(), 1.0f);
  }

  // Check indices
  EXPECT_EQ(out_geo.indices.size(), input_geo.indices.size());

  // Vertices should be at least equal to input (xatlas might split or keep
  // same) A planar quad shouldn't need splits, but xatlas might do it.
  EXPECT_GE(out_geo.vertices.size(), input_geo.vertices.size());
}

}  // namespace sh_baker
