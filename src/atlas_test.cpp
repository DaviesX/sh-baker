#include "atlas.h"

#include <gtest/gtest.h>

#include "scene.h"

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

TEST(AtlasTest, CalculateGeometryScales_Basic) {
  std::vector<Geometry> geometries(1);
  std::vector<Material> materials(1);

  // Mesh 0
  geometries[0].material_id = 0;
  geometries[0].texture_uvs = {{0, 0}, {1, 0}, {0, 1}};  // 1x1 tile

  // 100x100 texture
  materials[0].albedo.width = 100;
  materials[0].albedo.height = 100;

  float density_multiplier = 1.0f;
  std::vector<float> scales = atlas_internal::CalculateGeometryScales(
      geometries, materials, density_multiplier);

  ASSERT_EQ(scales.size(), 1);
  // sqrt(100 * 100) = 100
  // tile_count = 1
  // scale = 100 * 1 = 100
  EXPECT_NEAR(scales[0], 100.0f, 1e-4f);
}

TEST(AtlasTest, CalculateGeometryScales_Tiling) {
  std::vector<Geometry> geometries(1);
  std::vector<Material> materials(1);

  // Mesh 0: UVs range [0, 2] x [0, 2] -> 4 tiles
  geometries[0].material_id = 0;
  geometries[0].texture_uvs = {{0, 0}, {2, 0}, {0, 2}};

  // 10x10 texture
  materials[0].albedo.width = 10;
  materials[0].albedo.height = 10;

  float density_multiplier = 1.0f;
  std::vector<float> scales = atlas_internal::CalculateGeometryScales(
      geometries, materials, density_multiplier);

  ASSERT_EQ(scales.size(), 1);
  // sqrt(10 * 10) = 10
  // tile_count = 2 * 2 = 4
  // scale = 10 * sqrt(4) = 20
  EXPECT_NEAR(scales[0], 20.0f, 1e-4f);
}

TEST(AtlasTest, CalculateGeometryScales_Clamping) {
  // Create 3 meshes to establish a median.
  // Median scale will be 10.
  // One outlier will be huge, should be clamped.
  std::vector<Geometry> geometries(3);
  std::vector<Material> materials(3);

  // Mesh 0: Scale 10
  geometries[0].material_id = 0;
  geometries[0].texture_uvs = {{0, 0}, {1, 1}};
  materials[0].albedo.width = 10;
  materials[0].albedo.height = 10;

  // Mesh 1: Scale 10
  geometries[1].material_id = 1;
  geometries[1].texture_uvs = {{0, 0}, {1, 1}};
  materials[1].albedo.width = 10;
  materials[1].albedo.height = 10;

  // Mesh 2: Huge Scale (1000x1000 texture) -> Scale 1000
  geometries[2].material_id = 2;
  geometries[2].texture_uvs = {{0, 0}, {1, 1}};
  materials[2].albedo.width = 1000;
  materials[2].albedo.height = 1000;

  float density_multiplier = 1.0f;
  std::vector<float> scales = atlas_internal::CalculateGeometryScales(
      geometries, materials, density_multiplier);

  ASSERT_EQ(scales.size(), 3);
  EXPECT_NEAR(scales[0], 10.0f, 1e-4f);
  EXPECT_NEAR(scales[1], 10.0f, 1e-4f);

  // Median is 10. Max allowed is 5 * 10 = 50.
  EXPECT_NEAR(scales[2], 50.0f, 1e-4f);
}

TEST(AtlasTest, CalculateGeometryScales_DensityMultiplier) {
  std::vector<Geometry> geometries(1);
  std::vector<Material> materials(1);

  geometries[0].material_id = 0;
  geometries[0].texture_uvs = {{0, 0}, {1, 1}};
  materials[0].albedo.width = 100;
  materials[0].albedo.height = 100;

  float density_multiplier = 2.0f;
  std::vector<float> scales = atlas_internal::CalculateGeometryScales(
      geometries, materials, density_multiplier);

  ASSERT_EQ(scales.size(), 1);
  // 2.0 * 100 = 200
  EXPECT_NEAR(scales[0], 200.0f, 1e-4f);
}

TEST(AtlasTest, SkipParameterization) {
  Geometry input_geo;
  // Two triangles with distinct UV islands so they pack easily.
  input_geo.vertices = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0},
                        {2, 2, 0}, {3, 2, 0}, {2, 3, 0}};
  input_geo.normals.resize(6, Eigen::Vector3f(0, 0, 1));
  input_geo.tangents.resize(6, Eigen::Vector4f(1, 0, 0, 1));
  input_geo.indices = {0, 1, 2, 3, 4, 5};
  input_geo.texture_uvs = {{0.0f, 0.0f}, {0.4f, 0.0f}, {0.0f, 0.4f},
                           {0.6f, 0.6f}, {1.0f, 0.6f}, {0.6f, 1.0f}};

  input_geo.material_id = 0;

  Scene scene;
  scene.geometries.push_back(input_geo);
  Material mat;
  mat.albedo.width = 100;
  mat.albedo.height = 100;
  scene.materials.push_back(mat);

  // Call with skip_parameterization = true
  std::optional<AtlasResult> atlas_result =
      CreateAtlasGeometries(scene, 1024, 2, 1.0f, true);

  ASSERT_TRUE(atlas_result.has_value());
  EXPECT_GT(atlas_result->width, 0);
  EXPECT_GT(atlas_result->height, 0);

  const auto& out_geo = atlas_result->geometries[0];
  ASSERT_EQ(out_geo.lightmap_uvs.size(), 6);

  for (const auto& uv : out_geo.lightmap_uvs) {
    EXPECT_GE(uv.x(), 0.0f);
    EXPECT_LE(uv.x(), 1.0f);
    EXPECT_GE(uv.y(), 0.0f);
    EXPECT_LE(uv.y(), 1.0f);
  }
}

}  // namespace sh_baker
