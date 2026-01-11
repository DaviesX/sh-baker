#include "loader.h"

#include <gtest/gtest.h>

#include <filesystem>

namespace sh_baker {

TEST(LoaderTest, LoadCube) {
  // Assuming working directory is set correctly or absolute path used.
  std::filesystem::path input_path =
      "/Users/daviswen/sh-baker/data/cube/Cube.gltf";
  if (!std::filesystem::exists(input_path)) {
    // Fallback for relative path if running nearby
    input_path = "data/cube/Cube.gltf";
    if (!std::filesystem::exists(input_path)) {
      input_path = "../data/cube/Cube.gltf";
    }
  }

  ASSERT_TRUE(std::filesystem::exists(input_path))
      << "Test file not found: " << input_path;

  std::optional<Scene> scene = LoadScene(input_path);
  ASSERT_TRUE(scene.has_value());

  // Cube has 1 mesh, probably split into 1 geometry in our structure (assuming
  // 1 primitive).
  ASSERT_EQ(scene->geometries.size(), 1);

  const auto& geo = scene->geometries[0];

  // Check indices count (Cube usually has 36 indices)
  EXPECT_EQ(geo.indices.size(), 36);
  // Vertices might be 24 or 36 depending on flat shading export
  EXPECT_GT(geo.vertices.size(), 0);
  EXPECT_EQ(geo.vertices.size(), geo.normals.size());
  EXPECT_EQ(geo.vertices.size(), geo.texture_uvs.size());

  // Check material
  ASSERT_EQ(scene->materials.size(), 1);
  EXPECT_EQ(scene->materials[0].name, "Cube");
  // Check if texture loaded (Cube.gltf references Cube_BaseColor.png)
  EXPECT_GT(scene->materials[0].albedo.width, 0);
  EXPECT_GT(scene->materials[0].albedo.height, 0);

  // Verify file_path is populated and absolute
  ASSERT_TRUE(scene->materials[0].albedo.file_path.has_value());
  EXPECT_TRUE(scene->materials[0].albedo.file_path->is_absolute());
}

TEST(LoaderTest, MissingFile) {
  std::optional<Scene> scene = LoadScene("non_existent.gltf");
  EXPECT_FALSE(scene.has_value());
}

TEST(LoaderTest, LoadBoxFallbackColor) {
  std::filesystem::path input_path = "data/box/scene.gltf";
  ASSERT_TRUE(std::filesystem::exists(input_path))
      << "Test file not found: " << input_path;

  std::optional<Scene> scene = LoadScene(input_path);
  ASSERT_TRUE(scene.has_value());

  ASSERT_EQ(scene->materials.size(), 1);
  const auto& mat = scene->materials[0];
  EXPECT_EQ(mat.name, "Red");

  // Verify 1x1 fallback texture creation
  EXPECT_EQ(mat.albedo.width, 1);
  EXPECT_EQ(mat.albedo.height, 1);
  EXPECT_EQ(mat.albedo.channels, 4);
  ASSERT_EQ(mat.albedo.pixel_data.size(), 4);

  // Verify color (0.8 * 255 = 204)
  // Verify color (0.8 linear -> ~0.906 sRGB -> 231)
  EXPECT_NEAR(mat.albedo.pixel_data[0], 231, 1);
  EXPECT_EQ(mat.albedo.pixel_data[1], 0);
  EXPECT_EQ(mat.albedo.pixel_data[2], 0);
  EXPECT_EQ(mat.albedo.pixel_data[3], 255);
}

}  // namespace sh_baker
