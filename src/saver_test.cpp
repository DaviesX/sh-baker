#include "saver.h"

#include <gtest/gtest.h>
#include <tiny_gltf.h>

#include <filesystem>

#include "loader.h"
#include "scene.h"
#include "stb_image_write.h"
#include "tinyexr.h"

namespace sh_baker {

TEST(SaverTest, SaveCombinedImage) {
  SHTexture tex;
  tex.width = 16;
  tex.height = 16;
  tex.pixels.resize(16 * 16);

  Texture32F env_tex;
  env_tex.width = 16;
  env_tex.height = 16;
  env_tex.pixel_data.resize(16 * 16, 1.0f);

  // Fill with dummy data
  for (auto& sh : tex.pixels) {
    sh.coeffs[0] = Eigen::Vector3f(1.0f, 0.5f, 0.25f);
  }

  std::filesystem::path test_path = "test_output.exr";
  if (std::filesystem::exists(test_path)) {
    std::filesystem::remove(test_path);
  }

  bool success = SaveSHLightMap(tex, env_tex, test_path, SaveMode::kCombined);
  EXPECT_TRUE(success);
  EXPECT_TRUE(std::filesystem::exists(test_path));

  int verify_ret = IsEXR(test_path.string().c_str());
  EXPECT_EQ(verify_ret, TINYEXR_SUCCESS);

  if (std::filesystem::exists(test_path)) {
    std::filesystem::remove(test_path);
  }
}

TEST(SaverTest, SaveSplitChannels) {
  SHTexture tex;
  tex.width = 16;
  tex.height = 16;
  tex.pixels.resize(16 * 16);

  Texture32F env_tex;
  env_tex.width = 16;
  env_tex.height = 16;
  env_tex.pixel_data.resize(16 * 16, 0.5f);

  for (auto& sh : tex.pixels) {
    sh.coeffs[0] = Eigen::Vector3f(1.0f, 0.0f, 0.0f);  // L0
    sh.coeffs[1] = Eigen::Vector3f(0.0f, 1.0f, 0.0f);  // L1m1
  }

  std::filesystem::path test_path = "test_split.exr";
  // Expectations: test_split_L0.exr, test_split_L1m1.exr ...

  // Cleanup
  const char* suffixes[] = {"L0",   "L1m1", "L10", "L11", "L2m2",
                            "L2m1", "L20",  "L21", "L22", "EnvVisibility"};
  for (const char* suffix : suffixes) {
    std::string filename = std::string("test_split_") + suffix + ".exr";
    if (std::filesystem::exists(filename)) std::filesystem::remove(filename);
  }

  bool success =
      SaveSHLightMap(tex, env_tex, test_path, SaveMode::kSplitChannels);
  EXPECT_TRUE(success);

  for (const char* suffix : suffixes) {
    std::string filename = std::string("test_split_") + suffix + ".exr";
    EXPECT_TRUE(std::filesystem::exists(filename)) << "Missing " << filename;
    int verify_ret = IsEXR(filename.c_str());
    EXPECT_EQ(verify_ret, TINYEXR_SUCCESS);

    // Clean up
    if (std::filesystem::exists(filename)) std::filesystem::remove(filename);
  }
}

TEST(SaverTest, SaveSceneWithTexture) {
  // Setup temp directory
  std::filesystem::path temp_dir =
      std::filesystem::temp_directory_path() / "sh_baker_test_scene";
  std::filesystem::create_directories(temp_dir);

  // Create a dummy source texture file
  std::filesystem::path source_tex_dir = temp_dir / "source";
  std::filesystem::create_directories(source_tex_dir);
  std::filesystem::path source_tex_path = source_tex_dir / "test_albedo.png";

  {
    unsigned char pixels[] = {255, 0, 0};  // Red
    stbi_write_png(source_tex_path.string().c_str(), 1, 1, 3, pixels, 3);
  }

  // Create Scene
  Scene scene;
  Material mat;
  mat.name = "TestMat";
  // These dimensions are for display but we want to match the real file
  mat.albedo.width = 1;
  mat.albedo.height = 1;
  mat.albedo.file_path = source_tex_path;
  scene.materials.push_back(mat);

  // Add dummy geometry to trigger buffer generation
  Geometry geo;
  geo.vertices = {Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(1, 0, 0),
                  Eigen::Vector3f(0, 1, 0)};
  geo.indices = {0, 1, 2};
  geo.material_id = 0;
  scene.geometries.push_back(geo);

  std::filesystem::path output_gltf = temp_dir / "output" / "scene.gltf";
  std::filesystem::create_directories(output_gltf.parent_path());

  bool ret = SaveScene(scene, output_gltf);
  ASSERT_TRUE(ret);

  // Checks
  ASSERT_TRUE(std::filesystem::exists(output_gltf));

  // Check texture copy
  std::filesystem::path copied_tex_path =
      output_gltf.parent_path() / "test_albedo.png";
  EXPECT_TRUE(std::filesystem::exists(copied_tex_path));

  // Check bin file (External buffers)
  // TinyGLTF with !embedBuffers writes to a .bin file usually named after .gltf
  std::filesystem::path bin_path = output_gltf.parent_path() / "scene.bin";
  EXPECT_TRUE(std::filesystem::exists(bin_path));

  // Load back and verify
  tinygltf::Model model;
  tinygltf::TinyGLTF loader;
  std::string err, warn;
  bool load_ret =
      loader.LoadASCIIFromFile(&model, &err, &warn, output_gltf.string());
  ASSERT_TRUE(load_ret) << err;

  ASSERT_EQ(model.materials.size(), 1);
  EXPECT_EQ(model.materials[0].name, "TestMat");
  int tex_index =
      model.materials[0].pbrMetallicRoughness.baseColorTexture.index;
  ASSERT_GE(tex_index, 0);
  ASSERT_LT(tex_index, model.textures.size());

  int source_index = model.textures[tex_index].source;
  ASSERT_GE(source_index, 0);
  ASSERT_LT(source_index, model.images.size());

  EXPECT_EQ(model.images[source_index].uri, "test_albedo.png");

  // Cleanup
  std::filesystem::remove_all(temp_dir);
}

TEST(SaverTest, SaveSceneFallback1x1) {
  Scene scene;
  Material mat;
  mat.name = "FallbackMat";
  mat.albedo.width = 1;
  mat.albedo.height = 1;
  // sRGB color (255, 0, 0)
  mat.albedo.pixel_data = {255, 0, 0, 255};
  scene.materials.push_back(mat);

  std::filesystem::path temp_dir =
      std::filesystem::temp_directory_path() / "sh_baker_test_fallback";
  std::filesystem::create_directories(temp_dir);
  std::filesystem::path output_path = temp_dir / "fallback.gltf";

  SaveScene(scene, output_path);

  tinygltf::Model model;
  tinygltf::TinyGLTF loader;
  std::string err, warn;
  loader.LoadASCIIFromFile(&model, &err, &warn, output_path.string());

  ASSERT_EQ(model.materials.size(), 1);
  // Linearized Red: SRGBToLinear(1.0) = 1.0.
  auto baseColor = model.materials[0].pbrMetallicRoughness.baseColorFactor;
  EXPECT_EQ(baseColor.size(), 4);
  EXPECT_NEAR(baseColor[0], 1.0, 1e-4);
  EXPECT_NEAR(baseColor[1], 0.0, 1e-4);
  EXPECT_NEAR(baseColor[2], 0.0, 1e-4);
  EXPECT_NEAR(baseColor[3], 1.0, 1e-4);

  std::filesystem::remove_all(temp_dir);
}

TEST(SaverTest, SavePackedLuminance) {
  SHTexture tex;
  tex.width = 16;
  tex.height = 16;
  tex.pixels.resize(16 * 16);

  Texture32F env_tex;
  env_tex.width = 16;
  env_tex.height = 16;
  env_tex.pixel_data.resize(16 * 16, 0.2f);

  // Fill with dummy data
  for (auto& sh : tex.pixels) {
    // L0 = (1.0, 0.5, 0.25)
    sh.coeffs[0] = Eigen::Vector3f(1.0f, 0.5f, 0.25f);
    // L1m1 = (0.5, 0.5, 0.5)
    sh.coeffs[1] = Eigen::Vector3f(0.5f, 0.5f, 0.5f);
  }

  std::filesystem::path test_path = "test_packed.exr";

  // Cleanup
  for (int i = 0; i < 3; ++i) {
    std::string filename = "test_packed_packed_" + std::to_string(i) + ".exr";
    if (std::filesystem::exists(filename)) std::filesystem::remove(filename);
  }

  bool success =
      SaveSHLightMap(tex, env_tex, test_path, SaveMode::kLuminancePacked);
  EXPECT_TRUE(success);

  for (int i = 0; i < 3; ++i) {
    std::string filename = "test_packed_packed_" + std::to_string(i) + ".exr";
    EXPECT_TRUE(std::filesystem::exists(filename)) << "Missing " << filename;

    // Verify it is a valid EXR
    int verify_ret = IsEXR(filename.c_str());
    EXPECT_EQ(verify_ret, TINYEXR_SUCCESS);

    // Clean up
    if (std::filesystem::exists(filename)) std::filesystem::remove(filename);
  }
}

TEST(SaverTest, SaveComplexScene) {
  Scene scene;

  // 1. Materials
  for (int i = 0; i < 5; ++i) {
    Material mat;
    mat.name = "Mat_" + std::to_string(i);
    // 1x1 albedo to avoid file copy overhead in test
    mat.albedo.width = 1;
    mat.albedo.height = 1;
    mat.albedo.pixel_data = {255, 255, 255, 255};
    scene.materials.push_back(mat);
  }

  // 2. Geometries
  for (int i = 0; i < 3; ++i) {
    Geometry geo;
    geo.vertices = {Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(1, 0, 0),
                    Eigen::Vector3f(0, 1, 0)};
    geo.normals = {Eigen::Vector3f(0, 0, 1), Eigen::Vector3f(0, 0, 1),
                   Eigen::Vector3f(0, 0, 1)};
    geo.texture_uvs = {Eigen::Vector2f(0, 0), Eigen::Vector2f(1, 0),
                       Eigen::Vector2f(0, 1)};
    geo.indices = {0, 1, 2};
    geo.material_id = i;  // Use different materials
    scene.geometries.push_back(geo);
  }

  // 3. Lights
  Light pointLight;
  pointLight.type = Light::Type::Point;
  pointLight.position = Eigen::Vector3f(10, 10, 10);
  pointLight.intensity = 5.0f;
  scene.lights.push_back(pointLight);

  Light spotLight;
  spotLight.type = Light::Type::Spot;
  spotLight.position = Eigen::Vector3f(0, 5, 0);
  spotLight.direction = Eigen::Vector3f(0, -1, 0);
  // cos(angle)
  spotLight.cos_inner_cone = std::cos(0.5f);
  spotLight.cos_outer_cone = std::cos(0.8f);
  scene.lights.push_back(spotLight);

  Light dirLight;
  dirLight.type = Light::Type::Directional;
  dirLight.direction = Eigen::Vector3f(1, 0, 0);
  scene.lights.push_back(dirLight);

  // Setup path
  std::filesystem::path temp_dir =
      std::filesystem::temp_directory_path() / "sh_baker_test_complex";
  std::filesystem::create_directories(temp_dir);
  std::filesystem::path output_path = temp_dir / "complex.gltf";

  // Save
  bool ret = SaveScene(scene, output_path);
  ASSERT_TRUE(ret);

  // Load back using sh_baker::LoadScene
  auto loaded_scene_opt = LoadScene(output_path);
  ASSERT_TRUE(loaded_scene_opt.has_value())
      << "Failed to load saved scene from " << output_path;
  const Scene& loaded_scene = *loaded_scene_opt;

  // Checks
  EXPECT_EQ(loaded_scene.materials.size(), 5);
  EXPECT_EQ(loaded_scene.geometries.size(), 3);

  // Check Lights
  EXPECT_EQ(loaded_scene.lights.size(), 3);

  // Verify light types
  int point_count = 0;
  int spot_count = 0;
  int dir_count = 0;

  for (const auto& l : loaded_scene.lights) {
    if (l.type == Light::Type::Point) point_count++;
    if (l.type == Light::Type::Spot) spot_count++;
    if (l.type == Light::Type::Directional) dir_count++;
  }
  EXPECT_EQ(point_count, 1);
  EXPECT_EQ(spot_count, 1);
  EXPECT_EQ(dir_count, 1);

  // Cleanup
  std::filesystem::remove_all(temp_dir);
}

}  // namespace sh_baker
