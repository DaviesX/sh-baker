#include "saver.h"

#include <gtest/gtest.h>
#include <tiny_gltf.h>

#include <algorithm>
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
  std::filesystem::path irr_path = "test_output_irradiance.exr";
  if (std::filesystem::exists(test_path)) std::filesystem::remove(test_path);
  if (std::filesystem::exists(irr_path)) std::filesystem::remove(irr_path);

  bool success = SaveSHLightMap(tex, env_tex, test_path, SaveMode::kCombined);
  EXPECT_TRUE(success);
  EXPECT_TRUE(std::filesystem::exists(test_path));

  int verify_ret = IsEXR(test_path.string().c_str());
  EXPECT_EQ(verify_ret, TINYEXR_SUCCESS);

  if (std::filesystem::exists(test_path)) std::filesystem::remove(test_path);
  if (std::filesystem::exists(irr_path)) std::filesystem::remove(irr_path);
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
                            "L2m1", "L20",  "L21", "L22"};
  for (const char* suffix : suffixes) {
    std::string filename = std::string("test_split_") + suffix + ".exr";
    if (std::filesystem::exists(filename)) std::filesystem::remove(filename);
  }
  if (std::filesystem::exists("test_split_irradiance.exr"))
    std::filesystem::remove("test_split_irradiance.exr");

  bool success =
      SaveSHLightMap(tex, env_tex, test_path, SaveMode::kSplitChannels);
  EXPECT_TRUE(success);

  for (const char* suffix : suffixes) {
    std::string filename = std::string("test_split_") + suffix + ".exr";
    EXPECT_TRUE(std::filesystem::exists(filename)) << "Missing " << filename;

    // Verify EXR
    float* out;
    int width;
    int height;
    const char* err = nullptr;
    int ret = LoadEXR(&out, &width, &height, filename.c_str(), &err);
    EXPECT_EQ(ret, TINYEXR_SUCCESS) << err;
    free(out);
    // TODO: Verify L0 has 4 channels if possible with LoadEXR or check header
    // separately. TinyEXR LoadEXR returns flattened RGBA floats by default if I
    // recall correctly unless using custom loader. But here we just want to
    // know if it saved successfully.

    // Clean up
    if (std::filesystem::exists(filename)) std::filesystem::remove(filename);
  }

  // Verify that EnvVisibility file does NOT exist
  std::string env_filename = "test_split_EnvVisibility.exr";
  EXPECT_FALSE(std::filesystem::exists(env_filename));

  // Verify and cleanup Irradiance
  std::string irr_filename = "test_split_irradiance.exr";
  EXPECT_TRUE(std::filesystem::exists(irr_filename));
  if (std::filesystem::exists(irr_filename))
    std::filesystem::remove(irr_filename);
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
  if (std::filesystem::exists("test_packed_irradiance.exr"))
    std::filesystem::remove("test_packed_irradiance.exr");

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

  std::string irr_filename = "test_packed_irradiance.exr";
  EXPECT_TRUE(std::filesystem::exists(irr_filename));
  if (std::filesystem::exists(irr_filename))
    std::filesystem::remove(irr_filename);
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

TEST(SaverTest, SaveSceneEmission) {
  Scene scene;
  Material mat;
  mat.name = "EmissionMat";
  mat.albedo.width = 1;
  mat.albedo.height = 1;
  mat.albedo.pixel_data = {255, 255, 255, 255};

  // Emission Setup
  mat.emissive_factor = Eigen::Vector3f(1.0f, 0.5f, 0.0f);
  mat.emissive_strength = 5.0f;

  std::filesystem::path temp_dir =
      std::filesystem::temp_directory_path() / "sh_baker_test_emission";
  std::filesystem::create_directories(temp_dir);

  std::filesystem::path source_dir = temp_dir / "source";
  std::filesystem::create_directories(source_dir);

  // Emissive Texture (1x1 red)
  // We must save it to disk because SaveScene currently only supports
  // file-backed textures
  std::filesystem::path emissive_path = source_dir / "emissive.png";
  {
    unsigned char pixels[] = {255, 0, 0};
    stbi_write_png(emissive_path.string().c_str(), 1, 1, 3, pixels, 3);
  }

  mat.emissive_texture = Texture();
  mat.emissive_texture->width = 1;
  mat.emissive_texture->height = 1;
  mat.emissive_texture->channels = 3;
  mat.emissive_texture->pixel_data = {255, 0, 0};
  mat.emissive_texture->file_path = emissive_path;

  scene.materials.push_back(mat);

  std::filesystem::path output_path = temp_dir / "emission.gltf";

  bool ret = SaveScene(scene, output_path);
  ASSERT_TRUE(ret);

  // Load back using tinygltf directly to verify structure
  tinygltf::Model model;
  tinygltf::TinyGLTF loader;
  std::string err, warn;
  bool load_ret =
      loader.LoadASCIIFromFile(&model, &err, &warn, output_path.string());
  ASSERT_TRUE(load_ret) << err;

  ASSERT_EQ(model.materials.size(), 1);
  const auto& gmat = model.materials[0];

  // Verify Emissive Factor
  EXPECT_EQ(gmat.emissiveFactor.size(), 3);
  EXPECT_NEAR(gmat.emissiveFactor[0], 1.0, 1e-4);
  EXPECT_NEAR(gmat.emissiveFactor[1], 0.5, 1e-4);
  EXPECT_NEAR(gmat.emissiveFactor[2], 0.0, 1e-4);

  // Verify Emissive Strength Extension
  auto ext_it = gmat.extensions.find("KHR_materials_emissive_strength");
  ASSERT_NE(ext_it, gmat.extensions.end())
      << "Missing KHR_materials_emissive_strength extension";
  ASSERT_TRUE(ext_it->second.Has("emissiveStrength"));
  double strength = ext_it->second.Get("emissiveStrength").GetNumberAsDouble();
  EXPECT_NEAR(strength, 5.0, 1e-4);

  // Verify Emissive Texture
  ASSERT_GE(gmat.emissiveTexture.index, 0);

  std::filesystem::remove_all(temp_dir);
}

// The baker must pass the SH_material_layers extension through to its output,
// copying each layer/animMap texture and remapping its index, so the renderer
// can read the Quake 3 layer stack.
TEST(SaverTest, PassesMaterialLayersThrough) {
  std::filesystem::path temp_dir =
      std::filesystem::temp_directory_path() / "sh_baker_test_layers";
  std::filesystem::remove_all(temp_dir);
  std::filesystem::create_directories(temp_dir);

  // Source layer / animMap-frame textures on disk.
  auto write_png = [](const std::filesystem::path& p, unsigned char r) {
    unsigned char px[] = {r, 0, 0};
    stbi_write_png(p.string().c_str(), 1, 1, 3, px, 3);
  };
  std::filesystem::path layer0 = temp_dir / "layer0.png";
  std::filesystem::path layer1 = temp_dir / "layer1.png";
  std::filesystem::path frame1 = temp_dir / "frame1.png";
  write_png(layer0, 10);
  write_png(layer1, 20);
  write_png(frame1, 30);

  // Build a verbatim SH_material_layers Value with INPUT texture indices.
  auto str = [](const char* s) { return tinygltf::Value(std::string(s)); };
  tinygltf::Value::Object rgb_identity;
  rgb_identity["type"] = str("IDENTITY");

  tinygltf::Value::Object tex10;
  tex10["index"] = tinygltf::Value(10);
  tinygltf::Value::Object scale;
  scale["type"] = str("SCALE");
  scale["value"] = tinygltf::Value(tinygltf::Value::Array{tinygltf::Value(4.0),
                                                          tinygltf::Value(4.0)});
  tinygltf::Value::Object l0;
  l0["texture"] = tinygltf::Value(tex10);
  l0["blendSrc"] = str("ONE");
  l0["blendDst"] = str("ZERO");
  l0["rgbGen"] = tinygltf::Value(rgb_identity);
  l0["tcMod"] = tinygltf::Value(tinygltf::Value::Array{tinygltf::Value(scale)});
  l0["animFreq"] = tinygltf::Value(5.0);
  l0["animFrames"] = tinygltf::Value(
      tinygltf::Value::Array{tinygltf::Value(10), tinygltf::Value(15)});

  tinygltf::Value::Object tex11;
  tex11["index"] = tinygltf::Value(11);
  tinygltf::Value::Object l1;
  l1["texture"] = tinygltf::Value(tex11);
  l1["blendSrc"] = str("SRC_ALPHA");
  l1["blendDst"] = str("ONE_MINUS_SRC_ALPHA");
  l1["rgbGen"] = tinygltf::Value(rgb_identity);

  tinygltf::Value::Object ext;
  ext["surfaceBlend"] = str("OPAQUE");
  ext["cullMode"] = str("FRONT");
  ext["baseLayer"] = tinygltf::Value(1);
  ext["layers"] =
      tinygltf::Value(tinygltf::Value::Array{tinygltf::Value(l0),
                                             tinygltf::Value(l1)});

  Scene scene;
  Material mat;
  mat.name = "Layered";
  MaterialLayers ml;
  ml.extension = tinygltf::Value(ext);
  ml.texture_paths[10] = layer0;  // layer 0 + animFrames[0]
  ml.texture_paths[11] = layer1;  // layer 1
  ml.texture_paths[15] = frame1;  // animFrames[1]
  mat.layers = ml;
  scene.materials.push_back(mat);

  Geometry geo;
  geo.vertices = {Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(1, 0, 0),
                  Eigen::Vector3f(0, 1, 0)};
  geo.indices = {0, 1, 2};
  geo.material_id = 0;
  scene.geometries.push_back(geo);

  std::filesystem::path out = temp_dir / "output" / "scene.gltf";
  std::filesystem::create_directories(out.parent_path());
  ASSERT_TRUE(SaveScene(scene, out));

  // Layer / frame textures were copied next to the output.
  EXPECT_TRUE(std::filesystem::exists(out.parent_path() / "layer0.png"));
  EXPECT_TRUE(std::filesystem::exists(out.parent_path() / "layer1.png"));
  EXPECT_TRUE(std::filesystem::exists(out.parent_path() / "frame1.png"));

  // Reload and verify the extension survived with remapped indices.
  tinygltf::Model model;
  tinygltf::TinyGLTF loader;
  std::string err, warn;
  ASSERT_TRUE(loader.LoadASCIIFromFile(&model, &err, &warn, out.string()))
      << err;
  ASSERT_EQ(model.materials.size(), 1u);
  auto eit = model.materials[0].extensions.find("SH_material_layers");
  ASSERT_NE(eit, model.materials[0].extensions.end());
  const tinygltf::Value& rext = eit->second;

  EXPECT_EQ(rext.Get("surfaceBlend").Get<std::string>(), "OPAQUE");
  EXPECT_EQ(rext.Get("cullMode").Get<std::string>(), "FRONT");
  EXPECT_EQ(rext.Get("baseLayer").GetNumberAsInt(), 1);
  ASSERT_TRUE(rext.Get("layers").IsArray());
  ASSERT_EQ(rext.Get("layers").ArrayLen(), 2u);

  auto uri_of = [&](int tex_idx) -> std::string {
    EXPECT_GE(tex_idx, 0);
    EXPECT_LT(tex_idx, static_cast<int>(model.textures.size()));
    int src = model.textures[tex_idx].source;
    return model.images[src].uri;
  };

  const tinygltf::Value& r0 = rext.Get("layers").Get(0);
  EXPECT_EQ(r0.Get("blendSrc").Get<std::string>(), "ONE");
  EXPECT_EQ(r0.Get("blendDst").Get<std::string>(), "ZERO");
  EXPECT_DOUBLE_EQ(r0.Get("animFreq").GetNumberAsDouble(), 5.0);
  EXPECT_EQ(r0.Get("tcMod").Get(0).Get("type").Get<std::string>(), "SCALE");
  EXPECT_EQ(uri_of(r0.Get("texture").Get("index").GetNumberAsInt()),
            "layer0.png");
  // animFrames remapped: [layer0.png, frame1.png]; frame 0 dedups to the layer.
  ASSERT_EQ(r0.Get("animFrames").ArrayLen(), 2u);
  EXPECT_EQ(uri_of(r0.Get("animFrames").Get(0).GetNumberAsInt()), "layer0.png");
  EXPECT_EQ(uri_of(r0.Get("animFrames").Get(1).GetNumberAsInt()), "frame1.png");

  const tinygltf::Value& r1 = rext.Get("layers").Get(1);
  EXPECT_EQ(r1.Get("blendSrc").Get<std::string>(), "SRC_ALPHA");
  EXPECT_EQ(uri_of(r1.Get("texture").Get("index").GetNumberAsInt()),
            "layer1.png");

  EXPECT_NE(std::find(model.extensionsUsed.begin(), model.extensionsUsed.end(),
                      "SH_material_layers"),
            model.extensionsUsed.end());

  std::filesystem::remove_all(temp_dir);
}

// The loader must retain the SH_material_layers extension (with resolved source
// paths) so it survives a load -> save round trip.
TEST(SaverTest, LoaderRetainsMaterialLayers) {
  std::filesystem::path temp_dir =
      std::filesystem::temp_directory_path() / "sh_baker_test_layers_rt";
  std::filesystem::remove_all(temp_dir);
  std::filesystem::create_directories(temp_dir);

  std::filesystem::path layer0 = temp_dir / "rt_layer0.png";
  std::filesystem::path layer1 = temp_dir / "rt_layer1.png";
  {
    unsigned char px[] = {200, 100, 50};
    stbi_write_png(layer0.string().c_str(), 1, 1, 3, px, 3);
    unsigned char px2[] = {10, 20, 30};
    stbi_write_png(layer1.string().c_str(), 1, 1, 3, px2, 3);
  }

  auto str = [](const char* s) { return tinygltf::Value(std::string(s)); };
  auto make_layer = [&](int idx, const char* src, const char* dst) {
    tinygltf::Value::Object tex;
    tex["index"] = tinygltf::Value(idx);
    tinygltf::Value::Object rgb;
    rgb["type"] = str("IDENTITY");
    tinygltf::Value::Object lo;
    lo["texture"] = tinygltf::Value(tex);
    lo["blendSrc"] = str(src);
    lo["blendDst"] = str(dst);
    lo["rgbGen"] = tinygltf::Value(rgb);
    return lo;
  };
  tinygltf::Value::Object ext;
  ext["surfaceBlend"] = str("OPAQUE");
  ext["cullMode"] = str("FRONT");
  ext["baseLayer"] = tinygltf::Value(0);
  ext["layers"] = tinygltf::Value(tinygltf::Value::Array{
      tinygltf::Value(make_layer(7, "ONE", "ZERO")),
      tinygltf::Value(make_layer(8, "SRC_ALPHA", "ONE_MINUS_SRC_ALPHA"))});

  Scene scene;
  Material mat;
  mat.name = "Layered";
  mat.albedo.file_path = layer0;  // modern albedo placeholder
  mat.albedo.width = 1;
  mat.albedo.height = 1;
  MaterialLayers ml;
  ml.extension = tinygltf::Value(ext);
  ml.texture_paths[7] = layer0;
  ml.texture_paths[8] = layer1;
  mat.layers = ml;
  scene.materials.push_back(mat);

  // Geometry needs NORMAL + TEXCOORD_0 for the loader to accept the primitive.
  Geometry geo;
  geo.vertices = {Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(1, 0, 0),
                  Eigen::Vector3f(0, 1, 0)};
  geo.normals = {Eigen::Vector3f(0, 0, 1), Eigen::Vector3f(0, 0, 1),
                 Eigen::Vector3f(0, 0, 1)};
  geo.texture_uvs = {Eigen::Vector2f(0, 0), Eigen::Vector2f(1, 0),
                     Eigen::Vector2f(0, 1)};
  geo.indices = {0, 1, 2};
  geo.material_id = 0;
  scene.geometries.push_back(geo);

  std::filesystem::path out = temp_dir / "output" / "scene.gltf";
  std::filesystem::create_directories(out.parent_path());
  ASSERT_TRUE(SaveScene(scene, out));

  // Load the saved glTF back: the loader should capture the extension again.
  std::optional<Scene> loaded = LoadScene(out);
  ASSERT_TRUE(loaded.has_value());
  ASSERT_EQ(loaded->materials.size(), 1u);
  ASSERT_TRUE(loaded->materials[0].layers.has_value());

  const MaterialLayers& rl = *loaded->materials[0].layers;
  EXPECT_EQ(rl.extension.Get("baseLayer").GetNumberAsInt(), 0);
  ASSERT_TRUE(rl.extension.Get("layers").IsArray());
  EXPECT_EQ(rl.extension.Get("layers").ArrayLen(), 2u);
  // Both layer textures resolved to existing source files.
  EXPECT_EQ(rl.texture_paths.size(), 2u);
  for (const auto& [idx, p] : rl.texture_paths) {
    EXPECT_TRUE(std::filesystem::exists(p)) << p;
  }

  std::filesystem::remove_all(temp_dir);
}

}  // namespace sh_baker
