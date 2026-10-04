#include "loader.h"

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <tiny_gltf.h>

#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include "layer_composite.h"
#include "saver.h"

namespace sh_baker {
namespace {

// data/layers: exporter-shaped SH_material_layers (see make_fixture.py).
std::filesystem::path LayersFixture() {
  std::filesystem::path path = "data/layers/scene.gltf";
  if (!std::filesystem::exists(path)) path = "../data/layers/scene.gltf";
  return path;
}

// Collects the WARNING messages glog emits while it is registered.
class WarningCapture : public google::LogSink {
 public:
  WarningCapture() { google::AddLogSink(this); }
  ~WarningCapture() override { google::RemoveLogSink(this); }

  void send(google::LogSeverity severity, const char* /*full_filename*/,
            const char* /*base_filename*/, int /*line*/,
            const google::LogMessageTime& /*time*/, const char* message,
            size_t message_len) override {
    if (severity != google::GLOG_WARNING) return;
    std::lock_guard<std::mutex> lock(mutex_);
    warnings_.emplace_back(message, message_len);
  }

  bool Contains(const std::string& text) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const std::string& w : warnings_) {
      if (w.find(text) != std::string::npos) return true;
    }
    return false;
  }

 private:
  std::mutex mutex_;
  std::vector<std::string> warnings_;
};

}  // namespace

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

// Every tcMod kind the exporter writes loads with its parameters and wave
// function, in order.
TEST(LoaderTest, ReadsEveryTcModKind) {
  std::optional<Scene> scene = LoadScene(LayersFixture());
  ASSERT_TRUE(scene.has_value());
  ASSERT_EQ(scene->materials.size(), 2u);
  const Material& mat = scene->materials[0];
  ASSERT_TRUE(mat.layers.has_value());
  EXPECT_EQ(mat.layers->surface_blend, SurfaceBlend::kBlend);
  EXPECT_EQ(mat.layers->cull_mode, CullMode::kNone);
  EXPECT_EQ(mat.layers->base_layer, 1);
  ASSERT_EQ(mat.layers->layers.size(), 2u);

  const MaterialLayer& layer = mat.layers->layers[0];
  ASSERT_TRUE(layer.texture_path.has_value());
  EXPECT_EQ(layer.texture_path->filename(), "layer0.png");
  EXPECT_TRUE(layer.anim_frame_paths.empty());
  EXPECT_EQ(layer.blend_src, BlendFactor::kOne);
  EXPECT_EQ(layer.blend_dst, BlendFactor::kZero);
  EXPECT_EQ(layer.rgbgen.type, RgbGenType::kWave);
  EXPECT_EQ(layer.rgbgen.wave, WaveType::kSine);
  EXPECT_EQ(layer.rgbgen.base, 0.5f);
  EXPECT_EQ(layer.rgbgen.amplitude, 0.25f);
  EXPECT_EQ(layer.rgbgen.phase, 0.0f);
  EXPECT_EQ(layer.rgbgen.frequency, 1.0f);

  const std::vector<TcMod>& mods = layer.tcmods;
  ASSERT_EQ(mods.size(), 6u);
  EXPECT_EQ(mods[0].type, TcModType::kScale);
  EXPECT_EQ(mods[0].values, (std::vector<float>{2.0f, 3.0f}));
  EXPECT_EQ(mods[1].type, TcModType::kScroll);
  EXPECT_EQ(mods[1].values, (std::vector<float>{0.5f, 0.0f}));
  EXPECT_EQ(mods[2].type, TcModType::kRotate);
  EXPECT_EQ(mods[2].values, (std::vector<float>{30.0f}));
  EXPECT_EQ(mods[3].type, TcModType::kTurb);
  EXPECT_EQ(mods[3].values, (std::vector<float>{0.0f, 0.125f, 0.0f, 1.0f}));
  EXPECT_EQ(mods[3].wave, WaveType::kSine);
  EXPECT_EQ(mods[4].type, TcModType::kStretch);
  EXPECT_EQ(mods[4].values, (std::vector<float>{1.0f, 0.5f, 0.0f, 2.0f}));
  EXPECT_EQ(mods[4].wave, WaveType::kNone);
  EXPECT_EQ(mods[5].type, TcModType::kTransform);
  EXPECT_EQ(mods[5].values,
            (std::vector<float>{1.0f, 0.0f, 0.5f, 0.0f, 1.0f, 0.0f}));
}

// An animMap stage keeps its frequency and its frames' source images in order.
TEST(LoaderTest, ReadsAnimatedLayer) {
  std::optional<Scene> scene = LoadScene(LayersFixture());
  ASSERT_TRUE(scene.has_value());
  ASSERT_EQ(scene->materials.size(), 2u);
  const Material& mat = scene->materials[1];
  ASSERT_TRUE(mat.layers.has_value());
  EXPECT_EQ(mat.layers->surface_blend, SurfaceBlend::kAdd);
  ASSERT_EQ(mat.layers->layers.size(), 1u);

  const MaterialLayer& layer = mat.layers->layers[0];
  EXPECT_EQ(layer.anim_freq, 5.0f);
  ASSERT_EQ(layer.anim_frame_paths.size(), 2u);
  ASSERT_TRUE(layer.anim_frame_paths[0].has_value());
  ASSERT_TRUE(layer.anim_frame_paths[1].has_value());
  EXPECT_EQ(layer.anim_frame_paths[0]->filename(), "flame0.png");
  EXPECT_EQ(layer.anim_frame_paths[1]->filename(), "layer1.png");
  EXPECT_EQ(layer.anim_frame_paths[0], layer.texture_path);
  EXPECT_TRUE(std::filesystem::exists(*layer.anim_frame_paths[1]));
  EXPECT_EQ(layer.blend_src, BlendFactor::kOne);
  EXPECT_EQ(layer.blend_dst, BlendFactor::kOne);
}

// A WAVE rgbGen without `func` loads with no wave function, composites as SIN,
// and saves without `func` again.
TEST(LoaderTest, KeepsWaveWithoutFuncAbsent) {
  std::optional<Scene> scene = LoadScene(LayersFixture());
  ASSERT_TRUE(scene.has_value());
  ASSERT_TRUE(scene->materials[0].layers.has_value());
  ASSERT_EQ(scene->materials[0].layers->layers.size(), 2u);
  const RgbGen& gen = scene->materials[0].layers->layers[1].rgbgen;
  EXPECT_EQ(gen.type, RgbGenType::kWave);
  EXPECT_FALSE(gen.wave.has_value());
  EXPECT_EQ(gen.base, 0.75f);
  EXPECT_EQ(gen.amplitude, 0.25f);
  EXPECT_EQ(gen.phase, 0.25f);
  EXPECT_EQ(gen.frequency, 0.5f);

  RgbGen as_sine = gen;
  as_sine.wave = WaveType::kSine;
  EXPECT_EQ(EvalRgbGen(gen), EvalRgbGen(as_sine));

  std::filesystem::path temp_dir =
      std::filesystem::temp_directory_path() / "sh_baker_test_wave_no_func";
  std::filesystem::remove_all(temp_dir);
  std::filesystem::create_directories(temp_dir);
  std::filesystem::path out = temp_dir / "scene.gltf";
  ASSERT_TRUE(SaveScene(*scene, out));

  tinygltf::Model model;
  tinygltf::TinyGLTF gltf;
  std::string err, warn;
  ASSERT_TRUE(gltf.LoadASCIIFromFile(&model, &err, &warn, out.string())) << err;
  const tinygltf::Value& layers =
      model.materials[0].extensions.at("SH_material_layers").Get("layers");
  EXPECT_FALSE(layers.Get(1).Get("rgbGen").Has("func"));
  EXPECT_EQ(layers.Get(0).Get("rgbGen").Get("func").Get<std::string>(), "SIN");

  std::filesystem::remove_all(temp_dir);
}

// An unknown blend factor name warns and loads as ONE.
TEST(LoaderTest, UnknownBlendFactorDefaultsToOne) {
  std::filesystem::path temp_dir =
      std::filesystem::temp_directory_path() / "sh_baker_test_bogus_blend";
  std::filesystem::remove_all(temp_dir);
  std::filesystem::copy(LayersFixture().parent_path(), temp_dir,
                        std::filesystem::copy_options::recursive);
  std::filesystem::path gltf_path = temp_dir / "scene.gltf";
  std::string text;
  {
    std::ifstream in(gltf_path);
    std::stringstream buffer;
    buffer << in.rdbuf();
    text = buffer.str();
  }
  const std::string from = "\"blendSrc\": \"DST_COLOR\"";
  size_t pos = text.find(from);
  ASSERT_NE(pos, std::string::npos);
  text.replace(pos, from.size(), "\"blendSrc\": \"BOGUS\"");
  std::ofstream(gltf_path) << text;

  WarningCapture capture;
  std::optional<Scene> scene = LoadScene(gltf_path);
  ASSERT_TRUE(scene.has_value());
  ASSERT_TRUE(scene->materials[0].layers.has_value());
  EXPECT_EQ(scene->materials[0].layers->layers[1].blend_src, BlendFactor::kOne);
  EXPECT_TRUE(capture.Contains("BOGUS"));

  std::filesystem::remove_all(temp_dir);
}

}  // namespace sh_baker
