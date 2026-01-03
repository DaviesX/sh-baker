#include <gflags/gflags.h>
#include <glog/logging.h>

#include <filesystem>
#include <iostream>

#include "atlas.h"
#include "baker.h"
#include "dilation.h"
#include "loader.h"
#include "rasterizer.h"
#include "saver.h"

DEFINE_string(input, "", "Path to the input glTF file.");
DEFINE_int32(width, 1024, "Width of the output image.");
DEFINE_int32(height, 1024, "Height of the output image.");
DEFINE_int32(samples, 128, "Number of samples per pixel.");
DEFINE_int32(bounces, 3, "Number of bounces.");
DEFINE_int32(dilation, 0, "Number of dilation passes.");
DEFINE_string(output, "",
              "Folder to contain the output lightmap and glTF file.");

const char* kLightmapFileName = "lightmap.exr";
const char* kGLTFFileName = "scene.gltf";

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_logtostderr = 1;
  google::InitGoogleLogging(argv[0]);

  if (FLAGS_input.empty()) {
    std::cerr << "Usage: " << argv[0] << " --input <gltf_file>" << std::endl;
    return 1;
  }

  std::filesystem::path input_path(FLAGS_input);
  if (!std::filesystem::exists(input_path)) {
    LOG(ERROR) << "Input file does not exist: " << input_path;
    return 1;
  }

  LOG(INFO) << "Loading scene from: " << input_path;
  std::optional<sh_baker::Scene> scene_opt = sh_baker::LoadScene(input_path);

  if (!scene_opt) {
    LOG(ERROR) << "Failed to load scene.";
    return 1;
  }

  auto& scene = *scene_opt;
  LOG(INFO) << "Scene loaded successfully.";
  LOG(INFO) << "  Geometries: " << scene.geometries.size();
  LOG(INFO) << "  Materials: " << scene.materials.size();
  LOG(INFO) << "  Lights: " << scene.lights.size();

  // Generate Atlas
  LOG(INFO) << "Generating Lightmap UVs (xatlas)...";
  scene.geometries = sh_baker::CreateAtlasGeometries(scene.geometries);
  LOG(INFO)
      << "Atlas generation complete. New Geometries vertex counts adjusted.";

  // Configure Rasterizer
  sh_baker::RasterConfig raster_config;
  raster_config.width = FLAGS_width;
  raster_config.height = FLAGS_height;

  LOG(INFO) << "Rasterizing scene (" << raster_config.width << "x"
            << raster_config.height << ")...";
  auto surface_points = sh_baker::RasterizeScene(scene, raster_config);

  // Configure Baker
  sh_baker::BakeConfig config;
  config.samples = FLAGS_samples;
  config.bounces = FLAGS_bounces;

  // Bake
  LOG(INFO) << "Starting Bake (" << FLAGS_samples << " samples)...";
  auto start_time = std::chrono::high_resolution_clock::now();
  sh_baker::SHTexture texture =
      sh_baker::BakeSHLightMap(scene, surface_points, raster_config, config);
  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end_time - start_time;
  LOG(INFO) << "Bake complete in " << elapsed.count() << " seconds.";

  // Dilate
  if (FLAGS_dilation > 0) {
    LOG(INFO) << "Dilating " << FLAGS_dilation << " passes...";
    std::vector<uint8_t> validity_mask =
        sh_baker::CreateValidityMask(surface_points);
    sh_baker::Dilate(texture.width, texture.height, texture.pixels,
                     validity_mask, FLAGS_dilation);
  }

  // Save
  if (!FLAGS_output.empty()) {
    LOG(INFO) << "Saving output to: " << FLAGS_output;
    std::filesystem::path output_path(FLAGS_output);
    std::filesystem::create_directories(output_path);
    std::filesystem::path lightmap_path = output_path / kLightmapFileName;
    std::filesystem::path gltf_path = output_path / kGLTFFileName;
    if (!sh_baker::SaveSHLightMap(texture, lightmap_path)) {
      LOG(ERROR) << "Failed to save output.";
      return 1;
    }
    if (!sh_baker::SaveScene(scene, gltf_path)) {
      LOG(ERROR) << "Failed to save glTF.";
      return 1;
    }
  }

  return 0;
}
