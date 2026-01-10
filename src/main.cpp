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
DEFINE_int32(width, 2048, "Width of the output image.");
DEFINE_int32(height, 2048, "Height of the output image.");
DEFINE_int32(samples, 128, "Number of samples per pixel.");
DEFINE_int32(bounces, 3, "Number of bounces.");
DEFINE_int32(dilation, 16, "Number of dilation passes.");
DEFINE_string(output, "",
              "Folder to contain the output lightmap and glTF file.");
DEFINE_bool(split_channels, false,
            "If true, output 9 separate EXR files for SH coefficients.");
DEFINE_int32(supersample_scale, 1,
             "Scale factor for supersampling (e.g. 2 for 2x2).");

DEFINE_bool(luminance_only, false,
            "If true, compress Light map by storing only Luminance for L1/L2 "
            "coefficients (Packed into 3 textures).");

DEFINE_bool(debug_output, false,
            "If true, output debug information to the output folder.");

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
  if (scene.geometries.empty()) {
    LOG(ERROR) << "No geometries found in scene.";
    return 1;
  }

  std::optional<sh_baker::AtlasResult> atlas_result =
      sh_baker::CreateAtlasGeometries(scene.geometries, FLAGS_width,
                                      FLAGS_dilation);
  if (!atlas_result) {
    LOG(ERROR) << "Atlas generation failed (possibly could not fit charts).";
    return 1;
  }

  scene.geometries = atlas_result->geometries;
  LOG(INFO) << "Atlas generation complete. New Geometries vertex counts "
               "adjusted. Resolution adjusted to: "
            << atlas_result->width << "x" << atlas_result->height;

  // Configure Rasterizer
  sh_baker::RasterConfig raster_config;
  raster_config.width = FLAGS_width;  // Keep the requested resolution though
                                      // xatlas may prefer a different one.
  raster_config.height = FLAGS_height;
  raster_config.supersample_scale = FLAGS_supersample_scale;

  LOG(INFO) << "Rasterizing scene (" << raster_config.width << "x"
            << raster_config.height
            << ") scale: " << raster_config.supersample_scale << "...";
  auto surface_points = sh_baker::RasterizeScene(scene, raster_config);

  // Debug Output
  if (FLAGS_debug_output) {
    int scaled_w = raster_config.width * raster_config.supersample_scale;
    int scaled_h = raster_config.height * raster_config.supersample_scale;
    LOG(INFO) << "Generating Material Map (" << scaled_w << "x" << scaled_h
              << ")...";
    sh_baker::Texture mat_map =
        sh_baker::CreateMaterialMap(surface_points, scaled_w, scaled_h);

    std::filesystem::path out_dir = FLAGS_output.empty()
                                        ? std::filesystem::current_path()
                                        : std::filesystem::path(FLAGS_output);
    std::filesystem::create_directories(out_dir);
    std::filesystem::path map_path = out_dir / "material_map.png";
    if (sh_baker::SaveTexture(mat_map, map_path)) {
      LOG(INFO) << "Material map saved to: " << map_path;
    } else {
      LOG(ERROR) << "Failed to save material map to: " << map_path;
    }
  }

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

  std::vector<uint8_t> validity_mask =
      sh_baker::CreateValidityMask(surface_points);

  // Downsample if needed
  if (FLAGS_supersample_scale > 1) {
    LOG(INFO) << "Downsampling from scale " << FLAGS_supersample_scale << "...";
    texture = sh_baker::DownsampleSHTexture(texture, FLAGS_supersample_scale);
    validity_mask = sh_baker::DownsampleValidityMask(
        validity_mask, raster_config.width, raster_config.height,
        raster_config.supersample_scale);
  }

  // Dilate
  if (FLAGS_dilation > 0) {
    LOG(INFO) << "Dilating " << FLAGS_dilation << " passes...";
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

    sh_baker::SaveMode mode = sh_baker::SaveMode::kCombined;
    if (FLAGS_luminance_only) {
      mode = sh_baker::SaveMode::kLuminancePacked;
    } else if (FLAGS_split_channels) {
      mode = sh_baker::SaveMode::kSplitChannels;
    }

    if (!sh_baker::SaveSHLightMap(texture, lightmap_path, mode)) {
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
