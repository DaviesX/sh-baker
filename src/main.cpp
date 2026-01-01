#include <gflags/gflags.h>
#include <glog/logging.h>

#include <filesystem>
#include <iostream>

#include "loader.h"

DEFINE_string(input, "", "Path to the input glTF file.");

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
  auto scene_opt = sh_baker::LoadScene(input_path);

  if (!scene_opt) {
    LOG(ERROR) << "Failed to load scene.";
    return 1;
  }

  const auto& scene = *scene_opt;
  LOG(INFO) << "Scene loaded successfully.";
  LOG(INFO) << "  Geometries: " << scene.geometries.size();
  LOG(INFO) << "  Materials: " << scene.materials.size();
  LOG(INFO) << "  Lights: " << scene.lights.size();

  return 0;
}
