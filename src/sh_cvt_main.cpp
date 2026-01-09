#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <stb_image_write.h>
#include <tinyexr.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

DEFINE_string(input, "", "Input directory containing EXR files to process.");
DEFINE_bool(reinhard, false, "Apply Reinhard tone mapping (x / (1+x)).");

namespace fs = std::filesystem;

// Clamp float value to [0, 1]
float Clamp(float v) { return std::max(0.0f, std::min(1.0f, v)); }

// Convert float to byte [0, 255]
unsigned char FloatToByte(float v) {
  return static_cast<unsigned char>(Clamp(v) * 255.0f + 0.5f);
}

void ProcessFile(const fs::path& path) {
  const char* err = nullptr;
  EXRVersion exr_version;
  int ret = ParseEXRVersionFromFile(&exr_version, path.string().c_str());
  if (ret != 0) {
    LOG(ERROR) << "Invalid EXR file: " << path;
    return;
  }

  EXRHeader exr_header;
  InitEXRHeader(&exr_header);
  if (ParseEXRHeaderFromFile(&exr_header, &exr_version, path.string().c_str(),
                             &err) != 0) {
    LOG(ERROR) << "Failed to parse EXR header: " << path << " ("
               << (err ? err : "?") << ")";
    FreeEXRErrorMessage(err);
    return;
  }

  // Ensure we request FLOAT pixels for simplicity
  for (int i = 0; i < exr_header.num_channels; ++i) {
    exr_header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT;
  }

  EXRImage exr_image;
  InitEXRImage(&exr_image);
  if (LoadEXRImageFromFile(&exr_image, &exr_header, path.string().c_str(),
                           &err) != 0) {
    LOG(ERROR) << "Failed to load EXR image: " << path << " ("
               << (err ? err : "?") << ")";
    FreeEXRErrorMessage(err);
    FreeEXRHeader(&exr_header);
    return;
  }

  // Find L0 channels
  int idx_r = -1, idx_g = -1, idx_b = -1;

  // Strategy:
  // 1. Look for "L0.R", "L0.G", "L0.B"
  // 2. If not found, look for "R", "G", "B"

  auto FindChannel = [&](const std::string& name) -> int {
    for (int i = 0; i < exr_header.num_channels; ++i) {
      if (std::string(exr_header.channels[i].name) == name) return i;
    }
    return -1;
  };

  idx_r = FindChannel("L0.R");
  idx_g = FindChannel("L0.G");
  idx_b = FindChannel("L0.B");

  if (idx_r == -1 || idx_g == -1 || idx_b == -1) {
    // Fallback to standard RGB
    idx_r = FindChannel("R");
    idx_g = FindChannel("G");
    idx_b = FindChannel("B");
  }

  if (idx_r == -1 || idx_g == -1 || idx_b == -1) {
    LOG(ERROR)
        << "Could not find L0 channels (L0.R/G/B) or standard RGB channels in "
        << path;
    FreeEXRImage(&exr_image);
    FreeEXRHeader(&exr_header);
    return;
  }

  int width = exr_image.width;
  int height = exr_image.height;
  std::vector<unsigned char> png_data(width * height * 3);

  const float* ptr_r = reinterpret_cast<const float*>(exr_image.images[idx_r]);
  const float* ptr_g = reinterpret_cast<const float*>(exr_image.images[idx_g]);
  const float* ptr_b = reinterpret_cast<const float*>(exr_image.images[idx_b]);

  for (size_t i = 0; i < static_cast<size_t>(width * height); ++i) {
    float r = ptr_r[i];
    float g = ptr_g[i];
    float b = ptr_b[i];

    if (FLAGS_reinhard) {
      r = r / (1.0f + r);
      g = g / (1.0f + g);
      b = b / (1.0f + b);
    }

    png_data[i * 3 + 0] = FloatToByte(r);
    png_data[i * 3 + 1] = FloatToByte(g);
    png_data[i * 3 + 2] = FloatToByte(b);
  }

  // Construct output path
  // Requirement: "save it as a png (pick a fixed name) in the same folder"
  // Interpret as: InputFilename + "_L0.png"
  std::string stem = path.stem().string();
  fs::path out_path = path.parent_path() / (stem + "_L0.png");

  if (stbi_write_png(out_path.string().c_str(), width, height, 3,
                     png_data.data(), width * 3)) {
    LOG(INFO) << "Saved " << out_path;
  } else {
    LOG(ERROR) << "Failed to write PNG to " << out_path;
  }

  FreeEXRImage(&exr_image);
  FreeEXRHeader(&exr_header);
}

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  if (FLAGS_input.empty()) {
    LOG(ERROR) << "Please provide --input_dir";
    return 1;
  }

  fs::path input_dir(FLAGS_input);
  if (!fs::exists(input_dir) || !fs::is_directory(input_dir)) {
    LOG(ERROR) << "Input path is not a valid directory: " << FLAGS_input;
    return 1;
  }

  for (const auto& entry : fs::directory_iterator(input_dir)) {
    if (entry.is_regular_file() && entry.path().extension() == ".exr") {
      ProcessFile(entry.path());
    }
  }

  return 0;
}
