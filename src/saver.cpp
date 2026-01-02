#include "saver.h"

#include <glog/logging.h>

#include "tinyexr.h"

namespace sh_baker {

bool SaveSHLightMap(const SHTexture& sh_texture,
                    const std::filesystem::path& path) {
  if (sh_texture.pixels.empty() || sh_texture.width <= 0 ||
      sh_texture.height <= 0) {
    LOG(ERROR) << "Invalid SHTexture dimensions or empty pixels.";
    return false;
  }

  EXRHeader header;
  InitEXRHeader(&header);

  EXRImage image;
  InitEXRImage(&image);

  image.num_channels = 27;  // 9 coeffs * 3 (RGB)

  // Channel names
  // Conventional ordering or custom?
  // User requested: "Project result into 3rd-order SH... Output... secondary
  // binary file... or glTF extension" Later: "Can saving as an openexr image be
  // a good idea?" We will name channels clearly. Band 0: L0 Band 1: L1m1, L10,
  // L11 Band 2: L2m2, L2m1, L20, L21, L22 Suffixes: .R, .G, .B
  const char* channel_names[] = {
      "L0.R",   "L0.G",   "L0.B",    // 0
      "L1m1.R", "L1m1.G", "L1m1.B",  // 1
      "L10.R",  "L10.G",  "L10.B",   // 2
      "L11.R",  "L11.G",  "L11.B",   // 3
      "L2m2.R", "L2m2.G", "L2m2.B",  // 4
      "L2m1.R", "L2m1.G", "L2m1.B",  // 5
      "L20.R",  "L20.G",  "L20.B",   // 6
      "L21.R",  "L21.G",  "L21.B",   // 7
      "L22.R",  "L22.G",  "L22.B"    // 8
  };

  std::vector<float> channels[27];
  float* image_ptr[27];
  header.channels =
      (EXRChannelInfo*)malloc(sizeof(EXRChannelInfo) * image.num_channels);

  // Allocate pixel types
  header.pixel_types = (int*)malloc(sizeof(int) * image.num_channels);
  header.requested_pixel_types = (int*)malloc(sizeof(int) * image.num_channels);

  int num_pixels = sh_texture.width * sh_texture.height;

  // Split SoA
  for (int i = 0; i < 27; ++i) {
    channels[i].resize(num_pixels);
    image_ptr[i] = channels[i].data();

    // Map i to (coeff_idx, rgb_idx)
    int coeff_idx = i / 3;
    int rgb_idx = i % 3;

    for (int p = 0; p < num_pixels; ++p) {
      if (rgb_idx == 0)
        channels[i][p] = sh_texture.pixels[p].coeffs[coeff_idx].x();
      else if (rgb_idx == 1)
        channels[i][p] = sh_texture.pixels[p].coeffs[coeff_idx].y();
      else
        channels[i][p] = sh_texture.pixels[p].coeffs[coeff_idx].z();
    }

    // Set header info
    strncpy(header.channels[i].name, channel_names[i], 255);
    header.pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT;
    header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT;
  }

  image.images = (unsigned char**)image_ptr;
  image.width = sh_texture.width;
  image.height = sh_texture.height;

  header.num_channels = image.num_channels;
  header.compression_type = TINYEXR_COMPRESSIONTYPE_NONE;

  const char* err = nullptr;
  int ret = SaveEXRImageToFile(&image, &header, path.string().c_str(), &err);

  free(header.channels);
  free(header.pixel_types);
  free(header.requested_pixel_types);

  if (ret != TINYEXR_SUCCESS) {
    LOG(ERROR) << "SaveEXRImageToFile failed: "
               << (err ? err : "Unknown error");
    FreeEXRErrorMessage(err);
    return false;
  }

  return true;
}

}  // namespace sh_baker
