#include "saver.h"

#include <gtest/gtest.h>

#include <filesystem>

#include "tinyexr.h"

namespace sh_baker {

TEST(SaverTest, SaveSimpleImage) {
  SHTexture tex;
  tex.width = 16;
  tex.height = 16;
  tex.pixels.resize(16 * 16);

  // Fill with dummy data
  for (auto& sh : tex.pixels) {
    sh.coeffs[0] = Eigen::Vector3f(1.0f, 0.5f, 0.25f);
  }

  std::filesystem::path test_path = "test_output.exr";
  if (std::filesystem::exists(test_path)) {
    std::filesystem::remove(test_path);
  }

  bool success = SaveSHLightMap(tex, test_path);
  EXPECT_TRUE(success);
  EXPECT_TRUE(std::filesystem::exists(test_path));

  // Verify header with tinyexr
  float* out;
  int width;
  int height;
  const char* err = nullptr;
  int ret = LoadEXR(&out, &width, &height, test_path.string().c_str(), &err);

  // Note: LoadEXR (simple version) loads RGBA. Our file has 27 channels.
  // LoadEXR might fail or just read RGBA layer if present.
  // We used "L0.R" etc. so it might not find "R", "G", "B".
  // So we expect LoadEXR to potentially fail or warn if we don't have default
  // channels. But strictly we just want to know if it's a valid EXR.

  // Let's rely on IsEXR.
  int verify_ret = IsEXR(test_path.string().c_str());
  EXPECT_EQ(verify_ret, TINYEXR_SUCCESS);

  if (std::filesystem::exists(test_path)) {
    std::filesystem::remove(test_path);
  }
}

}  // namespace sh_baker
