#include "saver.h"

#include <gtest/gtest.h>

#include <filesystem>

#include "tinyexr.h"

namespace sh_baker {

TEST(SaverTest, SaveCombinedImage) {
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

  bool success = SaveSHLightMap(tex, test_path, SaveMode::kCombined);
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

  bool success = SaveSHLightMap(tex, test_path, SaveMode::kSplitChannels);
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

}  // namespace sh_baker
