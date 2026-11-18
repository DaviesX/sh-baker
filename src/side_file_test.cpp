#include "side_file.h"

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <stb_image.h>
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <string>
#include <vector>

namespace sh_baker {
namespace {

namespace fs = std::filesystem;

// A fresh directory per test, removed afterwards.
class SideFileTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const ::testing::TestInfo* info =
        ::testing::UnitTest::GetInstance()->current_test_info();
    dir_ = fs::temp_directory_path() /
           ("sh_side_file_" + std::string(info->name()) + "_" +
            std::to_string(getpid()));
    fs::remove_all(dir_);
    fs::create_directories(dir_);
  }
  void TearDown() override { fs::remove_all(dir_); }

  fs::path dir_;
};

using SideFileDeathTest = SideFileTest;

// Collects the ERROR messages glog emits while it is registered.
class ErrorCapture : public google::LogSink {
 public:
  ErrorCapture() { google::AddLogSink(this); }
  ~ErrorCapture() override { google::RemoveLogSink(this); }

  void send(google::LogSeverity severity, const char* /*full_filename*/,
            const char* /*base_filename*/, int /*line*/,
            const google::LogMessageTime& /*time*/, const char* message,
            size_t message_len) override {
    if (severity != google::GLOG_ERROR) return;
    std::lock_guard<std::mutex> lock(mutex_);
    errors_.emplace_back(message, message_len);
  }

  bool Contains(const std::string& text) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const std::string& e : errors_) {
      if (e.find(text) != std::string::npos) return true;
    }
    return false;
  }

  std::string All() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::string all;
    for (const std::string& e : errors_) all += e + "\n";
    return all;
  }

 private:
  std::mutex mutex_;
  std::vector<std::string> errors_;
};

std::vector<uint8_t> ReadBytes(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  return std::vector<uint8_t>(std::istreambuf_iterator<char>(in), {});
}

void WriteBytes(const fs::path& path, const std::vector<uint8_t>& bytes) {
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

void PutU16At(std::vector<uint8_t>* bytes, size_t offset, uint16_t v) {
  (*bytes)[offset] = v & 0xFF;
  (*bytes)[offset + 1] = v >> 8;
}

void PutU32At(std::vector<uint8_t>* bytes, size_t offset, uint32_t v) {
  for (int i = 0; i < 4; ++i) (*bytes)[offset + i] = (v >> (8 * i)) & 0xFF;
}

float Halfed(float v) { return static_cast<float>(Eigen::half(v)); }

uint32_t Bits(float v) {
  uint32_t b;
  std::memcpy(&b, &v, 4);
  return b;
}

// A valid page whose values vary by luxel, coefficient and channel.
SHSidePage MakePage(int width, int height, float seed) {
  SHSidePage page;
  page.width = width;
  page.height = height;
  const int n = width * height;
  page.sh.resize(n);
  page.visibility.resize(n);
  for (int j = 0; j < n; ++j) {
    for (int k = 0; k < 9; ++k) {
      for (int c = 0; c < 3; ++c) {
        float v = seed + 0.37f * j + 0.11f * k + 0.05f * c;
        // L0 stays non-negative; higher bands take both signs.
        page.sh[j].coeffs[k][c] = (k == 0 || (j + k + c) % 2 == 0) ? v : -v;
      }
    }
    page.visibility[j] = std::fmod(0.13f * j + seed, 1.0f);
  }
  return page;
}

SHHullMesh TwoHullMesh() {
  SHHullMesh mesh;
  mesh.positions = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {5, 5, 5}};
  mesh.indices = {0, 1, 2, 0, 2, 3, 1, 3, 4};
  mesh.hulls = {{0, 6}, {6, 3}};
  return mesh;
}

// The file of the "Golden bytes" scenario.
SHSideFile GoldenFile() {
  SHSideFile file;
  SHSidePage page;
  page.width = 1;
  page.height = 1;
  page.sh.resize(1);
  for (int k = 0; k < 9; ++k) {
    page.sh[0].coeffs[k] = Eigen::Vector3f(k, k + 0.5f, -float(k));
  }
  page.visibility = {0.25f};
  file.pages.push_back(page);
  file.hull_mesh.positions = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
  file.hull_mesh.indices = {0, 1, 2};
  file.hull_mesh.hulls = {{0, 3}};
  return file;
}

// Byte offsets into the golden file.
constexpr size_t kGoldenWidth = 24;
constexpr size_t kGoldenCoeffs = 32;  // coefficient k channel c: +6k + 2c
constexpr size_t kGoldenVisibility = 86;
constexpr size_t kGoldenVertices = 96;
constexpr size_t kGoldenIndices = 132;

void ExpectSameValues(const SHSideFile& written, const SHSideFile& read) {
  ASSERT_EQ(read.pages.size(), written.pages.size());
  for (size_t p = 0; p < written.pages.size(); ++p) {
    const SHSidePage& w = written.pages[p];
    const SHSidePage& r = read.pages[p];
    ASSERT_EQ(r.width, w.width) << "page " << p;
    ASSERT_EQ(r.height, w.height) << "page " << p;
    ASSERT_EQ(r.sh.size(), w.sh.size());
    ASSERT_EQ(r.visibility.size(), w.visibility.size());
    for (size_t j = 0; j < w.sh.size(); ++j) {
      for (int k = 0; k < 9; ++k) {
        for (int c = 0; c < 3; ++c) {
          ASSERT_EQ(Bits(r.sh[j].coeffs[k][c]),
                    Bits(Halfed(w.sh[j].coeffs[k][c])))
              << "page " << p << " luxel " << j << " coeff " << k;
        }
      }
      ASSERT_EQ(Bits(r.visibility[j]), Bits(Halfed(w.visibility[j])));
    }
  }
  const SHHullMesh& w = written.hull_mesh;
  const SHHullMesh& r = read.hull_mesh;
  ASSERT_EQ(r.positions.size(), w.positions.size());
  for (size_t v = 0; v < w.positions.size(); ++v) {
    EXPECT_EQ(r.positions[v], w.positions[v]);
  }
  EXPECT_EQ(r.indices, w.indices);
  ASSERT_EQ(r.hulls.size(), w.hulls.size());
  for (size_t h = 0; h < w.hulls.size(); ++h) {
    EXPECT_EQ(r.hulls[h].first_index, w.hulls[h].first_index);
    EXPECT_EQ(r.hulls[h].index_count, w.hulls[h].index_count);
  }
}

// --- Writer ---------------------------------------------------------------

TEST_F(SideFileTest, GoldenBytes) {
  // Built by hand from the layout in side_file.h.
  const std::vector<uint8_t> expected = {
      // Header: magic, version 1, 1 page, 1 hull, 3 vertices, 3 indices.
      'S', 'H', 'S', 'F', 1, 0, 0, 0, 1, 0, 0, 0,  //
      1, 0, 0, 0, 3, 0, 0, 0, 3, 0, 0, 0,          //
      // Page: width 1, height 1.
      1, 0, 0, 0, 1, 0, 0, 0,
      // Coefficient planes, (k, k + 0.5, -k) as binary16 R, G, B.
      0x00, 0x00, 0x00, 0x38, 0x00, 0x80,  // 0: 0, 0.5, -0
      0x00, 0x3C, 0x00, 0x3E, 0x00, 0xBC,  // 1: 1, 1.5, -1
      0x00, 0x40, 0x00, 0x41, 0x00, 0xC0,  // 2: 2, 2.5, -2
      0x00, 0x42, 0x00, 0x43, 0x00, 0xC2,  // 3: 3, 3.5, -3
      0x00, 0x44, 0x80, 0x44, 0x00, 0xC4,  // 4: 4, 4.5, -4
      0x00, 0x45, 0x80, 0x45, 0x00, 0xC5,  // 5: 5, 5.5, -5
      0x00, 0x46, 0x80, 0x46, 0x00, 0xC6,  // 6: 6, 6.5, -6
      0x00, 0x47, 0x80, 0x47, 0x00, 0xC7,  // 7: 7, 7.5, -7
      0x00, 0x48, 0x40, 0x48, 0x00, 0xC8,  // 8: 8, 8.5, -8
      // Visibility plane: 0.25.
      0x00, 0x34,
      // Hull range: first index 0, count 3.
      0, 0, 0, 0, 3, 0, 0, 0,
      // Vertices (0,0,0), (1,0,0), (0,1,0) as float32; 1.0f is 0x3F800000.
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,        //
      0, 0, 0x80, 0x3F, 0, 0, 0, 0, 0, 0, 0, 0,  //
      0, 0, 0, 0, 0, 0, 0x80, 0x3F, 0, 0, 0, 0,  //
      // Indices 0, 1, 2.
      0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0};
  ASSERT_EQ(expected.size(), 24u + 64u + 8u + 36u + 12u);
  ASSERT_EQ(expected.size(), 144u);

  const fs::path path = dir_ / "golden.shsf";
  ASSERT_TRUE(SaveSHSideFile(GoldenFile(), path));
  EXPECT_EQ(ReadBytes(path), expected);
}

// Writes one 1x1 page with `values` in coefficient 1 (any sign) and reads
// them back.
std::vector<float> RoundTripCoefficients(const fs::path& path,
                                         const std::vector<float>& values) {
  SHSideFile file;
  for (float v : values) {
    SHSidePage page = MakePage(1, 1, 0.5f);
    page.sh[0].coeffs[1] = Eigen::Vector3f(v, v, v);
    file.pages.push_back(page);
  }
  EXPECT_TRUE(SaveSHSideFile(file, path));
  std::optional<SHSideFile> read = LoadSHSideFile(path);
  std::vector<float> out;
  if (!read.has_value()) return out;
  for (const SHSidePage& page : read->pages) {
    out.push_back(page.sh[0].coeffs[1].x());
  }
  return out;
}

TEST_F(SideFileTest, RepresentableValuesRoundTripExactly) {
  const float smallest_subnormal = std::ldexp(1.0f, -24);
  const std::vector<float> values = {0.0f,    -0.5f,    1.0f,
                                     2048.0f, 65504.0f, smallest_subnormal};
  const std::vector<float> read =
      RoundTripCoefficients(dir_ / "exact.shsf", values);
  ASSERT_EQ(read.size(), values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    EXPECT_EQ(Bits(read[i]), Bits(values[i])) << values[i];
  }
  // The smallest subnormal is the bit pattern 0x0001 on disk.
  SHSideFile file = GoldenFile();
  file.pages[0].sh[0].coeffs[1].x() = smallest_subnormal;
  ASSERT_TRUE(SaveSHSideFile(file, dir_ / "subnormal.shsf"));
  const std::vector<uint8_t> bytes = ReadBytes(dir_ / "subnormal.shsf");
  EXPECT_EQ(bytes[kGoldenCoeffs + 6], 0x01);
  EXPECT_EQ(bytes[kGoldenCoeffs + 7], 0x00);
}

TEST_F(SideFileTest, RoundsToNearestEvenAndClamps) {
  const float tie = 1.0f + std::ldexp(1.0f, -11);
  const float above = 1.0f + 3.0f * std::ldexp(1.0f, -11);
  const std::vector<float> read =
      RoundTripCoefficients(dir_ / "round.shsf", {tie, above, 1e6f, -1e6f});
  ASSERT_EQ(read.size(), 4u);
  EXPECT_EQ(read[0], 1.0f);
  EXPECT_EQ(read[1], 1.0f + std::ldexp(1.0f, -9));
  EXPECT_EQ(read[2], 65504.0f);
  EXPECT_EQ(read[3], -65504.0f);
}

TEST_F(SideFileTest, WindingKept) {
  SHSideFile file;
  file.hull_mesh.positions = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
  file.hull_mesh.indices = {0, 1, 2};
  file.hull_mesh.hulls = {{0, 3}};
  ASSERT_TRUE(SaveSHSideFile(file, dir_ / "w.shsf"));
  std::optional<SHSideFile> read = LoadSHSideFile(dir_ / "w.shsf");
  ASSERT_TRUE(read.has_value());
  EXPECT_EQ(read->hull_mesh.indices, (std::vector<uint32_t>{0, 1, 2}));
}

TEST_F(SideFileDeathTest, WrongLuxelCount) {
  SHSideFile file;
  file.pages.push_back(MakePage(1, 1, 0.1f));
  file.pages.push_back(MakePage(4, 2, 0.1f));
  file.pages[1].sh.resize(7);
  EXPECT_DEATH(SaveSHSideFile(file, dir_ / "x"),
               "pages\\[1\\]\\.sh: 7 entries, expected 8");
}

TEST_F(SideFileDeathTest, WrongVisibilityCount) {
  SHSideFile file;
  file.pages.push_back(MakePage(4, 2, 0.1f));
  file.pages[0].visibility.resize(9);
  EXPECT_DEATH(SaveSHSideFile(file, dir_ / "x"),
               "pages\\[0\\]\\.visibility: 9 entries, expected 8");
}

TEST_F(SideFileDeathTest, ZeroWidth) {
  SHSideFile file;
  SHSidePage page;
  page.width = 0;
  page.height = 2;
  file.pages.push_back(page);
  EXPECT_DEATH(SaveSHSideFile(file, dir_ / "x"), "pages\\[0\\]\\.width: 0");
}

TEST_F(SideFileDeathTest, NonFiniteCoefficient) {
  SHSideFile file;
  file.pages.push_back(MakePage(2, 1, 0.1f));
  file.pages[0].sh[1].coeffs[2][1] = std::numeric_limits<float>::quiet_NaN();
  EXPECT_DEATH(SaveSHSideFile(file, dir_ / "x"),
               "pages\\[0\\]\\.sh\\[1\\]\\.coeffs\\[2\\]\\[1\\]: non-finite");
}

TEST_F(SideFileDeathTest, NonFiniteVisibility) {
  SHSideFile file;
  file.pages.push_back(MakePage(2, 1, 0.1f));
  file.pages[0].visibility[1] = std::numeric_limits<float>::infinity();
  EXPECT_DEATH(SaveSHSideFile(file, dir_ / "x"),
               "pages\\[0\\]\\.visibility\\[1\\]: non-finite");
}

TEST_F(SideFileDeathTest, NonFinitePosition) {
  SHSideFile file = GoldenFile();
  file.hull_mesh.positions[1].y() = std::numeric_limits<float>::infinity();
  EXPECT_DEATH(SaveSHSideFile(file, dir_ / "x"),
               "hull_mesh\\.positions\\[1\\]: non-finite position");
}

TEST_F(SideFileDeathTest, NotBakedMarker) {
  SHSideFile file;
  file.pages.push_back(MakePage(2, 2, 0.1f));
  file.pages[0].sh[3] = SHCoeffs(-1.0f);
  file.pages[0].visibility[3] = -1.0f;
  EXPECT_DEATH(SaveSHSideFile(file, dir_ / "x"),
               "pages\\[0\\]\\.visibility\\[3\\]: -1 outside \\[0, 1\\]");
}

TEST_F(SideFileDeathTest, VisibilityAboveOne) {
  SHSideFile file;
  file.pages.push_back(MakePage(1, 1, 0.1f));
  file.pages[0].visibility[0] = 1.5f;
  EXPECT_DEATH(SaveSHSideFile(file, dir_ / "x"),
               "pages\\[0\\]\\.visibility\\[0\\]: 1.5 outside \\[0, 1\\]");
}

TEST_F(SideFileDeathTest, NegativeL0) {
  SHSideFile file;
  file.pages.push_back(MakePage(1, 1, 0.1f));
  file.pages[0].sh[0].coeffs[0][2] = -0.25f;
  EXPECT_DEATH(SaveSHSideFile(file, dir_ / "x"),
               "pages\\[0\\]\\.sh\\[0\\]\\.coeffs\\[0\\]\\[2\\]: negative L0 "
               "component -0.25");
}

TEST_F(SideFileDeathTest, HullCountNotMultipleOfThree) {
  SHSideFile file;
  file.hull_mesh.positions = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
  file.hull_mesh.indices = {0, 1, 2, 0};
  file.hull_mesh.hulls = {{0, 4}};
  EXPECT_DEATH(SaveSHSideFile(file, dir_ / "x"),
               "hull_mesh\\.hulls\\[0\\]\\.index_count: 4 is not a positive "
               "multiple of 3");
}

TEST_F(SideFileDeathTest, ZeroCountHull) {
  SHSideFile file = GoldenFile();
  file.hull_mesh.hulls = {{0, 0}, {0, 3}};
  EXPECT_DEATH(SaveSHSideFile(file, dir_ / "x"),
               "hull_mesh\\.hulls\\[0\\]\\.index_count: 0 is not a positive");
}

TEST_F(SideFileDeathTest, GapBetweenHulls) {
  SHSideFile file;
  file.hull_mesh.positions = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
  file.hull_mesh.indices = {0, 1, 2, 0, 1, 2, 0, 1, 2};
  file.hull_mesh.hulls = {{0, 3}, {6, 3}};
  EXPECT_DEATH(SaveSHSideFile(file, dir_ / "x"),
               "hull_mesh\\.hulls\\[1\\]\\.first_index: 6, expected 3");
}

TEST_F(SideFileDeathTest, OverlappingHulls) {
  SHSideFile file;
  file.hull_mesh.positions = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
  file.hull_mesh.indices = {0, 1, 2, 0, 1, 2};
  file.hull_mesh.hulls = {{0, 3}, {0, 3}};
  EXPECT_DEATH(SaveSHSideFile(file, dir_ / "x"),
               "hull_mesh\\.hulls\\[1\\]\\.first_index: 0, expected 3");
}

TEST_F(SideFileDeathTest, RangesStopShort) {
  SHSideFile file;
  file.hull_mesh.positions = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
  file.hull_mesh.indices = {0, 1, 2, 0, 1, 2};
  file.hull_mesh.hulls = {{0, 3}};
  EXPECT_DEATH(SaveSHSideFile(file, dir_ / "x"),
               "hull_mesh\\.hulls: ranges cover 3 of 6 indices");
}

TEST_F(SideFileDeathTest, IndexOutOfRange) {
  SHSideFile file = GoldenFile();
  file.hull_mesh.indices[2] = 3;
  EXPECT_DEATH(SaveSHSideFile(file, dir_ / "x"),
               "hull_mesh\\.indices\\[2\\]: index 3 not below vertex count 3");
}

TEST_F(SideFileTest, MissingDirectoryReturnsFalse) {
  const fs::path path = dir_ / "no_such_dir" / "town.shsf";
  ErrorCapture errors;
  EXPECT_FALSE(SaveSHSideFile(GoldenFile(), path));
  EXPECT_TRUE(errors.Contains("town.shsf")) << errors.All();
  EXPECT_FALSE(fs::exists(dir_ / "no_such_dir"));
  EXPECT_TRUE(fs::is_empty(dir_));
}

TEST_F(SideFileTest, RenameFailureCleansUp) {
  const fs::path target = dir_ / "town.shsf";
  fs::create_directory(target);
  WriteBytes(target / "keep.txt", {'k'});
  ErrorCapture errors;
  EXPECT_FALSE(SaveSHSideFile(GoldenFile(), target));
  EXPECT_TRUE(errors.Contains("rename")) << errors.All();
  EXPECT_TRUE(fs::is_directory(target));
  EXPECT_EQ(ReadBytes(target / "keep.txt"), std::vector<uint8_t>{'k'});
  EXPECT_FALSE(fs::exists(dir_ / "town.shsf.tmp"));
  int entries = 0;
  for (const auto& e : fs::directory_iterator(dir_)) {
    (void)e;
    ++entries;
  }
  EXPECT_EQ(entries, 1);
}

TEST_F(SideFileTest, OverwriteReplacesBytes) {
  const fs::path path = dir_ / "town.shsf";
  SHSideFile first;
  first.pages.push_back(MakePage(3, 3, 0.2f));
  ASSERT_TRUE(SaveSHSideFile(first, path));
  const std::vector<uint8_t> first_bytes = ReadBytes(path);

  ASSERT_TRUE(SaveSHSideFile(GoldenFile(), path));
  const fs::path golden_path = dir_ / "golden_copy.shsf";
  ASSERT_TRUE(SaveSHSideFile(GoldenFile(), golden_path));
  EXPECT_NE(ReadBytes(path), first_bytes);
  EXPECT_EQ(ReadBytes(path), ReadBytes(golden_path));
  EXPECT_FALSE(fs::exists(dir_ / "town.shsf.tmp"));
}

// --- Reader ---------------------------------------------------------------

TEST_F(SideFileTest, RoundTripKeepsOrderAndSizes) {
  SHSideFile file;
  file.pages.push_back(MakePage(4, 2, 0.3f));
  file.pages.push_back(MakePage(1, 1, 7.0f));
  file.pages.push_back(MakePage(3, 5, 1.25f));
  file.hull_mesh = TwoHullMesh();
  const fs::path path = dir_ / "town.shsf";
  ASSERT_TRUE(SaveSHSideFile(file, path));
  EXPECT_EQ(fs::file_size(path), 24u + (8 + 56 * 8) + (8 + 56 * 1) +
                                     (8 + 56 * 15) + 8 * 2 + 12 * 5 + 4 * 9);
  std::optional<SHSideFile> read = LoadSHSideFile(path);
  ASSERT_TRUE(read.has_value());
  ASSERT_EQ(read->pages.size(), 3u);
  EXPECT_EQ(read->pages[0].width, 4);
  EXPECT_EQ(read->pages[0].height, 2);
  EXPECT_EQ(read->pages[1].width, 1);
  EXPECT_EQ(read->pages[1].height, 1);
  EXPECT_EQ(read->pages[2].width, 3);
  EXPECT_EQ(read->pages[2].height, 5);
  ExpectSameValues(file, *read);
}

TEST_F(SideFileTest, EmptyPartsAllowed) {
  SHSideFile hulls_only;
  hulls_only.hull_mesh = TwoHullMesh();
  ASSERT_TRUE(SaveSHSideFile(hulls_only, dir_ / "a.shsf"));
  std::optional<SHSideFile> a = LoadSHSideFile(dir_ / "a.shsf");
  ASSERT_TRUE(a.has_value());
  ExpectSameValues(hulls_only, *a);

  SHSideFile pages_only;
  pages_only.pages.push_back(MakePage(2, 3, 0.4f));
  ASSERT_TRUE(SaveSHSideFile(pages_only, dir_ / "b.shsf"));
  std::optional<SHSideFile> b = LoadSHSideFile(dir_ / "b.shsf");
  ASSERT_TRUE(b.has_value());
  ExpectSameValues(pages_only, *b);

  SHSideFile empty;
  ASSERT_TRUE(SaveSHSideFile(empty, dir_ / "c.shsf"));
  EXPECT_EQ(fs::file_size(dir_ / "c.shsf"), 24u);
  std::optional<SHSideFile> c = LoadSHSideFile(dir_ / "c.shsf");
  ASSERT_TRUE(c.has_value());
  EXPECT_TRUE(c->pages.empty());
  EXPECT_TRUE(c->hull_mesh.positions.empty());
}

TEST_F(SideFileTest, IrradianceReadsBackAsZero) {
  SHSideFile file;
  file.pages.push_back(MakePage(2, 1, 0.5f));
  file.pages[0].sh[0].irradiance = Eigen::Vector3f(3, 4, 5);
  file.pages[0].sh[1].irradiance = Eigen::Vector3f(1, 1, 1);
  ASSERT_TRUE(SaveSHSideFile(file, dir_ / "i.shsf"));
  std::optional<SHSideFile> read = LoadSHSideFile(dir_ / "i.shsf");
  ASSERT_TRUE(read.has_value());
  for (const SHCoeffs& sh : read->pages[0].sh) {
    EXPECT_EQ(sh.irradiance, Eigen::Vector3f::Zero());
  }
}

class SideFileRejectTest : public SideFileTest {
 protected:
  // Writes `file`, applies `mutate` to its bytes, and expects the read to
  // fail with a logged message containing `reason`.
  void ExpectRejected(const SHSideFile& file,
                      void (*mutate)(std::vector<uint8_t>*),
                      const std::string& reason) {
    const fs::path path = dir_ / "bad.shsf";
    ASSERT_TRUE(SaveSHSideFile(file, path));
    std::vector<uint8_t> bytes = ReadBytes(path);
    mutate(&bytes);
    WriteBytes(path, bytes);
    ExpectBytesRejected(bytes, reason);
  }

  void ExpectBytesRejected(const std::vector<uint8_t>& bytes,
                           const std::string& reason) {
    const fs::path path = dir_ / "bad.shsf";
    WriteBytes(path, bytes);
    ErrorCapture errors;
    EXPECT_FALSE(LoadSHSideFile(path).has_value());
    EXPECT_TRUE(errors.Contains(reason))
        << "expected \"" << reason << "\" in:\n"
        << errors.All();
  }
};

TEST_F(SideFileRejectTest, MissingFile) {
  ErrorCapture errors;
  EXPECT_FALSE(LoadSHSideFile(dir_ / "nothing.shsf").has_value());
  EXPECT_TRUE(errors.Contains("cannot open")) << errors.All();
}

TEST_F(SideFileRejectTest, BadMagic) {
  ExpectRejected(
      GoldenFile(), [](std::vector<uint8_t>* b) { (*b)[0] = 'X'; },
      "bad magic");
}

TEST_F(SideFileRejectTest, UnknownVersion) {
  ExpectRejected(
      GoldenFile(), [](std::vector<uint8_t>* b) { PutU32At(b, 4, 2); },
      "unknown version 2");
}

TEST_F(SideFileRejectTest, TruncatedFile) {
  ExpectRejected(
      GoldenFile(), [](std::vector<uint8_t>* b) { b->pop_back(); },
      "shorter than expected");
}

TEST_F(SideFileRejectTest, TrailingBytes) {
  ExpectRejected(
      GoldenFile(), [](std::vector<uint8_t>* b) { b->push_back(0); },
      "longer than expected");
}

TEST_F(SideFileRejectTest, ZeroPageWidth) {
  ExpectRejected(
      GoldenFile(),
      [](std::vector<uint8_t>* b) { PutU32At(b, kGoldenWidth, 0); },
      "page 0 has size 0 x 1");
}

TEST_F(SideFileRejectTest, HugeCountsInSmallFile) {
  std::vector<uint8_t> bytes = {'S', 'H', 'S', 'F', 1, 0, 0, 0};
  bytes.resize(24, 0xFF);
  ExpectBytesRejected(bytes, "shorter than expected");
}

TEST_F(SideFileRejectTest, HugePageInSmallFile) {
  std::vector<uint8_t> bytes(32, 0);
  std::memcpy(bytes.data(), "SHSF", 4);
  PutU32At(&bytes, 4, 1);
  PutU32At(&bytes, 8, 1);
  PutU32At(&bytes, 24, 0xFFFFFFFF);
  PutU32At(&bytes, 28, 0xFFFFFFFF);
  ExpectBytesRejected(bytes, "page 0 has size 4294967295 x 4294967295");
  // The largest sizes a page may have still fail the size walk, not an
  // allocation.
  PutU32At(&bytes, 24, 0x7FFFFFFF);
  PutU32At(&bytes, 28, 0x7FFFFFFF);
  ExpectBytesRejected(bytes, "does not fit in 32 bytes");
}

TEST_F(SideFileRejectTest, BadHullIndex) {
  ExpectRejected(
      GoldenFile(),
      [](std::vector<uint8_t>* b) { PutU32At(b, kGoldenIndices + 8, 3); },
      "hull index 3 at 2 not below vertex count 3");
}

TEST_F(SideFileRejectTest, HullRangeGap) {
  SHSideFile file;
  file.hull_mesh = TwoHullMesh();
  // Hull 1's range is the second (first, count) pair after the 24-byte
  // header; move its first index from 6 to 7.
  ExpectRejected(
      file, [](std::vector<uint8_t>* b) { PutU32At(b, 24 + 8, 7); },
      "hull 1 starts at index 7, expected 6");
}

TEST_F(SideFileRejectTest, InfinityInCoefficient) {
  ExpectRejected(
      GoldenFile(),
      [](std::vector<uint8_t>* b) {
        PutU16At(b, kGoldenCoeffs + 6 * 3 + 2, 0x7C00);
      },
      "non-finite binary16 0x7c00 at page 0 luxel 0 coefficient 3 channel 1");
}

TEST_F(SideFileRejectTest, NanInCoefficient) {
  ExpectRejected(
      GoldenFile(),
      [](std::vector<uint8_t>* b) { PutU16At(b, kGoldenCoeffs + 6, 0x7E00); },
      "non-finite binary16 0x7e00 at page 0 luxel 0 coefficient 1 channel 0");
}

TEST_F(SideFileRejectTest, VisibilityAboveOne) {
  ExpectRejected(
      GoldenFile(),
      [](std::vector<uint8_t>* b) { PutU16At(b, kGoldenVisibility, 0x3E00); },
      "visibility 0x3e00 (1.500000) outside [0, 1] at page 0 luxel 0");
}

TEST_F(SideFileRejectTest, NegativeL0) {
  ExpectRejected(
      GoldenFile(),
      // L0 green, 0.5, becomes -0.5.
      [](std::vector<uint8_t>* b) { PutU16At(b, kGoldenCoeffs + 2, 0xB800); },
      "negative L0 component -0.500000 at page 0 luxel 0 coefficient 0 "
      "channel 1");
}

TEST_F(SideFileRejectTest, NanHullPosition) {
  ExpectRejected(
      GoldenFile(),
      [](std::vector<uint8_t>* b) {
        PutU32At(b, kGoldenVertices + 12 + 4, 0x7FC00000);
      },
      "non-finite hull position (1.000000, nan, 0.000000) at vertex 1");
}

TEST_F(SideFileRejectTest, NegativeVisibility) {
  ExpectRejected(
      GoldenFile(),
      [](std::vector<uint8_t>* b) { PutU16At(b, kGoldenVisibility, 0xBC00); },
      "visibility 0xbc00 (-1.000000) outside [0, 1] at page 0 luxel 0");
}

TEST_F(SideFileRejectTest, ZeroCountHull) {
  // Hull 0 (first 0, count 6) becomes count 0, so hull 1 starts too late; the
  // count is checked first.
  SHSideFile file;
  file.hull_mesh = TwoHullMesh();
  ExpectRejected(
      file, [](std::vector<uint8_t>* b) { PutU32At(b, 24 + 4, 0); },
      "hull 0 has index count 0, not a positive multiple of 3");
}

TEST_F(SideFileRejectTest, HullCountNotMultipleOfThree) {
  SHSideFile file;
  file.hull_mesh = TwoHullMesh();
  ExpectRejected(
      file, [](std::vector<uint8_t>* b) { PutU32At(b, 24 + 12, 2); },
      "hull 1 has index count 2, not a positive multiple of 3");
}

TEST_F(SideFileRejectTest, RangesStopShort) {
  // Two hulls over 9 indices; drop hull 1 from the header and its range from
  // the file, leaving one 6-index hull over 9 indices.
  SHSideFile file;
  file.hull_mesh = TwoHullMesh();
  ExpectRejected(
      file,
      [](std::vector<uint8_t>* b) {
        PutU32At(b, 12, 1);
        b->erase(b->begin() + 32, b->begin() + 40);
      },
      "hull ranges cover 6 of 9 indices");
}

// --- L0 dump --------------------------------------------------------------

SHSidePage L0Page() {
  SHSidePage page = MakePage(4, 2, 0.0f);
  for (int y = 0; y < 2; ++y) {
    for (int x = 0; x < 4; ++x) {
      page.sh[y * 4 + x].coeffs[0] = Eigen::Vector3f(x, y, 1);
    }
  }
  return page;
}

TEST_F(SideFileTest, DumpMatchesL0) {
  const fs::path path = dir_ / "l0_page0.hdr";
  ASSERT_TRUE(SaveSHSidePageL0Hdr(L0Page(), path));
  int w = 0, h = 0, c = 0;
  float* data = stbi_loadf(path.c_str(), &w, &h, &c, 3);
  ASSERT_NE(data, nullptr);
  EXPECT_EQ(w, 4);
  EXPECT_EQ(h, 2);
  for (int y = 0; y < 2; ++y) {
    for (int x = 0; x < 4; ++x) {
      const Eigen::Vector3f expected(x, y, 1);
      const float tolerance = expected.maxCoeff() / 128.0f;
      for (int ch = 0; ch < 3; ++ch) {
        EXPECT_NEAR(data[(y * 4 + x) * 3 + ch], expected[ch], tolerance)
            << "pixel (" << x << ", " << y << ") channel " << ch;
      }
    }
  }
  stbi_image_free(data);
}

TEST_F(SideFileTest, DumpWritesNegativeL0AsZero) {
  SHSidePage page = MakePage(1, 1, 0.0f);
  page.sh[0].coeffs[0] = Eigen::Vector3f(-2.0f, 0.5f, 0.25f);
  const fs::path path = dir_ / "neg.hdr";
  ASSERT_TRUE(SaveSHSidePageL0Hdr(page, path));
  int w = 0, h = 0, c = 0;
  float* data = stbi_loadf(path.c_str(), &w, &h, &c, 3);
  ASSERT_NE(data, nullptr);
  EXPECT_EQ(data[0], 0.0f);
  EXPECT_NEAR(data[1], 0.5f, 0.5f / 128);
  EXPECT_NEAR(data[2], 0.25f, 0.5f / 128);
  stbi_image_free(data);
}

TEST_F(SideFileTest, DumpRefusesRend2Names) {
  for (const char* name : {"LM_0000.hdr", "lm_x.hdr", "Lm_1.hdr"}) {
    ErrorCapture errors;
    EXPECT_FALSE(SaveSHSidePageL0Hdr(L0Page(), dir_ / name)) << name;
    EXPECT_FALSE(fs::exists(dir_ / name)) << name;
    EXPECT_TRUE(errors.Contains("lm_")) << errors.All();
  }
  EXPECT_TRUE(SaveSHSidePageL0Hdr(L0Page(), dir_ / "l0_lm_.hdr"));
  EXPECT_TRUE(fs::exists(dir_ / "l0_lm_.hdr"));
}

TEST_F(SideFileTest, DumpMissingDirectoryReturnsFalse) {
  ErrorCapture errors;
  EXPECT_FALSE(SaveSHSidePageL0Hdr(L0Page(), dir_ / "missing" / "l0.hdr"));
  EXPECT_TRUE(errors.Contains("cannot write")) << errors.All();
}

}  // namespace
}  // namespace sh_baker
