#include "side_file.h"

#include <glog/logging.h>
#include <stb_image_write.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <system_error>

namespace sh_baker {

namespace {

constexpr char kMagic[4] = {'S', 'H', 'S', 'F'};
constexpr uint32_t kVersion = 1;
constexpr size_t kHeaderBytes = 24;
constexpr size_t kPageHeaderBytes = 8;
// 9 RGB coefficient planes plus one visibility plane, 2 bytes per value.
constexpr uint64_t kBytesPerLuxel = (9 * 3 + 1) * 2;
constexpr float kHalfMax = 65504.0f;
constexpr uint64_t kMaxPageSide = std::numeric_limits<int32_t>::max();

// --- Encoding -------------------------------------------------------------

void PutU16(uint16_t v, std::vector<uint8_t>* out) {
  out->push_back(static_cast<uint8_t>(v & 0xFF));
  out->push_back(static_cast<uint8_t>(v >> 8));
}

void PutU32(uint32_t v, std::vector<uint8_t>* out) {
  for (int shift = 0; shift < 32; shift += 8) {
    out->push_back(static_cast<uint8_t>((v >> shift) & 0xFF));
  }
}

void PutF32(float v, std::vector<uint8_t>* out) {
  PutU32(Eigen::numext::bit_cast<uint32_t>(v), out);
}

// Rounds to nearest, ties to even; finite values beyond the half range clamp.
uint16_t FloatToHalfBits(float v) {
  if (v > kHalfMax) v = kHalfMax;
  if (v < -kHalfMax) v = -kHalfMax;
  return Eigen::numext::bit_cast<uint16_t>(Eigen::half(v));
}

float HalfBitsToFloat(uint16_t bits) {
  return static_cast<float>(Eigen::numext::bit_cast<Eigen::half>(bits));
}

bool HalfBitsFinite(uint16_t bits) { return (bits & 0x7C00) != 0x7C00; }

uint32_t CheckedU32(size_t count, const char* field) {
  CHECK_LE(count, std::numeric_limits<uint32_t>::max())
      << field << ": count " << count << " does not fit in a uint32";
  return static_cast<uint32_t>(count);
}

// --- Writer checks --------------------------------------------------------

void CheckPageShape(const SHSidePage& page, size_t p) {
  CHECK_GE(page.width, 1) << "pages[" << p << "].width: " << page.width;
  CHECK_GE(page.height, 1) << "pages[" << p << "].height: " << page.height;
  const size_t luxels =
      static_cast<size_t>(page.width) * static_cast<size_t>(page.height);
  CHECK_EQ(page.sh.size(), luxels)
      << "pages[" << p << "].sh: " << page.sh.size() << " entries, expected "
      << luxels << " (" << page.width << " x " << page.height << ")";
  CHECK_EQ(page.visibility.size(), luxels)
      << "pages[" << p << "].visibility: " << page.visibility.size()
      << " entries, expected " << luxels << " (" << page.width << " x "
      << page.height << ")";
}

void CheckPageValues(const SHSidePage& page, size_t p) {
  for (size_t j = 0; j < page.sh.size(); ++j) {
    // Visibility first, so the bake's not-baked marker is reported as such.
    const float vis = page.visibility[j];
    CHECK(std::isfinite(vis)) << "pages[" << p << "].visibility[" << j
                              << "]: non-finite value " << vis;
    CHECK(vis >= 0.0f && vis <= 1.0f)
        << "pages[" << p << "].visibility[" << j << "]: " << vis
        << " outside [0, 1] (a luxel the bake skipped must be filled)";
    for (int k = 0; k < 9; ++k) {
      for (int c = 0; c < 3; ++c) {
        const float v = page.sh[j].coeffs[k][c];
        CHECK(std::isfinite(v))
            << "pages[" << p << "].sh[" << j << "].coeffs[" << k << "][" << c
            << "]: non-finite value " << v;
      }
    }
    for (int c = 0; c < 3; ++c) {
      const float l0 = page.sh[j].coeffs[0][c];
      CHECK_GE(l0, 0.0f) << "pages[" << p << "].sh[" << j << "].coeffs[0][" << c
                         << "]: negative L0 component " << l0;
    }
  }
}

void CheckHullMesh(const SHHullMesh& mesh) {
  CheckedU32(mesh.positions.size(), "hull_mesh.positions");
  CheckedU32(mesh.indices.size(), "hull_mesh.indices");
  CheckedU32(mesh.hulls.size(), "hull_mesh.hulls");
  for (size_t v = 0; v < mesh.positions.size(); ++v) {
    const Eigen::Vector3f& pos = mesh.positions[v];
    CHECK(pos.allFinite()) << "hull_mesh.positions[" << v
                           << "]: non-finite position (" << pos.x() << ", "
                           << pos.y() << ", " << pos.z() << ")";
  }
  uint64_t expected_first = 0;
  for (size_t h = 0; h < mesh.hulls.size(); ++h) {
    const SHHullRange& range = mesh.hulls[h];
    CHECK(range.index_count > 0 && range.index_count % 3 == 0)
        << "hull_mesh.hulls[" << h << "].index_count: " << range.index_count
        << " is not a positive multiple of 3";
    CHECK_EQ(range.first_index, expected_first)
        << "hull_mesh.hulls[" << h << "].first_index: " << range.first_index
        << ", expected " << expected_first;
    expected_first += range.index_count;
  }
  CHECK_EQ(expected_first, mesh.indices.size())
      << "hull_mesh.hulls: ranges cover " << expected_first << " of "
      << mesh.indices.size() << " indices";
  for (size_t i = 0; i < mesh.indices.size(); ++i) {
    CHECK_LT(mesh.indices[i], mesh.positions.size())
        << "hull_mesh.indices[" << i << "]: index " << mesh.indices[i]
        << " not below vertex count " << mesh.positions.size();
  }
}

std::vector<uint8_t> Serialize(const SHSideFile& file) {
  const SHHullMesh& mesh = file.hull_mesh;
  uint64_t total = kHeaderBytes;
  for (const SHSidePage& page : file.pages) {
    total += kPageHeaderBytes + kBytesPerLuxel * page.sh.size();
  }
  total += 8 * mesh.hulls.size() + 12 * mesh.positions.size() +
           4 * mesh.indices.size();

  std::vector<uint8_t> out;
  out.reserve(total);
  out.insert(out.end(), kMagic, kMagic + 4);
  PutU32(kVersion, &out);
  PutU32(CheckedU32(file.pages.size(), "pages"), &out);
  PutU32(static_cast<uint32_t>(mesh.hulls.size()), &out);
  PutU32(static_cast<uint32_t>(mesh.positions.size()), &out);
  PutU32(static_cast<uint32_t>(mesh.indices.size()), &out);
  for (const SHSidePage& page : file.pages) {
    PutU32(static_cast<uint32_t>(page.width), &out);
    PutU32(static_cast<uint32_t>(page.height), &out);
    for (int k = 0; k < 9; ++k) {
      for (const SHCoeffs& sh : page.sh) {
        for (int c = 0; c < 3; ++c)
          PutU16(FloatToHalfBits(sh.coeffs[k][c]), &out);
      }
    }
    for (float vis : page.visibility) PutU16(FloatToHalfBits(vis), &out);
  }
  for (const SHHullRange& range : mesh.hulls) {
    PutU32(range.first_index, &out);
    PutU32(range.index_count, &out);
  }
  for (const Eigen::Vector3f& pos : mesh.positions) {
    PutF32(pos.x(), &out);
    PutF32(pos.y(), &out);
    PutF32(pos.z(), &out);
  }
  for (uint32_t index : mesh.indices) PutU32(index, &out);
  DCHECK_EQ(out.size(), total);
  return out;
}

void RemoveQuietly(const std::filesystem::path& path) {
  std::error_code ec;
  std::filesystem::remove(path, ec);
}

// --- Reader ---------------------------------------------------------------

class ByteReader {
 public:
  explicit ByteReader(const std::vector<uint8_t>& bytes) : bytes_(bytes) {}

  size_t pos() const { return pos_; }
  void Seek(size_t pos) { pos_ = pos; }

  // Callers have checked that the bytes exist.
  uint16_t U16() {
    uint16_t v = static_cast<uint16_t>(bytes_[pos_] | (bytes_[pos_ + 1] << 8));
    pos_ += 2;
    return v;
  }
  uint32_t U32() {
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i) {
      v |= static_cast<uint32_t>(bytes_[pos_ + i]) << (8 * i);
    }
    pos_ += 4;
    return v;
  }
  float F32() { return Eigen::numext::bit_cast<float>(U32()); }

 private:
  const std::vector<uint8_t>& bytes_;
  size_t pos_ = 0;
};

bool ReadWholeFile(const std::filesystem::path& path,
                   std::vector<uint8_t>* bytes) {
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (f == nullptr) {
    LOG(ERROR) << "LoadSHSideFile: cannot open " << path << ": "
               << std::strerror(errno);
    return false;
  }
  bool ok = std::fseek(f, 0, SEEK_END) == 0;
  long size = ok ? std::ftell(f) : -1;
  ok = ok && size >= 0 && std::fseek(f, 0, SEEK_SET) == 0;
  if (ok) {
    bytes->resize(static_cast<size_t>(size));
    ok = bytes->empty() ||
         std::fread(bytes->data(), 1, bytes->size(), f) == bytes->size();
  }
  std::fclose(f);
  if (!ok) {
    LOG(ERROR) << "LoadSHSideFile: cannot read " << path << ": "
               << std::strerror(errno);
  }
  return ok;
}

}  // namespace

bool SaveSHSideFile(const SHSideFile& file, const std::filesystem::path& path) {
  CheckedU32(file.pages.size(), "pages");
  for (size_t p = 0; p < file.pages.size(); ++p) {
    CheckPageShape(file.pages[p], p);
    CheckPageValues(file.pages[p], p);
  }
  CheckHullMesh(file.hull_mesh);

  const std::vector<uint8_t> bytes = Serialize(file);
  std::filesystem::path tmp = path;
  tmp += ".tmp";

  std::FILE* f = std::fopen(tmp.c_str(), "wb");
  if (f == nullptr) {
    LOG(ERROR) << "SaveSHSideFile: cannot create " << tmp << " for " << path
               << ": " << std::strerror(errno);
    return false;
  }
  const bool written =
      std::fwrite(bytes.data(), 1, bytes.size(), f) == bytes.size();
  const bool closed = std::fclose(f) == 0;
  if (!written || !closed) {
    LOG(ERROR) << "SaveSHSideFile: cannot write " << tmp << " for " << path
               << ": " << std::strerror(errno);
    RemoveQuietly(tmp);
    return false;
  }
  std::error_code ec;
  std::filesystem::rename(tmp, path, ec);
  if (ec) {
    LOG(ERROR) << "SaveSHSideFile: cannot rename " << tmp << " to " << path
               << ": " << ec.message();
    RemoveQuietly(tmp);
    return false;
  }
  return true;
}

std::optional<SHSideFile> LoadSHSideFile(const std::filesystem::path& path) {
  std::vector<uint8_t> bytes;
  if (!ReadWholeFile(path, &bytes)) return std::nullopt;
  const uint64_t size = bytes.size();
  auto fail = [&path](const std::string& reason) {
    LOG(ERROR) << "LoadSHSideFile: " << path << ": " << reason;
    return std::nullopt;
  };
  auto shorter = [&](uint64_t expected) {
    return fail("file is shorter than expected: needs at least " +
                std::to_string(expected) + " bytes, has " +
                std::to_string(size));
  };

  if (size < kHeaderBytes) return shorter(kHeaderBytes);
  if (std::memcmp(bytes.data(), kMagic, 4) != 0) return fail("bad magic");
  ByteReader in(bytes);
  in.Seek(4);
  const uint32_t version = in.U32();
  if (version != kVersion) {
    return fail("unknown version " + std::to_string(version) +
                " (only version 1 is supported)");
  }
  const uint64_t page_count = in.U32();
  const uint64_t hull_count = in.U32();
  const uint64_t vertex_count = in.U32();
  const uint64_t index_count = in.U32();

  // Sizes first, without allocating: every count is at most 2^32 - 1, so
  // none of these sums can overflow 64 bits.
  const uint64_t tail = 8 * hull_count + 12 * vertex_count + 4 * index_count;
  const uint64_t minimum = kHeaderBytes + kPageHeaderBytes * page_count + tail;
  if (minimum > size) return shorter(minimum);
  const uint64_t pages_end = size - tail;
  uint64_t pos = kHeaderBytes;
  for (uint64_t p = 0; p < page_count; ++p) {
    if (pages_end - pos < kPageHeaderBytes) {
      return shorter(pos + kPageHeaderBytes + tail);
    }
    in.Seek(pos);
    const uint64_t width = in.U32();
    const uint64_t height = in.U32();
    if (width == 0 || height == 0 || width > kMaxPageSide ||
        height > kMaxPageSide) {
      return fail("page " + std::to_string(p) + " has size " +
                  std::to_string(width) + " x " + std::to_string(height));
    }
    pos += kPageHeaderBytes;
    // width * height < 2^62; compare before multiplying by the luxel size.
    const uint64_t luxels = width * height;
    if (luxels > (pages_end - pos) / kBytesPerLuxel) {
      return fail("file is shorter than expected: page " + std::to_string(p) +
                  " (" + std::to_string(width) + " x " +
                  std::to_string(height) + ") does not fit in " +
                  std::to_string(size) + " bytes");
    }
    pos += luxels * kBytesPerLuxel;
  }
  if (pos != pages_end) {
    return fail("file is longer than expected: " + std::to_string(pos + tail) +
                " bytes implied, has " + std::to_string(size));
  }

  // The sizes match the file; decode.
  SHSideFile file;
  file.pages.resize(page_count);
  in.Seek(kHeaderBytes);
  for (uint64_t p = 0; p < page_count; ++p) {
    SHSidePage& page = file.pages[p];
    page.width = static_cast<int>(in.U32());
    page.height = static_cast<int>(in.U32());
    const size_t luxels =
        static_cast<size_t>(page.width) * static_cast<size_t>(page.height);
    page.sh.resize(luxels);
    page.visibility.resize(luxels);
    for (int k = 0; k < 9; ++k) {
      for (size_t j = 0; j < luxels; ++j) {
        for (int c = 0; c < 3; ++c) {
          const uint16_t bits = in.U16();
          const float v = HalfBitsToFloat(bits);
          // Built only on failure: this loop runs for every value.
          auto where = [&] {
            return "page " + std::to_string(p) + " luxel " + std::to_string(j) +
                   " coefficient " + std::to_string(k) + " channel " +
                   std::to_string(c);
          };
          if (!HalfBitsFinite(bits)) {
            char hex[8];
            std::snprintf(hex, sizeof(hex), "0x%04x", bits);
            return fail("non-finite binary16 " + std::string(hex) + " at " +
                        where());
          }
          if (k == 0 && v < 0.0f) {
            return fail("negative L0 component " + std::to_string(v) + " at " +
                        where());
          }
          page.sh[j].coeffs[k][c] = v;
        }
      }
    }
    for (size_t j = 0; j < luxels; ++j) {
      const uint16_t bits = in.U16();
      const float v = HalfBitsToFloat(bits);
      if (!HalfBitsFinite(bits) || v < 0.0f || v > 1.0f) {
        char hex[8];
        std::snprintf(hex, sizeof(hex), "0x%04x", bits);
        return fail("visibility " + std::string(hex) + " (" +
                    std::to_string(v) + ") outside [0, 1] at page " +
                    std::to_string(p) + " luxel " + std::to_string(j));
      }
      page.visibility[j] = v;
    }
  }

  SHHullMesh& mesh = file.hull_mesh;
  mesh.hulls.resize(hull_count);
  uint64_t expected_first = 0;
  for (uint64_t h = 0; h < hull_count; ++h) {
    SHHullRange& range = mesh.hulls[h];
    range.first_index = in.U32();
    range.index_count = in.U32();
    if (range.first_index != expected_first) {
      return fail("hull " + std::to_string(h) + " starts at index " +
                  std::to_string(range.first_index) + ", expected " +
                  std::to_string(expected_first));
    }
    if (range.index_count == 0 || range.index_count % 3 != 0) {
      return fail("hull " + std::to_string(h) + " has index count " +
                  std::to_string(range.index_count) +
                  ", not a positive multiple of 3");
    }
    expected_first += range.index_count;
  }
  if (expected_first != index_count) {
    return fail("hull ranges cover " + std::to_string(expected_first) + " of " +
                std::to_string(index_count) + " indices");
  }
  mesh.positions.resize(vertex_count);
  for (uint64_t v = 0; v < vertex_count; ++v) {
    Eigen::Vector3f& pos3 = mesh.positions[v];
    pos3.x() = in.F32();
    pos3.y() = in.F32();
    pos3.z() = in.F32();
    if (!pos3.allFinite()) {
      return fail("non-finite hull position (" + std::to_string(pos3.x()) +
                  ", " + std::to_string(pos3.y()) + ", " +
                  std::to_string(pos3.z()) + ") at vertex " +
                  std::to_string(v));
    }
  }
  mesh.indices.resize(index_count);
  for (uint64_t i = 0; i < index_count; ++i) {
    const uint32_t index = in.U32();
    if (index >= vertex_count) {
      return fail("hull index " + std::to_string(index) + " at " +
                  std::to_string(i) + " not below vertex count " +
                  std::to_string(vertex_count));
    }
    mesh.indices[i] = index;
  }
  DCHECK_EQ(in.pos(), size);
  return file;
}

bool SaveSHSidePageL0Hdr(const SHSidePage& page,
                         const std::filesystem::path& path) {
  CheckPageShape(page, 0);
  std::string name = path.filename().string();
  if (name.size() >= 3 &&
      std::tolower(static_cast<unsigned char>(name[0])) == 'l' &&
      std::tolower(static_cast<unsigned char>(name[1])) == 'm' &&
      name[2] == '_') {
    LOG(ERROR) << "SaveSHSidePageL0Hdr: refusing " << path
               << ": rend2 auto-loads lm_*.hdr files";
    return false;
  }
  std::vector<float> rgb(page.sh.size() * 3);
  for (size_t j = 0; j < page.sh.size(); ++j) {
    for (int c = 0; c < 3; ++c) {
      rgb[j * 3 + c] = std::max(0.0f, page.sh[j].coeffs[0][c]);
    }
  }
  if (stbi_write_hdr(path.c_str(), page.width, page.height, 3, rgb.data()) ==
      0) {
    LOG(ERROR) << "SaveSHSidePageL0Hdr: cannot write " << path;
    return false;
  }
  return true;
}

}  // namespace sh_baker
