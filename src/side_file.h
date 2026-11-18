#ifndef SH_BAKER_SRC_SIDE_FILE_H_
#define SH_BAKER_SRC_SIDE_FILE_H_

#include <Eigen/Dense>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <vector>

#include "sh_coeffs.h"

namespace sh_baker {

// The SH side file: a level's baked SH lightmap pages and its solid hull mesh,
// written by q3map2 through this header and read by renderer_sh.
//
// Byte layout (version 1). Little-endian, no padding beyond what is listed,
// and the file ends exactly after the hull indices:
//
//   header, 24 bytes:
//     char[4]  magic 'S','H','S','F'
//     uint32   version (1)
//     uint32   page count
//     uint32   hull count
//     uint32   hull vertex count
//     uint32   hull index count
//   for each page, in order (8 + 56 * width * height bytes):
//     uint32   width   (>= 1)
//     uint32   height  (>= 1)
//     9 coefficient planes, in the order L0, L1m1, L10, L11, L2m2, L2m1, L20,
//       L21, L22 (real SH, the basis of sh_coeffs.cpp); each holds width *
//       height luxels of three binary16 values (R, G, B)
//     1 visibility plane of width * height binary16 values
//     Within a plane, luxels run x fastest, then y, starting at row 0.
//   hull ranges: per hull, uint32 first index, uint32 index count
//   hull vertices: per vertex, float32 x, y, z
//   hull indices: one uint32 per index
//
// Every section starts on a 4-byte boundary.
//
// The basis, for a unit direction (x, y, z) in the caller's world frame,
// pointing from the luxel towards the incoming light, with no Condon-Shortley
// phase (sh_coeffs.cpp):
//   L0   0.282095            L2m2 1.092548 x y
//   L1m1 0.488603 y          L2m1 1.092548 y z
//   L10  0.488603 z          L20  0.315392 (3 z^2 - 1)
//   L11  0.488603 x          L21  1.092548 x z
//                            L22  0.546274 (x^2 - y^2)
// Each coefficient is the projection of the incident radiance onto its basis
// function, the integral of L_i(w) Y(w) dw over the hemisphere above the
// surface; it is not convolved with the cosine lobe.
//
// Version rule: the writer always writes version 1, and the reader accepts
// only version 1. Any change to the layout or the meaning of the values takes
// a new version number.
//
// Values: SH coefficients and visibility are IEEE 754 binary16, rounded to
// nearest, ties to even. A finite value whose magnitude exceeds 65504 is
// clamped to +-65504. Positions are float32, stored as given. The reader
// decodes binary16 exactly.
//
// The file has no "not baked" value. Visibility is in [0, 1], and L0 is never
// negative, so a luxel still holding the bake's marker (SHCoeffs(-1),
// visibility -1) fails the writer's CHECK. The caller fills luxels the bake
// skipped (with zeros or by dilation) before writing.
//
// The hull mesh holds the un-shrunk closed hulls of the casting brushes, in
// the caller's world space and units. Each hull's triangles face outward,
// with front faces wound clockwise seen from outside (q3map2's winding). The
// file neither checks nor changes closure, convexity or winding.

// One hull: a contiguous run of `index_count` indices starting at
// `first_index`. Hulls are in order and cover the index list exactly.
struct SHHullRange {
  uint32_t first_index = 0;
  uint32_t index_count = 0;
};

struct SHHullMesh {
  std::vector<Eigen::Vector3f> positions;  // caller's world space and units
  std::vector<uint32_t> indices;           // outward, clockwise from outside
  std::vector<SHHullRange> hulls;          // contiguous, covering `indices`
};

// One page of width * height luxels, x fastest, then y.
struct SHSidePage {
  int width = 0, height = 0;
  // width * height. `irradiance` is not stored, and reads back as zero.
  std::vector<SHCoeffs> sh;
  std::vector<float> visibility;  // width * height, each in [0, 1]
};

struct SHSideFile {
  std::vector<SHSidePage> pages;
  SHHullMesh hull_mesh;
};

// Writes `file` to `path`. Input only a caller bug can produce fails a CHECK
// whose message names the field and, where there is one, the value and the
// page or hull index:
//   - a page width or height below 1;
//   - coefficient or visibility counts that differ from width * height;
//   - a non-finite coefficient, visibility value or position;
//   - a visibility outside [0, 1], or a negative L0 component;
//   - a hull index count that is not a multiple of 3, or zero for a hull;
//   - hull ranges that are not contiguous in order from index 0 to the end of
//     the index list;
//   - an index not below the vertex count;
//   - any count that does not fit in a uint32.
// The bytes are written to "<path>.tmp" and renamed to `path` once complete.
// If that fails, returns false with a logged reason, removes the temporary
// file and leaves any file at `path` unchanged. Two writers to the same path
// at once are not supported.
bool SaveSHSideFile(const SHSideFile& file, const std::filesystem::path& path);

// Reads a side file. Returns std::nullopt, with a LOG(ERROR) reason, for a
// file that cannot be opened, or with a wrong magic, an unknown version, a
// size other than what its header and page sizes imply, a page width or
// height of 0 or above 2^31 - 1, or values the writer would refuse. Sizes are
// checked against the file before any page or hull storage is allocated.
// Never aborts on file contents.
std::optional<SHSideFile> LoadSHSideFile(const std::filesystem::path& path);

// Writes `page`'s L0 RGB as a Radiance RGBE .hdr image for inspection: row 0
// is page row 0, values are unscaled, and negative components are written as
// 0. Refuses a file name beginning with "lm_" in any letter case (rend2
// auto-loads maps/<name>/lm_*.hdr), returning false and writing nothing.
// Returns false with a logged reason on an I/O failure. The page's sizes and
// counts are CHECKed as SaveSHSideFile does.
bool SaveSHSidePageL0Hdr(const SHSidePage& page,
                         const std::filesystem::path& path);

}  // namespace sh_baker

#endif  // SH_BAKER_SRC_SIDE_FILE_H_
