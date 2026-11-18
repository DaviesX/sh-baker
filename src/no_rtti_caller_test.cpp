// Built with -fno-exceptions -fno-rtti, as q3map2 is, and linked against
// libsh_baker.so (built with RTTI): sets both bake callbacks across that
// boundary and checks they behave as specified, then writes a side file from
// the bake's results and reads it back. Exits 0 on success.

#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include "baker.h"
#include "loader_q3map2.h"
#include "side_file.h"

#if defined(__cpp_rtti) || defined(__GXX_RTTI)
#error "no_rtti_caller_test must be compiled with -fno-rtti"
#endif
#if defined(__cpp_exceptions) || defined(__EXCEPTIONS)
#error "no_rtti_caller_test must be compiled with -fno-exceptions"
#endif

namespace {

sh_baker::Texture SolidTexture(int channels, std::vector<uint8_t> texel) {
  sh_baker::Texture t;
  t.width = 1;
  t.height = 1;
  t.channels = channels;
  t.pixel_data = std::move(texel);
  return t;
}

int Fail(const char* what) {
  std::fprintf(stderr, "no_rtti_caller_test: %s\n", what);
  return 1;
}

bool SameBits(float a, float b) { return std::memcmp(&a, &b, sizeof a) == 0; }

float Halfed(float v) { return static_cast<float>(Eigen::half(v)); }

// Writes one 4x2 page from the first 8 points the cancelled bake finished
// (points it skipped hold the not-baked marker, which the side file refuses)
// and one tetrahedron hull, reads the file back and compares.
int CheckSideFile(const sh_baker::BakeResult& result) {
  sh_baker::SHSideFile file;
  sh_baker::SHSidePage page;
  page.width = 4;
  page.height = 2;
  const size_t points = result.sh_texture.pixels.size();
  for (size_t i = 0; i < points && page.sh.size() < 8; ++i) {
    const float visibility =
        result.environment_visibility_texture.pixel_data[i];
    if (visibility < 0.0f) continue;
    page.sh.push_back(result.sh_texture.pixels[i]);
    page.visibility.push_back(visibility);
  }
  if (page.sh.size() < 8) return Fail("fewer than 8 baked points");
  bool any_light = false;
  for (const sh_baker::SHCoeffs& sh : page.sh) {
    if (sh.coeffs[0].maxCoeff() > 0.0f) any_light = true;
  }
  if (!any_light) return Fail("baked points hold no light");
  file.pages.push_back(page);
  sh_baker::SHHullMesh& hull = file.hull_mesh;
  hull.positions = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
  hull.indices = {0, 1, 2, 0, 3, 1, 1, 3, 2, 2, 3, 0};
  hull.hulls = {{0, 12}};

  const std::filesystem::path path =
      std::filesystem::temp_directory_path() /
      ("sh_no_rtti_side_file_" + std::to_string(getpid()) + ".shsf");
  if (!sh_baker::SaveSHSideFile(file, path)) return Fail("side file write");
  std::optional<sh_baker::SHSideFile> read = sh_baker::LoadSHSideFile(path);
  std::error_code ec;
  std::filesystem::remove(path, ec);
  if (!read.has_value()) return Fail("side file read");

  if (read->pages.size() != 1) return Fail("side file page count");
  const sh_baker::SHSidePage& r = read->pages[0];
  if (r.width != 4 || r.height != 2) return Fail("side file page size");
  if (r.sh.size() != 8 || r.visibility.size() != 8) {
    return Fail("side file luxel count");
  }
  for (size_t j = 0; j < 8; ++j) {
    for (int k = 0; k < 9; ++k) {
      for (int c = 0; c < 3; ++c) {
        if (!SameBits(r.sh[j].coeffs[k][c], Halfed(page.sh[j].coeffs[k][c]))) {
          return Fail("side file coefficient differs");
        }
      }
    }
    if (!SameBits(r.visibility[j], Halfed(page.visibility[j]))) {
      return Fail("side file visibility differs");
    }
  }
  const sh_baker::SHHullMesh& rh = read->hull_mesh;
  if (rh.positions != hull.positions) return Fail("side file positions");
  if (rh.indices != hull.indices) return Fail("side file indices");
  if (rh.hulls.size() != 1 || rh.hulls[0].first_index != 0 ||
      rh.hulls[0].index_count != 12) {
    return Fail("side file hull ranges");
  }
  std::printf("no_rtti_caller_test: side file round trip of 8 baked points\n");
  return 0;
}

}  // namespace

int main() {
  sh_baker::Scene scene;
  sh_baker::Material white;
  white.name = "white";
  white.albedo = SolidTexture(4, {255, 255, 255, 255});
  white.normal_texture = SolidTexture(3, {128, 128, 255});
  white.metallic_roughness_texture = SolidTexture(3, {0, 255, 0});
  scene.materials.push_back(white);

  // A floor facing +Z, in q3map2's clockwise winding.
  sh_baker::Q3Map2Surface floor;
  floor.positions = {{-2, -2, 0}, {2, -2, 0}, {-2, 2, 0}, {2, 2, 0}};
  floor.normals.assign(4, Eigen::Vector3f(0, 0, 1));
  floor.texture_uvs = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};
  floor.indices = {0, 2, 1, 1, 2, 3};
  floor.material_id = 0;
  sh_baker::AddQ3Map2Surface(floor, &scene);

  sh_baker::Light sun;
  sun.type = sh_baker::Light::Type::Directional;
  sun.direction = Eigen::Vector3f(0, 0, -1);
  sun.color = Eigen::Vector3f(1, 1, 1);
  sun.intensity = 1.0f;
  scene.lights.push_back(sun);

  constexpr size_t kPoints = 4096;
  std::vector<sh_baker::SurfacePoint> points(kPoints);
  for (size_t i = 0; i < kPoints; ++i) {
    points[i].position = Eigen::Vector3f(-1.5f + 3.0f * (i % 64) / 63.0f,
                                         -1.5f + 3.0f * (i / 64) / 63.0f, 0);
    points[i].normal = Eigen::Vector3f(0, 0, 1);
    points[i].tangent = Eigen::Vector4f(1, 0, 0, 1);
    points[i].material_id = 0;
  }

  sh_baker::RasterConfig raster_config;
  raster_config.width = static_cast<int>(kPoints);
  raster_config.height = 1;
  raster_config.supersample_scale = 1;

  std::atomic<bool> quarter_done{false};
  size_t last_done = 0;
  size_t reports = 0;
  bool increasing = true;
  sh_baker::BakeConfig config;
  config.samples = 16;
  config.bounces = 1;
  config.progress_fn = [&](size_t done, size_t total) {
    if (done <= last_done || done > total) increasing = false;
    last_done = done;
    ++reports;
    if (done * 4 >= total) quarter_done = true;
  };
  config.cancel_fn = [&] { return quarter_done.load(); };

  sh_baker::BakeResult result =
      sh_baker::BakeSHLightMap(scene, points, raster_config, config);

  if (reports == 0) return Fail("no progress reports");
  if (!increasing) return Fail("progress reports did not increase");
  if (!result.cancelled) return Fail("result not marked cancelled");
  if (last_done >= kPoints) return Fail("cancelled bake reported completion");
  std::printf("no_rtti_caller_test: %zu reports, cancelled at %zu of %zu\n",
              reports, last_done, kPoints);
  return CheckSideFile(result);
}
