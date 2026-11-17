// Built with -fno-exceptions -fno-rtti, as q3map2 is, and linked against
// libsh_baker.so (built with RTTI): sets both bake callbacks across that
// boundary and checks they behave as specified. Exits 0 on success.

#include <atomic>
#include <cstdio>
#include <vector>

#include "baker.h"
#include "loader_q3map2.h"

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
  return 0;
}
