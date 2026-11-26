#include "baker.h"

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "colorspace.h"
#include "loader_q3map2.h"
#include "rasterizer.h"
#include "sh_coeffs.h"

namespace sh_baker {

TEST(BakerTest, BakeSimpleQuad) {
  // Scene: A simple quad at Z=0, -1..1 xy range. Normal +Z.
  // Light: Directional light pointing -Z (towards the quad).

  Scene scene;

  Geometry quad;
  quad.vertices = {{-1, -1, 0}, {1, -1, 0}, {1, 1, 0}, {-1, 1, 0}};
  quad.normals = {{0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}};
  quad.texture_uvs = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
  quad.lightmap_uvs = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
  quad.indices = {0, 1, 2, 0, 2, 3};
  quad.material_id = 0;

  scene.geometries.push_back(quad);

  Material mat;
  mat.name = "white";
  mat.albedo.pixel_data = {255, 255, 255, 255};  // 1x1 white
  mat.albedo.width = 1;
  mat.albedo.height = 1;
  mat.albedo.channels = 4;
  scene.materials.push_back(mat);

  Light light;
  light.type = Light::Type::Directional;
  light.direction = Eigen::Vector3f(0, 0, -1);
  light.color = Eigen::Vector3f(1, 1, 1);
  light.intensity = 1.0f;
  scene.lights.push_back(light);

  // Create dummy surface points
  std::vector<SurfacePoint> surface_points(16);  // 4x4
  RasterConfig raster_config;
  raster_config.width = 4;
  raster_config.height = 4;

  // Center pixel (1, 1) -> index 5
  // UV 0.375, 0.375. Position -0.25, -0.25, 0. Normal 0, 0, 1.
  surface_points[5].material_id = 0;
  surface_points[5].position = Eigen::Vector3f(-0.25f, -0.25f, 0.0f);
  surface_points[5].normal = Eigen::Vector3f(0.0f, 0.0f, 1.0f);
  // Tangents
  surface_points[5].tangent = Eigen::Vector4f(1.0f, 0.0f, 0.0f, 1.0f);

  BakeConfig config;
  config.samples = 64;
  config.bounces = 0;  // Direct light only

  BakeResult output =
      BakeSHLightMap(scene, surface_points, raster_config, config);

  // Check center pixel
  SHCoeffs result = output.sh_texture.pixels[5];
  EXPECT_GT(result.coeffs[0].x(), 0.1f);
}

TEST(BakerTest, DownsampleSHTexture) {
  SHTexture input;
  input.width = 2;
  input.height = 2;
  input.pixels.resize(4);

  // All 1.0
  for (int i = 0; i < 4; ++i) {
    for (int c = 0; c < 9; ++c) {
      input.pixels[i].coeffs[c] = Eigen::Vector3f(1.0f, 1.0f, 1.0f);
    }
  }

  SHTexture output = DownsampleSHTexture(input, 2);
  EXPECT_EQ(output.width, 1);
  EXPECT_EQ(output.height, 1);
  EXPECT_EQ(output.pixels.size(), 1);

  for (int c = 0; c < 9; ++c) {
    EXPECT_FLOAT_EQ(output.pixels[0].coeffs[c].x(), 1.0f);
  }
}

namespace {

// A white material with the loader's default normal and metallic-roughness
// textures.
Material WhiteMaterial() {
  Material mat;
  mat.name = "white";
  mat.albedo.width = 1;
  mat.albedo.height = 1;
  mat.albedo.channels = 4;
  mat.albedo.pixel_data = {255, 255, 255, 255};
  mat.normal_texture.width = 1;
  mat.normal_texture.height = 1;
  mat.normal_texture.channels = 3;
  mat.normal_texture.pixel_data = {128, 128, 255};
  mat.metallic_roughness_texture.width = 1;
  mat.metallic_roughness_texture.height = 1;
  mat.metallic_roughness_texture.channels = 3;
  mat.metallic_roughness_texture.pixel_data = {0, 255, 0};
  return mat;
}

// A q3map2 surface: the rectangle [x0, x1] x [y0, y1] at height z, facing +Z
// or -Z. q3map2 winds front faces clockwise.
Q3Map2Surface RectXY(float x0, float x1, float y0, float y1, float z,
                     bool faces_up, int material_id) {
  Q3Map2Surface s;
  s.positions = {Eigen::Vector3f(x0, y0, z), Eigen::Vector3f(x1, y0, z),
                 Eigen::Vector3f(x0, y1, z), Eigen::Vector3f(x1, y1, z)};
  s.normals.assign(4, Eigen::Vector3f(0, 0, faces_up ? 1.0f : -1.0f));
  if (material_id >= 0) {
    s.texture_uvs = {Eigen::Vector2f(0, 0), Eigen::Vector2f(1, 0),
                     Eigen::Vector2f(0, 1), Eigen::Vector2f(1, 1)};
  }
  s.indices = faces_up ? std::vector<uint32_t>{0, 2, 1, 1, 2, 3}
                       : std::vector<uint32_t>{0, 1, 2, 2, 1, 3};
  s.material_id = material_id;
  return s;
}

// A closed, material-less box [lo, hi] with outward faces, as q3map2 would
// pass a solid brush hull.
Q3Map2Surface HullBox(const Eigen::Vector3f& lo, const Eigen::Vector3f& hi) {
  const Eigen::Vector3f d = hi - lo;
  // Each face: a corner and two edges whose cross product points outward.
  struct Face {
    Eigen::Vector3f origin, u, v;
  };
  const Face faces[] = {
      {Eigen::Vector3f(hi.x(), lo.y(), lo.z()), Eigen::Vector3f(0, d.y(), 0),
       Eigen::Vector3f(0, 0, d.z())},
      {lo, Eigen::Vector3f(0, 0, d.z()), Eigen::Vector3f(0, d.y(), 0)},
      {Eigen::Vector3f(lo.x(), hi.y(), lo.z()), Eigen::Vector3f(0, 0, d.z()),
       Eigen::Vector3f(d.x(), 0, 0)},
      {lo, Eigen::Vector3f(d.x(), 0, 0), Eigen::Vector3f(0, 0, d.z())},
      {Eigen::Vector3f(lo.x(), lo.y(), hi.z()), Eigen::Vector3f(d.x(), 0, 0),
       Eigen::Vector3f(0, d.y(), 0)},
      {lo, Eigen::Vector3f(0, d.y(), 0), Eigen::Vector3f(d.x(), 0, 0)},
  };
  Q3Map2Surface s;
  for (const Face& f : faces) {
    const uint32_t base = static_cast<uint32_t>(s.positions.size());
    const Eigen::Vector3f n = f.u.cross(f.v).normalized();
    s.positions.insert(
        s.positions.end(),
        {f.origin, f.origin + f.u, f.origin + f.v, f.origin + f.u + f.v});
    s.normals.insert(s.normals.end(), {n, n, n, n});
    // (0, 1, 2) is counter-clockwise seen from outside; q3map2 is clockwise.
    s.indices.insert(s.indices.end(),
                     {base, base + 2, base + 1, base + 1, base + 2, base + 3});
  }
  s.material_id = -1;
  return s;
}

SurfacePoint PointAt(const Eigen::Vector3f& position,
                     const Eigen::Vector3f& normal, int material_id) {
  SurfacePoint sp;
  sp.position = position;
  sp.normal = normal;
  sp.tangent = Eigen::Vector4f(1, 0, 0, 1);  // perpendicular to +-Z normals
  sp.material_id = material_id;
  return sp;
}

Light SunFromAbove() {
  Light light;
  light.type = Light::Type::Directional;
  light.direction = Eigen::Vector3f(0, 0, -1);
  light.color = Eigen::Vector3f(1, 1, 1);
  light.intensity = 1.0f;
  return light;
}

// Bakes `points` as an N x 1 buffer, direct light plus the first hit.
BakeResult BakePoints(const Scene& scene,
                      const std::vector<SurfacePoint>& points) {
  RasterConfig raster_config;
  raster_config.width = static_cast<int>(points.size());
  raster_config.height = 1;
  raster_config.supersample_scale = 1;
  BakeConfig config;
  config.samples = 64;
  config.bounces = 0;
  return BakeSHLightMap(scene, points, raster_config, config);
}

}  // namespace

TEST(BakerTest, IndirectOnlySkipsDirectAreaLightAndKeepsItsBounce) {
  Scene scene;
  scene.materials.push_back(WhiteMaterial());
  scene.materials.push_back(WhiteMaterial());
  Material& emitter = scene.materials.back();
  emitter.emissive_factor = Eigen::Vector3f::Ones();
  emitter.emissive_strength = 10.0f;
  AddQ3Map2Surface(RectXY(-4, 4, -4, 4, 0, true, 0), &scene);
  AddQ3Map2Surface(RectXY(-1, 1, -1, 1, 2, false, 1), &scene);

  Light area;
  area.type = Light::Type::Area;
  area.geometry = &scene.geometries.back();
  area.material = &emitter;
  area.color = emitter.emissive_factor;
  area.intensity = emitter.emissive_strength;
  area.area = SurfaceArea(*area.geometry);
  area.flux = Flux(area);
  scene.lights.push_back(area);

  const std::vector<SurfacePoint> points = {
      PointAt({0, 0, 0}, {0, 0, 1}, 0),
      PointAt({0, 0, 1}, {0, 0, -1}, 0),
  };
  RasterConfig raster_config;
  raster_config.width = 2;
  raster_config.height = 1;
  BakeConfig config;
  config.bounces = 0;
  const BakeResult full = BakeSHLightMap(scene, points, raster_config, config);
  config.indirect_only = true;
  const BakeResult indirect =
      BakeSHLightMap(scene, points, raster_config, config);

  EXPECT_GT(full.sh_texture.pixels[0].coeffs[0].x(), 0.0f);
  EXPECT_EQ(indirect.sh_texture.pixels[0].coeffs[0], Eigen::Vector3f::Zero());
  EXPECT_GT(indirect.sh_texture.pixels[1].coeffs[0].x(), 0.0f);
}

// Caller-supplied points from different surfaces bake in one call, each result
// at its point's index, with no lightmap UVs and no rasterization.
TEST(BakerTest, BakesPointsFromDifferentSurfacesInOneCall) {
  Scene scene;
  scene.materials.push_back(WhiteMaterial());
  AddQ3Map2Surface(RectXY(-1, 1, -1, 1, 0, /*faces_up=*/true, 0), &scene);
  AddQ3Map2Surface(RectXY(3, 5, -1, 1, 0, /*faces_up=*/false, 0), &scene);
  scene.lights.push_back(SunFromAbove());

  std::vector<SurfacePoint> points = {
      PointAt(Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(0, 0, 1), 0),
      PointAt(Eigen::Vector3f(4, 0, 0), Eigen::Vector3f(0, 0, -1), 0),
      PointAt(Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(0, 0, 1), -1),
  };
  BakeResult result = BakePoints(scene, points);

  ASSERT_EQ(result.sh_texture.width, 3);
  ASSERT_EQ(result.sh_texture.height, 1);
  ASSERT_EQ(result.sh_texture.pixels.size(), 3u);
  EXPECT_GT(result.sh_texture.pixels[0].coeffs[0].x(), 0.1f);
  EXPECT_GT(result.sh_texture.pixels[0].coeffs[0].x(),
            result.sh_texture.pixels[1].coeffs[0].x());
  for (const Eigen::Vector3f& c : result.sh_texture.pixels[2].coeffs) {
    EXPECT_EQ(c, Eigen::Vector3f::Constant(-1.0f));
  }
  EXPECT_EQ(result.environment_visibility_texture.pixel_data[2], -1.0f);
}

// A material-less hull behind a crack between thin surfaces blocks the light
// that leaks through it, and contributes no radiance of its own: this is how
// q3map2's solid brush hulls enter the bake.
TEST(BakerTest, HullBehindCrackBlocksLight) {
  // A floor facing up, and a ceiling at z = 1 facing down into the room, made
  // of two thin quads with a crack along x in (-0.1, 0.1). The sun shines
  // straight down through the crack onto the receiver at the origin.
  auto build_scene = [](bool with_hull) {
    Scene scene;
    scene.materials.push_back(WhiteMaterial());
    AddQ3Map2Surface(RectXY(-2, 2, -2, 2, 0, /*faces_up=*/true, 0), &scene);
    AddQ3Map2Surface(RectXY(-2, -0.1f, -2, 2, 1, /*faces_up=*/false, 0),
                     &scene);
    AddQ3Map2Surface(RectXY(0.1f, 2, -2, 2, 1, /*faces_up=*/false, 0), &scene);
    if (with_hull) {
      AddQ3Map2Surface(HullBox(Eigen::Vector3f(-0.3f, -0.3f, 0.85f),
                               Eigen::Vector3f(0.3f, 0.3f, 0.95f)),
                       &scene);
    }
    scene.lights.push_back(SunFromAbove());
    return scene;
  };
  const std::vector<SurfacePoint> points = {
      PointAt(Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(0, 0, 1), 0)};

  const float open =
      BakePoints(build_scene(false), points).sh_texture.pixels[0].coeffs[0].x();
  const float hulled =
      BakePoints(build_scene(true), points).sh_texture.pixels[0].coeffs[0].x();

  EXPECT_GT(open, 0.1f);
  EXPECT_GE(hulled, 0.0f);
  EXPECT_LT(hulled, 0.05f * open) << "open " << open << ", hulled " << hulled;
}

namespace {

// One point light 100 above an open floor, and floor points 200 and 300 from
// it.
struct RangeFixture {
  Scene scene;
  std::vector<SurfacePoint> points;
};

RangeFixture PointLightOverFloor(float range) {
  RangeFixture f;
  f.scene.materials.push_back(WhiteMaterial());
  AddQ3Map2Surface(RectXY(-400, 400, -400, 400, 0, /*faces_up=*/true, 0),
                   &f.scene);
  Light point;
  point.type = Light::Type::Point;
  point.position = Eigen::Vector3f(0, 0, 100);
  point.intensity = 1.0e4f;
  point.flux = Flux(point);
  point.range = range;
  f.scene.lights.push_back(point);
  const Eigen::Vector3f up(0, 0, 1);
  f.points = {
      PointAt(Eigen::Vector3f(std::sqrt(200.0f * 200.0f - 1.0e4f), 0, 0), up,
              0),
      PointAt(Eigen::Vector3f(std::sqrt(300.0f * 300.0f - 1.0e4f), 0, 0), up,
              0)};
  return f;
}

}  // namespace

TEST(BakerRangeTest, PointBeyondRangeIsDark) {
  RangeFixture f = PointLightOverFloor(256.0f);
  BakeResult r = BakePoints(f.scene, f.points);
  EXPECT_GT(r.sh_texture.pixels[0].coeffs[0].x(), 0.0f);
  for (const Eigen::Vector3f& c : r.sh_texture.pixels[1].coeffs) {
    EXPECT_EQ(c, Eigen::Vector3f::Zero());
  }
}

TEST(BakerRangeTest, DefaultRangeUnchanged) {
  RangeFixture f = PointLightOverFloor(Light().range);
  BakeResult r = BakePoints(f.scene, f.points);
  EXPECT_GT(r.sh_texture.pixels[0].coeffs[0].x(), 0.0f);
  EXPECT_GT(r.sh_texture.pixels[1].coeffs[0].x(), 0.0f);
}

namespace {

// A lit floor and `n` points on it, every `invalid_every`-th one invalid
// (0: none).
struct ProgressFixture {
  Scene scene;
  std::vector<SurfacePoint> points;
};

ProgressFixture FloorPoints(size_t n, size_t invalid_every = 0,
                            bool lit = true) {
  ProgressFixture f;
  f.scene.materials.push_back(WhiteMaterial());
  AddQ3Map2Surface(RectXY(-2, 2, -2, 2, 0, /*faces_up=*/true, 0), &f.scene);
  if (lit) f.scene.lights.push_back(SunFromAbove());
  f.points.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    const float x = -1.5f + 3.0f * float(i % 64) / 63.0f;
    const float y = -1.5f + 3.0f * float((i / 64) % 64) / 63.0f;
    const bool invalid = invalid_every > 0 && i % invalid_every == 0;
    f.points.push_back(PointAt(Eigen::Vector3f(x, y, 0),
                               Eigen::Vector3f(0, 0, 1), invalid ? -1 : 0));
  }
  return f;
}

// Bakes `points` as an N x 1 buffer with `config`.
BakeResult BakeWith(const Scene& scene, const std::vector<SurfacePoint>& points,
                    const BakeConfig& config) {
  RasterConfig raster_config;
  raster_config.width = static_cast<int>(points.size());
  raster_config.height = 1;
  raster_config.supersample_scale = 1;
  return BakeSHLightMap(scene, points, raster_config, config);
}

BakeConfig SmallBake() {
  BakeConfig config;
  config.samples = 16;
  config.bounces = 1;
  return config;
}

// Records progress reports and whether two calls ever overlapped.
struct ProgressLog {
  std::vector<std::pair<size_t, size_t>> reports;
  std::atomic<int> inside{0};
  std::atomic<bool> overlapped{false};

  std::function<void(size_t, size_t)> Callback() {
    return [this](size_t done, size_t total) {
      if (inside.fetch_add(1) != 0) overlapped = true;
      reports.emplace_back(done, total);
      inside.fetch_sub(1);
    };
  }
};

void ExpectBitIdentical(const BakeResult& a, const BakeResult& b) {
  ASSERT_EQ(a.sh_texture.pixels.size(), b.sh_texture.pixels.size());
  for (size_t i = 0; i < a.sh_texture.pixels.size(); ++i) {
    for (int k = 0; k < 9; ++k) {
      for (int c = 0; c < 3; ++c) {
        // Compare bit patterns, so -1 markers and any NaN compare too.
        uint32_t x, y;
        float fx = a.sh_texture.pixels[i].coeffs[k][c];
        float fy = b.sh_texture.pixels[i].coeffs[k][c];
        std::memcpy(&x, &fx, 4);
        std::memcpy(&y, &fy, 4);
        ASSERT_EQ(x, y) << "point " << i << " coeff " << k;
      }
    }
  }
  ASSERT_EQ(a.environment_visibility_texture.pixel_data.size(),
            b.environment_visibility_texture.pixel_data.size());
  EXPECT_EQ(std::memcmp(a.environment_visibility_texture.pixel_data.data(),
                        b.environment_visibility_texture.pixel_data.data(),
                        a.environment_visibility_texture.pixel_data.size() *
                            sizeof(float)),
            0);
}

}  // namespace

TEST(BakerProgressTest, ReportsIncreaseAndNeverOverlap) {
  ProgressFixture f = FloorPoints(1000);
  ProgressLog log;
  BakeConfig config = SmallBake();
  config.progress_fn = log.Callback();
  BakeResult result = BakeWith(f.scene, f.points, config);

  EXPECT_FALSE(log.overlapped);
  EXPECT_FALSE(result.cancelled);
  ASSERT_FALSE(log.reports.empty());
  for (size_t i = 0; i < log.reports.size(); ++i) {
    EXPECT_EQ(log.reports[i].second, 1000u);
    EXPECT_LE(log.reports[i].first, 1000u);
    if (i > 0) EXPECT_GT(log.reports[i].first, log.reports[i - 1].first);
  }
  EXPECT_EQ(log.reports.back(), std::make_pair(size_t{1000}, size_t{1000}));
}

TEST(BakerProgressTest, InvalidPointsCount) {
  // Points 0, 3, 6 and 9 are invalid.
  ProgressFixture f = FloorPoints(10, /*invalid_every=*/3);
  ProgressLog log;
  BakeConfig config = SmallBake();
  config.progress_fn = log.Callback();
  BakeWith(f.scene, f.points, config);
  ASSERT_FALSE(log.reports.empty());
  EXPECT_EQ(log.reports.back(), std::make_pair(size_t{10}, size_t{10}));
}

TEST(BakerProgressTest, AtMostOneReportPerPercent) {
  ProgressFixture f = FloorPoints(100000, 0, /*lit=*/false);
  ProgressLog log;
  BakeConfig config;
  config.samples = 1;
  config.bounces = 0;
  config.progress_fn = log.Callback();
  BakeWith(f.scene, f.points, config);
  EXPECT_FALSE(log.overlapped);
  EXPECT_LE(log.reports.size(), 101u);
  ASSERT_FALSE(log.reports.empty());
  EXPECT_EQ(log.reports.back(), std::make_pair(size_t{100000}, size_t{100000}));
}

TEST(BakerProgressTest, ZeroPointsReportNothing) {
  Scene scene;
  scene.materials.push_back(WhiteMaterial());
  AddQ3Map2Surface(RectXY(-1, 1, -1, 1, 0, /*faces_up=*/true, 0), &scene);
  ProgressLog log;
  BakeConfig config = SmallBake();
  config.progress_fn = log.Callback();
  RasterConfig raster_config;
  raster_config.width = 0;
  raster_config.height = 1;
  raster_config.supersample_scale = 1;
  BakeResult result = BakeSHLightMap(scene, {}, raster_config, config);
  EXPECT_TRUE(log.reports.empty());
  EXPECT_TRUE(result.sh_texture.pixels.empty());
  EXPECT_TRUE(result.environment_visibility_texture.pixel_data.empty());
  EXPECT_FALSE(result.cancelled);
}

TEST(BakerProgressTest, LibraryWritesNothingToStdout) {
  ProgressFixture f = FloorPoints(200);
  testing::internal::CaptureStdout();
  BakeWith(f.scene, f.points, SmallBake());
  EXPECT_EQ(testing::internal::GetCapturedStdout(), "");
}

TEST(BakerProgressTest, ProgressCallbackDoesNotChangeResults) {
  ProgressFixture f = FloorPoints(300, /*invalid_every=*/7);
  const BakeResult plain = BakeWith(f.scene, f.points, SmallBake());
  ProgressLog log;
  BakeConfig config = SmallBake();
  config.progress_fn = log.Callback();
  ExpectBitIdentical(plain, BakeWith(f.scene, f.points, config));
}

namespace {

bool IsMarker(const BakeResult& r, size_t i) {
  for (const Eigen::Vector3f& c : r.sh_texture.pixels[i].coeffs) {
    if (c != Eigen::Vector3f::Constant(-1.0f)) return false;
  }
  return r.environment_visibility_texture.pixel_data[i] == -1.0f;
}

bool SameBits(const BakeResult& a, const BakeResult& b, size_t i) {
  return std::memcmp(&a.sh_texture.pixels[i], &b.sh_texture.pixels[i],
                     sizeof(SHCoeffs)) == 0 &&
         std::memcmp(&a.environment_visibility_texture.pixel_data[i],
                     &b.environment_visibility_texture.pixel_data[i],
                     sizeof(float)) == 0;
}

}  // namespace

TEST(BakerCancelTest, CancelledBeforeStart) {
  ProgressFixture f = FloorPoints(500);
  ProgressLog log;
  BakeConfig config = SmallBake();
  config.progress_fn = log.Callback();
  int calls = 0;
  config.cancel_fn = [&calls] {
    ++calls;
    return true;
  };
  BakeResult result = BakeWith(f.scene, f.points, config);
  EXPECT_TRUE(result.cancelled);
  // The first point triggers cancellation, and no further polls occur.
  EXPECT_EQ(calls, 1);
  ASSERT_EQ(result.sh_texture.pixels.size(), 500u);
  for (size_t i = 0; i < 500; ++i) EXPECT_TRUE(IsMarker(result, i)) << i;
  for (const auto& report : log.reports) EXPECT_LT(report.first, report.second);
}

TEST(BakerCancelTest, CancelledMidwayKeepsExactResults) {
  constexpr size_t kPoints = 4096;
  ProgressFixture f = FloorPoints(kPoints, /*invalid_every=*/11);
  const BakeResult full = BakeWith(f.scene, f.points, SmallBake());

  std::atomic<bool> quarter_done{false};
  BakeConfig config = SmallBake();
  config.progress_fn = [&](size_t done, size_t total) {
    if (done * 4 >= total) quarter_done = true;
  };
  config.cancel_fn = [&] { return quarter_done.load(); };
  const BakeResult cut = BakeWith(f.scene, f.points, config);

  EXPECT_TRUE(cut.cancelled);
  size_t baked = 0;
  for (size_t i = 0; i < kPoints; ++i) {
    if (IsMarker(cut, i)) {
      continue;
    }
    ++baked;
    EXPECT_TRUE(SameBits(cut, full, i)) << "point " << i;
  }
  // Valid points are 10 in 11; at least a quarter of all points finished.
  EXPECT_GT(baked, 0u);
  EXPECT_LT(baked, kPoints * 10 / 11);
  // Every point the full bake left as a marker is a marker here too.
  for (size_t i = 0; i < kPoints; ++i) {
    if (IsMarker(full, i)) EXPECT_TRUE(IsMarker(cut, i)) << i;
  }
}

TEST(BakerCancelTest, NoPollAfterTrueAndPollsNeverOverlap) {
  ProgressFixture f = FloorPoints(4096);
  std::atomic<int> calls{0};
  std::atomic<int> after_true{0};
  std::atomic<int> inside{0};
  std::atomic<bool> overlapped{false};
  BakeConfig config = SmallBake();
  config.cancel_fn = [&] {
    if (inside.fetch_add(1) != 0) overlapped = true;
    const int n = ++calls;
    if (n > 100) ++after_true;
    if (n == 100) {
      // Hold the returning poll so other workers queue up behind it: a poll
      // that began after this true would show up as call 101.
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    inside.fetch_sub(1);
    return n >= 100;
  };
  BakeResult result = BakeWith(f.scene, f.points, config);
  EXPECT_TRUE(result.cancelled);
  EXPECT_EQ(calls.load(), 100);
  EXPECT_EQ(after_true.load(), 0);
  EXPECT_FALSE(overlapped);
}

TEST(BakerCancelTest, CancelCheckedBeforeValidity) {
  // All invalid: cancellation is checked before the first validity check.
  ProgressFixture f = FloorPoints(10, /*invalid_every=*/1);
  ProgressLog log;
  int calls = 0;
  BakeConfig config = SmallBake();
  config.progress_fn = log.Callback();
  config.cancel_fn = [&] { return ++calls >= 1; };
  BakeResult result = BakeWith(f.scene, f.points, config);
  EXPECT_TRUE(result.cancelled);
  EXPECT_EQ(calls, 1);
  EXPECT_TRUE(log.reports.empty());
}

TEST(BakerCancelTest, NeverTrueCancelLeavesResultUncancelledAndIdentical) {
  ProgressFixture f = FloorPoints(300, /*invalid_every=*/7);
  const BakeResult plain = BakeWith(f.scene, f.points, SmallBake());

  BakeConfig cancel_only = SmallBake();
  cancel_only.cancel_fn = [] { return false; };
  const BakeResult a = BakeWith(f.scene, f.points, cancel_only);
  EXPECT_FALSE(a.cancelled);
  ExpectBitIdentical(plain, a);

  ProgressLog log;
  BakeConfig both = SmallBake();
  both.progress_fn = log.Callback();
  both.cancel_fn = [] { return false; };
  const BakeResult b = BakeWith(f.scene, f.points, both);
  EXPECT_FALSE(b.cancelled);
  ExpectBitIdentical(plain, b);
  ASSERT_FALSE(log.reports.empty());
  EXPECT_EQ(log.reports.back(), std::make_pair(size_t{300}, size_t{300}));
}

// A fence over a floor, lit straight down by the sun: its RGBA albedo has
// alternating opaque (alpha 255) and cut-out (alpha 0) stripes, each 0.5 units
// wide. Floor points behind the open stripes get the full direct light, points
// behind the opaque stripes get none. The fence faces down, away from the sun,
// so it adds no light of its own.
TEST(BakerTest, CutOutFenceCastsStripedShadow) {
  constexpr int kStripes = 8;
  auto build_scene = [](bool with_fence) {
    Scene scene;
    scene.materials.push_back(WhiteMaterial());
    Material fence = WhiteMaterial();
    fence.name = "fence";
    fence.albedo.width = kStripes;
    fence.albedo.height = 1;
    fence.albedo.channels = 4;
    fence.albedo.pixel_data.clear();
    for (int i = 0; i < kStripes; ++i) {
      const uint8_t alpha = (i % 2 == 0) ? 255 : 0;
      fence.albedo.pixel_data.insert(fence.albedo.pixel_data.end(),
                                     {200, 200, 200, alpha});
    }
    scene.materials.push_back(fence);
    AddQ3Map2Surface(RectXY(-3, 3, -3, 3, 0, /*faces_up=*/true, 0), &scene);
    if (with_fence) {
      // u = (x + 2) / 4, so stripe i covers x in [-2 + 0.5 i, -1.5 + 0.5 i].
      AddQ3Map2Surface(RectXY(-2, 2, -2, 2, 1, /*faces_up=*/false, 1), &scene);
    }
    scene.lights.push_back(SunFromAbove());
    return scene;
  };
  std::vector<SurfacePoint> points;
  for (int i = 0; i < kStripes; ++i) {
    points.push_back(PointAt(Eigen::Vector3f(-1.75f + 0.5f * i, 0.0f, 0.0f),
                             Eigen::Vector3f(0, 0, 1), 0));
  }

  const BakeResult open = BakePoints(build_scene(false), points);
  const BakeResult fenced = BakePoints(build_scene(true), points);
  for (int i = 0; i < kStripes; ++i) {
    const float lit = open.sh_texture.pixels[i].coeffs[0].x();
    const float got = fenced.sh_texture.pixels[i].coeffs[0].x();
    ASSERT_GT(lit, 0.1f);
    if (i % 2 == 0) {
      EXPECT_LT(got, 0.01f * lit) << "opaque stripe " << i;
    } else {
      EXPECT_NEAR(got, lit, 0.01f * lit) << "open stripe " << i;
    }
  }
}

// A bake point behind a card of uniform alpha 128/255 keeps 1 - 128/255 of its
// direct light, from a point light and from the sun. The card faces down, away
// from the light, so it receives no direct light and adds none of its own to
// the receiver's first-hit term.
TEST(BakerTest, HalfCoveredCardAttenuatesBakedDirectLight) {
  Light point;
  point.type = Light::Type::Point;
  point.position = Eigen::Vector3f(0, 0, 4);
  point.color = Eigen::Vector3f(1, 1, 1);
  point.intensity = 16.0f;
  point.flux = 200.0f;
  for (const Light& light : {point, SunFromAbove()}) {
    auto build_scene = [&light](bool with_card) {
      Scene scene;
      scene.materials.push_back(WhiteMaterial());
      Material card = WhiteMaterial();
      card.name = "card";
      card.albedo.pixel_data = {200, 200, 200, 128};
      scene.materials.push_back(card);
      AddQ3Map2Surface(RectXY(-2, 2, -2, 2, 0, /*faces_up=*/true, 0), &scene);
      if (with_card) {
        AddQ3Map2Surface(RectXY(-1, 1, -1, 1, 1, /*faces_up=*/false, 1),
                         &scene);
      }
      scene.lights.push_back(light);
      return scene;
    };
    const std::vector<SurfacePoint> points = {
        PointAt(Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(0, 0, 1), 0)};

    const float open = BakePoints(build_scene(false), points)
                           .sh_texture.pixels[0]
                           .coeffs[0]
                           .x();
    const float covered = BakePoints(build_scene(true), points)
                              .sh_texture.pixels[0]
                              .coeffs[0]
                              .x();
    ASSERT_GT(open, 0.1f);
    EXPECT_NEAR(covered, (1.0f - 128.0f / 255.0f) * open, 0.02f * open)
        << "light type " << static_cast<int>(light.type) << ": open " << open
        << ", covered " << covered;
  }
}

namespace {

// A fixed scene with every light type and every bake path, for the defaults
// test: a white floor, a half-transparent card, a point light, a ranged
// smoothstep spot, a sun and an emissive quad as an area light, baked with
// bounces deep enough for Russian roulette. Built in place, since the area
// light points into the scene's own vectors.
void BuildDefaultsScene(Scene* scene) {
  scene->materials.push_back(WhiteMaterial());
  Material card = WhiteMaterial();
  card.name = "card";
  card.albedo.pixel_data = {200, 160, 120, 128};
  scene->materials.push_back(card);
  Material emitter = WhiteMaterial();
  emitter.name = "emitter";
  emitter.emissive_factor = Eigen::Vector3f(1.0f, 0.8f, 0.6f);
  emitter.emissive_strength = 6.0f;
  scene->materials.push_back(emitter);

  AddQ3Map2Surface(RectXY(-4, 4, -4, 4, 0, /*faces_up=*/true, 0), scene);
  AddQ3Map2Surface(RectXY(-4, 4, -4, 4, 4, /*faces_up=*/false, 0), scene);
  AddQ3Map2Surface(RectXY(-1, 0.5f, -1, 1, 1.5f, /*faces_up=*/false, 1), scene);
  AddQ3Map2Surface(RectXY(2, 3, -1, 1, 2, /*faces_up=*/false, 2), scene);

  Light point;
  point.type = Light::Type::Point;
  point.position = Eigen::Vector3f(-2.5f, 0.5f, 3.0f);
  point.color = Eigen::Vector3f(1.0f, 0.9f, 0.8f);
  point.intensity = 20.0f;
  point.flux = Flux(point);
  scene->lights.push_back(point);

  Light spot;
  spot.type = Light::Type::Spot;
  spot.position = Eigen::Vector3f(0.5f, 2.5f, 3.5f);
  spot.direction = Eigen::Vector3f(-0.2f, -0.3f, -1.0f).normalized();
  spot.color = Eigen::Vector3f(0.7f, 0.8f, 1.0f);
  spot.intensity = 40.0f;
  spot.cos_inner_cone = std::cos(20.0f * float(M_PI) / 180.0f);
  spot.cos_outer_cone = std::cos(40.0f * float(M_PI) / 180.0f);
  spot.cone_falloff = Light::ConeFalloff::kSmoothstep;
  spot.range = 6.0f;
  spot.flux = Flux(spot);
  scene->lights.push_back(spot);

  Light sun;
  sun.type = Light::Type::Directional;
  sun.direction = Eigen::Vector3f(0.4f, 0.2f, -1.0f).normalized();
  sun.color = Eigen::Vector3f(1.0f, 0.95f, 0.9f);
  sun.intensity = 0.5f;
  sun.flux = Flux(sun);
  scene->lights.push_back(sun);

  Light area;
  area.type = Light::Type::Area;
  area.geometry = &scene->geometries.back();
  area.material = &scene->materials.back();
  area.color = area.material->emissive_factor;
  area.intensity = area.material->emissive_strength;
  area.area = SurfaceArea(*area.geometry);
  area.flux = Flux(area);
  scene->lights.push_back(area);
}

std::vector<SurfacePoint> DefaultsPoints() {
  const Eigen::Vector3f up(0, 0, 1);
  return {
      PointAt({-3.0f, -2.0f, 0}, up, 0),
      PointAt({-0.5f, 0, 0}, up, 0),
      PointAt({0.2f, 0.3f, 0}, up, 0),
      PointAt({0, 0, 0}, up, -1),
      PointAt({2.5f, 0, 0}, up, 0),
      PointAt({3.5f, 3.5f, 0}, up, 0),
      PointAt({-0.2f, 0.1f, 1.5f}, Eigen::Vector3f(0, 0, -1), 1),
      PointAt({1.0f, -2.0f, 4.0f}, Eigen::Vector3f(0, 0, -1), 0),
  };
}

BakeConfig DefaultsConfig(bool indirect_only) {
  BakeConfig config;
  config.samples = 96;
  config.bounces = 4;
  config.num_light_samples = 2;
  config.indirect_only = indirect_only;
  return config;
}

uint32_t FloatBits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return u;
}

// Bit patterns of the bake of BuildDefaultsScene() at DefaultsPoints() with
// DefaultsConfig(), recorded from a Release build at 046ff2e, before
// SurfacePoint::full_sphere existed. Point 3 is skipped (material_id -1). They
// pin the default bake's output. Never regenerate them from a later commit.
//
// One exception: the `visibility` words of every baked point (all surface
// points) were re-recorded, the same way, from the commit that made surface
// points' environment visibility cosine-weighted (sum(e_i cos_i) /
// sum(cos_i) instead of the uniform fraction of first rays that escape). That
// commit changed nothing else in the bake, so the SH and irradiance words are
// still those of 046ff2e and keep pinning everything it must not change.
//
// The bits are those of one host's Release build (-march=native, -flto, -O3,
// GCC 15.2), so ExpectMatchesReference compares them bit for bit only when
// configured with -DSH_BAKER_BIT_EXACT_TESTS=ON on that host and toolchain;
// otherwise within a tolerance. To pin bits on another CPU or toolchain,
// re-record them from a Release build of 046ff2e itself (this test block and
// the definition copied onto it, printing FloatBits of each value), then check
// that this commit's build matches them.
struct DefaultsReference {
  uint32_t coeffs[27];  // 9 coefficients x RGB.
  uint32_t irradiance[3];
  uint32_t visibility;
};

constexpr DefaultsReference kDefaultsFull[] = {
    {{0x3f4fa47a, 0x3f3b2c0e, 0x3f27064b, 0x3f02d66e, 0x3ee94a38, 0x3ecd94c3,
      0x3f8d7001, 0x3f800a62, 0x3f65984b, 0x3e4ea132, 0x3e2dc9e3, 0x3e0e1c66,
      0x3e8bcc24, 0x3e78da4e, 0x3e5bb640, 0x3f55e4e6, 0x3f3f187c, 0x3f28b50e,
      0x3f4f0696, 0x3f3d6d4f, 0x3f2bd274, 0x3e8fb33c, 0x3e712080, 0x3e44245b,
      0xbe6f5206, 0xbe5aad83, 0xbe463156},
     {0x4010bca1, 0x400306fd, 0x3feaf33c},
     0x3f21fbdd},
    {{0x3fca8b63, 0x3fc18b44, 0x3fc4c6b7, 0x3f2ede9b, 0x3f3fbde6, 0x3f6369ef,
      0x400f7c0a, 0x4008b6b8, 0x400a942a, 0xbe04ad7a, 0xbdc18317, 0xbcb78a57,
      0x3da8207e, 0x3e0b9ec9, 0x3e67b52e, 0x3f9bdc30, 0x3fab2274, 0x3fcb0e7e,
      0x3fe8d45e, 0x3fdb6bdb, 0x3fdbb85e, 0xbeb10602, 0xbe89e459, 0xbe013b60,
      0x3d7be022, 0xbca6e673, 0xbe10fd68},
     {0x4092d4d8, 0x408be715, 0x408dcfab},
     0x3ec287a9},
    {{0x3fc5f19e, 0x3fb497b2, 0x3faae10c, 0x3eacd11c, 0x3ec75d7a, 0x3ef8ecab,
      0x40051b63, 0x3ff44e70, 0x3fe9af8e, 0xbc0e6d3d, 0xbd2629fe, 0xbd9066eb,
      0xbd90c3a8, 0xbd3bcb34, 0xbc4ba4a6, 0x3f240503, 0x3f3cca27, 0x3f6b361b,
      0x3fb8f3eb, 0x3faca31b, 0x3faa7528, 0xbdc77918, 0xbe04e158, 0xbe226b81,
      0x3f248974, 0x3efa7153, 0x3e9eac52},
     {0x4088363c, 0x407a0153, 0x406f2300},
     0x3ec73439},
    {{0xbf800000, 0xbf800000, 0xbf800000, 0xbf800000, 0xbf800000, 0xbf800000,
      0xbf800000, 0xbf800000, 0xbf800000, 0xbf800000, 0xbf800000, 0xbf800000,
      0xbf800000, 0xbf800000, 0xbf800000, 0xbf800000, 0xbf800000, 0xbf800000,
      0xbf800000, 0xbf800000, 0xbf800000, 0xbf800000, 0xbf800000, 0xbf800000,
      0xbf800000, 0xbf800000, 0xbf800000},
     {0x00000000, 0x00000000, 0x00000000},
     0xbf800000},
    {{0x3f9701b3, 0x3f7424d2, 0x3f3b91aa, 0xbce9f1ee, 0xbc9c18b7, 0xbc2f0848,
      0x3fed951f, 0x3fbff536, 0x3f92f14f, 0xbe808d5a, 0xbe55bdb4, 0xbe314fb0,
      0x3babbd82, 0xbb2e7d18, 0xbc04ff37, 0xbd64e48e, 0xbd255e67, 0xbcd25048,
      0x4000da85, 0x3fcfcf28, 0x3f9de9f6, 0xbebc868b, 0xbe9e9775, 0xbe8487ce,
      0x3d8c8298, 0x3d743256, 0x3d6700ce},
     {0x40731fdb, 0x40446f76, 0x40165ec2},
     0x3f029950},
    {{0x3ed9ac5e, 0x3eb699b6, 0x3e942698, 0xbed45a5c, 0xbeafa62e, 0xbe8b4b71,
      0x3ede7af8, 0x3eb9417e, 0x3e94f462, 0xbe96b234, 0xbe830e2c, 0xbe5fa790,
      0x3e96ebf7, 0x3e817d6e, 0x3e58b32e, 0xbf083ad2, 0xbee03fb4, 0xbeb0b55c,
      0x3d7d2417, 0x3d3b1904, 0x3cff797b, 0xbeb05170, 0xbe980315, 0xbe807bef,
      0xbdbc9654, 0xbd7b51fa, 0xbcf9d78b},
     {0x3f63ab78, 0x3f3d93b8, 0x3f186dde},
     0x3f303240},
    {{0x3fa7886b, 0x3f9c4086, 0x3f97ecea, 0x3e170d76, 0x3e441c4f, 0x3e87de57,
      0xbfc12f73, 0xbfb53ce6, 0xbfb1b5e6, 0x3e49659b, 0x3e14232a, 0x3db9910f,
      0xbe2e79c8, 0xbe215827, 0xbe19b8be, 0xbe45490f, 0xbe81d2ff, 0xbeb46c70,
      0x3f265ac4, 0x3f1f3843, 0x3f2078e3, 0xbe259570, 0xbde36207, 0xbd6b2ee8,
      0x3eb085af, 0x3e9359b9, 0x3e6780a0},
     {0x4045b108, 0x40397723, 0x4035db14},
     0x3e273b4e},
    {{0x3f5ac34e, 0x3f3d1497, 0x3f21d21e, 0x3f39343b, 0x3f213634, 0x3f0b11e5,
      0xbf5a2eea, 0xbf39fe57, 0xbf1d6535, 0xbe89e585, 0xbe882124, 0xbe873216,
      0xbeb4d8b4, 0xbeb184be, 0xbeaf62d4, 0xbf545ae2, 0xbf367322, 0xbf1beca0,
      0x3e4ee85e, 0x3e1d3033, 0x3df1ab93, 0x3c5aa730, 0x3d454b0e, 0x3daf408c,
      0x3cb97dd0, 0x3cb9badf, 0x3c9e55d0},
     {0x3fdf45c7, 0x3fbe54fc, 0x3fa11112},
     0x3f033f1a},
};

constexpr DefaultsReference kDefaultsIndirectOnly[] = {
    {{0x3f2a9d3e, 0x3f173420, 0x3f051468, 0x3ee9cd56, 0x3ece4776, 0x3eb52f56,
      0x3f74c626, 0x3f5a9577, 0x3f419104, 0x3e637b6c, 0x3e413707, 0x3e2452d3,
      0x3e04635c, 0x3dd58dae, 0x3dafb617, 0x3f577503, 0x3f400974, 0x3f29f57c,
      0x3f50fe3f, 0x3f3d7014, 0x3f29c446, 0x3ebb140e, 0x3ea2aecc, 0x3e8d4104,
      0xbe2222e7, 0xbe14f044, 0xbe071254},
     {0x3ffa7bc8, 0x3fdfaeb6, 0x3fc614e0},
     0x3f1281f0},
    {{0x3f0cde61, 0x3ee3fb1b, 0x3eb94dc2, 0x3d3511c2, 0x3d30100b, 0x3d27526d,
      0x3f52162a, 0x3f2a0ece, 0x3f0a8647, 0x3d123504, 0x3baff998, 0xbc86b8ac,
      0xbd83d2fa, 0xbd6bf85b, 0xbd4f0b02, 0x3dba5f3c, 0x3db50fbf, 0x3dabf97a,
      0x3f43e6ea, 0x3f1ec958, 0x3f0216e5, 0x3cc14f6a, 0xbcdd4cdb, 0xbd7c6328,
      0x3ddaf436, 0x3daff412, 0x3d8aab90},
     {0x3fd6fcaa, 0x3fae0648, 0x3f8dc176},
     0x3efa55f2},
    {{0x3f04148a, 0x3ed192f4, 0x3ea4ce4c, 0x3d85b0c7, 0x3d5d7ff2, 0x3d35e4bc,
      0x3f4327fc, 0x3f19aeb2, 0x3ef00730, 0xbdc99050, 0xbdb1da98, 0xbda63dd8,
      0xbcb84ee4, 0xbcae767c, 0xbca752b2, 0x3e02b29a, 0x3dd738eb, 0x3daf8b52,
      0x3f325fff, 0x3f09c68e, 0x3ed33367, 0xbe31c596, 0xbe1ac6ce, 0xbe0fd2fa,
      0x3e071d12, 0x3def8e68, 0x3dcda55a},
     {0x3fc7b554, 0x3f9d4466, 0x3f75a07c},
     0x3ee2c62d},
    {{0xbf800000, 0xbf800000, 0xbf800000, 0xbf800000, 0xbf800000, 0xbf800000,
      0xbf800000, 0xbf800000, 0xbf800000, 0xbf800000, 0xbf800000, 0xbf800000,
      0xbf800000, 0xbf800000, 0xbf800000, 0xbf800000, 0xbf800000, 0xbf800000,
      0xbf800000, 0xbf800000, 0xbf800000, 0xbf800000, 0xbf800000, 0xbf800000,
      0xbf800000, 0xbf800000, 0xbf800000},
     {0x00000000, 0x00000000, 0x00000000},
     0xbf800000},
    {{0x3ed906a7, 0x3eb5280c, 0x3e940330, 0xbd90178c, 0xbd65fdbb, 0xbd2c850e,
      0x3f0d7b03, 0x3eed2864, 0x3ec1d097, 0xbeae9666, 0xbe918531, 0xbe70cacb,
      0x3cd8865f, 0x3ca00814, 0x3c5613c4, 0xbe0a369e, 0xbdde5d1e, 0xbda84ab3,
      0x3eb7878f, 0x3e9b48aa, 0x3e7d0f6c, 0xbf03423d, 0xbedd07cf, 0xbeb81c24,
      0x3e2d1140, 0x3e0da4ca, 0x3deab36f},
     {0x3f90c7d7, 0x3f72b08b, 0x3f4655ef},
     0x3f03ceb9},
    {{0x3e8292e1, 0x3e5d5bb3, 0x3e3830d2, 0xbe32a607, 0xbe18ab14, 0xbe003878,
      0x3ea8b123, 0x3e8e275a, 0x3e6afe2b, 0xbe1fae88, 0xbe086b66, 0xbde4ba4c,
      0x3dbeaf30, 0x3da2f51b, 0x3d87f89d, 0xbe91ebc2, 0xbe78de6c, 0xbe509016,
      0x3e540017, 0x3e2f11de, 0x3e0d44cf, 0xbe73f67f, 0xbe4e6463, 0xbe2b2690,
      0xbbd6ccdf, 0xbb9850d8, 0xbb57d2c0},
     {0x3f2ca076, 0x3f117834, 0x3ef07968},
     0x3f445a42},
    {{0x3fb34516, 0x3faaabcf, 0x3faa8834, 0x3ece269b, 0x3ede8afb, 0x3f022a72,
      0xbfde8e83, 0xbfd522a4, 0xbfd720a6, 0x3dd906a7, 0x3db35a74, 0x3d983117,
      0x3dc13528, 0x3ddd7ff6, 0x3e079028, 0xbf1f8569, 0xbf2b80d2, 0xbf4825d6,
      0x3f833bc1, 0x3f7f335b, 0x3f843e18, 0xbe06b284, 0xbddd937b, 0xbdb8efc2,
      0x3d7c6ef3, 0x3cf0b850, 0xbc8f7aec},
     {0x4063bf7a, 0x405a1b5a, 0x405c253e},
     0x3df22195},
    {{0x3ef7c4ce, 0x3edd969b, 0x3ec644d8, 0x3e54639b, 0x3e3eea46, 0x3e2eff88,
      0xbf2e6ab3, 0xbf1b9a3f, 0xbf0ac0f6, 0xbcc67eb3, 0xbce0f476, 0xbd002ae2,
      0x3a7be8b0, 0xbb8bd336, 0xbc27f118, 0xbea7b576, 0xbe95e987, 0xbe88c49f,
      0x3f0e24af, 0x3efbe926, 0x3ede90a8, 0x3c92e548, 0x3cd9d050, 0x3d14c0c0,
      0xbd0aded8, 0xbd0bb2f7, 0xbd1bb9d4},
     {0x3fb27c34, 0x3f9f3b68, 0x3f8dfd80},
     0x3edb3476},
};

// Compares a bake with a reference: bit for bit when SH_BAKER_TEST_BIT_EXACT is
// defined (a Release build with SH_BAKER_BIT_EXACT_TESTS=ON, on the host and
// toolchain the reference was recorded with), and within a small tolerance
// otherwise, where other compiler flags or CPUs (no -march=native, so no FMA
// contraction, or another vector width) round differently.
void ExpectMatchesReference(const BakeResult& r,
                            const DefaultsReference (&ref)[8]) {
  ASSERT_EQ(r.sh_texture.pixels.size(), 8u);
  auto expect = [](float got, uint32_t want_bits, const char* what, size_t i,
                   int k) {
#ifdef SH_BAKER_TEST_BIT_EXACT
    EXPECT_EQ(FloatBits(got), want_bits)
        << what << " of point " << i << ", element " << k << ": got " << got;
#else
    float want;
    std::memcpy(&want, &want_bits, 4);
    EXPECT_NEAR(got, want, 1e-4f + 1e-3f * std::abs(want))
        << what << " of point " << i << ", element " << k;
#endif
  };
  for (size_t i = 0; i < 8; ++i) {
    const SHCoeffs& sh = r.sh_texture.pixels[i];
    for (int k = 0; k < 27; ++k) {
      expect(sh.coeffs[k / 3][k % 3], ref[i].coeffs[k], "coefficient", i, k);
    }
    for (int c = 0; c < 3; ++c) {
      expect(sh.irradiance[c], ref[i].irradiance[c], "irradiance", i, c);
    }
    expect(r.environment_visibility_texture.pixel_data[i], ref[i].visibility,
           "visibility", i, 0);
  }
}

}  // namespace

// With every new field at its default, the bake is the one recorded before
// SurfacePoint::full_sphere existed: every light type, the alpha card,
// bounces, a skipped point, in a full and an indirect-only bake.
TEST(BakerDefaultsTest, DefaultBakeMatchesRecordedReference) {
  Scene scene;
  BuildDefaultsScene(&scene);
  const std::vector<SurfacePoint> points = DefaultsPoints();
  ExpectMatchesReference(BakeWith(scene, points, DefaultsConfig(false)),
                         kDefaultsFull);
  ExpectMatchesReference(BakeWith(scene, points, DefaultsConfig(true)),
                         kDefaultsIndirectOnly);
}

namespace {

// A full-sphere probe: no normal, no tangent.
SurfacePoint ProbeAt(const Eigen::Vector3f& position) {
  SurfacePoint sp;
  sp.position = position;
  sp.material_id = 0;
  sp.full_sphere = true;
  return sp;
}

// The real SH basis in sh_coeffs.cpp's order, at unit direction w.
std::array<float, 9> ShBasis(const Eigen::Vector3f& w) {
  const float x = w.x(), y = w.y(), z = w.z();
  return {0.282095f,
          0.488603f * y,
          0.488603f * z,
          0.488603f * x,
          1.092548f * x * y,
          1.092548f * y * z,
          0.315392f * (3.0f * z * z - 1.0f),
          1.092548f * x * z,
          0.546274f * (x * x - y * y)};
}

// A material that reflects nothing.
Material BlackMaterial() {
  Material mat = WhiteMaterial();
  mat.name = "black";
  mat.albedo.pixel_data = {0, 0, 0, 255};
  return mat;
}

// The closed box [lo, hi] with every face facing inward, as one surface.
Q3Map2Surface InwardBox(const Eigen::Vector3f& lo, const Eigen::Vector3f& hi,
                        int material_id) {
  Q3Map2Surface s = HullBox(lo, hi);
  for (Eigen::Vector3f& n : s.normals) n = -n;
  for (size_t t = 0; t < s.indices.size(); t += 3) {
    std::swap(s.indices[t + 1], s.indices[t + 2]);
  }
  s.texture_uvs.assign(s.positions.size(), Eigen::Vector2f::Zero());
  s.material_id = material_id;
  return s;
}

}  // namespace

// A probe midway between two point lights on the X axis, with no geometry,
// gets the analytic full-sphere projection of two delta lights:
// sum over lights of (I / r^2) color Y(w). The +X light is magenta and the -X
// light cyan, so red sees only +X, green only -X, and blue both, whose L1
// along X cancels.
//
// Tolerance: the only noise is which light each NEE pick chooses (pdf about
// 1/2 each, by symmetry); the scene has no geometry, so traced rays add
// nothing. 256 samples x 64 picks (confidence 0, so every sample is taken)
// give 16384 picks; the fraction of picks per light has sigma 0.5/128, so a
// red or green coefficient has sigma 2 a |Y| 0.5/128 = 0.008 a |Y|, and blue's
// L1 twice that. 0.1 a |Y_k| is over 6 sigma; coefficients whose basis is 0 on
// the axis are exactly 0.
TEST(BakerFullSphereTest, ProbeBetweenOpposedPointLights) {
  Scene scene;
  scene.materials.push_back(WhiteMaterial());
  const float kIntensity = 50.0f;
  const float kDist = 2.0f;
  for (float sign : {1.0f, -1.0f}) {
    Light point;
    point.type = Light::Type::Point;
    point.position = Eigen::Vector3f(sign * kDist, 0, 0);
    point.color =
        sign > 0 ? Eigen::Vector3f(1, 0, 1) : Eigen::Vector3f(0, 1, 1);
    point.intensity = kIntensity;
    point.flux = Flux(point);
    scene.lights.push_back(point);
  }
  BakeConfig config;
  config.samples = 256;
  config.bounces = 2;
  config.num_light_samples = 64;
  config.confidence_threshold = 0.0f;
  const BakeResult r =
      BakeWith(scene, {ProbeAt(Eigen::Vector3f::Zero())}, config);

  const float a = kIntensity / (kDist * kDist);
  const std::array<float, 9> yp = ShBasis(Eigen::Vector3f::UnitX());
  const std::array<float, 9> ym = ShBasis(-Eigen::Vector3f::UnitX());
  const SHCoeffs& sh = r.sh_texture.pixels[0];
  for (int k = 0; k < 9; ++k) {
    const Eigen::Vector3f want(a * yp[k], a * ym[k], a * (yp[k] + ym[k]));
    const float tol = 0.1f * a * std::abs(yp[k]) + 1e-6f;
    for (int c = 0; c < 3; ++c) {
      EXPECT_NEAR(sh.coeffs[k][c], want[c], tol)
          << "coefficient " << k << ", channel " << c;
    }
  }
  // L0 of blue sees both lights in every pick: 2 a Y00 up to the light tree's
  // float asymmetry.
  EXPECT_NEAR(sh.coeffs[0].z(), 2.0f * a * 0.282095f, 1e-3f * a);
}

// With confidence_threshold 0, a point whose per-sample L0 is constant takes
// exactly `samples` rays instead of stopping after the 64-sample warm-up. The
// probe sees one unshadowed point light (its next-event sample is the same
// every time) and a material-less floor that returns exactly 0, so every
// sample is identical. Its environment visibility, escaped rays / rays taken,
// shows the ray count: with 97 rays (prime) and some but not all rays
// escaping, it is a multiple of 1/97 and not of 1/64.
TEST(BakerFullSphereTest, ZeroThresholdTakesEverySampleAtConstantPoint) {
  Scene scene;
  scene.materials.push_back(BlackMaterial());
  AddQ3Map2Surface(RectXY(-1000, 1000, -1000, 1000, 0.0f, /*faces_up=*/true,
                          /*material_id=*/-1),
                   &scene);
  Light point;
  point.type = Light::Type::Point;
  point.position = Eigen::Vector3f(0, 0, 3);
  point.color = Eigen::Vector3f(1, 1, 1);
  point.intensity = 10.0f;
  point.flux = Flux(point);
  scene.lights.push_back(point);

  constexpr int kSamples = 97;
  BakeConfig config;
  config.samples = kSamples;
  config.bounces = 1;
  const std::vector<SurfacePoint> probe = {ProbeAt(Eigen::Vector3f(0, 0, 1))};

  // Distance of v * n from the nearest integer.
  auto off_grid = [](float v, int n) {
    return std::abs(v * n - std::round(v * n));
  };

  // Default threshold: stops at the warm-up.
  const float v_default = BakeWith(scene, probe, config)
                              .environment_visibility_texture.pixel_data[0];
  EXPECT_GT(v_default, 0.0f);
  EXPECT_LT(v_default, 1.0f);
  EXPECT_LT(off_grid(v_default, 64), 1e-4f) << v_default;

  for (float threshold : {0.0f, -1.0f}) {
    config.confidence_threshold = threshold;
    const float v = BakeWith(scene, probe, config)
                        .environment_visibility_texture.pixel_data[0];
    EXPECT_GT(v, 0.0f) << threshold;
    EXPECT_LT(v, 1.0f) << threshold;
    EXPECT_LT(off_grid(v, kSamples), 1e-4f) << threshold << ": " << v;
    EXPECT_GT(off_grid(v, 64), 1e-3f) << threshold << ": " << v;
  }
}

// A sun exactly edge-on to the probe's normal field still adds its full
// projection E Y(w); a surface point with that normal gets nothing from it.
// The light tree picks the only light with pdf 1, so there is no noise.
TEST(BakerFullSphereTest, EdgeOnSunCounted) {
  Scene scene;
  scene.materials.push_back(WhiteMaterial());
  Light sun;
  sun.type = Light::Type::Directional;
  sun.direction = Eigen::Vector3f(1, 0, 0);  // Light arrives from -X.
  sun.color = Eigen::Vector3f(1.0f, 0.5f, 0.25f);
  sun.intensity = 2.0f;
  sun.flux = Flux(sun);
  scene.lights.push_back(sun);

  SurfacePoint probe = ProbeAt(Eigen::Vector3f::Zero());
  probe.normal = Eigen::Vector3f(0, 0, 1);
  const SurfacePoint surface = PointAt(Eigen::Vector3f::Zero(), {0, 0, 1}, 0);
  const BakeResult r = BakePoints(scene, {probe, surface});

  const Eigen::Vector3f e = sun.intensity * sun.color;
  const std::array<float, 9> y = ShBasis(-Eigen::Vector3f::UnitX());
  const SHCoeffs& sh = r.sh_texture.pixels[0];
  for (int k = 0; k < 9; ++k) {
    for (int c = 0; c < 3; ++c) {
      EXPECT_NEAR(sh.coeffs[k][c], e[c] * y[k], 1e-5f)
          << "coefficient " << k << ", channel " << c;
    }
  }
  EXPECT_GT(sh.coeffs[0].x(), 0.0f);
  EXPECT_EQ(r.sh_texture.pixels[1].coeffs[0], Eigen::Vector3f::Zero());
}

// A probe at the centre of a closed box of inward-facing emitters of uniform
// radiance L and albedo 0 sees L in every direction: L0 = 4 pi L Y00 =
// 2 sqrt(pi) L, and every L1 and L2 coefficient is 0. (BakeSHLightMap never
// reads Scene::environment, so a sky can't stand in for the box.)
//
// Tolerance: all light comes from NEE on the box, whose per-pick estimator
// 24 L cos / d^2 Y_k(w) (uniform area sampling of the 2 x 2 x 2 box) has a
// standard deviation of 1.35 L for L0 and at most 4.2 L for L1 and L2
// (measured by Monte Carlo over the box). 4096 samples x 16 picks give 65536
// picks, so sigma is 0.0053 L for L0 and 0.0164 L for the others; the
// tolerances are about 5 sigma.
TEST(BakerFullSphereTest, ProbeInUniformEmissiveBox) {
  Scene scene;
  scene.materials.push_back(BlackMaterial());
  Material emitter = BlackMaterial();
  emitter.name = "emitter";
  const float kRadiance = 3.0f;
  emitter.emissive_factor = Eigen::Vector3f(1.0f, 0.5f, 0.25f);
  emitter.emissive_strength = kRadiance;
  scene.materials.push_back(emitter);
  AddQ3Map2Surface(InwardBox(Eigen::Vector3f(-1, -1, -1),
                             Eigen::Vector3f(1, 1, 1), /*material_id=*/1),
                   &scene);
  Light area;
  area.type = Light::Type::Area;
  area.geometry = &scene.geometries.back();
  area.material = &scene.materials.back();
  area.color = area.material->emissive_factor;
  area.intensity = area.material->emissive_strength;
  area.area = SurfaceArea(*area.geometry);
  area.flux = Flux(area);
  scene.lights.push_back(area);

  BakeConfig config;
  config.samples = 4096;
  config.bounces = 0;
  config.num_light_samples = 16;
  config.confidence_threshold = 0.0f;
  const BakeResult r =
      BakeWith(scene, {ProbeAt(Eigen::Vector3f::Zero())}, config);

  const SHCoeffs& sh = r.sh_texture.pixels[0];
  for (int c = 0; c < 3; ++c) {
    const float l = kRadiance * emitter.emissive_factor[c];
    EXPECT_NEAR(sh.coeffs[0][c], 2.0f * std::sqrt(float(M_PI)) * l, 0.03f * l)
        << "channel " << c;
    for (int k = 1; k < 9; ++k) {
      EXPECT_NEAR(sh.coeffs[k][c], 0.0f, 0.08f * l)
          << "coefficient " << k << ", channel " << c;
    }
  }
  // Every first ray hits the box.
  EXPECT_EQ(r.environment_visibility_texture.pixel_data[0], 0.0f);
}

// Indirect light at a probe is the sum of the two hemispheres: the full-sphere
// projection equals the projections of surface points at the same position
// facing +Z and -Z added together, for every coefficient. This checks the
// probe's uniform-sphere rays and their 4 pi weight against the hemisphere
// path. (Visibility has no such identity: a surface point's is cosine-
// weighted and a probe's uniform. BakerVisibilityTest covers both.)
//
// Tolerance: each point is baked kCopies times at different indices, which
// seeds each copy independently, and the tolerance is 5 standard errors of the
// difference of the means, from the copies' own spread.
TEST(BakerFullSphereTest, IndirectProbeIsSumOfHemispheres) {
  Scene scene;
  BuildDefaultsScene(&scene);
  constexpr int kCopies = 16;
  const Eigen::Vector3f pos(0.5f, -0.5f, 2.0f);
  SurfacePoint up = PointAt(pos, {0, 0, 1}, 0);
  SurfacePoint down = PointAt(pos, {0, 0, -1}, 0);
  std::vector<SurfacePoint> points;
  for (int i = 0; i < kCopies; ++i) {
    points.insert(points.end(), {ProbeAt(pos), up, down});
  }
  BakeConfig config = DefaultsConfig(/*indirect_only=*/true);
  config.samples = 512;
  config.confidence_threshold = 0.0f;
  const BakeResult r = BakeWith(scene, points, config);

  // Mean and squared standard error of the mean of `value(copy)`.
  auto stats = [&](int which, auto value) {
    double sum = 0, sum2 = 0;
    for (int i = 0; i < kCopies; ++i) {
      const double v = value(3 * i + which);
      sum += v;
      sum2 += v * v;
    }
    const double mean = sum / kCopies;
    const double var = (sum2 - kCopies * mean * mean) / (kCopies - 1);
    return std::pair<double, double>(mean, std::max(0.0, var) / kCopies);
  };
  auto expect_sum = [&](auto value, double scale, const char* what) {
    const auto [p, sp2] = stats(0, value);
    const auto [u, su2] = stats(1, value);
    const auto [d, sd2] = stats(2, value);
    const double tol = 5.0 * std::sqrt(sp2 + su2 + sd2) + 1e-5;
    EXPECT_NEAR(p, scale * (u + d), tol) << what;
  };
  for (int k = 0; k < 9; ++k) {
    for (int c = 0; c < 3; ++c) {
      expect_sum([&](size_t i) { return r.sh_texture.pixels[i].coeffs[k][c]; },
                 1.0,
                 ("coefficient " + std::to_string(k) + ", channel " +
                  std::to_string(c))
                     .c_str());
    }
  }
  // Not trivially zero: the floor's bounce light reaches the probe.
  EXPECT_GT(
      stats(0, [&](size_t i) { return r.sh_texture.pixels[i].coeffs[0].x(); })
          .first,
      0.05);
}

// Probes appended after the surface points leave every surface point's result
// bit-identical, since each point is seeded by its index, in a full and an
// indirect-only bake. The probes' results are finite.
TEST(BakerFullSphereTest, AppendedProbesLeaveSurfacePointsUnchanged) {
  Scene scene;
  BuildDefaultsScene(&scene);
  const std::vector<SurfacePoint> surface = DefaultsPoints();
  std::vector<SurfacePoint> with_probes = surface;
  with_probes.push_back(ProbeAt(Eigen::Vector3f(0, 0, 2)));
  with_probes.push_back(ProbeAt(Eigen::Vector3f(2.5f, 0, 1)));
  with_probes.push_back(ProbeAt(Eigen::Vector3f(-3, 3, 3)));

  for (bool indirect_only : {false, true}) {
    const BakeResult a =
        BakeWith(scene, surface, DefaultsConfig(indirect_only));
    BakeResult b = BakeWith(scene, with_probes, DefaultsConfig(indirect_only));
    for (size_t i = surface.size(); i < with_probes.size(); ++i) {
      const SHCoeffs& sh = b.sh_texture.pixels[i];
      for (const Eigen::Vector3f& c : sh.coeffs) {
        EXPECT_TRUE(c.allFinite()) << "probe " << i;
      }
      EXPECT_GT(sh.coeffs[0].x(), 0.0f) << "probe " << i;
      EXPECT_TRUE(
          std::isfinite(b.environment_visibility_texture.pixel_data[i]));
    }
    b.sh_texture.pixels.resize(surface.size());
    b.environment_visibility_texture.pixel_data.resize(surface.size());
    ExpectBitIdentical(a, b);
  }
}

// A full-sphere point ignores its tangent: a zero tangent and normal, which
// would give a surface point NaN directions, bake to finite results with
// bounces through the scene.
TEST(BakerFullSphereTest, ZeroTangentGivesNoNaN) {
  Scene scene;
  BuildDefaultsScene(&scene);
  SurfacePoint probe = ProbeAt(Eigen::Vector3f(0.5f, -0.5f, 2.0f));
  ASSERT_EQ(probe.tangent, Eigen::Vector4f::Zero());
  ASSERT_EQ(probe.normal, Eigen::Vector3f::Zero());
  const BakeResult r = BakeWith(scene, {probe}, DefaultsConfig(false));
  const SHCoeffs& sh = r.sh_texture.pixels[0];
  for (const Eigen::Vector3f& c : sh.coeffs) {
    EXPECT_TRUE(c.allFinite());
  }
  EXPECT_TRUE(sh.irradiance.allFinite());
  EXPECT_GT(sh.coeffs[0].x(), 0.0f);
  const float vis = r.environment_visibility_texture.pixel_data[0];
  EXPECT_TRUE(std::isfinite(vis));
  EXPECT_GT(vis, 0.0f);  // The box is open at the sides.
  EXPECT_LT(vis, 1.0f);  // The floor and ceiling block some rays.
}

// A full-sphere probe with a nonzero normal still counts light from every
// direction: it gives the light tree a zero normal, so a light edge-on to or
// below the normal is picked, and the point and spot estimators skip their
// cos_n rejection. With normal +Z, a red point light edge-on at +X, a green
// point light below at -Z and a blue spot below at (0, -1, -1) aimed at the
// probe each add (I / r^2) color Y(w), the full-sphere projection of a delta
// light (the probe is on the spot's axis, where its cone factor is 1).
//
// Tolerance: the only noise is which light each NEE pick chooses; there is no
// geometry. With 512 samples x 64 picks (32768 picks), a channel's coefficient
// has relative sigma sqrt((1 - p) / (p n)), under 0.017 for any pick
// probability p >= 0.1, so 0.1 a |Y_k| is over 5 sigma. Coefficients whose
// basis is 0 in a light's direction are exactly 0 in its channel.
TEST(BakerFullSphereTest, ProbeWithNormalCountsLightsEdgeOnAndBelow) {
  Scene scene;
  scene.materials.push_back(WhiteMaterial());
  const float kIntensity = 50.0f;
  const float kDist = 2.0f;
  const Eigen::Vector3f spot_dir = Eigen::Vector3f(0, -1, -1).normalized();
  struct Placed {
    Light::Type type;
    Eigen::Vector3f w;  // Direction from the probe to the light.
    Eigen::Vector3f color;
  };
  const Placed placed[] = {
      {Light::Type::Point, Eigen::Vector3f::UnitX(), Eigen::Vector3f(1, 0, 0)},
      {Light::Type::Point, -Eigen::Vector3f::UnitZ(), Eigen::Vector3f(0, 1, 0)},
      {Light::Type::Spot, spot_dir, Eigen::Vector3f(0, 0, 1)},
  };
  for (const Placed& p : placed) {
    Light light;
    light.type = p.type;
    light.position = kDist * p.w;
    light.direction = -p.w;  // Spot only: aimed at the probe.
    light.color = p.color;
    light.intensity = kIntensity;
    light.flux = Flux(light);
    scene.lights.push_back(light);
  }
  BakeConfig config;
  config.samples = 512;
  config.bounces = 0;
  config.num_light_samples = 64;
  config.confidence_threshold = 0.0f;
  SurfacePoint probe = ProbeAt(Eigen::Vector3f::Zero());
  probe.normal = Eigen::Vector3f(0, 0, 1);
  const BakeResult r = BakeWith(scene, {probe}, config);

  const float a = kIntensity / (kDist * kDist);
  const SHCoeffs& sh = r.sh_texture.pixels[0];
  for (int c = 0; c < 3; ++c) {
    const std::array<float, 9> y = ShBasis(placed[c].w);
    for (int k = 0; k < 9; ++k) {
      EXPECT_NEAR(sh.coeffs[k][c], a * y[k], 0.1f * a * std::abs(y[k]) + 1e-6f)
          << "coefficient " << k << ", channel " << c;
    }
    EXPECT_GT(sh.coeffs[0][c], 0.0f) << "channel " << c;
  }
}

// An emissive quad below a full-sphere probe's normal adds its full-sphere
// projection, the integral over the quad of L cos_light / d^2 Y(w) dA: the area
// estimator skips its cos_n rejection for the probe. The reference integral is
// a 256 x 256 midpoint sum over the quad.
//
// Tolerance: the light tree picks the only light, so the only noise is where
// on the 0.5 x 0.5 quad, 2 below the probe, each NEE sample lands. Over the
// quad L cos_light / d^2 Y_k(w) varies by under 0.06 L for every k (|w_x|,
// |w_y| <= 0.125), so 256 samples x 16 picks give a sigma under 0.001 L; the
// tolerance is 0.005 L.
TEST(BakerFullSphereTest, ProbeWithNormalCountsAreaLightBelow) {
  Scene scene;
  scene.materials.push_back(BlackMaterial());
  Material emitter = BlackMaterial();
  emitter.name = "emitter";
  const float kRadiance = 4.0f;
  emitter.emissive_factor = Eigen::Vector3f(1.0f, 0.5f, 0.25f);
  emitter.emissive_strength = kRadiance;
  scene.materials.push_back(emitter);
  const float kHalf = 0.25f;
  const float kDepth = -2.0f;
  AddQ3Map2Surface(RectXY(-kHalf, kHalf, -kHalf, kHalf, kDepth,
                          /*faces_up=*/true, /*material_id=*/1),
                   &scene);
  Light area;
  area.type = Light::Type::Area;
  area.geometry = &scene.geometries.back();
  area.material = &scene.materials.back();
  area.color = area.material->emissive_factor;
  area.intensity = area.material->emissive_strength;
  area.area = SurfaceArea(*area.geometry);
  area.flux = Flux(area);
  scene.lights.push_back(area);

  BakeConfig config;
  config.samples = 256;
  config.bounces = 0;
  config.num_light_samples = 16;
  config.confidence_threshold = 0.0f;
  SurfacePoint probe = ProbeAt(Eigen::Vector3f::Zero());
  probe.normal = Eigen::Vector3f(0, 0, 1);
  const BakeResult r = BakeWith(scene, {probe}, config);

  // Projection per unit radiance.
  constexpr int kGrid = 256;
  const double cell = 2.0 * kHalf / kGrid;
  std::array<double, 9> want{};
  for (int i = 0; i < kGrid; ++i) {
    for (int j = 0; j < kGrid; ++j) {
      const Eigen::Vector3f q(-kHalf + (i + 0.5) * cell,
                              -kHalf + (j + 0.5) * cell, kDepth);
      const float d2 = q.squaredNorm();
      const Eigen::Vector3f w = q / std::sqrt(d2);
      const double weight = std::abs(w.z()) / d2 * cell * cell;
      const std::array<float, 9> y = ShBasis(w);
      for (int k = 0; k < 9; ++k) want[k] += weight * y[k];
    }
  }
  const SHCoeffs& sh = r.sh_texture.pixels[0];
  for (int c = 0; c < 3; ++c) {
    const float l = kRadiance * emitter.emissive_factor[c];
    for (int k = 0; k < 9; ++k) {
      EXPECT_NEAR(sh.coeffs[k][c], l * want[k], 0.005f * l)
          << "coefficient " << k << ", channel " << c;
    }
  }
  EXPECT_GT(sh.coeffs[0].x(), 0.01f * kRadiance);
}

// --- Sky radiance on escape ---
//
// Every test here sets confidence_threshold 0 (each point takes exactly
// `samples` rays, so sky radiance can't change the sample count) and
// firefly_clamp 0. "Infinite" floors and ceilings are squares of side 2000
// centred on the point, at most 20 units from it, well inside the first ray's
// tfar of 1e4: the band they miss lies within about 1.2 degrees of the horizon.

namespace {

constexpr float kSqrtPi = 1.7724538509f;
constexpr float kPiF = 3.14159265359f;

// A uniform texture environment of radiance `value`.
Environment UniformSky(const Eigen::Vector3f& value) {
  Environment env;
  env.type = Environment::Type::Texture;
  env.texture.width = 16;
  env.texture.height = 8;
  env.texture.channels = 3;
  env.texture.pixel_data.resize(16 * 8 * 3);
  for (size_t i = 0; i < env.texture.pixel_data.size(); ++i) {
    env.texture.pixel_data[i] = value[i % 3];
  }
  env.sh_coeffs = ProjectEnvironmentToSH(env);
  return env;
}

BakeConfig SkyBake(int samples) {
  BakeConfig config;
  config.samples = samples;
  config.bounces = 1;
  config.confidence_threshold = 0.0f;
  config.firefly_clamp = 0.0f;
  return config;
}

// A square of side 2000 centred on the origin's XY at height z.
Q3Map2Surface BigSquare(float z, bool faces_up, int material_id) {
  return RectXY(-1000, 1000, -1000, 1000, z, faces_up, material_id);
}

// A material whose albedo is the sRGB byte `v` in every channel.
Material GreyMaterial(uint8_t v) {
  Material mat = WhiteMaterial();
  mat.name = "grey";
  mat.albedo.pixel_data = {v, v, v, 255};
  return mat;
}

// A fully transparent (alpha 0) material: an alpha cut-out.
Material CutOutMaterial() {
  Material mat = WhiteMaterial();
  mat.name = "cutout";
  mat.albedo.pixel_data = {255, 255, 255, 0};
  return mat;
}

// A down-facing square ceiling of side 2000 at height z, centred on the
// origin's XY, with a circular hole of radius r at its centre (a 256-gon
// inscribed in the circle).
Q3Map2Surface CeilingWithHole(float z, float r, int material_id) {
  constexpr int kSegments = 256;
  constexpr float kHalf = 1000.0f;
  Q3Map2Surface s;
  for (int k = 0; k < kSegments; ++k) {
    const float a = 2.0f * float(M_PI) * k / kSegments;
    const Eigen::Vector2f d(std::cos(a), std::sin(a));
    // Where the ray from the centre along d meets the square's boundary.
    const float t = kHalf / std::max(std::abs(d.x()), std::abs(d.y()));
    s.positions.push_back(Eigen::Vector3f(r * d.x(), r * d.y(), z));
    s.positions.push_back(Eigen::Vector3f(t * d.x(), t * d.y(), z));
  }
  for (int k = 0; k < kSegments; ++k) {
    const uint32_t i0 = 2 * k, o0 = 2 * k + 1;
    const uint32_t i1 = 2 * ((k + 1) % kSegments), o1 = i1 + 1;
    // Counter-clockwise seen from above: q3map2's front face pointing down,
    // as RectXY winds a down-facing rectangle.
    s.indices.insert(s.indices.end(), {i0, o0, o1, i0, o1, i1});
  }
  s.normals.assign(s.positions.size(), Eigen::Vector3f(0, 0, -1));
  s.texture_uvs.assign(s.positions.size(), Eigen::Vector2f::Zero());
  s.material_id = material_id;
  return s;
}

}  // namespace

// An up-facing point under an opaque ceiling 10 units up, with a circular hole
// above it whose edge is 60 degrees from the normal: the cosine-weighted
// visibility is sin^2(60) = 0.75, where the uniform fraction would be
// 1 - cos(60) = 0.5. The ratio estimator's sigma at 4096 samples is about
// 0.006, so 0.02 is over 3 sigma.
TEST(BakerVisibilityTest, SkylightHoleIsCosineWeighted) {
  const float kHeight = 10.0f;
  const float kRadius = kHeight * std::tan(60.0f * float(M_PI) / 180.0f);
  Scene scene;
  scene.materials.push_back(WhiteMaterial());
  AddQ3Map2Surface(RectXY(-1000, 1000, -1000, 1000, 0, /*faces_up=*/true, 0),
                   &scene);
  AddQ3Map2Surface(CeilingWithHole(kHeight, kRadius, 0), &scene);
  const std::vector<SurfacePoint> points{
      PointAt(Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(0, 0, 1), 0)};

  const BakeResult r = BakeWith(scene, points, SkyBake(4096));
  EXPECT_NEAR(r.environment_visibility_texture.pixel_data[0], 0.75f, 0.02f);
}

// A point with nothing above it has visibility exactly 1, and one inside a
// closed box exactly 0.
TEST(BakerVisibilityTest, FullyOpenAndFullyClosed) {
  Scene open;
  open.materials.push_back(WhiteMaterial());
  AddQ3Map2Surface(RectXY(-1000, 1000, -1000, 1000, 0, /*faces_up=*/true, 0),
                   &open);
  Scene closed;
  closed.materials.push_back(WhiteMaterial());
  AddQ3Map2Surface(
      InwardBox(Eigen::Vector3f(-2, -2, -1), Eigen::Vector3f(2, 2, 3), 0),
      &closed);
  const std::vector<SurfacePoint> points{
      PointAt(Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(0, 0, 1), 0)};

  EXPECT_EQ(BakeWith(open, points, SkyBake(256))
                .environment_visibility_texture.pixel_data[0],
            1.0f);
  EXPECT_EQ(BakeWith(closed, points, SkyBake(256))
                .environment_visibility_texture.pixel_data[0],
            0.0f);
}

// A full-sphere probe 10 units above a floor keeps the uniform fraction of its
// rays that escape: 0.5, plus at most 0.005 for the band below the horizon
// that misses the floor. Sigma at 4096 samples is 0.5 / 64 = 0.008; 0.02 is
// 2.5 sigma, and the bake is deterministic.
TEST(BakerVisibilityTest, ProbeAboveFloorIsUniform) {
  Scene scene;
  scene.materials.push_back(WhiteMaterial());
  AddQ3Map2Surface(RectXY(-1000, 1000, -1000, 1000, 0, /*faces_up=*/true, 0),
                   &scene);
  const std::vector<SurfacePoint> points{ProbeAt(Eigen::Vector3f(0, 0, 10))};

  const BakeResult r = BakeWith(scene, points, SkyBake(4096));
  EXPECT_NEAR(r.environment_visibility_texture.pixel_data[0], 0.5f, 0.02f);
}

// An up-facing point on an open floor sees the uniform sky over its whole
// hemisphere: L0 = L 2 pi Y00 = sqrt(pi) L and irradiance pi L. Every sample's
// L0 is the same, so L0 has no noise; irradiance per sample is 2 pi L cos,
// with sigma pi L / sqrt(3) = 1.81 L, so 1024 samples give sigma 0.057 L.
TEST(BakerSkyTest, OpenFloorUnderUniformSky) {
  const Eigen::Vector3f kL(0.5f, 1.0f, 2.0f);
  Scene scene;
  scene.materials.push_back(WhiteMaterial());
  AddQ3Map2Surface(BigSquare(0, /*faces_up=*/true, 0), &scene);
  scene.environment = UniformSky(kL);
  const std::vector<SurfacePoint> points{
      PointAt(Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(0, 0, 1), 0)};

  const BakeResult on = BakeWith(scene, points, SkyBake(1024));
  const SHCoeffs& sh = on.sh_texture.pixels[0];
  for (int c = 0; c < 3; ++c) {
    EXPECT_NEAR(sh.coeffs[0][c], kSqrtPi * kL[c], 1e-3f * kL[c]) << c;
    EXPECT_NEAR(sh.irradiance[c], kPiF * kL[c], 0.3f * kL[c]) << c;
  }

  BakeConfig indirect = SkyBake(1024);
  indirect.indirect_only = true;
  const BakeResult off = BakeWith(scene, points, indirect);
  EXPECT_EQ(off.environment_visibility_texture.pixel_data[0], 1.0f);
  EXPECT_EQ(off.sh_texture.pixels[0].coeffs[0], Eigen::Vector3f::Zero());
}

// A down-facing point 10 units above a Lambertian floor of albedo a under a
// uniform sky, with no lights: the floor sees the whole sky, so its radiance
// is a L, and the point's L0 is sqrt(pi) a L. Each sample that hits the floor
// gives exactly a L (cosine-sampled bounce, brdf a / pi, cos / pdf = pi), so
// the only error is the band within atan(10 / 1000) of the horizon that
// misses the floor and sees L: at most 1% of the samples, adding at most
// 0.01 (1 - a) / a of L0.
TEST(BakerSkyTest, BouncedSkyLight) {
  const Eigen::Vector3f kL(1.0f, 2.0f, 4.0f);
  const uint8_t kAlbedoByte = 188;
  const float a = SRGBToLinear(kAlbedoByte);  // about 0.5
  Scene scene;
  scene.materials.push_back(GreyMaterial(kAlbedoByte));
  AddQ3Map2Surface(BigSquare(0, /*faces_up=*/true, 0), &scene);
  scene.environment = UniformSky(kL);
  const std::vector<SurfacePoint> points{
      PointAt(Eigen::Vector3f(0, 0, 10), Eigen::Vector3f(0, 0, -1), 0)};

  const BakeResult on = BakeWith(scene, points, SkyBake(1024));
  for (int c = 0; c < 3; ++c) {
    EXPECT_NEAR(on.sh_texture.pixels[0].coeffs[0][c], kSqrtPi * a * kL[c],
                0.02f * kSqrtPi * a * kL[c])
        << c;
  }

  BakeConfig indirect = SkyBake(1024);
  indirect.indirect_only = true;
  const BakeResult bounced = BakeWith(scene, points, indirect);
  for (int c = 0; c < 3; ++c) {
    EXPECT_NEAR(bounced.sh_texture.pixels[0].coeffs[0][c],
                kSqrtPi * a * kL[c], 0.02f * kSqrtPi * a * kL[c])
        << c;
  }
}

// Sky light passes an alpha-0 quad: an up-facing point under one gets the
// same L0 as without it. Both are exact (every sample sees L), so they agree
// to float precision.
TEST(BakerSkyTest, SkyThroughCutOut) {
  const Eigen::Vector3f kL(1.0f, 1.0f, 1.0f);
  Scene open;
  open.materials.push_back(WhiteMaterial());
  open.materials.push_back(CutOutMaterial());
  AddQ3Map2Surface(BigSquare(0, /*faces_up=*/true, 0), &open);
  open.environment = UniformSky(kL);
  Scene covered = open;
  AddQ3Map2Surface(BigSquare(5, /*faces_up=*/false, 1), &covered);
  const std::vector<SurfacePoint> points{
      PointAt(Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(0, 0, 1), 0)};

  const BakeResult a = BakeWith(open, points, SkyBake(256));
  const BakeResult b = BakeWith(covered, points, SkyBake(256));
  for (int c = 0; c < 3; ++c) {
    EXPECT_NEAR(b.sh_texture.pixels[0].coeffs[0][c],
                a.sh_texture.pixels[0].coeffs[0][c], 1e-4f)
        << c;
  }
  EXPECT_NEAR(b.sh_texture.pixels[0].coeffs[0].x(), kSqrtPi, 1e-3f);

  BakeConfig indirect = SkyBake(256);
  indirect.indirect_only = true;
  const BakeResult bounced = BakeWith(covered, points, indirect);
  EXPECT_EQ(bounced.sh_texture.pixels[0].coeffs[0], Eigen::Vector3f::Zero());
}

// A Z-up scene: an environment bright only near its +Y pole, with a frame
// mapping world +Z to +Y, lights a point facing world +Z and not one facing
// world -Z.
TEST(BakerSkyTest, ZUpSceneUsesEnvironmentFrame) {
  Environment env;
  env.type = Environment::Type::Texture;
  env.texture.width = 16;
  env.texture.height = 8;
  env.texture.channels = 3;
  env.texture.pixel_data.assign(16 * 8 * 3, 0.0f);
  // Row 0 (theta < pi / 8 from +Y) is bright.
  for (int i = 0; i < 16 * 3; ++i) env.texture.pixel_data[i] = 10.0f;
  env.to_environment << 1, 0, 0,  //
      0, 0, 1,                    //
      0, -1, 0;
  env.sh_coeffs = ProjectEnvironmentToSH(env);

  Scene scene;
  scene.materials.push_back(WhiteMaterial());
  scene.environment = env;
  const std::vector<SurfacePoint> points{
      PointAt(Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(0, 0, 1), 0),
      PointAt(Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(0, 0, -1), 0)};

  const BakeResult r = BakeWith(scene, points, SkyBake(1024));
  EXPECT_GT(r.sh_texture.pixels[0].coeffs[0].x(), 0.1f);
  EXPECT_GT(r.sh_texture.pixels[0].irradiance.x(), 0.1f);
  EXPECT_EQ(r.sh_texture.pixels[1].coeffs[0], Eigen::Vector3f::Zero());
  EXPECT_EQ(r.sh_texture.pixels[1].irradiance, Eigen::Vector3f::Zero());
}

// Visibility doesn't depend on sky radiance: a full bake of a scene with
// lights and cut-outs gives bit-identical visibility with or without a sky.
TEST(BakerSkyTest, VisibilityIndependentOfSkyRadiance) {
  Scene scene;
  scene.materials.push_back(WhiteMaterial());
  scene.materials.push_back(CutOutMaterial());
  AddQ3Map2Surface(BigSquare(0, /*faces_up=*/true, 0), &scene);
  AddQ3Map2Surface(RectXY(-2, 2, -2, 2, 3, /*faces_up=*/false, 0), &scene);
  AddQ3Map2Surface(RectXY(2, 6, -2, 2, 3, /*faces_up=*/false, 1), &scene);
  scene.lights.push_back(SunFromAbove());
  Scene no_env = scene;
  scene.environment = UniformSky(Eigen::Vector3f(1, 2, 3));
  std::vector<SurfacePoint> points;
  for (int i = 0; i < 8; ++i) {
    points.push_back(PointAt(Eigen::Vector3f(-3.0f + i, 0.5f, 0),
                             Eigen::Vector3f(0, 0, 1), 0));
  }
  points.push_back(ProbeAt(Eigen::Vector3f(0, 0, 1)));

  BakeConfig config = SkyBake(128);
  config.bounces = 3;
  const BakeResult on = BakeWith(scene, points, config);
  const BakeResult none = BakeWith(no_env, points, config);

  const auto& v_on = on.environment_visibility_texture.pixel_data;
  const auto& v_none = none.environment_visibility_texture.pixel_data;
  ASSERT_EQ(v_on.size(), v_none.size());
  EXPECT_EQ(std::memcmp(v_on.data(), v_none.data(), v_on.size() * 4), 0);
  EXPECT_GT(on.sh_texture.pixels[0].coeffs[0].x(),
            none.sh_texture.pixels[0].coeffs[0].x());
}

}  // namespace sh_baker
