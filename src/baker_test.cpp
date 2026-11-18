#include "baker.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iterator>
#include <thread>
#include <utility>
#include <vector>

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

// --- Split punctual-direct result ---

// A point at `position` with unit `normal` and a tangent perpendicular to it.
SurfacePoint PointFacing(const Eigen::Vector3f& position,
                         const Eigen::Vector3f& normal, int material_id = 0) {
  SurfacePoint sp;
  sp.position = position;
  sp.normal = normal.normalized();
  Eigen::Vector3f t = sp.normal.unitOrthogonal();
  sp.tangent = Eigen::Vector4f(t.x(), t.y(), t.z(), 1.0f);
  sp.material_id = material_id;
  return sp;
}

// A split bake of up to `samples` samples per point. The confidence
// threshold is 0, so adaptive sampling stops only on the absolute noise floor
// and the sample count really is `samples` for lit points.
BakeConfig SplitConfig(int samples) {
  BakeConfig config;
  config.samples = samples;
  config.bounces = 1;
  config.confidence_threshold = 0.0f;
  config.split_punctual_direct = true;
  return config;
}

Eigen::Vector3f PunctualIrradiance(const BakeResult& r, size_t i) {
  const std::vector<float>& d = r.punctual_direct_irradiance.pixel_data;
  return Eigen::Vector3f(d[i * 3 + 0], d[i * 3 + 1], d[i * 3 + 2]);
}

uint32_t Bits(float f) {
  uint32_t b;
  std::memcpy(&b, &f, 4);
  return b;
}

// A floor, a casting point light, a sun and an emissive quad.
struct SplitFixture {
  Scene scene;
  std::vector<SurfacePoint> points;
};

SplitFixture PointSunAndEmitter() {
  SplitFixture f;
  Scene& scene = f.scene;
  scene.materials.push_back(WhiteMaterial());
  Material glow = WhiteMaterial();
  glow.name = "glow";
  glow.emissive_factor = Eigen::Vector3f(1.0f, 0.7f, 0.4f);
  glow.emissive_strength = 4.0f;
  scene.materials.push_back(glow);
  AddQ3Map2Surface(RectXY(-4, 4, -4, 4, 0, /*faces_up=*/true, 0), &scene);
  AddQ3Map2Surface(RectXY(-4, -1, -4, 4, 3, /*faces_up=*/false, 0), &scene);
  AddQ3Map2Surface(RectXY(1, 2, 1, 2, 2, /*faces_up=*/false, 1), &scene);
  const Geometry& emitter = scene.geometries.back();

  Light point;
  point.type = Light::Type::Point;
  point.position = Eigen::Vector3f(-1.5f, -1.0f, 2.0f);
  point.color = Eigen::Vector3f(0.7f, 0.85f, 1.0f);
  point.intensity = 6.0f;
  point.flux = Flux(point);
  scene.lights.push_back(point);

  Light sun;
  sun.type = Light::Type::Directional;
  sun.direction = Eigen::Vector3f(-0.4f, 0.3f, -1.0f).normalized();
  sun.intensity = 1.5f;
  sun.flux = Flux(sun);
  scene.lights.push_back(sun);

  Light area;
  area.type = Light::Type::Area;
  area.geometry = &emitter;
  area.material = &scene.materials[1];
  area.area = SurfaceArea(emitter);
  area.flux = Flux(area);
  scene.lights.push_back(area);

  const Eigen::Vector3f up(0, 0, 1);
  f.points = {PointFacing(Eigen::Vector3f(0, 0, 0), up),
              PointFacing(Eigen::Vector3f(1.5f, 1.5f, 0), up),
              PointFacing(Eigen::Vector3f(-2, -2, 0), up),
              PointFacing(Eigen::Vector3f(-3, 2, 0), up),
              PointFacing(Eigen::Vector3f(-2, 0, 3), -up)};
  return f;
}

}  // namespace

// Main result plus the punctual SH equals a normal bake within its noise.
TEST(BakerSplitTest, SplitPlusPunctualEqualsFullBake) {
  SplitFixture f = PointSunAndEmitter();
  BakeConfig full_config = SplitConfig(1024);
  full_config.split_punctual_direct = false;
  const BakeResult full = BakeWith(f.scene, f.points, full_config);
  const BakeResult split = BakeWith(f.scene, f.points, SplitConfig(1024));
  EXPECT_TRUE(full.punctual_direct_sh_texture.pixels.empty());
  ASSERT_EQ(split.punctual_direct_sh_texture.pixels.size(), f.points.size());
  for (size_t i = 0; i < f.points.size(); ++i) {
    const Eigen::Vector3f expected = full.sh_texture.pixels[i].coeffs[0];
    const Eigen::Vector3f sum =
        split.sh_texture.pixels[i].coeffs[0] +
        split.punctual_direct_sh_texture.pixels[i].coeffs[0];
    for (int c = 0; c < 3; ++c) {
      ASSERT_GT(expected[c], 0.0f) << "point " << i;
      EXPECT_LT(std::abs(sum[c] - expected[c]) / expected[c], 0.05f)
          << "point " << i << " channel " << c << ": " << sum[c] << " vs "
          << expected[c];
    }
    // The main result really lost the punctual direct light.
    EXPECT_LT(split.sh_texture.pixels[i].coeffs[0].x(), expected.x())
        << "point " << i;
  }
}

TEST(BakerSplitTest, ExactPointLightIrradiance) {
  Scene scene;
  scene.materials.push_back(WhiteMaterial());
  AddQ3Map2Surface(RectXY(-4, 4, -4, 4, 0, /*faces_up=*/true, 0), &scene);
  Light point;
  point.type = Light::Type::Point;
  point.position = Eigen::Vector3f(0.5f, -0.25f, 3.0f);
  point.intensity = 7.0f;
  point.flux = Flux(point);
  scene.lights.push_back(point);
  const std::vector<SurfacePoint> points = {
      PointFacing(Eigen::Vector3f(0.5f, -0.25f, 0), Eigen::Vector3f(0, 0, 1))};

  const BakeResult r = BakeWith(scene, points, SplitConfig(16));
  const float expected = 7.0f / (3.0f * 3.0f);
  const Eigen::Vector3f e = PunctualIrradiance(r, 0);
  for (int c = 0; c < 3; ++c) {
    EXPECT_NEAR(e[c], expected, 1e-5f * expected) << c;
    // Straight overhead: the SH holds E_perp * Y(+Z).
    EXPECT_NEAR(r.punctual_direct_sh_texture.pixels[0].coeffs[0][c],
                expected * 0.282095f, 1e-5f * expected);
    EXPECT_NEAR(r.punctual_direct_sh_texture.pixels[0].coeffs[2][c],
                expected * 0.488603f, 1e-5f * expected);
  }
  // The floor bakes no punctual light into its main result: bounces off the
  // open floor escape, and there is no area light.
  EXPECT_EQ(r.sh_texture.pixels[0].coeffs[0], Eigen::Vector3f::Zero());
}

TEST(BakerSplitTest, PunctualDirectIsDeterministic) {
  SplitFixture f = PointSunAndEmitter();
  const BakeResult few = BakeWith(f.scene, f.points, SplitConfig(16));
  const BakeResult many = BakeWith(f.scene, f.points, SplitConfig(1024));
  ASSERT_EQ(few.punctual_direct_irradiance.pixel_data.size(),
            many.punctual_direct_irradiance.pixel_data.size());
  for (size_t i = 0; i < few.punctual_direct_irradiance.pixel_data.size();
       ++i) {
    EXPECT_EQ(Bits(few.punctual_direct_irradiance.pixel_data[i]),
              Bits(many.punctual_direct_irradiance.pixel_data[i]))
        << i;
  }
  for (size_t i = 0; i < f.points.size(); ++i) {
    EXPECT_GT(PunctualIrradiance(few, i).x(), 0.0f) << i;
    for (int k = 0; k < 9; ++k) {
      for (int c = 0; c < 3; ++c) {
        EXPECT_EQ(Bits(few.punctual_direct_sh_texture.pixels[i].coeffs[k][c]),
                  Bits(many.punctual_direct_sh_texture.pixels[i].coeffs[k][c]))
            << "point " << i << " coeff " << k;
      }
    }
  }
}

TEST(BakerSplitTest, SkippedPointsKeepTheMarker) {
  SplitFixture f = PointSunAndEmitter();
  f.points[1].material_id = -1;
  const BakeResult r = BakeWith(f.scene, f.points, SplitConfig(16));
  ASSERT_EQ(r.punctual_direct_irradiance.channels, 3u);
  ASSERT_EQ(r.punctual_direct_irradiance.pixel_data.size(),
            3 * f.points.size());
  for (int k = 0; k < 9; ++k) {
    EXPECT_EQ(r.sh_texture.pixels[1].coeffs[k], Eigen::Vector3f::Constant(-1));
    EXPECT_EQ(r.punctual_direct_sh_texture.pixels[1].coeffs[k],
              Eigen::Vector3f::Constant(-1));
  }
  EXPECT_EQ(PunctualIrradiance(r, 1), Eigen::Vector3f::Constant(-1));
  EXPECT_GT(PunctualIrradiance(r, 0).x(), 0.0f);
}

// A cancelled bake leaves unbaked points with the marker in the split results
// too.
TEST(BakerSplitTest, CancelledPointsKeepTheMarker) {
  SplitFixture f = PointSunAndEmitter();
  BakeConfig config = SplitConfig(16);
  config.cancel_fn = [] { return true; };
  const BakeResult r = BakeWith(f.scene, f.points, config);
  EXPECT_TRUE(r.cancelled);
  for (size_t i = 0; i < f.points.size(); ++i) {
    EXPECT_EQ(r.punctual_direct_sh_texture.pixels[i].coeffs[0],
              Eigen::Vector3f::Constant(-1));
    EXPECT_EQ(PunctualIrradiance(r, i), Eigen::Vector3f::Constant(-1));
  }
}

// A smoothstep spot's punctual irradiance at t = 0.25 is 0.15625 of its
// on-axis value at the same distance and incidence.
TEST(BakerSplitTest, SmoothstepMidpoint) {
  Scene scene;
  scene.materials.push_back(WhiteMaterial());
  Light spot;
  spot.type = Light::Type::Spot;
  spot.position = Eigen::Vector3f(0, 0, 0);
  spot.direction = Eigen::Vector3f(1, 0, 0);
  spot.intensity = 50.0f;
  spot.cos_outer_cone = std::cos(45.0f * float(M_PI) / 180.0f);
  spot.cos_inner_cone = std::cos(30.0f * float(M_PI) / 180.0f);
  spot.cone_falloff = Light::ConeFalloff::kSmoothstep;
  spot.flux = Flux(spot);
  scene.lights.push_back(spot);
  const float theta =
      std::acos(spot.cos_outer_cone +
                0.25f * (spot.cos_inner_cone - spot.cos_outer_cone));
  const Eigen::Vector3f axis(1, 0, 0);
  const Eigen::Vector3f off(std::cos(theta), std::sin(theta), 0);
  const std::vector<SurfacePoint> points = {PointFacing(4.0f * axis, -axis),
                                            PointFacing(4.0f * off, -off)};
  const BakeResult r = BakeWith(scene, points, SplitConfig(16));
  const float on_axis = PunctualIrradiance(r, 0).x();
  ASSERT_GT(on_axis, 0.0f);
  EXPECT_NEAR(PunctualIrradiance(r, 1).x() / on_axis, 0.15625f,
              1e-4f * 0.15625f);
}

// Punctual irradiance through a blocker: 0 (opaque) or 1 - 128/255 (card)
// of the open value for a casting light, and the open value for a light that
// casts no shadows.
TEST(BakerSplitTest, NonCastingLightIgnoresBlockers) {
  auto irradiance = [](int blocker_alpha, bool casts) {
    Scene scene;
    scene.materials.push_back(WhiteMaterial());
    Material card = WhiteMaterial();
    card.albedo.pixel_data[3] =
        static_cast<uint8_t>(blocker_alpha < 0 ? 255 : blocker_alpha);
    scene.materials.push_back(card);
    AddQ3Map2Surface(RectXY(-4, 4, -4, 4, 0, /*faces_up=*/true, 0), &scene);
    if (blocker_alpha >= 0) {
      AddQ3Map2Surface(RectXY(-1, 1, -1, 1, 1.5f, /*faces_up=*/true, 1),
                       &scene);
    }
    Light point;
    point.type = Light::Type::Point;
    point.position = Eigen::Vector3f(0, 0, 3);
    point.intensity = 9.0f;
    point.flux = Flux(point);
    point.casts_shadows = casts;
    scene.lights.push_back(point);
    const std::vector<SurfacePoint> points = {
        PointFacing(Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(0, 0, 1))};
    return PunctualIrradiance(BakeWith(scene, points, SplitConfig(16)), 0).x();
  };
  const float open = irradiance(-1, true);
  ASSERT_NEAR(open, 1.0f, 1e-5f);
  EXPECT_EQ(irradiance(255, true), 0.0f);
  EXPECT_EQ(irradiance(255, false), open);
  EXPECT_NEAR(irradiance(128, true), (1.0f - 128.0f / 255.0f) * open, 1e-6f);
  EXPECT_EQ(irradiance(128, false), open);
}

namespace {

// A fixed scene that exercises every light type, alpha transmittance, bounces
// and a skipped point. Its results before the editor-light-model change are
// recorded in kDefaultsReference*.
struct DefaultsFixture {
  Scene scene;
  std::vector<SurfacePoint> points;
};

DefaultsFixture DefaultsScene() {
  DefaultsFixture f;
  Scene& scene = f.scene;
  scene.materials.push_back(WhiteMaterial());  // 0: receiver
  Material card = WhiteMaterial();             // 1: half-transparent card
  card.name = "card";
  card.albedo.pixel_data = {200, 160, 120, 128};
  scene.materials.push_back(card);
  Material glow = WhiteMaterial();  // 2: emitter
  glow.name = "glow";
  glow.emissive_factor = Eigen::Vector3f(1.0f, 0.8f, 0.5f);
  glow.emissive_strength = 5.0f;
  scene.materials.push_back(glow);

  AddQ3Map2Surface(RectXY(-4, 4, -4, 4, 0, /*faces_up=*/true, 0), &scene);
  AddQ3Map2Surface(RectXY(-4, 0, -4, 4, 3, /*faces_up=*/false, 0), &scene);
  AddQ3Map2Surface(RectXY(-1, 1, -1, 1, 1.5f, /*faces_up=*/true, 1), &scene);
  AddQ3Map2Surface(RectXY(2, 3, 2, 3, 2.5f, /*faces_up=*/false, 2), &scene);
  const Geometry& emitter = scene.geometries.back();

  Light sun;
  sun.type = Light::Type::Directional;
  sun.direction = Eigen::Vector3f(0.3f, 0.2f, -1.0f).normalized();
  sun.color = Eigen::Vector3f(1.0f, 0.9f, 0.8f);
  sun.intensity = 2.0f;
  sun.flux = Flux(sun);
  scene.lights.push_back(sun);

  Light point;
  point.type = Light::Type::Point;
  point.position = Eigen::Vector3f(-2, 1, 2);
  point.color = Eigen::Vector3f(0.6f, 0.8f, 1.0f);
  point.intensity = 5.0f;
  point.flux = Flux(point);
  scene.lights.push_back(point);

  Light spot;
  spot.type = Light::Type::Spot;
  spot.position = Eigen::Vector3f(1, -1, 2.8f);
  spot.direction = Eigen::Vector3f(0, 0, -1);
  spot.intensity = 8.0f;
  spot.cos_inner_cone = std::cos(20.0f * float(M_PI) / 180.0f);
  spot.cos_outer_cone = std::cos(35.0f * float(M_PI) / 180.0f);
  spot.flux = Flux(spot);
  scene.lights.push_back(spot);

  Light area;
  area.type = Light::Type::Area;
  area.geometry = &emitter;
  area.material = &scene.materials[2];
  area.area = SurfaceArea(emitter);
  area.flux = Flux(area);
  scene.lights.push_back(area);

  const Eigen::Vector3f up(0, 0, 1), down(0, 0, -1);
  f.points = {
      PointAt(Eigen::Vector3f(0, 0, 0), up, 0),
      PointAt(Eigen::Vector3f(1, -1, 0), up, 0),
      PointAt(Eigen::Vector3f(-2.5f, 1.5f, 0), up, 0),
      PointAt(Eigen::Vector3f(3, 3, 0), up, -1),
      PointAt(Eigen::Vector3f(2.5f, 2.5f, 0), up, 0),
      PointAt(Eigen::Vector3f(-2, -2, 3), down, 0),
  };
  return f;
}

BakeConfig DefaultsConfig(bool indirect_only) {
  BakeConfig config;
  config.samples = 32;
  config.bounces = 2;
  config.indirect_only = indirect_only;
  return config;
}

// DefaultsScene()'s SH (9 RGB coefficients) and environment visibility per
// point, as bit patterns, baked by sh-baker at ce5419a, before Light::range,
// cone_falloff, casts_shadows and split_punctual_direct existed. Regenerate
// only from a build of that commit.
constexpr uint32_t kDefaultsReferenceFull[] = {
    // point 0
    0x3ebfacd1,
    0x3ede4c1e,
    0x3efcea1a,
    0xbd2869e9,
    0xbbe839e0,
    0x3cdcc7fe,
    0x3f015f8c,
    0x3f13d6b0,
    0x3f264d3f,
    0xbe3c964d,
    0xbe833bdd,
    0xbea82b0c,
    0xbe0ed2a1,
    0xbe3b9550,
    0xbe685b57,
    0xbddeb472,
    0xbd60b1a6,
    0xba7c0ce0,
    0x3eb5760f,
    0x3ec53604,
    0x3ed4f631,
    0xbe8a64f3,
    0xbec46953,
    0xbefe6bda,
    0x3dc3373d,
    0x3e11da29,
    0x3e42176a,
    0x3f300000,
    // point 1
    0x3f779465,
    0x3f648844,
    0x3f50f69e,
    0xbde7275c,
    0xbdc27cf8,
    0xbda89e1a,
    0x3fc03501,
    0x3fb045f8,
    0x3fa04ab4,
    0xbee3607e,
    0xbed9f4a4,
    0xbed2343e,
    0x3dc42cd5,
    0x3d7b5394,
    0x3cbbe9c9,
    0xbeb1e9ba,
    0xbe9d1e8e,
    0xbe8b3f22,
    0x3fca826b,
    0x3fb7be00,
    0x3fa58e55,
    0xbf617e02,
    0xbf52b246,
    0xbf452a6a,
    0x3d1fbb12,
    0x3d4cadda,
    0x3d86fca5,
    0x3f300000,
    // point 2
    0x3ed3f160,
    0x3efa7d94,
    0x3f0d827e,
    0xbda56501,
    0xbe01a234,
    0xbe3433b3,
    0x3f116c8f,
    0x3f357a84,
    0x3f573455,
    0x3e8d6c3b,
    0x3e8c4316,
    0x3e820b7e,
    0x3c3cb396,
    0xbcb0824e,
    0xbd7c06a4,
    0xbe4e94e6,
    0xbe935a19,
    0xbec13805,
    0x3ef55ea1,
    0x3f2a9dc1,
    0x3f5bddf2,
    0x3ec0567d,
    0x3ed0d8b7,
    0x3ed86df2,
    0x3e1ca83a,
    0x3dec12f6,
    0x3d7989da,
    0x3f480000,
    // point 3
    0xbf800000,
    0xbf800000,
    0xbf800000,
    0xbf800000,
    0xbf800000,
    0xbf800000,
    0xbf800000,
    0xbf800000,
    0xbf800000,
    0xbf800000,
    0xbf800000,
    0xbf800000,
    0xbf800000,
    0xbf800000,
    0xbf800000,
    0xbf800000,
    0xbf800000,
    0xbf800000,
    0xbf800000,
    0xbf800000,
    0xbf800000,
    0xbf800000,
    0xbf800000,
    0xbf800000,
    0xbf800000,
    0xbf800000,
    0xbf800000,
    0xbf800000,
    // point 4
    0x3f5cb3b1,
    0x3f40f831,
    0x3f1c2309,
    0xbe337e74,
    0xbe27ea80,
    0xbe1ced08,
    0x3fb2929a,
    0x3f9a6104,
    0x3f74c406,
    0xbe8ab039,
    0xbe87016c,
    0xbe841d43,
    0x3e13b78d,
    0x3e0f3fe1,
    0x3e0add4a,
    0xbea8e368,
    0xbe9b2695,
    0xbe8e0f9c,
    0x3fcd1d47,
    0x3fae5207,
    0x3f85a74b,
    0xbf002c13,
    0xbef2a10c,
    0xbee6af88,
    0x3d8ec1dd,
    0x3d9cdda1,
    0x3dab8bfc,
    0x3f600000,
    // point 5
    0x3e405484,
    0x3e7c16ec,
    0x3e9b88ff,
    0x3e093a95,
    0x3e32becb,
    0x3e5b81b9,
    0xbe40a2b2,
    0xbe7e6881,
    0xbe9de207,
    0xbd3ab750,
    0xbd843ca0,
    0xbdac8258,
    0x3cb20bf8,
    0x3cb8abd6,
    0x3cb5baab,
    0xbd8343ef,
    0xbdab9212,
    0xbdd356dc,
    0x3d454f26,
    0x3d872d45,
    0x3dac67f6,
    0x3dc6c005,
    0x3e05de7d,
    0x3e28931f,
    0xbe229388,
    0xbe5855ba,
    0xbe870292,
    0x3f400000,
};
constexpr uint32_t kDefaultsReferenceIndirectOnly[] = {
    // point 0
    0x3df0dee6,
    0x3e07e9e7,
    0x3e12cf00,
    0x3d85537d,
    0x3d7fd0e7,
    0x3d62353e,
    0x3e1158d0,
    0x3e28a9b1,
    0x3e3b55aa,
    0xbd071e48,
    0xbd88ad39,
    0xbdd68eb5,
    0x3cfd8a2a,
    0x3c381790,
    0xbc6706ea,
    0x3dc05892,
    0x3dbf4323,
    0x3db1e001,
    0x3d821415,
    0x3dab1ade,
    0x3dd3da1e,
    0xbd944740,
    0xbe015bd4,
    0xbe3e5379,
    0x3d3211e7,
    0x3d75e834,
    0x3d9da969,
    0x3f380000,
    // point 1
    0x3d85fea3,
    0x3d8aa7c6,
    0x3d87d0b1,
    0x3d7b8734,
    0x3d63c197,
    0x3d37ce75,
    0x3d8f8740,
    0x3d995c2f,
    0x3d9c1095,
    0xbc2321f5,
    0xbcc079e5,
    0xbd1f8ab4,
    0x3c784f25,
    0x3b264ef8,
    0xbc5b6fc3,
    0x3d9ecd80,
    0x3d91d01b,
    0x3d70ecb4,
    0x3c76dd86,
    0x3cb579e7,
    0x3cf2b037,
    0xbca14637,
    0xbd1ec302,
    0xbd767eb1,
    0xbcb1e0c9,
    0xbc1e9496,
    0x3bc399d4,
    0x3f400000,
    // point 2
    0x3e08a30b,
    0x3e2bb533,
    0x3e4cd092,
    0xbd47f547,
    0xbd8beefa,
    0xbdb52175,
    0x3e368d88,
    0x3e6b662c,
    0x3e8f5f4b,
    0x3d41d824,
    0x3d42d363,
    0x3d37e039,
    0xbc0d9924,
    0xbc922dfd,
    0xbce744e0,
    0xbdc0bb2a,
    0xbe03c3cc,
    0xbe27c754,
    0x3e00e7fa,
    0x3e30bc11,
    0x3e617972,
    0x3d69673c,
    0x3d787611,
    0x3d7bc138,
    0x3c723b27,
    0x3baa062f,
    0xbbe9a66b,
    0x3f280000,
    // point 3
    0xbf800000,
    0xbf800000,
    0xbf800000,
    0xbf800000,
    0xbf800000,
    0xbf800000,
    0xbf800000,
    0xbf800000,
    0xbf800000,
    0xbf800000,
    0xbf800000,
    0xbf800000,
    0xbf800000,
    0xbf800000,
    0xbf800000,
    0xbf800000,
    0xbf800000,
    0xbf800000,
    0xbf800000,
    0xbf800000,
    0xbf800000,
    0xbf800000,
    0xbf800000,
    0xbf800000,
    0xbf800000,
    0xbf800000,
    0xbf800000,
    0xbf800000,
    // point 4
    0x3e638fd3,
    0x3e390a33,
    0x3df0a512,
    0xbbc0d5d7,
    0xbb8b64ce,
    0xbaff84f2,
    0x3ec00e68,
    0x3e9aea7f,
    0x3e459dee,
    0xbbeed0bd,
    0xbc2278e5,
    0xbc4e25f9,
    0xbb28bddc,
    0xbb503831,
    0xbb748d9a,
    0xbc5f984e,
    0xbc2835de,
    0xbbb0eee1,
    0x3eed87de,
    0x3ebd9235,
    0x3e6b8c0e,
    0xbbf25378,
    0xbc2b240b,
    0xbc5eeaab,
    0x3ba4019e,
    0x3beba87c,
    0x3c1b3f44,
    0x3f680000,
    // point 5
    0x3e9b0a50,
    0x3ea7887c,
    0x3eb54f4c,
    0xbde8e543,
    0xbdf10598,
    0xbdfc0ec0,
    0xbedc123e,
    0xbeef36e4,
    0xbf022c72,
    0x3d9145a6,
    0x3d876a44,
    0x3d82e55c,
    0x3bb44fec,
    0x3c82c501,
    0x3cc9221e,
    0x3e62d904,
    0x3e6ae4f8,
    0x3e74d713,
    0x3eb4f21f,
    0x3ec762eb,
    0x3edb893c,
    0xbdc6a4dd,
    0xbdb2e5d7,
    0xbda9f207,
    0x3d8497bc,
    0x3d8dec80,
    0x3d9baa8a,
    0x3f300000,
};

void ExpectMatchesReference(const BakeResult& r, const uint32_t* reference,
                            size_t reference_size) {
  ASSERT_EQ(reference_size, r.sh_texture.pixels.size() * 28);
  for (size_t i = 0; i < r.sh_texture.pixels.size(); ++i) {
    const uint32_t* expected = reference + i * 28;
    for (int k = 0; k < 9; ++k) {
      for (int c = 0; c < 3; ++c) {
        EXPECT_EQ(Bits(r.sh_texture.pixels[i].coeffs[k][c]),
                  expected[k * 3 + c])
            << "point " << i << " coeff " << k << " channel " << c;
      }
    }
    EXPECT_EQ(Bits(r.environment_visibility_texture.pixel_data[i]),
              expected[27])
        << "point " << i;
  }
}

}  // namespace

// With every new field at its default, a bake gives the same bits as before
// the editor-light-model change, for a normal and an indirect-only bake.
TEST(BakerDefaultsTest, DefaultsKeepExistingOutput) {
  for (bool indirect_only : {false, true}) {
    const DefaultsFixture f = DefaultsScene();
    const BakeResult r =
        BakeWith(f.scene, f.points, DefaultsConfig(indirect_only));
    EXPECT_TRUE(r.punctual_direct_sh_texture.pixels.empty());
    EXPECT_TRUE(r.punctual_direct_irradiance.pixel_data.empty());
    if (indirect_only) {
      ExpectMatchesReference(r, kDefaultsReferenceIndirectOnly,
                             std::size(kDefaultsReferenceIndirectOnly));
    } else {
      ExpectMatchesReference(r, kDefaultsReferenceFull,
                             std::size(kDefaultsReferenceFull));
    }
  }
}

}  // namespace sh_baker
