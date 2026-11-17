#include "baker.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <functional>
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
  // The first poll comes before any ray-tracing setup, and none follows it.
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
  // All invalid. Calls 1 and 2 are the pre-setup and post-build polls; call 3
  // is the first point's.
  ProgressFixture f = FloorPoints(10, /*invalid_every=*/1);
  ProgressLog log;
  int calls = 0;
  BakeConfig config = SmallBake();
  config.progress_fn = log.Callback();
  config.cancel_fn = [&] { return ++calls >= 3; };
  BakeResult result = BakeWith(f.scene, f.points, config);
  EXPECT_TRUE(result.cancelled);
  EXPECT_EQ(calls, 3);
  EXPECT_TRUE(log.reports.empty());
}

TEST(BakerCancelTest, CancelledAfterBuildSkipsLoop) {
  ProgressFixture f = FloorPoints(100);
  int calls = 0;
  BakeConfig config = SmallBake();
  config.cancel_fn = [&] { return ++calls >= 2; };
  BakeResult result = BakeWith(f.scene, f.points, config);
  EXPECT_TRUE(result.cancelled);
  EXPECT_EQ(calls, 2);
  for (size_t i = 0; i < 100; ++i) EXPECT_TRUE(IsMarker(result, i)) << i;
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

}  // namespace sh_baker
