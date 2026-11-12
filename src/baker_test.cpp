#include "baker.h"

#include <gtest/gtest.h>

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

}  // namespace sh_baker
