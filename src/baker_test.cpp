#include "baker.h"

#include <gtest/gtest.h>

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

}  // namespace sh_baker
