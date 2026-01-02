#include "baker.h"

#include <gtest/gtest.h>

#include "sh_coeffs.h"

namespace sh_baker {

TEST(BakerTest, BakeSimpleQuad) {
  // Scene: A simple quad at Z=0, -1..1 xy range. Normal +Z.
  // Light: Directional light pointing -Z (towards the quad).

  Scene scene;

  Geometry quad;
  quad.vertices = {{-1, -1, 0}, {1, -1, 0}, {1, 1, 0}, {-1, 1, 0}};
  quad.normals = {{0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1}};
  quad.uvs = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
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

  // Sky: Black
  scene.sky.sun_intensity = 0.0f;
  scene.sky.sun_color = Eigen::Vector3f::Zero();

  // Light: Directional from +Z
  // In our baker logic (Hit Sky or Emission), explicit lights in list are
  // ignored currently. We rely on Sky or Emissive Surfaces. Let's make the Sky
  // bright from +Z direction to simulate the light.
  scene.sky.sun_intensity = 1.0f;
  scene.sky.sun_color = Eigen::Vector3f(1.0f, 1.0f, 1.0f);
  scene.sky.sun_direction = Eigen::Vector3f(0, 0, 1);

  BakeConfig config;
  config.width = 4;
  config.height = 4;
  config.samples = 64;
  config.bounces = 0;  // Direct light only

  SHTexture output = BakeSHLightMap(scene, config);

  // Check center pixel
  // Index 5 (1,1) or similar.
  // We expect non-zero result.
  SHCoeffs result = output.pixels[0];
  EXPECT_GT(result.coeffs[0].x(), 0.1f);
}

}  // namespace sh_baker
