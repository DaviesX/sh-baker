#include "tracer.h"

#include <gtest/gtest.h>

#include "light.h"
#include "occlusion.h"
#include "scene.h"

namespace sh_baker {

class TracerTest : public ::testing::Test {
 protected:
  void SetUp() override { device = rtcNewDevice(nullptr); }

  void TearDown() override { rtcReleaseDevice(device); }

  RTCDevice device;
};

TEST_F(TracerTest, TraceSky) {
  Scene scene;
  // Empty scene
  RTCScene rtc_scene = BuildBVH(scene, device);

  bool sky_hit = false;
  auto on_sky = [&]() { sky_hit = true; };

  TraceConfig config(rtc_scene, scene, /*light_tree=*/nullptr, 1, 1, on_sky);

  std::mt19937 rng(42);
  Ray ray;
  ray.origin = Eigen::Vector3f(0, 0, 0);
  ray.direction = Eigen::Vector3f(0, 0, 1);

  Eigen::Vector3f result = Trace(config, ray, 0, rng);

  EXPECT_EQ(result, Eigen::Vector3f::Zero());
  EXPECT_TRUE(sky_hit);

  rtcReleaseScene(rtc_scene);
}

TEST_F(TracerTest, TraceObjectNoLight) {
  Scene scene;
  Geometry geo;
  // Triangle at Z=2
  geo.vertices = {{-1, -1, 2}, {1, -1, 2}, {0, 1, 2}};
  geo.indices = {0, 1, 2};
  geo.material_id = 0;
  scene.geometries.push_back(geo);

  Material mat;
  mat.name = "default";
  scene.materials.push_back(mat);

  RTCScene rtc_scene = BuildBVH(scene, device);

  bool sky_hit = false;
  auto on_sky = [&]() { sky_hit = true; };

  TraceConfig config(rtc_scene, scene, /*light_tree=*/nullptr, 1, 1, on_sky);

  std::mt19937 rng(42);
  Ray ray;
  ray.origin = Eigen::Vector3f(0, 0, 0);
  ray.direction = Eigen::Vector3f(0, 0, 1);  // Towards Z=2

  // With no lights and default material (black?), it should return black.
  Eigen::Vector3f result = Trace(config, ray, 0, rng);

  EXPECT_EQ(result, Eigen::Vector3f::Zero());
  EXPECT_FALSE(sky_hit);

  rtcReleaseScene(rtc_scene);
}

TEST_F(TracerTest, TraceThroughTransparentSurface) {
  Scene scene;
  Material transparent;
  transparent.albedo.width = transparent.albedo.height = 1;
  transparent.albedo.channels = 4;
  transparent.albedo.pixel_data = {255, 255, 255, 0};
  scene.materials.push_back(transparent);
  Material receiver;
  receiver.albedo.width = receiver.albedo.height = 1;
  receiver.albedo.channels = 4;
  receiver.albedo.pixel_data = {255, 255, 255, 255};
  scene.materials.push_back(receiver);

  Geometry card;
  card.vertices = {{-1, -1, 1}, {0, 1, 1}, {1, -1, 1}};
  card.indices = {0, 1, 2};
  card.material_id = 0;
  scene.geometries.push_back(card);
  Geometry target = card;
  for (auto& vertex : target.vertices) vertex.z() = 2;
  target.material_id = 1;
  scene.geometries.push_back(target);

  RTCScene rtc_scene = BuildBVH(scene, device);
  Light light;
  light.type = Light::Type::Point;
  light.position = Eigen::Vector3f(0, 0, 0);
  light.color = Eigen::Vector3f::Ones();
  light.intensity = 100.0f;
  light.flux = 1000.0f;
  std::vector<Light> lights{light};
  LightTree tree;
  tree.Build(lights);
  TraceConfig config(rtc_scene, scene, &tree, 1, 1, [] {});
  Ray ray;
  ray.origin = Eigen::Vector3f::Zero();
  ray.direction = Eigen::Vector3f(0, 0, 1);
  std::mt19937 rng(42);

  EXPECT_GT(Trace(config, ray, 0, rng).x(), 0.0f);
  rtcReleaseScene(rtc_scene);
}

}  // namespace sh_baker
