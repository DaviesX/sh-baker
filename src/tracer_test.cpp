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

namespace {

// A uniform texture environment of radiance `value`.
Environment UniformEnvironment(const Eigen::Vector3f& value) {
  Environment env;
  env.type = Environment::Type::Texture;
  env.texture.width = 8;
  env.texture.height = 4;
  env.texture.channels = 3;
  env.texture.pixel_data.resize(8 * 4 * 3);
  for (size_t i = 0; i < env.texture.pixel_data.size(); ++i) {
    env.texture.pixel_data[i] = value[i % 3];
  }
  return env;
}

}  // namespace

TEST_F(TracerTest, TraceSkyReturnsEnvironmentRadiance) {
  Scene scene;
  RTCScene rtc_scene = BuildBVH(scene, device);
  scene.environment = UniformEnvironment(Eigen::Vector3f(0.5f, 1.0f, 2.0f));

  bool sky_hit = false;
  TraceConfig config(
      rtc_scene, scene, /*light_tree=*/nullptr, 1, 1, [&]() { sky_hit = true; });
  std::mt19937 rng(42);
  Ray ray;
  ray.origin = Eigen::Vector3f(0, 0, 0);
  ray.direction = Eigen::Vector3f(0.6f, 0.0f, 0.8f);

  Eigen::Vector3f result = Trace(config, ray, 0, rng);

  EXPECT_TRUE(result.isApprox(Eigen::Vector3f(0.5f, 1.0f, 2.0f), 1e-6f))
      << result.transpose();
  EXPECT_TRUE(sky_hit);
  rtcReleaseScene(rtc_scene);
}

TEST_F(TracerTest, TraceSkyWithoutEnvironmentReturnsZero) {
  Scene scene;
  RTCScene rtc_scene = BuildBVH(scene, device);

  TraceConfig config(rtc_scene, scene, /*light_tree=*/nullptr, 1, 1, [] {});
  std::mt19937 rng(42);
  Ray ray;
  ray.origin = Eigen::Vector3f(0, 0, 0);
  ray.direction = Eigen::Vector3f(0, 0, 1);

  EXPECT_EQ(Trace(config, ray, 0, rng), Eigen::Vector3f::Zero());
  rtcReleaseScene(rtc_scene);
}

TEST_F(TracerTest, TraceSkyThroughCutOut) {
  Scene scene;
  Material transparent;
  transparent.albedo.width = transparent.albedo.height = 1;
  transparent.albedo.channels = 4;
  transparent.albedo.pixel_data = {255, 255, 255, 0};
  scene.materials.push_back(transparent);

  Geometry card;
  card.vertices = {{-1, -1, 1}, {0, 1, 1}, {1, -1, 1}};
  card.indices = {0, 1, 2};
  card.material_id = 0;
  scene.geometries.push_back(card);
  RTCScene rtc_scene = BuildBVH(scene, device);

  scene.environment = UniformEnvironment(Eigen::Vector3f(3.0f, 2.0f, 1.0f));
  bool sky_hit = false;
  TraceConfig config(
      rtc_scene, scene, /*light_tree=*/nullptr, 1, 1, [&]() { sky_hit = true; });
  std::mt19937 rng(42);
  Ray ray;
  ray.origin = Eigen::Vector3f::Zero();
  ray.direction = Eigen::Vector3f(0, 0, 1);

  Eigen::Vector3f result = Trace(config, ray, 0, rng);

  EXPECT_TRUE(result.isApprox(Eigen::Vector3f(3.0f, 2.0f, 1.0f), 1e-6f))
      << result.transpose();
  // The cut-out still blocks the first ray for visibility.
  EXPECT_FALSE(sky_hit);
  rtcReleaseScene(rtc_scene);
}

}  // namespace sh_baker
