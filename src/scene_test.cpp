#include "scene.h"

#include <embree4/rtcore.h>
#include <gtest/gtest.h>

namespace sh_baker {

class SceneTest : public ::testing::Test {
 protected:
  void SetUp() override { device = rtcNewDevice(nullptr); }

  void TearDown() override { rtcReleaseDevice(device); }

  RTCDevice device;
};

TEST_F(SceneTest, BuildBVHWithTriangle) {
  Scene scene;
  Geometry geo;

  // Single triangle
  geo.vertices = {{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}};
  geo.indices = {0, 1, 2};

  scene.geometries.push_back(geo);

  RTCScene rtc_scene = BuildBVH(scene, device);
  ASSERT_NE(rtc_scene, nullptr);

  rtcReleaseScene(rtc_scene);
}

TEST_F(SceneTest, BuildEmptyScene) {
  Scene scene;
  RTCScene rtc_scene = BuildBVH(scene, device);
  ASSERT_NE(rtc_scene, nullptr);
  rtcReleaseScene(rtc_scene);
}

}  // namespace sh_baker
