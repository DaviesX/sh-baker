#include "light_tree.h"

#include <gtest/gtest.h>

#include <random>

namespace sh_baker {
namespace {

TEST(LightTreeInternalTest, UnionEmptyBounds) {
  light_tree_internal::LightBounds a;
  light_tree_internal::LightBounds b;
  b.bounds =
      Eigen::AlignedBox3f(Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(1, 1, 1));
  b.phi = 100.0f;
  b.axis = Eigen::Vector3f::UnitZ();
  b.cos_theta_o = 0.5f;

  auto result = light_tree_internal::Union(a, b);

  EXPECT_FLOAT_EQ(result.phi, 100.0f);
  EXPECT_EQ(result.axis, Eigen::Vector3f::UnitZ());
  EXPECT_FLOAT_EQ(result.cos_theta_o, 0.5f);
}

TEST(LightTreeInternalTest, UnionTwoBounds) {
  light_tree_internal::LightBounds a;
  a.bounds =
      Eigen::AlignedBox3f(Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(1, 1, 1));
  a.phi = 50.0f;
  a.axis = Eigen::Vector3f::UnitX();
  a.cos_theta_o = 0.8f;

  light_tree_internal::LightBounds b;
  b.bounds =
      Eigen::AlignedBox3f(Eigen::Vector3f(2, 2, 2), Eigen::Vector3f(3, 3, 3));
  b.phi = 50.0f;
  b.axis = Eigen::Vector3f::UnitY();
  b.cos_theta_o = 0.8f;

  auto result = light_tree_internal::Union(a, b);

  EXPECT_FLOAT_EQ(result.phi, 100.0f);
  EXPECT_EQ(result.bounds.min(), Eigen::Vector3f(0, 0, 0));
  EXPECT_EQ(result.bounds.max(), Eigen::Vector3f(3, 3, 3));
  // The axis should be between UnitX and UnitY.
  EXPECT_GT(result.axis.norm(), 0.9f);
}

TEST(LightTreeInternalTest, ImportancePointLight) {
  light_tree_internal::LightBounds lb;
  lb.bounds =
      Eigen::AlignedBox3f(Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(0, 0, 0));
  lb.phi = 100.0f;
  lb.axis = Eigen::Vector3f::UnitY();
  lb.cos_theta_o = -1.0f;  // Full sphere.
  lb.cos_theta_e = -1.0f;
  lb.two_sided = true;

  Eigen::Vector3f p(10, 0, 0);
  Eigen::Vector3f n = Eigen::Vector3f::UnitX();

  float importance = light_tree_internal::Importance(lb, p, n);
  EXPECT_GT(importance, 0.0f);
}

TEST(LightTreeInternalTest, ImportanceZeroWhenBehind) {
  light_tree_internal::LightBounds lb;
  lb.bounds =
      Eigen::AlignedBox3f(Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(0, 0, 0));
  lb.phi = 100.0f;
  lb.axis = Eigen::Vector3f::UnitX();  // Light points in +X.
  lb.cos_theta_o = 1.0f;               // Single direction.
  lb.cos_theta_e = 0.9f;               // Narrow cone.
  lb.two_sided = false;

  // Point is in -X direction from light.
  Eigen::Vector3f p(-10, 0, 0);
  Eigen::Vector3f n = -Eigen::Vector3f::UnitX();

  float importance = light_tree_internal::Importance(lb, p, n);
  EXPECT_FLOAT_EQ(importance, 0.0f);
}

TEST(LightTreeInternalTest, ComputeLightBoundsPointLight) {
  Light light;
  light.type = Light::Type::Point;
  light.position = Eigen::Vector3f(5, 5, 5);
  light.color = Eigen::Vector3f::Ones();
  light.intensity = 100.0f;
  // Flux for point light is intensity * 4 * PI (roughly, for test purposes).
  light.flux = light.intensity * 4.0f * 3.14159f;

  auto lb = light_tree_internal::ComputeLightBounds(light);

  EXPECT_EQ(lb.bounds.min(), light.position);
  EXPECT_EQ(lb.bounds.max(), light.position);
  EXPECT_NEAR(lb.phi, 4 * M_PI * 100.0f, 1e-2f);
  EXPECT_FLOAT_EQ(lb.cos_theta_o, -1.0f);  // Full sphere.
  EXPECT_TRUE(lb.two_sided);
}

TEST(LightTreeInternalTest, ComputeLightBoundsSpotLight) {
  Light light;
  light.type = Light::Type::Spot;
  light.position = Eigen::Vector3f(0, 5, 0);
  light.direction = -Eigen::Vector3f::UnitY();
  light.color = Eigen::Vector3f::Ones();
  light.intensity = 50.0f;
  light.cos_outer_cone = 0.7f;
  light.flux = light.intensity * 1.0f;  // Dummy flux for test.

  auto lb = light_tree_internal::ComputeLightBounds(light);

  EXPECT_EQ(lb.bounds.min(), light.position);
  EXPECT_EQ(lb.bounds.max(), light.position);
  EXPECT_FLOAT_EQ(lb.phi, 50.0f);
  EXPECT_EQ(lb.axis, light.direction);
  EXPECT_FLOAT_EQ(lb.cos_theta_e, 0.7f);
  EXPECT_FALSE(lb.two_sided);
}

TEST(LightTreeInternalTest, ComputeLightBoundsDirectionalReturnsZeroPhi) {
  Light light;
  light.type = Light::Type::Directional;
  light.direction = Eigen::Vector3f::UnitY();
  light.color = Eigen::Vector3f::Ones();
  light.intensity = 100.0f;

  auto lb = light_tree_internal::ComputeLightBounds(light);

  // Directional lights are infinite and should not be in the BVH.
  EXPECT_FLOAT_EQ(lb.phi, 0.0f);
}

TEST(LightTreeTest, BuildEmptyLights) {
  LightTree tree;
  tree.Build({});

  EXPECT_TRUE(tree.Empty());
  EXPECT_EQ(tree.NumLights(), 0);
}

TEST(LightTreeTest, BuildSingleLight) {
  Light light;
  light.type = Light::Type::Point;
  light.position = Eigen::Vector3f(0, 0, 0);
  light.color = Eigen::Vector3f::Ones();
  light.intensity = 100.0f;
  light.flux = 1000.0f;

  LightTree tree;
  std::vector<Light> lights = {light};
  tree.Build(lights);

  EXPECT_FALSE(tree.Empty());
  EXPECT_EQ(tree.NumLights(), 1);
  EXPECT_EQ(tree.Nodes().size(), 1);
  EXPECT_TRUE(tree.Nodes()[0].is_leaf);
  EXPECT_EQ(tree.Nodes()[0].light, &lights[0]);
}

TEST(LightTreeTest, BuildTwoLights) {
  Light light1;
  light1.type = Light::Type::Point;
  light1.position = Eigen::Vector3f(0, 0, 0);
  light1.color = Eigen::Vector3f::Ones();
  light1.intensity = 100.0f;
  light1.flux = 1000.0f;

  Light light2;
  light2.type = Light::Type::Point;
  light2.position = Eigen::Vector3f(10, 0, 0);
  light2.color = Eigen::Vector3f::Ones();
  light2.intensity = 100.0f;
  light2.flux = 1000.0f;

  LightTree tree;
  tree.Build({light1, light2});

  EXPECT_FALSE(tree.Empty());
  EXPECT_EQ(tree.NumLights(), 2);
  // Should have 3 nodes: 1 interior + 2 leaves.
  EXPECT_EQ(tree.Nodes().size(), 3);
  EXPECT_FALSE(tree.Nodes()[0].is_leaf);  // Root is interior.
}

TEST(LightTreeTest, SampleSingleLight) {
  Light light;
  light.type = Light::Type::Point;
  light.position = Eigen::Vector3f(0, 0, 0);
  light.color = Eigen::Vector3f::Ones();
  light.intensity = 100.0f;
  light.flux = 1000.0f;

  std::vector<Light> lights = {light};
  LightTree tree;
  tree.Build(lights);

  Eigen::Vector3f p(5, 0, 0);
  Eigen::Vector3f n = Eigen::Vector3f::UnitX();

  auto result = tree.Sample(p, n, 0.5f);

  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->light, &lights[0]);
  EXPECT_FLOAT_EQ(result->pdf, 1.0f);
}

TEST(LightTreeTest, SampleTwoLightsCloserHasHigherProbability) {
  Light light1;
  light1.type = Light::Type::Point;
  light1.position = Eigen::Vector3f(0, 0, 0);
  light1.color = Eigen::Vector3f::Ones();
  light1.intensity = 100.0f;
  light1.flux = 1000.0f;

  Light light2;
  light2.type = Light::Type::Point;
  light2.position = Eigen::Vector3f(100, 0, 0);  // Far away.
  light2.color = Eigen::Vector3f::Ones();
  light2.intensity = 100.0f;
  light2.flux = 1000.0f;

  std::vector<Light> lights = {light1, light2};
  LightTree tree;
  tree.Build(lights);

  Eigen::Vector3f p(1, 0, 0);  // Closer to light1.
  Eigen::Vector3f n = -Eigen::Vector3f::UnitX();

  // Sample many times and count.
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  int count0 = 0, count1 = 0;
  for (int i = 0; i < 1000; ++i) {
    auto result = tree.Sample(p, n, dist(rng));
    if (!result.has_value()) continue;
    if (result->light == &lights[0])
      ++count0;
    else if (result->light == &lights[1])
      ++count1;
  }

  // Light1 should be sampled more often because it's closer.
  EXPECT_GT(count0, count1);
}

TEST(LightTreeTest, PdfSingleLight) {
  Light light;
  light.type = Light::Type::Point;
  light.position = Eigen::Vector3f(0, 0, 0);
  light.color = Eigen::Vector3f::Ones();
  light.intensity = 100.0f;
  light.flux = 1000.0f;

  std::vector<Light> lights = {light};
  LightTree tree;
  tree.Build(lights);

  Eigen::Vector3f p(5, 0, 0);
  Eigen::Vector3f n = Eigen::Vector3f::UnitX();

  float pdf = tree.Pdf(p, n, &lights[0]);
  EXPECT_FLOAT_EQ(pdf, 1.0f);
}

TEST(LightTreeTest, PdfSumToOne) {
  Light light1;
  light1.type = Light::Type::Point;
  light1.position = Eigen::Vector3f(0, 0, 0);
  light1.color = Eigen::Vector3f::Ones();
  light1.intensity = 100.0f;
  light1.flux = 1000.0f;

  Light light2;
  light2.type = Light::Type::Point;
  light2.position = Eigen::Vector3f(10, 0, 0);
  light2.color = Eigen::Vector3f::Ones();
  light2.intensity = 100.0f;
  light2.flux = 1000.0f;

  std::vector<Light> lights = {light1, light2};
  LightTree tree;
  tree.Build(lights);

  Eigen::Vector3f p(5, 5, 0);
  Eigen::Vector3f n = Eigen::Vector3f::UnitY();

  float pdf0 = tree.Pdf(p, n, &lights[0]);
  float pdf1 = tree.Pdf(p, n, &lights[1]);

  EXPECT_NEAR(pdf0 + pdf1, 1.0f, 1e-5f);
}

TEST(LightTreeTest, SamplePdfConsistency) {
  // Build a tree with multiple lights.
  std::vector<Light> lights;
  for (int i = 0; i < 8; ++i) {
    Light light;
    light.type = Light::Type::Point;
    light.position = Eigen::Vector3f(static_cast<float>(i * 5), 0, 0);
    light.color = Eigen::Vector3f::Ones();
    light.intensity = 100.0f * (i + 1);
    light.flux = light.intensity * 10.0f;
    lights.push_back(light);
  }

  LightTree tree;
  tree.Build(lights);

  Eigen::Vector3f p(20, 5, 0);
  Eigen::Vector3f n = Eigen::Vector3f::UnitY();

  // Verify that Sample returns the correct PDF.
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  for (int i = 0; i < 100; ++i) {
    float u = dist(rng);
    auto result = tree.Sample(p, n, u);
    if (result.has_value() && result->light != nullptr) {
      float pdf = tree.Pdf(p, n, result->light);
      EXPECT_NEAR(result->pdf, pdf, 1e-5f);
    }
  }
}

TEST(LightTreeTest, DirectionalLightsIncluded) {
  Light point_light;
  point_light.type = Light::Type::Point;
  point_light.position = Eigen::Vector3f(0, 0, 0);
  point_light.color = Eigen::Vector3f::Ones();
  point_light.intensity = 100.0f;
  point_light.flux = 1000.0f;

  Light dir_light;
  dir_light.type = Light::Type::Directional;
  dir_light.direction = -Eigen::Vector3f::UnitY();
  dir_light.color = Eigen::Vector3f::Ones();
  dir_light.intensity = 1000.0f;

  std::vector<Light> lights = {point_light, dir_light};
  LightTree tree;
  tree.Build(lights);

  // Both lights should be in the tree (1 in BVH, 1 directional).
  EXPECT_FALSE(tree.Empty());
  EXPECT_EQ(tree.NumLights(), 2);
  EXPECT_EQ(tree.NumDirectionalLights(), 1);
  EXPECT_EQ(tree.Nodes().size(), 1);  // Single leaf for point light in BVH.
  EXPECT_TRUE(tree.Nodes()[0].is_leaf);
  EXPECT_EQ(tree.Nodes()[0].light, &lights[0]);  // Pointer to point_light.
}

}  // namespace

// Tests for Area Lights
namespace {

TEST(LightTreeInternalTest, ComputeLightBoundsAreaLight) {
  // Create a simple triangle geometry for the area light.
  Geometry geo;
  geo.vertices = {Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(2, 0, 0),
                  Eigen::Vector3f(1, 0, 2)};
  geo.normals = {Eigen::Vector3f(0, 1, 0), Eigen::Vector3f(0, 1, 0),
                 Eigen::Vector3f(0, 1, 0)};
  geo.indices = {0, 1, 2};

  Light area_light;
  area_light.type = Light::Type::Area;
  area_light.geometry = &geo;
  area_light.color = Eigen::Vector3f::Ones();
  area_light.intensity = 10.0f;
  area_light.flux = 20.0f;  // 10 * 2
  area_light.area = 2.0f;   // Triangle area.

  auto lb = light_tree_internal::ComputeLightBounds(area_light);

  // Bounds should contain all vertices.
  EXPECT_LE(lb.bounds.min().x(), 0.0f);
  EXPECT_LE(lb.bounds.min().z(), 0.0f);
  EXPECT_GE(lb.bounds.max().x(), 2.0f);
  EXPECT_GE(lb.bounds.max().z(), 2.0f);

  // Phi should be intensity * maxCoeff * area.
  EXPECT_FLOAT_EQ(lb.phi, 10.0f * 1.0f * 2.0f);

  // Axis should be approximately the average normal (0, 1, 0).
  EXPECT_NEAR(lb.axis.y(), 1.0f, 1e-5f);

  // Area lights are two-sided.
  EXPECT_TRUE(lb.two_sided);

  // Emission cone should be full sphere for two-sided.
  EXPECT_FLOAT_EQ(lb.cos_theta_e, -1.0f);
}

TEST(LightTreeInternalTest, ComputeLightBoundsAreaLightNoGeometry) {
  Light area_light;
  area_light.type = Light::Type::Area;
  area_light.geometry = nullptr;  // No geometry!
  area_light.color = Eigen::Vector3f::Ones();
  area_light.intensity = 10.0f;
  area_light.area = 2.0f;

  auto lb = light_tree_internal::ComputeLightBounds(area_light);

  // Without geometry, phi should be 0 (light won't be in tree).
  EXPECT_FLOAT_EQ(lb.phi, 0.0f);
}

TEST(LightTreeTest, BuildAreaLight) {
  // Create geometry for area light.
  Geometry geo;
  geo.vertices = {Eigen::Vector3f(5, 5, 0), Eigen::Vector3f(7, 5, 0),
                  Eigen::Vector3f(6, 5, 2)};
  geo.normals = {Eigen::Vector3f(0, -1, 0), Eigen::Vector3f(0, -1, 0),
                 Eigen::Vector3f(0, -1, 0)};
  geo.indices = {0, 1, 2};

  Light area_light;
  area_light.type = Light::Type::Area;
  area_light.geometry = &geo;
  area_light.color = Eigen::Vector3f::Ones();
  area_light.intensity = 50.0f;
  area_light.flux = 100.0f;
  area_light.area = 2.0f;

  std::vector<Light> lights = {area_light};
  LightTree tree;
  tree.Build(lights);

  EXPECT_FALSE(tree.Empty());
  EXPECT_EQ(tree.NumLights(), 1);
  EXPECT_EQ(tree.Nodes().size(), 1);
  EXPECT_TRUE(tree.Nodes()[0].is_leaf);
  EXPECT_EQ(tree.Nodes()[0].light, &lights[0]);
}

TEST(LightTreeTest, SampleAreaLight) {
  // Create geometry for area light at Y=10.
  Geometry geo;
  geo.vertices = {Eigen::Vector3f(-1, 10, -1), Eigen::Vector3f(1, 10, -1),
                  Eigen::Vector3f(0, 10, 1)};
  geo.normals = {Eigen::Vector3f(0, -1, 0), Eigen::Vector3f(0, -1, 0),
                 Eigen::Vector3f(0, -1, 0)};
  geo.indices = {0, 1, 2};

  Light area_light;
  area_light.type = Light::Type::Area;
  area_light.geometry = &geo;
  area_light.color = Eigen::Vector3f::Ones();
  area_light.intensity = 100.0f;
  area_light.flux = 200.0f;
  area_light.area = 2.0f;

  std::vector<Light> lights = {area_light};
  LightTree tree;
  tree.Build(lights);

  // Point below the area light, looking up.
  Eigen::Vector3f p(0, 0, 0);
  Eigen::Vector3f n(0, 1, 0);

  // Sample should return the area light.
  auto result = tree.Sample(p, n, 0.5f);

  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->light, &lights[0]);
  EXPECT_FLOAT_EQ(result->pdf, 1.0f);
}

TEST(LightTreeTest, SampleAreaLightFromBehind) {
  // Area light at Y=10, normal pointing DOWN (-Y).
  // This tests two-sided behavior.
  Geometry geo;
  geo.vertices = {Eigen::Vector3f(-1, 10, -1), Eigen::Vector3f(1, 10, -1),
                  Eigen::Vector3f(0, 10, 1)};
  geo.normals = {Eigen::Vector3f(0, -1, 0), Eigen::Vector3f(0, -1, 0),
                 Eigen::Vector3f(0, -1, 0)};
  geo.indices = {0, 1, 2};

  Light area_light;
  area_light.type = Light::Type::Area;
  area_light.geometry = &geo;
  area_light.color = Eigen::Vector3f::Ones();
  area_light.intensity = 100.0f;
  area_light.flux = 200.0f;
  area_light.area = 2.0f;

  std::vector<Light> lights = {area_light};
  LightTree tree;
  tree.Build(lights);

  // Point ABOVE the area light (behind it relative to its normal).
  // Since area lights are two-sided, this should still sample the light.
  Eigen::Vector3f p(0, 20, 0);
  Eigen::Vector3f n(0, -1, 0);  // Looking down at the light.

  auto result = tree.Sample(p, n, 0.5f);

  // Two-sided area light should still be sampled from behind.
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(result->light, &lights[0]);
  EXPECT_GT(result->pdf, 0.0f);
}

TEST(LightTreeTest, AreaLightImportanceIsNonZeroFromBothSides) {
  // Create geometry for area light.
  Geometry geo;
  geo.vertices = {Eigen::Vector3f(-1, 0, -1), Eigen::Vector3f(1, 0, -1),
                  Eigen::Vector3f(0, 0, 1)};
  geo.normals = {Eigen::Vector3f(0, 1, 0), Eigen::Vector3f(0, 1, 0),
                 Eigen::Vector3f(0, 1, 0)};
  geo.indices = {0, 1, 2};

  Light area_light;
  area_light.type = Light::Type::Area;
  area_light.geometry = &geo;
  area_light.color = Eigen::Vector3f::Ones();
  area_light.intensity = 100.0f;
  area_light.flux = 200.0f;
  area_light.area = 2.0f;

  auto lb = light_tree_internal::ComputeLightBounds(area_light);

  // Point above (front side).
  Eigen::Vector3f p_above(0, 5, 0);
  Eigen::Vector3f n_above(0, -1, 0);
  float importance_above =
      light_tree_internal::Importance(lb, p_above, n_above);

  // Point below (back side).
  Eigen::Vector3f p_below(0, -5, 0);
  Eigen::Vector3f n_below(0, 1, 0);
  float importance_below =
      light_tree_internal::Importance(lb, p_below, n_below);

  // Both should have non-zero importance (two-sided).
  EXPECT_GT(importance_above, 0.0f);
  EXPECT_GT(importance_below, 0.0f);
}

TEST(LightTreeTest, MixedPointAndAreaLights) {
  // Create geometry for area light.
  Geometry geo;
  geo.vertices = {Eigen::Vector3f(9, 10, -1), Eigen::Vector3f(11, 10, -1),
                  Eigen::Vector3f(10, 10, 1)};
  geo.normals = {Eigen::Vector3f(0, -1, 0), Eigen::Vector3f(0, -1, 0),
                 Eigen::Vector3f(0, -1, 0)};
  geo.indices = {0, 1, 2};

  Light point_light;
  point_light.type = Light::Type::Point;
  point_light.position = Eigen::Vector3f(0, 10, 0);
  point_light.color = Eigen::Vector3f::Ones();
  point_light.intensity = 100.0f;
  point_light.flux = 1000.0f;

  Light area_light;
  area_light.type = Light::Type::Area;
  area_light.geometry = &geo;
  area_light.color = Eigen::Vector3f::Ones();
  area_light.intensity = 100.0f;
  area_light.flux = 200.0f;
  area_light.area = 2.0f;

  std::vector<Light> lights = {point_light, area_light};
  LightTree tree;
  tree.Build(lights);

  EXPECT_FALSE(tree.Empty());
  EXPECT_EQ(tree.NumLights(), 2);
  // Should have 3 nodes: 1 interior + 2 leaves.
  EXPECT_EQ(tree.Nodes().size(), 3);

  // Sample from a point between them.
  Eigen::Vector3f p(5, 0, 0);
  Eigen::Vector3f n(0, 1, 0);

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  int count_point = 0, count_area = 0;
  for (int i = 0; i < 1000; ++i) {
    auto result = tree.Sample(p, n, dist(rng));
    if (!result.has_value()) continue;
    if (result->light == &lights[0])
      ++count_point;
    else if (result->light == &lights[1])
      ++count_area;
  }

  // Both lights should be sampled.
  EXPECT_GT(count_point, 0);
  EXPECT_GT(count_area, 0);
}

}  // namespace
}  // namespace sh_baker
