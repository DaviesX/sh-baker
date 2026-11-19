#include "light_tree.h"

#include <gtest/gtest.h>

#include <random>

#include "light.h"

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

// --- Range-aware light selection ---

TEST(LightTreeInternalTest, ComputeLightBoundsCarriesRange) {
  Light point;
  point.type = Light::Type::Point;
  point.flux = 100.0f;
  EXPECT_EQ(light_tree_internal::ComputeLightBounds(point).max_range, 0.0f);
  point.range = 64.0f;
  EXPECT_EQ(light_tree_internal::ComputeLightBounds(point).max_range, 64.0f);
  Light spot = point;
  spot.type = Light::Type::Spot;
  EXPECT_EQ(light_tree_internal::ComputeLightBounds(spot).max_range, 64.0f);
}

TEST(LightTreeInternalTest, UnionRange) {
  light_tree_internal::LightBounds empty;
  light_tree_internal::LightBounds a;
  a.bounds =
      Eigen::AlignedBox3f(Eigen::Vector3f::Zero(), Eigen::Vector3f::Zero());
  a.phi = 10.0f;
  a.max_range = 64.0f;
  light_tree_internal::LightBounds b = a;
  b.max_range = 128.0f;
  light_tree_internal::LightBounds unranged = a;
  unranged.max_range = 0.0f;

  EXPECT_EQ(light_tree_internal::Union(a, b).max_range, 128.0f);
  EXPECT_EQ(light_tree_internal::Union(b, a).max_range, 128.0f);
  EXPECT_EQ(light_tree_internal::Union(a, unranged).max_range, 0.0f);
  EXPECT_EQ(light_tree_internal::Union(unranged, b).max_range, 0.0f);
  // An empty side leaves the other's range.
  EXPECT_EQ(light_tree_internal::Union(empty, a).max_range, 64.0f);
  EXPECT_EQ(light_tree_internal::Union(a, empty).max_range, 64.0f);
}

TEST(LightTreeInternalTest, ImportanceZeroBeyondRange) {
  light_tree_internal::LightBounds lb;
  lb.bounds =
      Eigen::AlignedBox3f(Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(10, 0, 0));
  lb.phi = 100.0f;
  lb.cos_theta_o = -1.0f;
  lb.cos_theta_e = -1.0f;
  lb.two_sided = true;
  const Eigen::Vector3f n = Eigen::Vector3f::Zero();
  const Eigen::Vector3f near_end(10 + 63, 0, 0), far_end(10 + 65, 0, 0);

  EXPECT_GT(light_tree_internal::Importance(lb, far_end, n), 0.0f);
  lb.max_range = 64.0f;
  // Range is measured from the bounds, not their centre.
  EXPECT_GT(light_tree_internal::Importance(lb, near_end, n), 0.0f);
  EXPECT_EQ(light_tree_internal::Importance(lb, far_end, n), 0.0f);
}

// At the range boundary, a light that still lights the point (by the
// incoming-radiance functions' own range test) must keep nonzero importance,
// or it would be sampled with probability 0 and its light lost.
TEST(LightTreeInternalTest, ImportanceKeepsLightsLitAtTheRangeBoundary) {
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> u(-1.0f, 1.0f);
  std::uniform_real_distribution<float> range_dist(1.0f, 500.0f);
  int lit = 0;
  for (int i = 0; i < 200000; ++i) {
    Light light;
    light.type = (i % 2) ? Light::Type::Point : Light::Type::Spot;
    light.position = 1000.0f * Eigen::Vector3f(u(rng), u(rng), u(rng));
    light.range = range_dist(rng);
    Eigen::Vector3f d(u(rng), u(rng), u(rng));
    if (d.norm() < 1e-3f) continue;
    d.normalize();
    light.direction = d;  // The spot points at the shaded point.
    light.intensity = 1.0f;
    light.flux = Flux(light);
    const Eigen::Vector3f p = light.position + light.range * d;
    const Eigen::Vector3f n = -d;
    Ray ray;
    const Eigen::Vector3f radiance =
        light.type == Light::Type::Point
            ? light_internal::PointLightIncomingRadiance(light, p, n, &ray)
                  .radiance
            : light_internal::SpotLightIncomingRadiance(light, p, n, &ray)
                  .radiance;
    if (radiance.isZero()) continue;
    ++lit;
    EXPECT_GT(light_tree_internal::Importance(
                  light_tree_internal::ComputeLightBounds(light), p, n),
              0.0f)
        << "range " << light.range << " distance "
        << (light.position - p).norm();
  }
  EXPECT_GT(lit, 1000);
}

// One point light of range 64 and a sun.
struct RangedAndSun {
  std::vector<Light> lights;
  LightTree tree;

  RangedAndSun() {
    Light point;
    point.type = Light::Type::Point;
    point.position = Eigen::Vector3f(0, 0, 0);
    point.intensity = 100.0f;
    point.flux = 1000.0f;
    point.range = 64.0f;
    Light sun;
    sun.type = Light::Type::Directional;
    sun.direction = -Eigen::Vector3f::UnitZ();
    sun.intensity = 3.0f;
    lights = {point, sun};
    tree.Build(lights);
  }
  const Light* point() const { return &lights[0]; }
  const Light* sun() const { return &lights[1]; }
};

TEST(LightTreeTest, OutOfRangeLightsNeverSampled) {
  RangedAndSun f;
  const Eigen::Vector3f p(500, 0, 0), n = Eigen::Vector3f::UnitZ();
  for (int i = 0; i < 1000; ++i) {
    const float u = (i + 0.5f) / 1000.0f;
    std::optional<SampledLight> s = f.tree.Sample(p, n, u);
    ASSERT_TRUE(s.has_value()) << u;
    EXPECT_EQ(s->light, f.sun()) << u;
    EXPECT_EQ(s->pdf, 1.0f) << u;
  }
  EXPECT_EQ(f.tree.Pdf(p, n, f.point()), 0.0f);
  EXPECT_EQ(f.tree.Pdf(p, n, f.sun()), 1.0f);
}

TEST(LightTreeTest, InRangeKeepsTheFixedDirectionalShare) {
  RangedAndSun f;
  const Eigen::Vector3f p(30, 0, 10), n = Eigen::Vector3f::UnitZ();
  EXPECT_EQ(f.tree.Pdf(p, n, f.sun()), 0.5f);
  EXPECT_EQ(f.tree.Pdf(p, n, f.point()), 0.5f);
  EXPECT_EQ(f.tree.Sample(p, n, 0.25f)->light, f.sun());
  EXPECT_EQ(f.tree.Sample(p, n, 0.75f)->light, f.point());
}

TEST(LightTreeTest, SampleAndPdfAgreeInAndOutOfRange) {
  // Ranged point and spot lights of different ranges along X, and a sun.
  std::vector<Light> lights;
  for (int i = 0; i < 6; ++i) {
    Light light;
    light.type = i % 2 ? Light::Type::Spot : Light::Type::Point;
    light.position = Eigen::Vector3f(40.0f * i, 0, 0);
    light.direction = -Eigen::Vector3f::UnitZ();
    light.intensity = 100.0f;
    light.flux = 1000.0f;
    light.range = 30.0f + 10.0f * i;
    lights.push_back(light);
  }
  Light sun;
  sun.type = Light::Type::Directional;
  sun.direction = -Eigen::Vector3f::UnitZ();
  lights.push_back(sun);
  LightTree tree;
  tree.Build(lights);

  const Eigen::Vector3f n = Eigen::Vector3f::UnitZ();
  const Eigen::Vector3f points[] = {
      Eigen::Vector3f(0, 0, -10),    // in range of light 0
      Eigen::Vector3f(85, 0, -20),   // between lights 2 and 3
      Eigen::Vector3f(200, 0, -30),  // near the last
      Eigen::Vector3f(100, 300, 0),  // out of every range
      Eigen::Vector3f(-500, 0, 0),   // out of every range
  };
  for (const Eigen::Vector3f& p : points) {
    // Out-of-range lights have pdf 0. A node's range test is conservative (its
    // bounds and largest range), so the total can fall short of 1 where a
    // reachable node holds only unreachable lights; Sample() then returns
    // nothing, with the same probability.
    float total = 0.0f;
    for (const Light& light : lights) {
      const float pdf = tree.Pdf(p, -n, &light);
      if (light.range > 0.0f && (light.position - p).norm() > light.range) {
        EXPECT_EQ(pdf, 0.0f) << p.transpose();
      }
      total += pdf;
    }
    EXPECT_LE(total, 1.0f + 1e-5f) << p.transpose();
    EXPECT_GE(tree.Pdf(p, -n, &lights.back()), 0.5f) << p.transpose();
    for (int i = 0; i < 200; ++i) {
      const float u = (i + 0.5f) / 200.0f;
      std::optional<SampledLight> s = tree.Sample(p, -n, u);
      if (!s) continue;
      EXPECT_NEAR(s->pdf, tree.Pdf(p, -n, s->light), 1e-6f) << p.transpose();
      if (s->light->range > 0.0f) {
        EXPECT_LE((s->light->position - p).norm(), s->light->range)
            << p.transpose();
      }
    }
  }
}

// The root's union cone of two anti-parallel spots is not conservative (it
// keeps one axis with a 90 degree spread), so the root's full importance can
// be 0 at a point one of the spots lights. The directional share must then
// stay fixed: only the range test may hand the whole choice to the sun.
TEST(LightTreeTest, InRangeRootWithZeroOrientationImportanceKeepsTheShare) {
  std::vector<Light> lights;
  for (int i = 0; i < 2; ++i) {
    Light spot;
    spot.type = Light::Type::Spot;
    spot.position = Eigen::Vector3f(10.0f * i, 0, 0);
    spot.direction = Eigen::Vector3f(0, 0, i == 0 ? 1.0f : -1.0f);
    spot.intensity = 100.0f;
    spot.range = 100.0f;
    spot.flux = Flux(spot);
    lights.push_back(spot);
  }
  Light sun;
  sun.type = Light::Type::Directional;
  sun.direction = -Eigen::Vector3f::UnitZ();
  sun.intensity = 1.0f;
  sun.flux = Flux(sun);
  lights.push_back(sun);
  LightTree tree;
  tree.Build(lights);

  // Above the up spot and below the down spot, facing each.
  const std::pair<Eigen::Vector3f, Eigen::Vector3f> cases[] = {
      {Eigen::Vector3f(0, 0, 20), -Eigen::Vector3f::UnitZ()},
      {Eigen::Vector3f(10, 0, -20), Eigen::Vector3f::UnitZ()},
  };
  for (int i = 0; i < 2; ++i) {
    const auto& [p, n] = cases[i];
    Ray ray;
    ASSERT_GT(light_internal::SpotLightIncomingRadiance(lights[i], p, n, &ray)
                  .radiance.x(),
              0.0f);
    EXPECT_GT(tree.Pdf(p, n, &lights[i]), 0.0f) << i;
    EXPECT_EQ(tree.Pdf(p, n, &lights[2]), 0.5f) << i;
    int picked = 0;
    for (int k = 0; k < 1000; ++k) {
      std::optional<SampledLight> s = tree.Sample(p, n, (k + 0.5f) / 1000.0f);
      if (s && s->light == &lights[i]) {
        ++picked;
        EXPECT_FLOAT_EQ(s->pdf, tree.Pdf(p, n, s->light));
      }
    }
    EXPECT_GT(picked, 0) << i;
  }
}

TEST(LightTreeTest, OutOfRangeWithoutDirectionalSamplesNothing) {
  Light point;
  point.type = Light::Type::Point;
  point.flux = 1000.0f;
  point.range = 64.0f;
  std::vector<Light> lights = {point};
  LightTree tree;
  tree.Build(lights);
  const Eigen::Vector3f p(500, 0, 0), n = Eigen::Vector3f::UnitZ();
  EXPECT_FALSE(tree.Sample(p, n, 0.5f).has_value());
  EXPECT_EQ(tree.Pdf(p, n, &lights[0]), 0.0f);
  EXPECT_TRUE(tree.Sample(Eigen::Vector3f(10, 0, 5), n, 0.5f).has_value());
}

}  // namespace
}  // namespace sh_baker
