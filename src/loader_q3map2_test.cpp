#include "loader_q3map2.h"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "scene.h"

namespace sh_baker {
namespace {

// A unit quad in the XY plane, wound clockwise as seen from +Z: q3map2's
// front face is +Z.
Q3Map2Surface Quad(int material_id) {
  Q3Map2Surface s;
  s.positions = {Eigen::Vector3f(0, 0, 0), Eigen::Vector3f(1, 0, 0),
                 Eigen::Vector3f(0, 1, 0), Eigen::Vector3f(1, 1, 0)};
  s.normals.assign(4, Eigen::Vector3f(0, 0, 1));
  // Texture U follows +X and V follows +Y.
  s.texture_uvs = {Eigen::Vector2f(0, 0), Eigen::Vector2f(1, 0),
                   Eigen::Vector2f(0, 1), Eigen::Vector2f(1, 1)};
  s.indices = {0, 2, 1, 1, 2, 3};
  s.material_id = material_id;
  return s;
}

Scene SceneWithMaterials(int count) {
  Scene scene;
  for (int i = 0; i < count; ++i) {
    Material mat;
    mat.name = "mat" + std::to_string(i);
    scene.materials.push_back(mat);
  }
  return scene;
}

TEST(LoaderQ3Map2Test, ValidLitSurface) {
  Scene scene = SceneWithMaterials(1);
  Light point;
  point.type = Light::Type::Point;
  scene.lights.push_back(point);

  Q3Map2Surface quad = Quad(0);
  AddQ3Map2Surface(quad, &scene);

  ASSERT_EQ(scene.geometries.size(), 1u);
  const Geometry& geo = scene.geometries[0];
  EXPECT_EQ(geo.vertices, quad.positions);
  EXPECT_EQ(geo.texture_uvs, quad.texture_uvs);
  EXPECT_EQ(geo.material_id, 0);
  EXPECT_TRUE(geo.lightmap_uvs.empty());
  EXPECT_TRUE(geo.transform.matrix().isIdentity());
  EXPECT_EQ(geo.tangents.size(), quad.positions.size());
  ASSERT_EQ(scene.lights.size(), 1u);
  EXPECT_EQ(scene.lights[0].type, Light::Type::Point);
  ASSERT_EQ(scene.materials.size(), 1u);
  EXPECT_EQ(scene.materials[0].name, "mat0");
}

TEST(LoaderQ3Map2Test, FlipsTriangles) {
  Scene scene = SceneWithMaterials(1);
  Q3Map2Surface quad = Quad(0);
  quad.indices = {0, 1, 2, 2, 1, 3};
  AddQ3Map2Surface(quad, &scene);
  EXPECT_EQ(scene.geometries[0].indices,
            (std::vector<uint32_t>{0, 2, 1, 2, 3, 1}));
}

TEST(LoaderQ3Map2Test, OccluderWithoutUvs) {
  Scene scene;
  Q3Map2Surface quad = Quad(-1);
  quad.texture_uvs.clear();
  AddQ3Map2Surface(quad, &scene);
  ASSERT_EQ(scene.geometries.size(), 1u);
  EXPECT_EQ(scene.geometries[0].material_id, -1);
  EXPECT_EQ(scene.geometries[0].texture_uvs,
            std::vector<Eigen::Vector2f>(4, Eigen::Vector2f::Zero()));
}

TEST(LoaderQ3Map2DeathTest, IndexOutOfRange) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  Scene scene = SceneWithMaterials(1);
  Q3Map2Surface quad = Quad(0);
  quad.indices[4] = 4;
  EXPECT_DEATH(AddQ3Map2Surface(quad, &scene), "indices\\[4\\]: index 4");
}

TEST(LoaderQ3Map2DeathTest, UnknownMaterial) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  Scene scene = SceneWithMaterials(2);
  EXPECT_DEATH(AddQ3Map2Surface(Quad(3), &scene), "material_id: 3");
}

TEST(LoaderQ3Map2DeathTest, LitSurfaceWithoutTextureUvs) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  Scene scene = SceneWithMaterials(1);
  Q3Map2Surface quad = Quad(0);
  quad.texture_uvs.clear();
  EXPECT_DEATH(AddQ3Map2Surface(quad, &scene), "texture_uvs");
}

TEST(LoaderQ3Map2DeathTest, NonFiniteTextureUv) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  Scene scene = SceneWithMaterials(1);
  Q3Map2Surface quad = Quad(0);
  quad.texture_uvs[2] =
      Eigen::Vector2f(std::numeric_limits<float>::quiet_NaN(), 0);
  EXPECT_DEATH(AddQ3Map2Surface(quad, &scene), "texture_uvs\\[2\\]");
}

TEST(LoaderQ3Map2DeathTest, SurfaceAfterAreaLight) {
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  Scene scene = SceneWithMaterials(1);
  AddQ3Map2Surface(Quad(0), &scene);
  Light area;
  area.type = Light::Type::Area;
  area.geometry = &scene.geometries[0];
  scene.lights.push_back(area);
  EXPECT_DEATH(AddQ3Map2Surface(Quad(0), &scene), "assembly order");
}

TEST(LoaderQ3Map2Test, TilingUvsAccepted) {
  Scene scene = SceneWithMaterials(1);
  Q3Map2Surface quad = Quad(0);
  quad.texture_uvs = {Eigen::Vector2f(-3, -3), Eigen::Vector2f(40, -3),
                      Eigen::Vector2f(-3, 40), Eigen::Vector2f(40, 40)};
  AddQ3Map2Surface(quad, &scene);
  EXPECT_EQ(scene.geometries[0].texture_uvs, quad.texture_uvs);
}

TEST(LoaderQ3Map2Test, NormalizesNormals) {
  Scene scene = SceneWithMaterials(1);
  Q3Map2Surface quad = Quad(0);
  quad.normals.assign(4, Eigen::Vector3f(0, 0, 2));
  AddQ3Map2Surface(quad, &scene);
  EXPECT_EQ(scene.geometries[0].normals,
            std::vector<Eigen::Vector3f>(4, Eigen::Vector3f(0, 0, 1)));
}

TEST(LoaderQ3Map2Test, RepairsZeroNormal) {
  Scene scene = SceneWithMaterials(1);
  Q3Map2Surface quad = Quad(0);  // clockwise as seen from +Z
  quad.normals[3] = Eigen::Vector3f::Zero();
  AddQ3Map2Surface(quad, &scene);
  EXPECT_TRUE(scene.geometries[0].normals[3].isApprox(Eigen::Vector3f(0, 0, 1)))
      << scene.geometries[0].normals[3].transpose();
}

TEST(LoaderQ3Map2Test, DropsRepeatedIndex) {
  Scene scene = SceneWithMaterials(1);
  Q3Map2Surface quad = Quad(0);
  quad.indices = {0, 1, 2, 2, 1, 3, 0, 0, 1};
  AddQ3Map2Surface(quad, &scene);
  EXPECT_EQ(scene.geometries[0].indices,
            (std::vector<uint32_t>{0, 2, 1, 2, 3, 1}));
}

TEST(LoaderQ3Map2Test, TangentsFollowTextureU) {
  Scene scene = SceneWithMaterials(1);
  AddQ3Map2Surface(Quad(0), &scene);
  const Geometry& geo = scene.geometries[0];
  ASSERT_EQ(geo.tangents.size(), 4u);
  for (const Eigen::Vector4f& t : geo.tangents) {
    EXPECT_TRUE(t.head<3>().isApprox(Eigen::Vector3f(1, 0, 0), 1e-4f))
        << t.transpose();
    EXPECT_EQ(std::abs(t.w()), 1.0f) << t.transpose();
  }
}

TEST(LoaderQ3Map2Test, FallbackTangentsForOccluder) {
  Scene scene;
  Q3Map2Surface quad = Quad(-1);
  quad.texture_uvs.clear();
  AddQ3Map2Surface(quad, &scene);
  const Geometry& geo = scene.geometries[0];
  ASSERT_EQ(geo.tangents.size(), 4u);
  for (size_t v = 0; v < geo.tangents.size(); ++v) {
    const Eigen::Vector3f t = geo.tangents[v].head<3>();
    EXPECT_NEAR(t.norm(), 1.0f, 1e-4f);
    EXPECT_NEAR(t.dot(geo.normals[v]), 0.0f, 1e-4f);
    EXPECT_EQ(geo.tangents[v].w(), 1.0f);
  }
}

}  // namespace
}  // namespace sh_baker
