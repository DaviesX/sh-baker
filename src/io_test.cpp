#include "io.h"

#include <gtest/gtest.h>
#include <filesystem>

namespace sh_baker {

TEST(IOTest, LoadCube) {
    // Assuming the test runs from the project root or build dir. 
    // We try to locate data/cube/Cube.gltf.
    // If running via cmake, CTEST_WORKING_DIRECTORY might be set or we rely on relative path.
    // The user rules say project root is /Users/daviswen/sh-baker.
    // The test binary is likely in build/sh_baker_test.
    // So ../data/cube/Cube.gltf might be needed or absolute path. 
    // I'll try absolute path for robustness in this environment.
    
    std::filesystem::path input_path = "/Users/daviswen/sh-baker/data/cube/Cube.gltf";
    ASSERT_TRUE(std::filesystem::exists(input_path)) << "Test file not found: " << input_path;

    std::optional<Scene> scene = LoadScene(input_path);
    ASSERT_TRUE(scene.has_value());

    // Cube has 36 indices (6 faces * 2 triangles * 3 vertices) or less if indexed?
    // Accessor 0 (indices) has count 36.
    EXPECT_EQ(scene->indices.size(), 36);
    
    // Accessor 1 (Position) has count 36.
    // It seems the cube is not fully interconnected (flat shading duplicate verts) or just exported that way.
    EXPECT_EQ(scene->vertices.size(), 36);
    EXPECT_EQ(scene->normals.size(), 36);
    EXPECT_EQ(scene->uvs.size(), 36);
}

TEST(IOTest, MissingFile) {
    std::optional<Scene> scene = LoadScene("non_existent.gltf");
    EXPECT_FALSE(scene.has_value());
}

}  // namespace sh_baker
