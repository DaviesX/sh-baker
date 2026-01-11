#ifndef SH_BAKER_SRC_SCENE_H_
#define SH_BAKER_SRC_SCENE_H_

#include <embree4/rtcore.h>

#include <Eigen/Dense>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace sh_baker {

// --- Texture ---
struct Texture {
  // If set, the texture is loaded from a file. This denotes the provenance of
  // the texture.
  std::optional<std::filesystem::path> file_path;

  uint32_t width = 0;
  uint32_t height = 0;
  uint32_t channels = 0;
  std::vector<uint8_t> pixel_data;
};

// --- Material ---
struct Material {
  std::string name;

  // Albedo / Transparency
  Texture albedo;
  Texture normal_texture;
  Texture metallic_roughness_texture;  // Metallic in B, Roughness in G

  // Emission (for Area Lights).
  float emission_intensity = 0.0f;
};

// --- Geometry ---
struct Geometry {
  std::vector<Eigen::Vector3f> vertices;
  std::vector<Eigen::Vector3f> normals;
  std::vector<Eigen::Vector2f> texture_uvs;
  std::vector<Eigen::Vector2f> lightmap_uvs;
  std::vector<Eigen::Vector4f> tangents;  // xyz + w (sign)

  std::vector<uint32_t> indices;

  int material_id = -1;  // Index into Scene::materials
  Eigen::Affine3f transform = Eigen::Affine3f::Identity();
};

// --- Light ---
struct Light {
  enum class Type { Point, Directional, Spot, Area };
  Type type;

  Eigen::Vector3f position = Eigen::Vector3f::Zero();
  Eigen::Vector3f direction = Eigen::Vector3f(0, 0, -1);
  Eigen::Vector3f color = Eigen::Vector3f::Ones();
  float intensity = 1.0f;

  float cos_inner_cone = 1.0f;
  float cos_outer_cone = 0.70710678118654752440f;  // cos(pi/4)

  // Area Light Parameters
  float area = 0.0f;
  const Material* material = nullptr;
  const Geometry* geometry = nullptr;
  int geometry_index = -1;  // For internal use: index into Scene::geometries.
};

struct Environment {
  enum class Type { Texture, Preetham };
  Type type;

  // For Texture type (HDRi)
  Texture texture;
  // CDFs for Importance Sampling
  std::vector<float> marginal_cdf;                   // V-axis
  std::vector<std::vector<float>> conditional_cdfs;  // U-axis per V

  // For Preetham type
  Eigen::Vector3f sun_direction = Eigen::Vector3f(0, 1, 0);
  float turbidity = 2.5f;

  float intensity = 1.0f;
};

// --- Scene ---
struct Scene {
  std::vector<Geometry> geometries;
  std::vector<Material> materials;
  std::vector<Light> lights;
  std::optional<Environment> environment;
};

// Builds the CDFs for the environment texture if present.
void BuildEnvironmentCDF(Environment& env);

// Transforms the geometry by the transform matrix.
std::vector<Eigen::Vector3f> TransformedVertices(const Geometry& geometry);
std::vector<Eigen::Vector3f> TransformedNormals(const Geometry& geometry);
std::vector<Eigen::Vector4f> TransformedTangents(const Geometry& geometry);

// Builds an Embree BVH from the scene geometries.
RTCScene BuildBVH(const Scene& scene, RTCDevice device);

// Releases the Embree BVH.
void ReleaseBVH(RTCScene scene);

}  // namespace sh_baker

#endif  // SH_BAKER_SRC_SCENE_H_
