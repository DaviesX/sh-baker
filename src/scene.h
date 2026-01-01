#ifndef SH_BAKER_SRC_SCENE_H_
#define SH_BAKER_SRC_SCENE_H_

#include <embree4/rtcore.h>

#include <Eigen/Dense>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace sh_baker {

// --- Texture ---
struct Texture {
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
  float roughness = 0.9f;
  float metallic = 0.0f;

  // Emission (for Area Lights).
  float emission_intensity = 0.0f;
};

// --- Geometry ---
struct Geometry {
  std::vector<Eigen::Vector3f> vertices;
  std::vector<Eigen::Vector3f> normals;
  std::vector<Eigen::Vector2f> uvs;

  std::vector<uint32_t> indices;

  uint32_t material_id = 0;  // Index into Scene::materials
  Eigen::Affine3f transform = Eigen::Affine3f::Identity();
};

// --- Light ---
struct Light {
  enum class Type { Point, Directional, Spot };
  Type type;

  Eigen::Vector3f position = Eigen::Vector3f::Zero();
  Eigen::Vector3f direction = Eigen::Vector3f(0, 0, -1);
  Eigen::Vector3f color = Eigen::Vector3f::Ones();
  float intensity = 1.0f;

  float inner_cone_angle = 0.0f;
  float outer_cone_angle = 0.785398f;  // pi/4
};

// --- SkyModel ---
struct SkyModel {
  Eigen::Vector3f sun_direction = Eigen::Vector3f(0, 1, 0).normalized();
  Eigen::Vector3f sun_color = Eigen::Vector3f::Ones();
  float sun_intensity = 1.0f;
};

// --- Scene ---
struct Scene {
  std::vector<Geometry> geometries;
  std::vector<Material> materials;
  std::vector<Light> lights;
  SkyModel sky;
};

// Builds an Embree BVH from the scene geometries.
RTCScene BuildBVH(const Scene& scene, RTCDevice device);

}  // namespace sh_baker

#endif  // SH_BAKER_SRC_SCENE_H_
