#pragma once

#include <Eigen/Dense>
#include <filesystem>
#include <vector>

#include "scene.h"
// visualizer_utils.h defines GL headers
#include "visualizer_utils.h"

namespace sh_baker {

class RadianceRenderer {
 public:
  RadianceRenderer() = default;
  ~RadianceRenderer();

  // Loads shaders, uploads geometry, loads all textures (albedo, normal, MR,
  // SH).
  bool Init(const Scene& scene, const std::filesystem::path& input_dir);

  // Define static uniforms like texture units. Call this once after Init or if
  // needed. Although Init covers it, sometimes we might want to re-bind if
  // other passes mess up texture units? But typically Draw sets state.

  // Draws the scene geometry.
  // Requires Scene to access transforms (assuming 1:1 mapping with internal
  // meshes).
  void Draw(const Scene& scene, const Eigen::Matrix4f& vp,
            const Eigen::Vector3f& cam_pos);

  GLuint GetProgram() const { return program_; }

 private:
  struct RenderMesh {
    GLuint vao;
    GLsizei count;
    int material_id;
    // We could store transform here to avoid passing Scene to Draw,
    // but the Scene object might be updated? For now, we follow existing
    // pattern.
  };

  GLuint program_ = 0;
  std::vector<RenderMesh> meshes_;

  // Texture Resources
  std::vector<GLuint> albedo_textures_;
  std::vector<GLuint> normal_textures_;
  std::vector<GLuint> mr_textures_;
  std::vector<GLuint> sh_textures_;

  bool use_packed_luminance_ = false;

  void UploadGeometry(const Scene& scene);
  void LoadMaterials(const Scene& scene);
  void LoadSH(const std::filesystem::path& input_dir);
};

}  // namespace sh_baker
