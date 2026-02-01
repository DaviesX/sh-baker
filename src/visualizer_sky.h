#pragma once

#include <Eigen/Dense>

// We need GL headers. Since visualizer_main uses GLFW directly:
#define GL_GLEXT_PROTOTYPES
#include <GLFW/glfw3.h>

namespace sh_baker {

class SkyRenderer {
 public:
  SkyRenderer() = default;
  ~SkyRenderer();

  // Initialize the renderer (buffers, VAO).
  // Does NOT compile the shader; you must call SetProgram.
  void Init();

  // Set the shader program to use.
  void SetProgram(GLuint program);

  // Set the environment parameters.
  void SetEnvironment(bool use_preetham, const Eigen::Vector3f& sun_dir,
                      GLuint texture_id);

  // Draw the skybox.
  // Note: view matrix should contain rotation but likely not translation
  // (handled internally or by caller). The original implementation removed
  // translation from view matrix inside DrawSky.
  void Draw(const Eigen::Matrix4f& view, const Eigen::Matrix4f& proj);

 private:
  GLuint program_ = 0;
  GLuint vao_ = 0;
  GLuint vbo_ = 0;
  GLuint ebo_ = 0;

  bool use_preetham_ = false;
  Eigen::Vector3f sun_dir_ = Eigen::Vector3f(0, 1, 0);
  GLuint texture_id_ = 0;
};

}  // namespace sh_baker
