#pragma once

#define GL_GLEXT_PROTOTYPES
#include <GLFW/glfw3.h>

namespace sh_baker {

class ExposureComputer {
 public:
  ExposureComputer() = default;
  ~ExposureComputer();

  // Initializes the framebuffer and texture for luminance computation.
  // Requires shaders to be loaded? No, maybe it loads its own shader?
  // The original code InitLuminanceFramebuffer didn't load shader, it was
  // loaded in main. To be self contained, we should probably load shader here
  // or pass it. Let's have Init load the shader.
  bool Init();

  // Computes the average log luminance from the input HDR texture.
  // quad_vao: The VAO of a screen-space quad (shared resource).
  // input_hdr_texture: The scene color texture to sample from.
  void Compute(GLuint quad_vao, GLuint input_hdr_texture);

  // Returns the 1x1 texture containing the average log luminance.
  GLuint GetLuminanceTexture() const { return texture_; }

 private:
  GLuint fbo_ = 0;
  GLuint texture_ = 0;
  GLuint program_ = 0;

  static const int kSize = 256;
};

}  // namespace sh_baker
