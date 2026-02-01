#pragma once

#define GL_GLEXT_PROTOTYPES
#include <GLFW/glfw3.h>

namespace sh_baker {

class BloomRenderer {
 public:
  BloomRenderer() = default;
  ~BloomRenderer();

  // Initialize framebuffers and shaders.
  // width, height: The size of the source HDR buffer (bloom will be half size).
  bool Init(int width, int height);

  // Resizes internal buffers if window size changes.
  // For now, we assume size is constant or Init is called again (with cleanup).
  // But strictly, we might need Resize(w, h).

  // Computes the bloom texture.
  // quad_vao: Screen quad VAO.
  // input_hdr_texture: Main scene HDR color.
  // luminance_texture: Average log luminance texture (for thresholding).
  void Compute(GLuint quad_vao, GLuint input_hdr_texture,
               GLuint luminance_texture);

  // Returns the final blurred bloom texture.
  GLuint GetBloomTexture() const;

 private:
  int width_ = 0;
  int height_ = 0;

  GLuint bright_program_ = 0;
  GLuint blur_program_ = 0;

  GLuint fbos_[2] = {0, 0};
  GLuint textures_[2] = {0, 0};
};

}  // namespace sh_baker
