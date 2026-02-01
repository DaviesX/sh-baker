#pragma once

#define GL_GLEXT_PROTOTYPES
#include <GLFW/glfw3.h>

namespace sh_baker {

class ToneMapper {
 public:
  ToneMapper() = default;
  ~ToneMapper();

  bool Init();

  // Renders the final combined image to the default framebuffer (screen)
  // or currently bound framebuffer.
  void Draw(GLuint quad_vao, GLuint scene_texture, GLuint luminance_texture,
            GLuint bloom_texture);

 private:
  GLuint program_ = 0;
};

}  // namespace sh_baker
