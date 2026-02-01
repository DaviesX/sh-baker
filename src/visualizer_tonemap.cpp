#include "visualizer_tonemap.h"

#include <glog/logging.h>

#include "visualizer_utils.h"

namespace sh_baker {

ToneMapper::~ToneMapper() {
  if (program_) glDeleteProgram(program_);
}

bool ToneMapper::Init() {
  program_ = CreateShaderProgram("glsl/post.vert", "glsl/post.frag");
  if (!program_) {
    LOG(ERROR) << "Failed to create tonemap program";
    return false;
  }
  return true;
}

void ToneMapper::Draw(GLuint quad_vao, GLuint scene_texture,
                      GLuint luminance_texture, GLuint bloom_texture) {
  if (!program_) return;

  glUseProgram(program_);
  glDisable(GL_DEPTH_TEST);

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, scene_texture);
  glUniform1i(glGetUniformLocation(program_, "u_ScreenTexture"), 0);

  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, luminance_texture);
  glUniform1i(glGetUniformLocation(program_, "u_LumTexture"), 1);

  glActiveTexture(GL_TEXTURE2);
  glBindTexture(GL_TEXTURE_2D, bloom_texture);
  glUniform1i(glGetUniformLocation(program_, "u_BloomTexture"), 2);

  glBindVertexArray(quad_vao);
  glDrawArrays(GL_TRIANGLES, 0, 6);

  glEnable(GL_DEPTH_TEST);
}

}  // namespace sh_baker
