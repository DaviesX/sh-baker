#include "visualizer_bloom.h"

#include <glog/logging.h>

#include "visualizer_utils.h"

namespace sh_baker {

BloomRenderer::~BloomRenderer() {
  if (bright_program_) glDeleteProgram(bright_program_);
  if (blur_program_) glDeleteProgram(blur_program_);
  if (fbos_[0]) glDeleteFramebuffers(2, fbos_);
  if (textures_[0]) glDeleteTextures(2, textures_);
}

bool BloomRenderer::Init(int width, int height) {
  width_ = width / 2;
  height_ = height / 2;

  // 1. Init Framebuffers
  if (fbos_[0]) {
    glDeleteFramebuffers(2, fbos_);
    glDeleteTextures(2, textures_);
  }
  glGenFramebuffers(2, fbos_);
  glGenTextures(2, textures_);

  for (int i = 0; i < 2; i++) {
    glBindFramebuffer(GL_FRAMEBUFFER, fbos_[i]);
    glBindTexture(GL_TEXTURE_2D, textures_[i]);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, width_, height_, 0, GL_RGBA,
                 GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                           textures_[i], 0);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
      LOG(ERROR) << "Bloom Framebuffer " << i << " not complete!";
      return false;
    }
  }
  glBindFramebuffer(GL_FRAMEBUFFER, 0);

  // 2. Load Shaders
  bright_program_ = CreateShaderProgram("glsl/post.vert", "glsl/bright.frag");
  blur_program_ = CreateShaderProgram("glsl/post.vert", "glsl/blur.frag");

  if (!bright_program_ || !blur_program_) {
    LOG(ERROR) << "Failed to load bloom shaders";
    return false;
  }

  return true;
}

void BloomRenderer::Compute(GLuint quad_vao, GLuint input_hdr_texture,
                            GLuint luminance_texture) {
  if (!bright_program_ || !blur_program_) return;

  // 1. Extraction (Bright Pass)
  glViewport(0, 0, width_, height_);
  glBindFramebuffer(GL_FRAMEBUFFER, fbos_[0]);
  glUseProgram(bright_program_);

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, input_hdr_texture);
  glUniform1i(glGetUniformLocation(bright_program_, "u_HdrTex"), 0);

  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, luminance_texture);
  glUniform1i(glGetUniformLocation(bright_program_, "u_LumTexture"), 1);

  glBindVertexArray(quad_vao);
  glDrawArrays(GL_TRIANGLES, 0, 6);

  // 2. Blur (Ping Pong)
  glUseProgram(blur_program_);
  bool horizontal = true;
  int amount = 10;  // Increased amount for smoother bloom or match original '2'
                    // * loop logic?
  // Original was: amount = 2 loops.
  //   for (int i = 0; i < amount; i++) {
  //     draw(horizontal)
  //     horizontal = !horizontal
  //   }
  // That means it did H-blur then V-blur (if loop=2).
  // Actually, standard pingpong usually needs even number to land result in a
  // specific FBO? The original loop ran 2 times. i=0 horizontal=true. i=1
  // horizontal=false. Result ends in fbos_[0] (since i=1 draws to fbos_[0]).
  // Wait, original:
  // horizontal = true (initially)
  // i=0: draw to fbo[1], read tex[0]. horz=true. set horz=false.
  // i=1: draw to fbo[0], read tex[1]. horz=false. set horz=true.
  // Final result in tex[0]. Correct.

  amount = 2;  // Match original

  for (int i = 0; i < amount; i++) {
    glBindFramebuffer(GL_FRAMEBUFFER, fbos_[horizontal ? 1 : 0]);
    glUniform1i(glGetUniformLocation(blur_program_, "u_Horizontal"),
                horizontal);
    glUniform1i(glGetUniformLocation(blur_program_, "u_Image"), 0);

    glActiveTexture(GL_TEXTURE0);
    // Bind texture from OPPOSITE FBO (previous pass result) (or extraction
    // result which is in textures_[0]) i=0: read [0], write [1]. i=1: read [1],
    // write [0].
    glBindTexture(GL_TEXTURE_2D, textures_[horizontal ? 0 : 1]);

    glDrawArrays(GL_TRIANGLES, 0, 6);
    horizontal = !horizontal;
  }

  glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

GLuint BloomRenderer::GetBloomTexture() const {
  // Result ends in textures_[0] after loop
  return textures_[0];
}

}  // namespace sh_baker
