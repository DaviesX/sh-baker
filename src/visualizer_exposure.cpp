#include "visualizer_exposure.h"

#include <glog/logging.h>

#include "visualizer_utils.h"

namespace sh_baker {

ExposureComputer::~ExposureComputer() {
  if (fbo_) glDeleteFramebuffers(1, &fbo_);
  if (texture_) glDeleteTextures(1, &texture_);
  if (program_) glDeleteProgram(program_);
}

bool ExposureComputer::Init() {
  // 1. Create Framebuffer and Texture
  if (fbo_) {
    glDeleteFramebuffers(1, &fbo_);
    glDeleteTextures(1, &texture_);
  }
  glGenFramebuffers(1, &fbo_);
  glBindFramebuffer(GL_FRAMEBUFFER, fbo_);

  glGenTextures(1, &texture_);
  glBindTexture(GL_TEXTURE_2D, texture_);
  // R16F is sufficient for log luminance
  glTexImage2D(GL_TEXTURE_2D, 0, GL_R16F, kSize, kSize, 0, GL_RED, GL_FLOAT,
               NULL);
  // Mipmaps needed for average
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                  GL_LINEAR_MIPMAP_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                         texture_, 0);

  if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
    LOG(ERROR) << "Luminance Framebuffer not complete!";
    return false;
  }

  glBindFramebuffer(GL_FRAMEBUFFER, 0);

  // 2. Load Shader
  program_ = CreateShaderProgram("glsl/post.vert", "glsl/lum.frag");
  if (!program_) {
    LOG(ERROR) << "Failed to create luminance program";
    return false;
  }

  return true;
}

void ExposureComputer::Compute(GLuint quad_vao, GLuint input_hdr_texture) {
  if (!program_) {
    LOG(WARNING) << "ExposureComputer not initialized!";
    return;
  }

  glBindFramebuffer(GL_FRAMEBUFFER, fbo_);
  glViewport(0, 0, kSize, kSize);
  glUseProgram(program_);
  glDisable(GL_DEPTH_TEST);

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, input_hdr_texture);
  glUniform1i(glGetUniformLocation(program_, "u_HdrTex"), 0);

  glBindVertexArray(quad_vao);
  glDrawArrays(GL_TRIANGLES, 0, 6);

  // Generate Mipmaps to average
  glBindTexture(GL_TEXTURE_2D, texture_);
  glGenerateMipmap(GL_TEXTURE_2D);

  // Restore state if necessary, but caller (DrawPostProcess) handles its own
  // viewport/fbo usually Ideally we unbind FBO
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

}  // namespace sh_baker
