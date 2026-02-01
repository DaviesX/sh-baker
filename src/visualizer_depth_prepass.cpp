#include "visualizer_depth_prepass.h"

#include <glog/logging.h>

#include "visualizer_utils.h"

namespace sh_baker {

DepthPrepass::~DepthPrepass() {
  if (program_) glDeleteProgram(program_);
}

bool DepthPrepass::Init() {
  program_ = CreateShaderProgram("glsl/depth.vert", "glsl/depth.frag");
  if (!program_) {
    LOG(ERROR) << "Failed to create depth prepass program.";
    return false;
  }
  return true;
}

void DepthPrepass::Draw(const Scene& scene,
                        const RadianceRenderer& radiance_renderer,
                        const Eigen::Matrix4f& vp) {
  if (!program_) return;

  glUseProgram(program_);

  // Disable Color Writes
  glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
  // Enable Depth Writes & Test
  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LESS);
  glDepthMask(GL_TRUE);

  const auto& meshes = radiance_renderer.GetMeshes();

  for (size_t i = 0; i < meshes.size(); ++i) {
    glBindVertexArray(meshes[i].vao);

    const auto& geo = scene.geometries[i];
    Eigen::Matrix4f model = geo.transform.matrix();
    Eigen::Matrix4f mvp = vp * model;

    glUniformMatrix4fv(glGetUniformLocation(program_, "u_MVP"), 1, GL_FALSE,
                       mvp.data());

    glDrawElements(GL_TRIANGLES, meshes[i].count, GL_UNSIGNED_INT, 0);
  }

  // Restore State
  glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
  // Main pass should use GL_EQUAL or GL_LEQUAL.
  // We leave depth func for caller to set, but we must restore masks.
  glUseProgram(0);
}

}  // namespace sh_baker
