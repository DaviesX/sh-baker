#include "visualizer_sky.h"

#include <glog/logging.h>

#include "visualizer_utils.h"

namespace sh_baker {

SkyRenderer::~SkyRenderer() {
  if (vao_ != 0) glDeleteVertexArrays(1, &vao_);
  if (vbo_ != 0) glDeleteBuffers(1, &vbo_);
  if (ebo_ != 0) glDeleteBuffers(1, &ebo_);
}

void SkyRenderer::Init() {
  if (vao_ != 0) return;

  float skyboxVertices[] = {// positions
                            -1.0f, 1.0f,  -1.0f, -1.0f, -1.0f, -1.0f,
                            1.0f,  -1.0f, -1.0f, 1.0f,  1.0f,  -1.0f,
                            -1.0f, -1.0f, 1.0f,  -1.0f, 1.0f,  1.0f,
                            1.0f,  -1.0f, 1.0f,  1.0f,  1.0f,  1.0f};

  unsigned int skyboxIndices[] = {0, 1, 2, 2, 3, 0, 4, 1, 0, 0, 5, 4,
                                  2, 6, 7, 7, 3, 2, 4, 5, 7, 7, 6, 4,
                                  0, 3, 7, 7, 5, 0, 1, 4, 2, 2, 4, 6};

  glGenVertexArrays(1, &vao_);
  glGenBuffers(1, &vbo_);
  glGenBuffers(1, &ebo_);

  glBindVertexArray(vao_);

  glBindBuffer(GL_ARRAY_BUFFER, vbo_);
  glBufferData(GL_ARRAY_BUFFER, sizeof(skyboxVertices), &skyboxVertices,
               GL_STATIC_DRAW);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(skyboxIndices), &skyboxIndices,
               GL_STATIC_DRAW);

  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

  glBindVertexArray(0);
}

void SkyRenderer::SetProgram(GLuint program) { program_ = program; }

void SkyRenderer::UpdateAndBind(const sh_baker::Environment* env,
                                GLuint mesh_program) {
  // 1. Calculate and Bind SH Uniforms for the Mesh Program
  std::vector<float> sky_sh_data;
  sky_sh_data.reserve(9 * 3);

  if (env) {
    const auto& sh = env->sh_coeffs;
    for (int i = 0; i < 9; ++i) {
      sky_sh_data.push_back(sh.coeffs[i].x());
      sky_sh_data.push_back(sh.coeffs[i].y());
      sky_sh_data.push_back(sh.coeffs[i].z());
    }
  } else {
    // Placeholder: L0 = 3.5449 (convolution with Y00=0.282 yields ~1.0)
    float c0 = 3.5449f;
    sky_sh_data.push_back(c0);
    sky_sh_data.push_back(c0);
    sky_sh_data.push_back(c0);
    for (int i = 1; i < 9; ++i) {
      sky_sh_data.push_back(0.0f);
      sky_sh_data.push_back(0.0f);
      sky_sh_data.push_back(0.0f);
    }
  }

  // Bind to the mesh program (which renders objects lit by sky)
  GLint skySHLoc = glGetUniformLocation(mesh_program, "u_SkySH");
  if (skySHLoc != -1) {
    glUniform3fv(skySHLoc, 9, sky_sh_data.data());
  }

  // 2. Set Internal State for Drawing Skybox (Background)
  if (env) {
    if (env->type == sh_baker::Environment::Type::Texture) {
      // LoadTexture is now available via visualizer_utils.h
      GLuint tex = LoadTexture(env->texture);
      SetEnvironment(false, Eigen::Vector3f::Zero(), tex);
    } else {
      LOG(INFO) << "Using Preetham Sky (Scene). Sun Dir: "
                << env->sun_direction.transpose();
      SetEnvironment(true, env->sun_direction, 0);
    }
  } else {
    LOG(WARNING) << "No environment found in scene. Using default sky.";
    // Keep defaults (preetham=false, tex=0) or set explicitly?
    // Current defaults in constructor are safe.
  }
}

void SkyRenderer::SetEnvironment(bool use_preetham,
                                 const Eigen::Vector3f& sun_dir,
                                 GLuint texture_id) {
  use_preetham_ = use_preetham;
  sun_dir_ = sun_dir;
  texture_id_ = texture_id;
}

void SkyRenderer::Draw(const Eigen::Matrix4f& view,
                       const Eigen::Matrix4f& proj) {
  if (vao_ == 0) Init();
  if (program_ == 0) {
    LOG(WARNING) << "SkyRenderer program not set!";
    return;
  }

  glUseProgram(program_);

  glDepthFunc(GL_LEQUAL);

  // View matrix for skybox should remove translation
  Eigen::Matrix4f viewSky = view;
  viewSky(0, 3) = 0;
  viewSky(1, 3) = 0;
  viewSky(2, 3) = 0;
  Eigen::Matrix4f mvpSky = proj * viewSky;

  glUniformMatrix4fv(glGetUniformLocation(program_, "u_MVP"), 1, GL_FALSE,
                     mvpSky.data());

  glUniform1i(glGetUniformLocation(program_, "u_UsePreetham"), use_preetham_);

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, texture_id_);
  glUniform1i(glGetUniformLocation(program_, "u_SkyboxTex"), 0);

  glUniform3fv(glGetUniformLocation(program_, "u_SunDir"), 1, sun_dir_.data());

  glBindVertexArray(vao_);
  glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);
  glBindVertexArray(0);

  glDepthFunc(GL_LESS);
}

}  // namespace sh_baker
