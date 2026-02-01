#include "visualizer_utils.h"

#include <fstream>
#include <iostream>
#include <sstream>

#include "tinyexr.h"

#ifndef GL_TEXTURE_MAX_ANISOTROPY_EXT
#define GL_TEXTURE_MAX_ANISOTROPY_EXT 0x84FE
#define GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT 0x84FF
#endif

// We need to know external globals or pass them in?
// g_MaxAnisotropy is used in LoadTexture.
// We should probably pass anisotropy as a parameter or query it?
// For simplicity in this refactor, I will query it or defaulting it.
// Actually, let's use a static variable or just query GL state if possible?
// GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT is the *capability*.
// We want to set the PARAMETER.
// Let's assume we want max possible.
static float GetMaxAnisotropy() {
  static float max_aniso = 0.0f;
  if (max_aniso == 0.0f) {
    if (glfwExtensionSupported("GL_EXT_texture_filter_anisotropic")) {
      glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &max_aniso);
    } else {
      max_aniso = 1.0f;
    }
  }
  return max_aniso;
}

std::string ReadFile(const std::string& path) {
  std::ifstream t(path);
  if (!t.is_open()) {
    LOG(ERROR) << "Failed to open file: " << path;
    return "";
  }
  std::stringstream buffer;
  buffer << t.rdbuf();
  return buffer.str();
}

GLuint CompileShader(GLenum type, const std::string& source) {
  GLuint shader = glCreateShader(type);
  const char* src = source.c_str();
  glShaderSource(shader, 1, &src, nullptr);
  glCompileShader(shader);

  GLint success;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
  if (!success) {
    char infoLog[512];
    glGetShaderInfoLog(shader, 512, nullptr, infoLog);
    LOG(ERROR) << "Shader compilation failed:\n" << infoLog;
    return 0;
  }
  return shader;
}

GLuint CreateShaderProgram(const std::string& vertPath,
                           const std::string& fragPath) {
  std::string vertSrc = ReadFile(vertPath);
  std::string fragSrc = ReadFile(fragPath);
  if (vertSrc.empty() || fragSrc.empty()) return 0;

  GLuint vertex = CompileShader(GL_VERTEX_SHADER, vertSrc);
  GLuint fragment = CompileShader(GL_FRAGMENT_SHADER, fragSrc);
  if (!vertex || !fragment) return 0;

  GLuint program = glCreateProgram();
  glAttachShader(program, vertex);
  glAttachShader(program, fragment);
  glLinkProgram(program);

  GLint success;
  glGetProgramiv(program, GL_LINK_STATUS, &success);
  if (!success) {
    char infoLog[512];
    glGetProgramInfoLog(program, 512, nullptr, infoLog);
    LOG(ERROR) << "Program linking failed:\n" << infoLog;
    return 0;
  }
  glDeleteShader(vertex);
  glDeleteShader(fragment);
  return program;
}

GLuint LoadTexture(const sh_baker::Texture& tex) {
  GLuint tid;
  glGenTextures(1, &tid);
  glBindTexture(GL_TEXTURE_2D, tid);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, tex.width, tex.height, 0,
               tex.channels == 4 ? GL_RGBA : GL_RGB, GL_UNSIGNED_BYTE,
               tex.pixel_data.data());
  glGenerateMipmap(GL_TEXTURE_2D);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                  GL_LINEAR_MIPMAP_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

  float aniso = GetMaxAnisotropy();
  if (aniso > 1.0f) {
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, aniso);
  }
  return tid;
}

GLuint LoadTexture(const sh_baker::Texture32F& tex) {
  GLuint tid;
  glGenTextures(1, &tid);
  glBindTexture(GL_TEXTURE_2D, tid);
  // Upload as 16F
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, tex.width, tex.height, 0,
               tex.channels == 4 ? GL_RGBA : GL_RGB, GL_FLOAT,
               tex.pixel_data.data());
  glGenerateMipmap(GL_TEXTURE_2D);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                  GL_LINEAR_MIPMAP_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

  float aniso = GetMaxAnisotropy();
  if (aniso > 1.0f) {
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, aniso);
  }
  return tid;
}

GLuint LoadEXRTexture(const std::string& path) {
  float* out;
  int width;
  int height;
  const char* err = nullptr;

  int ret = LoadEXR(&out, &width, &height, path.c_str(), &err);
  if (ret != TINYEXR_SUCCESS) {
    if (err) {
      LOG(ERROR) << "LoadEXR failed: " << err;
      FreeEXRErrorMessage(err);
    }
    return 0;
  }

  GLuint tid;
  glGenTextures(1, &tid);
  glBindTexture(GL_TEXTURE_2D, tid);

  // Upload as RGB float
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, width, height, 0, GL_RGBA,
               GL_FLOAT, out);
  glGenerateMipmap(GL_TEXTURE_2D);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                  GL_LINEAR_MIPMAP_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

  float aniso = GetMaxAnisotropy();
  if (aniso > 1.0f) {
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, aniso);
  }

  free(out);
  return tid;
}

GLuint CreatePlaceholderTexture(float r, float g, float b) {
  GLuint tid;
  glGenTextures(1, &tid);
  glBindTexture(GL_TEXTURE_2D, tid);
  // GL_RGBA16F to match EXR
  float color[4] = {r, g, b, 1.0f};
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, 1, 1, 0, GL_RGBA, GL_FLOAT, color);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  return tid;
}
