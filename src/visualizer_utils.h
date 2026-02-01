#ifndef SH_BAKER_SRC_VISUALIZER_UTILS_H_
#define SH_BAKER_SRC_VISUALIZER_UTILS_H_

#define GL_GLEXT_PROTOTYPES
// #define GL_SILENCE_DEPRECATION // Already defined in most cpp files, but good
// to have
#include <GLFW/glfw3.h>
#include <glog/logging.h>

#include <string>

#include "scene.h"  // For Texture/Texture32F

// --- Helper Functions ---

std::string ReadFile(const std::string& path);

GLuint CompileShader(GLenum type, const std::string& source);

GLuint CreateShaderProgram(const std::string& vertPath,
                           const std::string& fragPath);

GLuint LoadTexture(const sh_baker::Texture& tex);

GLuint LoadTexture(const sh_baker::Texture32F& tex);

GLuint LoadEXRTexture(const std::string& path);

GLuint CreatePlaceholderTexture(float r, float g, float b);

#endif  // SH_BAKER_SRC_VISUALIZER_UTILS_H_
