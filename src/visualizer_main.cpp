#define GL_GLEXT_PROTOTYPES
#define GL_SILENCE_DEPRECATION
#include <GLFW/glfw3.h>
#include <gflags/gflags.h>
#include <glog/logging.h>

#include <Eigen/Dense>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "loader.h"
#include "scene.h"
#include "tinyexr.h"

#ifndef GL_TEXTURE_MAX_ANISOTROPY_EXT
#define GL_TEXTURE_MAX_ANISOTROPY_EXT 0x84FE
#define GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT 0x84FF
#endif

// --- Constants ---
const int kWindowWidth = 1280;
const int kWindowHeight = 720;

// --- Flags ---
DEFINE_string(input, "",
              "Path to the input folder containing scene.gltf and "
              "lightmap_*.exr files.");

// --- Globals ---
// --- Globals (Additional) ---
GLuint g_MeshProgram = 0;
GLuint g_SkyProgram = 0;
GLuint g_PostProgram = 0;

// HDR Framebuffer (MSAA)
GLuint g_HdrFBO_MS = 0;
GLuint g_HdrColorTexture_MS = 0;
GLuint g_HdrDepthRBO_MS = 0;

// HDR Framebuffer (Resolve / Post-Process Input)
GLuint g_HdrFBO_Resolve = 0;
GLuint g_HdrColorTexture_Resolve = 0;

// Luminance Framebuffer (Auto Exposure)
GLuint g_LumProgram = 0;
GLuint g_LumFBO = 0;
GLuint g_LumTexture = 0;
const int kLumSize = 256;

// Bloom Framebuffers (Ping Pong)
GLuint g_BrightProgram = 0;
GLuint g_BlurProgram = 0;
GLuint g_BloomFBO[2] = {0, 0};
GLuint g_BloomTextures[2] = {0, 0};
int kBloomWidth = 0;
int kBloomHeight = 0;

void InitBloomFramebuffers(int width, int height) {
  kBloomWidth = width / 2;
  kBloomHeight = height / 2;

  glGenFramebuffers(2, g_BloomFBO);
  glGenTextures(2, g_BloomTextures);

  for (int i = 0; i < 2; i++) {
    glBindFramebuffer(GL_FRAMEBUFFER, g_BloomFBO[i]);
    glBindTexture(GL_TEXTURE_2D, g_BloomTextures[i]);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, kBloomWidth, kBloomHeight, 0,
                 GL_RGBA, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                           g_BloomTextures[i], 0);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
      LOG(ERROR) << "Bloom Framebuffer " << i << " not complete!";
  }
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

// Screen Quad
GLuint g_QuadVAO = 0;
GLuint g_QuadVBO = 0;

// Skybox Data (Global for access by DrawSky, setup in main)
GLuint g_SkyboxTexture = 0;
bool g_UsePreetham = false;
Eigen::Vector3f g_SunDir(0, 1, 0);
float g_SkyIntensity = 1.0f;
static GLuint g_CubeVAO = 0;
static GLuint g_CubeVBO = 0;

// ... (RenderMesh struct, vector globals unchanged) ...

void InitScreenQuad() {
  if (g_QuadVAO == 0) {
    float quadVertices[] = {// positions   // texCoords
                            -1.0f, 1.0f, 0.0f, 1.0f,  -1.0f, -1.0f,
                            0.0f,  0.0f, 1.0f, -1.0f, 1.0f,  0.0f,

                            -1.0f, 1.0f, 0.0f, 1.0f,  1.0f,  -1.0f,
                            1.0f,  0.0f, 1.0f, 1.0f,  1.0f,  1.0f};
    glGenVertexArrays(1, &g_QuadVAO);
    glGenBuffers(1, &g_QuadVBO);
    glBindVertexArray(g_QuadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, g_QuadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices,
                 GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                          (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                          (void*)(2 * sizeof(float)));
  }
}

void InitLuminanceFramebuffer() {
  if (g_LumFBO) {
    glDeleteFramebuffers(1, &g_LumFBO);
    glDeleteTextures(1, &g_LumTexture);
  }
  glGenFramebuffers(1, &g_LumFBO);
  glBindFramebuffer(GL_FRAMEBUFFER, g_LumFBO);

  glGenTextures(1, &g_LumTexture);
  glBindTexture(GL_TEXTURE_2D, g_LumTexture);
  // R16F is sufficient for log luminance
  glTexImage2D(GL_TEXTURE_2D, 0, GL_R16F, kLumSize, kLumSize, 0, GL_RED,
               GL_FLOAT, NULL);
  // Mipmaps needed for average
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                  GL_LINEAR_MIPMAP_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                         g_LumTexture, 0);

  if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    LOG(ERROR) << "Luminance Framebuffer not complete!";

  glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void InitHdrFramebuffer(int width, int height) {
  // 1. MSAA Framebuffer
  if (g_HdrFBO_MS) {
    glDeleteFramebuffers(1, &g_HdrFBO_MS);
    glDeleteTextures(1, &g_HdrColorTexture_MS);
    glDeleteRenderbuffers(1, &g_HdrDepthRBO_MS);
  }
  glGenFramebuffers(1, &g_HdrFBO_MS);
  glBindFramebuffer(GL_FRAMEBUFFER, g_HdrFBO_MS);

  glGenTextures(1, &g_HdrColorTexture_MS);
  glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, g_HdrColorTexture_MS);
  glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, 4, GL_RGBA16F, width,
                          height, GL_TRUE);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                         GL_TEXTURE_2D_MULTISAMPLE, g_HdrColorTexture_MS, 0);

  glGenRenderbuffers(1, &g_HdrDepthRBO_MS);
  glBindRenderbuffer(GL_RENDERBUFFER, g_HdrDepthRBO_MS);
  glRenderbufferStorageMultisample(GL_RENDERBUFFER, 4, GL_DEPTH_COMPONENT24,
                                   width, height);
  glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                            GL_RENDERBUFFER, g_HdrDepthRBO_MS);

  if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    LOG(ERROR) << "MSAA Framebuffer not complete!";

  // 2. Resolve Framebuffer (Standard Texture)
  if (g_HdrFBO_Resolve) {
    glDeleteFramebuffers(1, &g_HdrFBO_Resolve);
    glDeleteTextures(1, &g_HdrColorTexture_Resolve);
  }
  glGenFramebuffers(1, &g_HdrFBO_Resolve);
  glBindFramebuffer(GL_FRAMEBUFFER, g_HdrFBO_Resolve);

  glGenTextures(1, &g_HdrColorTexture_Resolve);
  glBindTexture(GL_TEXTURE_2D, g_HdrColorTexture_Resolve);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, width, height, 0, GL_RGBA,
               GL_FLOAT, NULL);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                         g_HdrColorTexture_Resolve, 0);

  if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    LOG(ERROR) << "Resolve Framebuffer not complete!";

  glBindFramebuffer(GL_FRAMEBUFFER, 0);

  // Init Luminance
  InitLuminanceFramebuffer();

  // Init Bloom
  InitBloomFramebuffers(width, height);
}

void DrawPostProcess(int width, int height) {
  if (g_QuadVAO == 0) InitScreenQuad();

  // 1. Resolve MSAA to Texture
  glBindFramebuffer(GL_READ_FRAMEBUFFER, g_HdrFBO_MS);
  glBindFramebuffer(GL_DRAW_FRAMEBUFFER, g_HdrFBO_Resolve);
  glBlitFramebuffer(0, 0, width, height, 0, 0, width, height,
                    GL_COLOR_BUFFER_BIT, GL_NEAREST);

  // 2. Compute Average Log Luminance
  glBindFramebuffer(GL_FRAMEBUFFER, g_LumFBO);
  glViewport(0, 0, kLumSize, kLumSize);
  glUseProgram(g_LumProgram);
  glDisable(GL_DEPTH_TEST);

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, g_HdrColorTexture_Resolve);
  glUniform1i(glGetUniformLocation(g_LumProgram, "u_HdrTex"), 0);

  glBindVertexArray(g_QuadVAO);
  glDrawArrays(GL_TRIANGLES, 0, 6);

  // Generate Mipmaps to average
  glBindTexture(GL_TEXTURE_2D, g_LumTexture);
  glGenerateMipmap(GL_TEXTURE_2D);

  // 3. Bloom Extraction (Bright Pass)
  glViewport(0, 0, kBloomWidth, kBloomHeight);
  glBindFramebuffer(GL_FRAMEBUFFER, g_BloomFBO[0]);
  glUseProgram(g_BrightProgram);

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, g_HdrColorTexture_Resolve);
  glUniform1i(glGetUniformLocation(g_BrightProgram, "u_HdrTex"), 0);

  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, g_LumTexture);
  glUniform1i(glGetUniformLocation(g_BrightProgram, "u_LumTexture"),
              1);  // Uses mipmapped lum

  glDrawArrays(GL_TRIANGLES, 0, 6);

  // 4. Bloom Blur (Ping Pong)
  glUseProgram(g_BlurProgram);
  bool horizontal = true;
  int amount = 2;

  for (int i = 0; i < amount; i++) {
    glBindFramebuffer(GL_FRAMEBUFFER, g_BloomFBO[horizontal ? 1 : 0]);
    glUniform1i(glGetUniformLocation(g_BlurProgram, "u_Horizontal"),
                horizontal);
    glUniform1i(glGetUniformLocation(g_BlurProgram, "u_Image"), 0);

    glActiveTexture(GL_TEXTURE0);
    // Bind texture from OPPOSITE FBO (previous pass result)
    glBindTexture(GL_TEXTURE_2D, g_BloomTextures[horizontal ? 0 : 1]);

    glDrawArrays(GL_TRIANGLES, 0, 6);
    horizontal = !horizontal;
  }

  // 5. Render Final Post Process to Screen
  glEnable(GL_FRAMEBUFFER_SRGB);
  glBindFramebuffer(GL_FRAMEBUFFER, 0);  // Back to default for drawing quad
  glViewport(0, 0, width, height);       // Restore viewport

  glUseProgram(g_PostProgram);
  glDisable(GL_DEPTH_TEST);  // Already disabled, but good to be explicit

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, g_HdrColorTexture_Resolve);
  glUniform1i(glGetUniformLocation(g_PostProgram, "u_ScreenTexture"), 0);

  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, g_LumTexture);
  glUniform1i(glGetUniformLocation(g_PostProgram, "u_LumTexture"), 1);

  glActiveTexture(GL_TEXTURE2);
  glBindTexture(GL_TEXTURE_2D, g_BloomTextures[0]);  // Final blur result
  glUniform1i(glGetUniformLocation(g_PostProgram, "u_BloomTexture"), 2);

  glBindVertexArray(g_QuadVAO);
  glDrawArrays(GL_TRIANGLES, 0, 6);

  glEnable(GL_DEPTH_TEST);
}

struct RenderMesh {
  GLuint vao;
  GLsizei count;
  int material_id;
};
std::vector<RenderMesh> g_Meshes;

std::vector<GLuint> g_AlbedoTextures;
std::vector<GLuint> g_NormalTextures;
std::vector<GLuint> g_MRTextures;
std::vector<GLuint> g_SHTextures;

float g_MaxAnisotropy = 1.0f;

// Camera
Eigen::Vector3f g_CamPos(0, 0, 5);
Eigen::Vector3f g_CamTarget(0, 0, 0);
float g_CamYaw = 0.0f;
float g_CamPitch = 0.0f;
float g_CamDist = 5.0f;
bool g_MousePressed = false;
double g_LastMouseX, g_LastMouseY;

// --- Helper Functions ---

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
  if (g_MaxAnisotropy > 1.0f) {
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT,
                    g_MaxAnisotropy);
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
  if (g_MaxAnisotropy > 1.0f) {
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT,
                    g_MaxAnisotropy);
  }
  return tid;
}

GLuint LoadTextureVariant(
    const std::variant<sh_baker::Texture, sh_baker::Texture32F>& tex_var) {
  return std::visit([](const auto& t) { return LoadTexture(t); }, tex_var);
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
  if (g_MaxAnisotropy > 1.0f) {
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT,
                    g_MaxAnisotropy);
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

void ProcessInput(GLFWwindow* window) {
  if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    glfwSetWindowShouldClose(window, true);
}

void MouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
  if (button == GLFW_MOUSE_BUTTON_LEFT) {
    if (action == GLFW_PRESS) {
      g_MousePressed = true;
      glfwGetCursorPos(window, &g_LastMouseX, &g_LastMouseY);
    } else {
      g_MousePressed = false;
    }
  }
}

void CursorPosCallback(GLFWwindow* window, double xpos, double ypos) {
  if (g_MousePressed) {
    float dx = static_cast<float>(xpos - g_LastMouseX);
    float dy = static_cast<float>(ypos - g_LastMouseY);

    g_CamYaw -= dx * 0.01f;
    g_CamPitch -= dy * 0.01f;

    // Clamp pitch
    g_CamPitch = std::max(-1.5f, std::min(1.5f, g_CamPitch));

    g_LastMouseX = xpos;
    g_LastMouseY = ypos;
  }
}

void ScrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
  g_CamDist -= static_cast<float>(yoffset) * 0.5f;
  if (g_CamDist < 0.1f) g_CamDist = 0.1f;
}

void InitSkyboxGeometry() {
  if (g_CubeVAO == 0) {
    float skyboxVertices[] = {// positions
                              -1.0f, 1.0f,  -1.0f, -1.0f, -1.0f, -1.0f,
                              1.0f,  -1.0f, -1.0f, 1.0f,  1.0f,  -1.0f,
                              -1.0f, -1.0f, 1.0f,  -1.0f, 1.0f,  1.0f,
                              1.0f,  -1.0f, 1.0f,  1.0f,  1.0f,  1.0f};
    unsigned int skyboxIndices[] = {0, 1, 2, 2, 3, 0, 4, 1, 0, 0, 5, 4,
                                    2, 6, 7, 7, 3, 2, 4, 5, 7, 7, 6, 4,
                                    0, 3, 7, 7, 5, 0, 1, 4, 2, 2, 4, 6};
    glGenVertexArrays(1, &g_CubeVAO);
    glGenBuffers(1, &g_CubeVBO);
    GLuint cubeEBO;
    glGenBuffers(1, &cubeEBO);
    glBindVertexArray(g_CubeVAO);
    glBindBuffer(GL_ARRAY_BUFFER, g_CubeVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(skyboxVertices), &skyboxVertices,
                 GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, cubeEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(skyboxIndices), &skyboxIndices,
                 GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float),
                          (void*)0);
  }
}

void DrawMeshes(const sh_baker::Scene& scene, const Eigen::Matrix4f& vp,
                const Eigen::Vector3f& cam_pos) {
  glUseProgram(g_MeshProgram);

  // Pass CamPos to Shader
  glUniform3fv(glGetUniformLocation(g_MeshProgram, "u_CamPos"), 1,
               cam_pos.data());

  // Bind SH Textures (Static uniforms, but Texture Units need binding)
  for (size_t i = 0; i < g_SHTextures.size(); ++i) {
    glActiveTexture(GL_TEXTURE1 + i);
    glBindTexture(GL_TEXTURE_2D, g_SHTextures[i]);
  }

  // Draw Meshes
  for (size_t i = 0; i < g_Meshes.size(); ++i) {
    glBindVertexArray(g_Meshes[i].vao);

    const auto& geo = scene.geometries[i];
    Eigen::Matrix4f model = geo.transform.matrix();
    Eigen::Matrix4f mvp = vp * model;

    glUniformMatrix4fv(glGetUniformLocation(g_MeshProgram, "u_MVP"), 1,
                       GL_FALSE, mvp.data());
    glUniformMatrix4fv(glGetUniformLocation(g_MeshProgram, "u_Model"), 1,
                       GL_FALSE, model.data());

    int mat_id = g_Meshes[i].material_id;
    CHECK_GT(mat_id, -1);
    CHECK_LT(mat_id, scene.materials.size());

    // Albedo
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, g_AlbedoTextures[mat_id]);

    // Normal
    glActiveTexture(GL_TEXTURE4);
    glBindTexture(GL_TEXTURE_2D, g_NormalTextures[mat_id]);

    // MR
    glActiveTexture(GL_TEXTURE5);
    glBindTexture(GL_TEXTURE_2D, g_MRTextures[mat_id]);

    glDrawElements(GL_TRIANGLES, g_Meshes[i].count, GL_UNSIGNED_INT, 0);
  }
}

void DrawSky(const Eigen::Matrix4f& view, const Eigen::Matrix4f& proj) {
  if (g_CubeVAO == 0) InitSkyboxGeometry();

  glUseProgram(g_SkyProgram);

  glDepthFunc(GL_LEQUAL);

  // View matrix for skybox should remove translation
  Eigen::Matrix4f viewSky = view;
  viewSky(0, 3) = 0;
  viewSky(1, 3) = 0;
  viewSky(2, 3) = 0;
  Eigen::Matrix4f mvpSky = proj * viewSky;

  glUniformMatrix4fv(glGetUniformLocation(g_SkyProgram, "u_MVP"), 1, GL_FALSE,
                     mvpSky.data());

  glUniform1i(glGetUniformLocation(g_SkyProgram, "u_UsePreetham"),
              g_UsePreetham);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, g_SkyboxTexture);
  glUniform1i(glGetUniformLocation(g_SkyProgram, "u_SkyboxTex"), 0);
  glUniform3fv(glGetUniformLocation(g_SkyProgram, "u_SunDir"), 1,
               g_SunDir.data());
  glUniform1f(glGetUniformLocation(g_SkyProgram, "u_SkyIntensity"),
              g_SkyIntensity);

  glBindVertexArray(g_CubeVAO);
  glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, 0);
  glDepthFunc(GL_LESS);
}

int main(int argc, char* argv[]) {
  FLAGS_logtostderr = 1;
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  if (FLAGS_input.empty()) {
    std::cerr << "Usage: " << argv[0] << " --input <folder>" << std::endl;
    return 1;
  }

  std::filesystem::path input_dir(FLAGS_input);
  if (!std::filesystem::exists(input_dir) ||
      !std::filesystem::is_directory(input_dir)) {
    LOG(ERROR) << "Input is not a valid directory: " << FLAGS_input;
    return 1;
  }

  // --- Init GLFW ---
  if (!glfwInit()) return -1;

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_SAMPLES, 4);  // 4x MSAA
#ifdef __APPLE__
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

  GLFWwindow* window = glfwCreateWindow(
      kWindowWidth, kWindowHeight, "SH Baker Visualizer", nullptr, nullptr);
  if (!window) {
    glfwTerminate();
    return -1;
  }
  glfwMakeContextCurrent(window);
  glfwSetMouseButtonCallback(window, MouseButtonCallback);
  glfwSetCursorPosCallback(window, CursorPosCallback);
  glfwSetScrollCallback(window, ScrollCallback);

  // glEnable(GL_FRAMEBUFFER_SRGB);
  glEnable(GL_MULTISAMPLE);  // Enable MSAA locally if supported, but FBO is
                             // single sample for now.

  // Check for Anisotropic Filtering support
  if (glfwExtensionSupported("GL_EXT_texture_filter_anisotropic")) {
    glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, &g_MaxAnisotropy);
    LOG(INFO) << "Anisotropic Filtering Enabled. Max Anisotropy: "
              << g_MaxAnisotropy;
  } else {
    LOG(WARNING) << "Anisotropic Filtering NOT supported.";
  }

  // --- Init HDR FBO ---
  InitHdrFramebuffer(kWindowWidth, kWindowHeight);

  // --- Load Scene ---
  auto scene_path = input_dir / "scene.gltf";
  LOG(INFO) << "Loading scene: " << scene_path;
  auto scene_opt = sh_baker::LoadScene(scene_path);
  if (!scene_opt) {
    LOG(ERROR) << "Failed to load scene";
    return 1;
  }
  const auto& scene = *scene_opt;
  LOG(INFO) << "Scene loaded successfully.";
  LOG(INFO) << "  Geometries: " << scene.geometries.size();
  LOG(INFO) << "  Materials: " << scene.materials.size();
  LOG(INFO) << "  Lights: " << scene.lights.size();

  // --- Upload Geometry ---
  for (const auto& geo : scene.geometries) {
    RenderMesh mesh;
    mesh.count = static_cast<GLsizei>(geo.indices.size());
    mesh.material_id = geo.material_id;

    glGenVertexArrays(1, &mesh.vao);
    glBindVertexArray(mesh.vao);

    GLuint vbo[5];  // Pos, Normal, UV0, UV1, Tangent
    GLuint ebo;

    glGenBuffers(5, vbo);
    glGenBuffers(1, &ebo);

    // 0: Position
    glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
    glBufferData(GL_ARRAY_BUFFER, geo.vertices.size() * sizeof(Eigen::Vector3f),
                 geo.vertices.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

    // 1: Normal
    glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
    glBufferData(GL_ARRAY_BUFFER, geo.normals.size() * sizeof(Eigen::Vector3f),
                 geo.normals.data(), GL_STATIC_DRAW);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);

    // 2: UV0
    if (!geo.texture_uvs.empty()) {
      glBindBuffer(GL_ARRAY_BUFFER, vbo[2]);
      glBufferData(GL_ARRAY_BUFFER,
                   geo.texture_uvs.size() * sizeof(Eigen::Vector2f),
                   geo.texture_uvs.data(), GL_STATIC_DRAW);
      glEnableVertexAttribArray(2);
      glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, (void*)0);
    }

    // 3: UV1
    if (!geo.lightmap_uvs.empty()) {
      glBindBuffer(GL_ARRAY_BUFFER, vbo[3]);
      glBufferData(GL_ARRAY_BUFFER,
                   geo.lightmap_uvs.size() * sizeof(Eigen::Vector2f),
                   geo.lightmap_uvs.data(), GL_STATIC_DRAW);
      glEnableVertexAttribArray(3);
      glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, 0, (void*)0);
    }

    // 4: Tangent
    if (!geo.tangents.empty()) {
      glBindBuffer(GL_ARRAY_BUFFER, vbo[4]);
      glBufferData(GL_ARRAY_BUFFER,
                   geo.tangents.size() * sizeof(Eigen::Vector4f),
                   geo.tangents.data(), GL_STATIC_DRAW);
      glEnableVertexAttribArray(4);
      glVertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, 0, (void*)0);
    }

    // EBO
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, geo.indices.size() * sizeof(uint32_t),
                 geo.indices.data(), GL_STATIC_DRAW);

    g_Meshes.push_back(mesh);
  }

  // --- Load Materials ---
  for (const auto& mat : scene.materials) {
    // Albedo
    if (!mat.albedo.pixel_data.empty()) {
      g_AlbedoTextures.push_back(LoadTexture(mat.albedo));
    } else {
      g_AlbedoTextures.push_back(0);
    }
    // Normal
    if (!mat.normal_texture.pixel_data.empty()) {
      g_NormalTextures.push_back(LoadTexture(mat.normal_texture));
    } else {
      g_NormalTextures.push_back(0);
    }
    // Metallic/Roughness
    if (!mat.metallic_roughness_texture.pixel_data.empty()) {
      g_MRTextures.push_back(LoadTexture(mat.metallic_roughness_texture));
    } else {
      g_MRTextures.push_back(0);
    }
  }

  // --- Load SH Textures ---
  const char* kCoeffSuffixes[] = {"L0",   "L1m1", "L10", "L11", "L2m2",
                                  "L2m1", "L20",  "L21", "L22"};

  bool use_packed_luminance = false;

  // Check if packed file exists
  if (std::filesystem::exists(input_dir / "lightmap_packed_0.exr")) {
    use_packed_luminance = true;
    LOG(INFO) << "Detected Packed Luminance SH Lightmaps.";

    for (int i = 0; i < 3; ++i) {
      std::string filename = "lightmap_packed_" + std::to_string(i) + ".exr";
      std::filesystem::path p = input_dir / filename;
      GLuint tid = LoadEXRTexture(p.string());
      if (tid == 0) LOG(WARNING) << "Failed to load SH texture: " << p;
      g_SHTextures.push_back(tid);
    }
  } else if (std::filesystem::exists(input_dir / "lightmap_L0.exr")) {
    LOG(INFO) << "Using Standard SH Lightmaps (9 files).";
    for (int i = 0; i < 9; ++i) {
      std::string filename =
          "lightmap_" + std::string(kCoeffSuffixes[i]) + ".exr";
      std::filesystem::path p = input_dir / filename;
      GLuint tid = LoadEXRTexture(p.string());
      if (tid == 0) {
        LOG(WARNING) << "Failed to load SH texture: " << p;
      }
      g_SHTextures.push_back(tid);
    }
  } else {
    LOG(INFO) << "No SH Lightmaps found. Using 1x1 Placeholders (Luminance "
                 "Only).";
    // L0 = Approx 1.0 radiance
    // Y00 = 0.282. Coeff = Radiance / Y00
    // Reconstruction: Color = C0 * Y00 ...
    // If we want Color=1, C0 * 0.282 = 1 => C0 = 3.54
    float c0 = 3.5449f;
    g_SHTextures.push_back(CreatePlaceholderTexture(c0, c0, c0));
    for (int i = 1; i < 9; ++i) {
      g_SHTextures.push_back(CreatePlaceholderTexture(0.0f, 0.0f, 0.0f));
    }
  }

  // --- Setup Shaders ---
  // --- Setup Shaders ---
  g_MeshProgram = CreateShaderProgram("glsl/viz.vert", "glsl/viz.frag");
  g_SkyProgram = CreateShaderProgram("glsl/sky.vert", "glsl/sky.frag");
  g_PostProgram = CreateShaderProgram("glsl/post.vert", "glsl/post.frag");
  g_LumProgram = CreateShaderProgram("glsl/post.vert", "glsl/lum.frag");
  g_BrightProgram = CreateShaderProgram("glsl/post.vert", "glsl/bright.frag");
  g_BlurProgram = CreateShaderProgram("glsl/post.vert", "glsl/blur.frag");

  if (!g_MeshProgram || !g_SkyProgram || !g_PostProgram || !g_LumProgram ||
      !g_BrightProgram || !g_BlurProgram)
    return 1;

  glEnable(GL_DEPTH_TEST);

  // --- Setup Mesh Program Static Uniforms ---
  glUseProgram(g_MeshProgram);

  // Set Mode Uniform
  glUniform1i(glGetUniformLocation(g_MeshProgram, "u_UsePackedLuminance"),
              use_packed_luminance ? 1 : 0);

  // Set Material Sampler Units (Static)
  glUniform1i(glGetUniformLocation(g_MeshProgram, "u_AlbedoTex"), 0);
  glUniform1i(glGetUniformLocation(g_MeshProgram, "u_NormalTex"), 4);
  glUniform1i(glGetUniformLocation(g_MeshProgram, "u_MRTex"), 5);

  // Bind SH Textures (Static)
  if (use_packed_luminance) {
    for (int i = 0; i < 3; ++i) {
      glActiveTexture(GL_TEXTURE1 + i);
      glBindTexture(GL_TEXTURE_2D, g_SHTextures[i]);
      std::string u_name = "u_PackedTex" + std::to_string(i);
      glUniform1i(glGetUniformLocation(g_MeshProgram, u_name.c_str()), 1 + i);
    }
  } else {
    for (int i = 0; i < 9; ++i) {
      glActiveTexture(GL_TEXTURE1 + i);
      glBindTexture(GL_TEXTURE_2D, g_SHTextures[i]);
      std::string u_name = "u_" + std::string(kCoeffSuffixes[i]);
      glUniform1i(glGetUniformLocation(g_MeshProgram, u_name.c_str()), 1 + i);
    }
  }

  // --- Load Skybox Data ---
  if (!scene.environment) {
    LOG(ERROR) << "No environment found.";
    // Fallback?
    g_UsePreetham = true;
    g_SunDir = Eigen::Vector3f(0.2f, 0.8f, 0.2f).normalized();
    g_SkyIntensity = 1.0f;
  } else {
    if (scene.environment->type == sh_baker::Environment::Type::Texture) {
      const auto& tex_var = scene.environment->texture;
      bool valid =
          std::visit([](const auto& t) { return t.width > 0; }, tex_var);
      if (valid) {
        LOG(INFO) << "Loading Skybox Texture...";
        g_SkyboxTexture = LoadTextureVariant(tex_var);
        // We probably also want to set g_IrradianceTexture from this if we had
        // SH? But for now just g_SkyboxTexture.
      }
    } else {
      g_UsePreetham = true;
      g_SunDir = scene.environment->sun_direction;
      g_SkyIntensity = scene.environment->intensity;
      LOG(INFO) << "Using Preetham Sky (Scene). Sun Dir: "
                << g_SunDir.transpose();
    }
  }

  // --- Main Loop ---
  while (!glfwWindowShouldClose(window)) {
    ProcessInput(window);

    // 1. Render to HDR FBO (MSAA)
    glBindFramebuffer(GL_FRAMEBUFFER, g_HdrFBO_MS);
    glViewport(0, 0, kWindowWidth, kWindowHeight);  // Fixed size for now
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Camera Matrix
    Eigen::Vector3f cam_pos_cartesian;
    cam_pos_cartesian.x() =
        g_CamDist * std::sin(g_CamYaw) * std::cos(g_CamPitch);
    cam_pos_cartesian.z() =
        g_CamDist * std::cos(g_CamYaw) * std::cos(g_CamPitch);
    cam_pos_cartesian.y() = g_CamDist * std::sin(g_CamPitch);

    // Update global camera position for shader uniform
    g_CamPos = cam_pos_cartesian;

    // LookAt
    Eigen::Vector3f f = (g_CamTarget - cam_pos_cartesian).normalized();
    Eigen::Vector3f up(0, 1, 0);  // World Up
    Eigen::Vector3f s = f.cross(up).normalized();
    Eigen::Vector3f u = s.cross(f);

    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();
    view(0, 0) = s.x();
    view(0, 1) = s.y();
    view(0, 2) = s.z();
    view(1, 0) = u.x();
    view(1, 1) = u.y();
    view(1, 2) = u.z();
    view(2, 0) = -f.x();
    view(2, 1) = -f.y();
    view(2, 2) = -f.z();
    view(0, 3) = -s.dot(cam_pos_cartesian);
    view(1, 3) = -u.dot(cam_pos_cartesian);
    view(2, 3) = f.dot(cam_pos_cartesian);

    float aspect = (float)kWindowWidth / (float)kWindowHeight;
    float fov = 45.0f * M_PI / 180.0f;
    float tanHalfFov = std::tan(fov / 2.0f);
    float zNear = 0.1f;
    float zFar = 100.0f;
    Eigen::Matrix4f proj = Eigen::Matrix4f::Zero();
    proj(0, 0) = 1.0f / (aspect * tanHalfFov);
    proj(1, 1) = 1.0f / tanHalfFov;
    proj(2, 2) = -(zFar + zNear) / (zFar - zNear);
    proj(2, 3) = -(2.0f * zFar * zNear) / (zFar - zNear);
    proj(3, 2) = -1.0f;

    Eigen::Matrix4f vp = proj * view;

    DrawMeshes(scene, vp, g_CamPos);

    DrawSky(view, proj);

    // 2. Render Post Process to Screen
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    // Use actual window size for screen output
    int w, h;
    glfwGetFramebufferSize(window, &w, &h);
    glViewport(0, 0, w, h);
    glClearColor(1.0f, 0.0f, 1.0f, 1.0f);  // Pink debug if quad fails
    glClear(GL_COLOR_BUFFER_BIT |
            GL_DEPTH_BUFFER_BIT);  // Depth not needed but good hygiene

    DrawPostProcess(kWindowWidth, kWindowHeight);

    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  glfwTerminate();
  return 0;
}
