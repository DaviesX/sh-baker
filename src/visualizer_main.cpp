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
#include "visualizer_camera.h"
#include "visualizer_control.h"
#include "visualizer_exposure.h"
#include "visualizer_radiance.h"
#include "visualizer_sky.h"
#include "visualizer_utils.h"

// --- Constants ---
const int kWindowWidth = 1280;
const int kWindowHeight = 720;

// --- Flags ---
DEFINE_string(input, "",
              "Path to the input folder containing scene.gltf and "
              "lightmap_*.exr files.");

// --- Globals ---
GLuint g_PostProgram = 0;

// HDR Framebuffer (MSAA)
GLuint g_HdrFBO_MS = 0;
GLuint g_HdrColorTexture_MS = 0;
GLuint g_HdrDepthRBO_MS = 0;

// HDR Framebuffer (Resolve / Post-Process Input)
GLuint g_HdrFBO_Resolve = 0;
GLuint g_HdrColorTexture_Resolve = 0;

// Luminance Framebuffer (Auto Exposure) - Moved to ExposureComputer
sh_baker::ExposureComputer g_ExposureComputer;

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

// Skybox Data - Moved to SkyRenderer
sh_baker::SkyRenderer g_SkyRenderer;
sh_baker::RadianceRenderer g_RadianceRenderer;

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
  if (!g_ExposureComputer.Init()) {
    LOG(ERROR) << "Failed to init Exposure Computer";
  }

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
  g_ExposureComputer.Compute(g_QuadVAO, g_HdrColorTexture_Resolve);

  // 3. Bloom Extraction (Bright Pass)
  glViewport(0, 0, kBloomWidth, kBloomHeight);
  glBindFramebuffer(GL_FRAMEBUFFER, g_BloomFBO[0]);
  glUseProgram(g_BrightProgram);

  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, g_HdrColorTexture_Resolve);
  glUniform1i(glGetUniformLocation(g_BrightProgram, "u_HdrTex"), 0);

  glActiveTexture(GL_TEXTURE1);
  glBindTexture(GL_TEXTURE_2D, g_ExposureComputer.GetLuminanceTexture());
  glUniform1i(glGetUniformLocation(g_BrightProgram, "u_LumTexture"), 1);

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
  glBindTexture(GL_TEXTURE_2D, g_ExposureComputer.GetLuminanceTexture());
  glUniform1i(glGetUniformLocation(g_PostProgram, "u_LumTexture"), 1);

  glActiveTexture(GL_TEXTURE2);
  glBindTexture(GL_TEXTURE_2D, g_BloomTextures[0]);  // Final blur result
  glUniform1i(glGetUniformLocation(g_PostProgram, "u_BloomTexture"), 2);

  glBindVertexArray(g_QuadVAO);
  glDrawArrays(GL_TRIANGLES, 0, 6);

  glEnable(GL_DEPTH_TEST);
}

// Camera
sh_baker::Camera g_Camera(Eigen::Vector3f(0.0f, 0.0f, 5.0f));
sh_baker::InputController g_InputController(g_Camera);
float g_LastFrame = 0.0f;

// --- Helper Functions ---

void MouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
  g_InputController.MouseButtonCallback(window, button, action, mods);
}

void CursorPosCallback(GLFWwindow* window, double xpos, double ypos) {
  g_InputController.CursorPosCallback(window, xpos, ypos);
}

void ScrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
  g_InputController.ScrollCallback(window, xoffset, yoffset);
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

  // --- Init Radiance Renderer ---
  if (!g_RadianceRenderer.Init(scene, input_dir)) {
    LOG(ERROR) << "Failed to init Radiance Render";
    return 1;
  }

  // --- Setup Shaders (Post Process) ---
  g_PostProgram = CreateShaderProgram("glsl/post.vert", "glsl/post.frag");
  GLuint skyProgram = CreateShaderProgram("glsl/sky.vert", "glsl/sky.frag");
  g_SkyRenderer.SetProgram(skyProgram);
  // g_LumProgram removed, handled by ExposureComputer
  g_BrightProgram = CreateShaderProgram("glsl/post.vert", "glsl/bright.frag");
  g_BlurProgram = CreateShaderProgram("glsl/post.vert", "glsl/blur.frag");

  if (!g_PostProgram || !skyProgram || !g_BrightProgram || !g_BlurProgram)
    return 1;

  glEnable(GL_DEPTH_TEST);

  // --- Set Sky SH Uniforms and Skybox State ---
  g_SkyRenderer.UpdateAndBind(scene.environment ? &*scene.environment : nullptr,
                              g_RadianceRenderer.GetProgram());

  // --- Main Loop ---
  while (!glfwWindowShouldClose(window)) {
    float currentFrame = static_cast<float>(glfwGetTime());
    float deltaTime = currentFrame - g_LastFrame;
    g_LastFrame = currentFrame;

    g_InputController.ProcessInput(window, deltaTime);

    // 1. Render to HDR FBO (MSAA)
    glBindFramebuffer(GL_FRAMEBUFFER, g_HdrFBO_MS);
    glViewport(0, 0, kWindowWidth, kWindowHeight);  // Fixed size for now
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    Eigen::Matrix4f view = g_Camera.GetViewMatrix();

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

    g_RadianceRenderer.Draw(scene, vp, g_Camera.Position());

    g_SkyRenderer.Draw(view, proj);

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
