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

// --- Constants ---
const int kWindowWidth = 1280;
const int kWindowHeight = 720;

// --- Flags ---
DEFINE_string(input, "",
              "Path to the input folder containing scene.gltf and "
              "lightmap_*.exr files.");

// --- Globals ---
GLuint g_ShaderProgram = 0;
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
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
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
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

  free(out);
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

int main(int argc, char* argv[]) {
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

  // --- Load Scene ---
  auto scene_path = input_dir / "scene.gltf";
  LOG(INFO) << "Loading scene: " << scene_path;
  auto scene_opt = sh_baker::LoadScene(scene_path);
  if (!scene_opt) {
    LOG(ERROR) << "Failed to load scene";
    return 1;
  }
  const auto& scene = *scene_opt;

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
  } else {
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
  }

  // --- Shader ---
  g_ShaderProgram = CreateShaderProgram("glsl/viz.vert", "glsl/viz.frag");
  if (!g_ShaderProgram) return 1;

  glEnable(GL_DEPTH_TEST);

  glUseProgram(g_ShaderProgram);

  // Set Mode Uniform
  glUniform1i(glGetUniformLocation(g_ShaderProgram, "u_UsePackedLuminance"),
              use_packed_luminance ? 1 : 0);

  // Set Material Sampler Units (Static)
  glUniform1i(glGetUniformLocation(g_ShaderProgram, "u_AlbedoTex"), 0);
  glUniform1i(glGetUniformLocation(g_ShaderProgram, "u_NormalTex"), 4);
  glUniform1i(glGetUniformLocation(g_ShaderProgram, "u_MRTex"), 5);

  // Bind SH Textures (Static)
  if (use_packed_luminance) {
    for (int i = 0; i < 3; ++i) {
      glActiveTexture(GL_TEXTURE1 + i);
      glBindTexture(GL_TEXTURE_2D, g_SHTextures[i]);
      std::string u_name = "u_PackedTex" + std::to_string(i);
      glUniform1i(glGetUniformLocation(g_ShaderProgram, u_name.c_str()), 1 + i);
    }
  } else {
    for (int i = 0; i < 9; ++i) {
      glActiveTexture(GL_TEXTURE1 + i);
      glBindTexture(GL_TEXTURE_2D, g_SHTextures[i]);
      std::string u_name = "u_" + std::string(kCoeffSuffixes[i]);
      glUniform1i(glGetUniformLocation(g_ShaderProgram, u_name.c_str()), 1 + i);
    }
  }

  // --- Main Loop ---
  while (!glfwWindowShouldClose(window)) {
    ProcessInput(window);

    int w, h;
    glfwGetFramebufferSize(window, &w, &h);
    glViewport(0, 0, w, h);
    glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
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

    float aspect = (float)w / (float)h;
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

    // Pass CamPos to Shader
    glUniform3fv(glGetUniformLocation(g_ShaderProgram, "u_CamPos"), 1,
                 g_CamPos.data());

    // Draw Meshes
    for (size_t i = 0; i < g_Meshes.size(); ++i) {
      glBindVertexArray(g_Meshes[i].vao);

      const auto& geo = scene.geometries[i];
      Eigen::Matrix4f model = geo.transform.matrix();
      Eigen::Matrix4f mvp = vp * model;

      glUniformMatrix4fv(glGetUniformLocation(g_ShaderProgram, "u_MVP"), 1,
                         GL_FALSE, mvp.data());
      glUniformMatrix4fv(glGetUniformLocation(g_ShaderProgram, "u_Model"), 1,
                         GL_FALSE, model.data());

      int mat_id = g_Meshes[i].material_id;
      if (mat_id >= 0 && mat_id < scene.materials.size()) {
        const auto& mat_data = scene.materials[mat_id];

        // Albedo
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, g_AlbedoTextures[mat_id]);
        glUniform1i(glGetUniformLocation(g_ShaderProgram, "u_HasAlbedo"),
                    g_AlbedoTextures[mat_id] != 0 ? 1 : 0);
        glUniform3f(glGetUniformLocation(g_ShaderProgram, "u_AlbedoColor"), 1,
                    1, 1);

        // Normal
        glActiveTexture(GL_TEXTURE4);
        glBindTexture(GL_TEXTURE_2D, g_NormalTextures[mat_id]);
        glUniform1i(glGetUniformLocation(g_ShaderProgram, "u_HasNormal"),
                    g_NormalTextures[mat_id] != 0 ? 1 : 0);

        // MR
        glActiveTexture(GL_TEXTURE5);
        glBindTexture(GL_TEXTURE_2D, g_MRTextures[mat_id]);
        glUniform1i(glGetUniformLocation(g_ShaderProgram, "u_HasMR"),
                    g_MRTextures[mat_id] != 0 ? 1 : 0);

        glUniform1f(glGetUniformLocation(g_ShaderProgram, "u_Metallic"),
                    mat_data.metallic);
        glUniform1f(glGetUniformLocation(g_ShaderProgram, "u_Roughness"),
                    mat_data.roughness);

      } else {
        glUniform1i(glGetUniformLocation(g_ShaderProgram, "u_HasAlbedo"), 0);
        glUniform3f(glGetUniformLocation(g_ShaderProgram, "u_AlbedoColor"), 1,
                    0, 1);
        glUniform1i(glGetUniformLocation(g_ShaderProgram, "u_HasNormal"), 0);
        glUniform1i(glGetUniformLocation(g_ShaderProgram, "u_HasMR"), 0);
        glUniform1f(glGetUniformLocation(g_ShaderProgram, "u_Metallic"), 0.0f);
        glUniform1f(glGetUniformLocation(g_ShaderProgram, "u_Roughness"), 0.5f);
      }

      glDrawElements(GL_TRIANGLES, g_Meshes[i].count, GL_UNSIGNED_INT, 0);
    }

    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  glfwTerminate();
  return 0;
}
