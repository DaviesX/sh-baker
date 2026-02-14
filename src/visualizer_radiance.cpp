#include "visualizer_radiance.h"

#include <glog/logging.h>

#include <string>

namespace sh_baker {

RadianceRenderer::~RadianceRenderer() {
  if (program_) glDeleteProgram(program_);
  for (auto& mesh : meshes_) {
    glDeleteVertexArrays(1, &mesh.vao);
  }
  // Textures should ideally be deleted too, but for this simple visualizer we
  // rely on OS cleanup or simple destructor. Implementing proper cleanup:
  glDeleteTextures(albedo_textures_.size(), albedo_textures_.data());
  glDeleteTextures(mr_textures_.size(), mr_textures_.data());
  glDeleteTextures(sh_textures_.size(), sh_textures_.data());
  if (irradiance_texture_) glDeleteTextures(1, &irradiance_texture_);
}

bool RadianceRenderer::Init(const Scene& scene,
                            const std::filesystem::path& input_dir) {
  // 1. Compile Shaders
  program_ = CreateShaderProgram("glsl/viz.vert", "glsl/viz.frag");
  if (!program_) {
    LOG(ERROR) << "Failed to create mesh program.";
    return false;
  }

  // 2. Upload Geometry
  UploadGeometry(scene);

  // 3. Load Materials
  LoadMaterials(scene);

  // 4. Load SH
  LoadSH(input_dir);

  // 5. Setup Static Uniforms
  glUseProgram(program_);

  // Set Mode Uniform
  glUniform1i(glGetUniformLocation(program_, "u_UsePackedLuminance"),
              use_packed_luminance_ ? 1 : 0);

  // Set Material Sampler Units (Static)
  glUniform1i(glGetUniformLocation(program_, "u_AlbedoTex"), 0);
  glUniform1i(glGetUniformLocation(program_, "u_NormalTex"), 4);
  glUniform1i(glGetUniformLocation(program_, "u_MRTex"), 5);

  // Bind SH Sampler Units
  // Note: Actual texture binding happens in Draw or here if we assume they stay
  // bound to these units. We will assume units 1..3 or 1..9 are reserved for
  // SH.
  const char* kCoeffSuffixes[] = {"L0",   "L1m1", "L10", "L11", "L2m2",
                                  "L2m1", "L20",  "L21", "L22"};
  if (use_packed_luminance_) {
    for (int i = 0; i < 3; ++i) {
      std::string u_name = "u_PackedTex" + std::to_string(i);
      glUniform1i(glGetUniformLocation(program_, u_name.c_str()), 1 + i);
    }
  } else {
    for (int i = 0; i < 9; ++i) {
      std::string u_name = "u_" + std::string(kCoeffSuffixes[i]);
      glUniform1i(glGetUniformLocation(program_, u_name.c_str()), 1 + i);
    }
  }

  // Irradiance Texture Unit (10)
  glUniform1i(glGetUniformLocation(program_, "u_IrradianceTex"), 10);

  glUseProgram(0);
  return true;
}

void RadianceRenderer::UploadGeometry(const Scene& scene) {
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
    // Note: logic in main used geo.indices.size(), assuming correct here.
    // Corrected the copy-paste potential error in sizeof.

    meshes_.push_back(mesh);

    // Cleanup VBOs? They are bound to VAO now, but if we don't delete them or
    // store them, they leak until context destruction. Ideally we store VBO IDs
    // too. For this refactor, we are mimicking previous behavior which
    // leaked/didn't track them explicitly in `main` (declared locally in loop).
    // Actually, `glDeleteBuffers` after `glVertexAttribPointer` is WRONG if
    // ARRAY_BUFFER. `glVertexAttribPointer` binds the buffer currently bound to
    // target. So we can delete the VBO handle if we want, but usually we keep
    // them. Since we don't store them, they will persist until context death.
    glBindVertexArray(0);
    glDeleteBuffers(5, vbo);
    glDeleteBuffers(1, &ebo);
  }
}

void RadianceRenderer::LoadMaterials(const Scene& scene) {
  for (const auto& mat : scene.materials) {
    // Albedo
    if (!mat.albedo.pixel_data.empty()) {
      albedo_textures_.push_back(LoadTexture(mat.albedo));
    } else {
      albedo_textures_.push_back(0);
    }
    // Normal
    if (!mat.normal_texture.pixel_data.empty()) {
      normal_textures_.push_back(LoadTexture(mat.normal_texture));
    } else {
      normal_textures_.push_back(0);
    }
    // Metallic/Roughness
    if (!mat.metallic_roughness_texture.pixel_data.empty()) {
      mr_textures_.push_back(LoadTexture(mat.metallic_roughness_texture));
    } else {
      mr_textures_.push_back(0);
    }
  }
}

void RadianceRenderer::LoadSH(const std::filesystem::path& input_dir) {
  const char* kCoeffSuffixes[] = {"L0",   "L1m1", "L10", "L11", "L2m2",
                                  "L2m1", "L20",  "L21", "L22"};

  // Check if packed file exists
  if (std::filesystem::exists(input_dir / "lightmap_packed_0.exr")) {
    use_packed_luminance_ = true;
    LOG(INFO) << "Detected Packed Luminance SH Lightmaps.";

    for (int i = 0; i < 3; ++i) {
      std::string filename = "lightmap_packed_" + std::to_string(i) + ".exr";
      std::filesystem::path p = input_dir / filename;
      GLuint tid = LoadEXRTexture(p.string());
      if (tid == 0) LOG(WARNING) << "Failed to load SH texture: " << p;
      sh_textures_.push_back(tid);
    }
  } else if (std::filesystem::exists(input_dir / "lightmap_L0.exr")) {
    use_packed_luminance_ = false;
    LOG(INFO) << "Using Standard SH Lightmaps (9 files).";
    for (int i = 0; i < 9; ++i) {
      std::string filename =
          "lightmap_" + std::string(kCoeffSuffixes[i]) + ".exr";
      std::filesystem::path p = input_dir / filename;
      GLuint tid = LoadEXRTexture(p.string());
      if (tid == 0) {
        LOG(WARNING) << "Failed to load SH texture: " << p;
      }
      sh_textures_.push_back(tid);
    }
  } else {
    use_packed_luminance_ = false;
    LOG(INFO) << "No SH Lightmaps found. Using 1x1 Placeholders (Luminance "
                 "Only).";
    float c0 = 3.5449f;
    sh_textures_.push_back(CreatePlaceholderTexture(c0, c0, c0));
    for (int i = 1; i < 9; ++i) {
      sh_textures_.push_back(CreatePlaceholderTexture(0.0f, 0.0f, 0.0f));
    }
  }

  // Load Irradiance Map
  if (std::filesystem::exists(input_dir / "lightmap_irradiance.exr")) {
    irradiance_texture_ =
        LoadEXRTexture((input_dir / "lightmap_irradiance.exr").string());
    if (irradiance_texture_ == 0) {
      LOG(WARNING) << "Failed to load irradiance texture.";
    } else {
      LOG(INFO) << "Loaded Irradiance Map.";
    }
  } else {
    // Placeholder
    irradiance_texture_ = CreatePlaceholderTexture(0.5f, 0.5f, 0.5f);
    LOG(INFO) << "No Irradiance Map found. Using Placeholder.";
  }
}

void RadianceRenderer::Draw(const Scene& scene, const Eigen::Matrix4f& vp,
                            const Eigen::Vector3f& cam_pos) {
  if (!program_) return;

  glUseProgram(program_);

  // Pass CamPos
  glUniform3fv(glGetUniformLocation(program_, "u_CamPos"), 1, cam_pos.data());

  // Bind SH Textures
  for (size_t i = 0; i < sh_textures_.size(); ++i) {
    glActiveTexture(GL_TEXTURE1 + i);
    glBindTexture(GL_TEXTURE_2D, sh_textures_[i]);
  }

  // Bind Irradiance
  glActiveTexture(GL_TEXTURE10);
  glBindTexture(GL_TEXTURE_2D, irradiance_texture_);

  // Update Directional Flag
  glUniform1i(glGetUniformLocation(program_, "u_ShowDirectional"),
              show_directional_ ? 1 : 0);

  // Draw Meshes
  for (size_t i = 0; i < meshes_.size(); ++i) {
    glBindVertexArray(meshes_[i].vao);

    const auto& geo = scene.geometries[i];
    Eigen::Matrix4f model = geo.transform.matrix();
    Eigen::Matrix4f mvp = vp * model;

    glUniformMatrix4fv(glGetUniformLocation(program_, "u_MVP"), 1, GL_FALSE,
                       mvp.data());
    glUniformMatrix4fv(glGetUniformLocation(program_, "u_Model"), 1, GL_FALSE,
                       model.data());

    int mat_id = meshes_[i].material_id;

    // Albedo
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, albedo_textures_[mat_id]);

    // Normal
    glActiveTexture(GL_TEXTURE4);
    glBindTexture(GL_TEXTURE_2D, normal_textures_[mat_id]);

    // MR
    glActiveTexture(GL_TEXTURE5);
    glBindTexture(GL_TEXTURE_2D, mr_textures_[mat_id]);

    glDrawElements(GL_TRIANGLES, meshes_[i].count, GL_UNSIGNED_INT, 0);
  }
}

}  // namespace sh_baker
