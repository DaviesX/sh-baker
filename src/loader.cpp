#include "loader.h"

#include <algorithm>
#include <cmath>
#include <iostream>

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <glog/logging.h>

#include <Eigen/Geometry>

#include "colorspace.h"
#include "tiny_gltf.h"

namespace sh_baker {
namespace {

// Helper to convert array to Eigen matrix/vector
Eigen::Affine3f NodeToTransform(const tinygltf::Node& node) {
  Eigen::Affine3f transform = Eigen::Affine3f::Identity();

  if (node.matrix.size() == 16) {
    Eigen::Matrix4f mat;
    for (int i = 0; i < 16; ++i) mat(i) = static_cast<float>(node.matrix[i]);
    transform = Eigen::Affine3f(mat);
  } else {
    if (node.translation.size() == 3) {
      transform.translate(
          Eigen::Vector3f(static_cast<float>(node.translation[0]),
                          static_cast<float>(node.translation[1]),
                          static_cast<float>(node.translation[2])));
    }
    if (node.rotation.size() == 4) {
      // glTF quat is x, y, z, w. Eigen Quat constructor is w, x, y, z
      Eigen::Quaternionf q(static_cast<float>(node.rotation[3]),
                           static_cast<float>(node.rotation[0]),
                           static_cast<float>(node.rotation[1]),
                           static_cast<float>(node.rotation[2]));
      transform.rotate(q);
    }
    if (node.scale.size() == 3) {
      transform.scale(Eigen::Vector3f(static_cast<float>(node.scale[0]),
                                      static_cast<float>(node.scale[1]),
                                      static_cast<float>(node.scale[2])));
    }
  }
  return transform;
}

void ProcessPrimitive(const tinygltf::Model& model,
                      const tinygltf::Primitive& primitive,
                      const Eigen::Affine3f& transform, Scene* scene) {
  Geometry geo;
  geo.transform = transform;
  geo.material_id = std::max(0, primitive.material);  // Default to 0 if -1

  // Get Position
  if (primitive.attributes.find("POSITION") == primitive.attributes.end()) {
    DLOG(WARNING) << "Primitive missing POSITION attribute";
    return;
  }

  const tinygltf::Accessor& pos_accessor =
      model.accessors[primitive.attributes.at("POSITION")];
  const tinygltf::BufferView& pos_view =
      model.bufferViews[pos_accessor.bufferView];
  const tinygltf::Buffer& pos_buffer = model.buffers[pos_view.buffer];
  const float* pos_data = reinterpret_cast<const float*>(
      &pos_buffer.data[pos_view.byteOffset + pos_accessor.byteOffset]);
  size_t vertex_count = pos_accessor.count;
  int pos_stride = pos_accessor.ByteStride(pos_view)
                       ? (pos_accessor.ByteStride(pos_view) / sizeof(float))
                       : 3;

  // Get Normal
  const float* norm_data = nullptr;
  int norm_stride = 0;
  if (primitive.attributes.find("NORMAL") != primitive.attributes.end()) {
    const tinygltf::Accessor& norm_accessor =
        model.accessors[primitive.attributes.at("NORMAL")];
    const tinygltf::BufferView& norm_view =
        model.bufferViews[norm_accessor.bufferView];
    const tinygltf::Buffer& norm_buffer = model.buffers[norm_view.buffer];
    norm_data = reinterpret_cast<const float*>(
        &norm_buffer.data[norm_view.byteOffset + norm_accessor.byteOffset]);
    norm_stride = norm_accessor.ByteStride(norm_view)
                      ? (norm_accessor.ByteStride(norm_view) / sizeof(float))
                      : 3;
  }

  // Get UV0
  const float* uv0_data = nullptr;
  int uv0_stride = 0;
  if (primitive.attributes.find("TEXCOORD_0") != primitive.attributes.end()) {
    const tinygltf::Accessor& uv_accessor =
        model.accessors[primitive.attributes.at("TEXCOORD_0")];
    const tinygltf::BufferView& uv_view =
        model.bufferViews[uv_accessor.bufferView];
    const tinygltf::Buffer& uv_buffer = model.buffers[uv_view.buffer];
    uv0_data = reinterpret_cast<const float*>(
        &uv_buffer.data[uv_view.byteOffset + uv_accessor.byteOffset]);
    uv0_stride = uv_accessor.ByteStride(uv_view)
                     ? (uv_accessor.ByteStride(uv_view) / sizeof(float))
                     : 2;
  }

  // Get UV1
  const float* uv1_data = nullptr;
  int uv1_stride = 0;
  if (primitive.attributes.find("TEXCOORD_1") != primitive.attributes.end()) {
    const tinygltf::Accessor& uv_accessor =
        model.accessors[primitive.attributes.at("TEXCOORD_1")];
    const tinygltf::BufferView& uv_view =
        model.bufferViews[uv_accessor.bufferView];
    const tinygltf::Buffer& uv_buffer = model.buffers[uv_view.buffer];
    uv1_data = reinterpret_cast<const float*>(
        &uv_buffer.data[uv_view.byteOffset + uv_accessor.byteOffset]);
    uv1_stride = uv_accessor.ByteStride(uv_view)
                     ? (uv_accessor.ByteStride(uv_view) / sizeof(float))
                     : 2;
  }

  // Indices
  if (primitive.indices > -1) {
    const tinygltf::Accessor& index_accessor =
        model.accessors[primitive.indices];
    const tinygltf::BufferView& index_view =
        model.bufferViews[index_accessor.bufferView];
    const tinygltf::Buffer& index_buffer = model.buffers[index_view.buffer];
    const uint8_t* index_data_ptr =
        &index_buffer.data[index_view.byteOffset + index_accessor.byteOffset];
    int stride = index_accessor.ByteStride(index_view);

    geo.indices.reserve(index_accessor.count);
    for (size_t i = 0; i < index_accessor.count; ++i) {
      uint32_t val = 0;
      if (index_accessor.componentType ==
          TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) {
        val = *(index_data_ptr + i * stride);
      } else if (index_accessor.componentType ==
                 TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
        val = *reinterpret_cast<const uint16_t*>(index_data_ptr + i * stride);
      } else if (index_accessor.componentType ==
                 TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
        val = *reinterpret_cast<const uint32_t*>(index_data_ptr + i * stride);
      }
      geo.indices.push_back(val);
    }
  }

  // Vertices
  geo.vertices.reserve(vertex_count);
  geo.normals.reserve(vertex_count);
  geo.texture_uvs.reserve(vertex_count);
  if (uv1_data) {
    geo.lightmap_uvs.reserve(vertex_count);
  }

  for (size_t i = 0; i < vertex_count; ++i) {
    geo.vertices.emplace_back(pos_data[i * pos_stride + 0],
                              pos_data[i * pos_stride + 1],
                              pos_data[i * pos_stride + 2]);

    if (norm_data) {
      geo.normals.emplace_back(norm_data[i * norm_stride + 0],
                               norm_data[i * norm_stride + 1],
                               norm_data[i * norm_stride + 2]);
    } else {
      geo.normals.emplace_back(0, 1, 0);
    }

    if (uv0_data) {
      geo.texture_uvs.emplace_back(uv0_data[i * uv0_stride + 0],
                                   uv0_data[i * uv0_stride + 1]);
    } else {
      geo.texture_uvs.emplace_back(0, 0);
    }

    if (uv1_data) {
      geo.lightmap_uvs.emplace_back(uv1_data[i * uv1_stride + 0],
                                    uv1_data[i * uv1_stride + 1]);
    }
  }

  scene->geometries.push_back(std::move(geo));

  // Check for emission and create Area Light if needed
  Geometry& added_geo = scene->geometries.back();
  const Material& mat = scene->materials[added_geo.material_id];

  if (mat.emission_intensity > 0.0f) {
    Light area_light;
    area_light.type = Light::Type::Area;
    // We can safely reference the material because it has been populated
    // altogether, but it isn't the case for geometry. We will assign the
    // pointer once all geometries are added.
    area_light.material = &mat;
    area_light.geometry_index = static_cast<int>(scene->geometries.size()) - 1;

    // Surface Area
    float total_area = 0.0f;

    for (size_t i = 0; i < added_geo.indices.size(); i += 3) {
      uint32_t i0 = added_geo.indices[i];
      uint32_t i1 = added_geo.indices[i + 1];
      uint32_t i2 = added_geo.indices[i + 2];

      if (i0 < added_geo.vertices.size() && i1 < added_geo.vertices.size() &&
          i2 < added_geo.vertices.size()) {
        const Eigen::Vector3f& v0 = added_geo.vertices[i0];
        const Eigen::Vector3f& v1 = added_geo.vertices[i1];
        const Eigen::Vector3f& v2 = added_geo.vertices[i2];

        float tri_area = 0.5f * (v1 - v0).cross(v2 - v0).norm();
        total_area += tri_area;
      }
    }
    area_light.area = total_area;

    // Intensity. Color will come from the material's albedo texture.
    area_light.intensity = mat.emission_intensity;

    scene->lights.push_back(std::move(area_light));
  }
}

void ProcessMaterials(const tinygltf::Model& model,
                      const std::filesystem::path& base_path, Scene* scene) {
  if (model.materials.empty()) {
    Material default_mat;
    default_mat.name = "default";
    scene->materials.push_back(default_mat);
    return;
  }

  for (const auto& gltf_mat : model.materials) {
    Material mat;
    mat.name = gltf_mat.name;
    mat.roughness =
        static_cast<float>(gltf_mat.pbrMetallicRoughness.roughnessFactor);
    mat.metallic =
        static_cast<float>(gltf_mat.pbrMetallicRoughness.metallicFactor);

    // Emission
    // Our ioquake3 exporter will ensure that the
    // KHR_materials_emissive_strength extension is used for all emissive
    // materials.
    auto emissive_strength_it =
        gltf_mat.extensions.find("KHR_materials_emissive_strength");
    if (emissive_strength_it != gltf_mat.extensions.end()) {
      const auto& emissive_strength = emissive_strength_it->second;
      if (emissive_strength.Has("emissiveStrength")) {
        mat.emission_intensity = float(
            emissive_strength.Get("emissiveStrength").GetNumberAsDouble());
      }
    }

    // Texture (Base Color)
    int tex_idx = gltf_mat.pbrMetallicRoughness.baseColorTexture.index;
    bool texture_loaded = false;
    if (tex_idx >= 0 && tex_idx < model.textures.size()) {
      int img_idx = model.textures[tex_idx].source;
      if (img_idx >= 0 && img_idx < model.images.size()) {
        const auto& img = model.images[img_idx];
        mat.albedo.width = img.width;
        mat.albedo.height = img.height;
        mat.albedo.channels = img.component;
        mat.albedo.pixel_data = img.image;  // Copy data
        if (!img.uri.empty()) {
          std::filesystem::path uri_path(img.uri);
          if (uri_path.is_absolute()) {
            mat.albedo.file_path = uri_path;
          } else {
            mat.albedo.file_path =
                std::filesystem::absolute(base_path / uri_path);
          }
        }
        texture_loaded = true;
      }
    }

    if (!texture_loaded) {
      // Create 1x1 texture from baseColorFactor (Linear -> sRGB)
      const auto& color = gltf_mat.pbrMetallicRoughness.baseColorFactor;
      mat.albedo.width = 1;
      mat.albedo.height = 1;
      mat.albedo.channels = 4;
      mat.albedo.pixel_data.resize(4);

      // baseColorFactor is RGBA (4 items)
      if (color.size() == 4) {
        mat.albedo.pixel_data[0] = LinearToSRGB(static_cast<float>(color[0]));
        mat.albedo.pixel_data[1] = LinearToSRGB(static_cast<float>(color[1]));
        mat.albedo.pixel_data[2] = LinearToSRGB(static_cast<float>(color[2]));
        mat.albedo.pixel_data[3] = static_cast<unsigned char>(
            std::rint(std::clamp(color[3], 0.0, 1.0) * 255.0));
      } else {
        // Fallback white
        mat.albedo.pixel_data = {255, 255, 255, 255};
      }
    }

    scene->materials.push_back(std::move(mat));
  }
}

void ProcessLight(const tinygltf::Model& model,
                  const tinygltf::Value& light_obj,
                  const Eigen::Affine3f& transform, Scene* scene) {
  // Parse KHR_lights_punctual object
  if (!light_obj.IsObject()) return;

  Light l;

  // Position/Direction from transform
  l.position = transform.translation();
  // Default direction is -Z. Apply rotation to it.
  l.direction = transform.rotation() * Eigen::Vector3f(0, 0, -1);
  l.direction.normalize();

  if (light_obj.Has("name")) {
    // We could store name if Light struct had one
  }

  std::string type_str;
  if (light_obj.Has("type")) {
    type_str = light_obj.Get("type").Get<std::string>();
  }

  if (type_str == "point")
    l.type = Light::Type::Point;
  else if (type_str == "directional")
    l.type = Light::Type::Directional;
  else if (type_str == "spot")
    l.type = Light::Type::Spot;

  if (light_obj.Has("color")) {
    const auto& color_arr = light_obj.Get("color");
    if (color_arr.IsArray() && color_arr.ArrayLen() == 3) {
      l.color = Eigen::Vector3f(color_arr.Get(0).Get<double>(),
                                color_arr.Get(1).Get<double>(),
                                color_arr.Get(2).Get<double>());
    }
  }

  if (light_obj.Has("intensity")) {
    l.intensity = static_cast<float>(light_obj.Get("intensity").Get<double>());
  }

  if (l.type == Light::Type::Spot && light_obj.Has("spot")) {
    const auto& spot = light_obj.Get("spot");
    if (spot.Has("innerConeAngle")) {
      float inner_cone_angle =
          static_cast<float>(spot.Get("innerConeAngle").Get<double>());
      l.cos_inner_cone = std::cos(inner_cone_angle);
    }
    if (spot.Has("outerConeAngle")) {
      float outer_cone_angle =
          static_cast<float>(spot.Get("outerConeAngle").Get<double>());
      l.cos_outer_cone = std::cos(outer_cone_angle);
    }
  }

  scene->lights.push_back(std::move(l));
}

void TraverseNodes(const tinygltf::Model& model, int node_index,
                   const Eigen::Affine3f& parent_transform, Scene* scene) {
  const tinygltf::Node& node = model.nodes[node_index];

  Eigen::Affine3f global_transform = parent_transform * NodeToTransform(node);

  // Mesh
  if (node.mesh >= 0) {
    const tinygltf::Mesh& mesh = model.meshes[node.mesh];
    for (const auto& primitive : mesh.primitives) {
      ProcessPrimitive(model, primitive, global_transform, scene);
    }
  }

  // Light (KHR_lights_punctual)
  // The extension is defined on the node as "extensions": {
  // "KHR_lights_punctual": { "light": 0 }
  // }
  auto lights_punctual_it = node.extensions.find("KHR_lights_punctual");
  if (lights_punctual_it != node.extensions.end()) {
    const auto& ext = lights_punctual_it->second;
    if (ext.Has("light")) {
      int light_idx = ext.Get("light").Get<int>();
      if (model.extensions.find("KHR_lights_punctual") !=
          model.extensions.end()) {
        const auto& model_ext = model.extensions.at("KHR_lights_punctual");
        if (model_ext.Has("lights")) {
          const auto& lights_arr = model_ext.Get("lights");
          if (lights_arr.IsArray() && light_idx < lights_arr.ArrayLen()) {
            ProcessLight(model, lights_arr.Get(light_idx), global_transform,
                         scene);
          }
        }
      }
    }
  }

  for (int child : node.children) {
    TraverseNodes(model, child, global_transform, scene);
  }
}

}  // namespace

std::optional<Scene> LoadScene(const std::filesystem::path& gltf_file) {
  tinygltf::Model model;
  tinygltf::TinyGLTF loader;
  std::string err;
  std::string warn;

  bool ret = false;
  if (gltf_file.extension() == ".glb") {
    ret = loader.LoadBinaryFromFile(&model, &err, &warn, gltf_file.string());
  } else {
    ret = loader.LoadASCIIFromFile(&model, &err, &warn, gltf_file.string());
  }

  if (!warn.empty()) {
    DLOG(WARNING) << "TinyGLTF warning: " << warn;
  }
  if (!err.empty()) {
    LOG(ERROR) << "TinyGLTF error: " << err;
  }
  if (!ret) {
    return std::nullopt;
  }

  Scene scene;

  ProcessMaterials(model, gltf_file.parent_path(), &scene);

  const tinygltf::Scene& gltf_scene =
      model.scenes[model.defaultScene > -1 ? model.defaultScene : 0];
  for (int node_index : gltf_scene.nodes) {
    TraverseNodes(model, node_index, Eigen::Affine3f::Identity(), &scene);
  }

  // Set geometry pointers for area lights because all geometries have been
  // added. The geometry vector is frozen from this point on.
  for (auto& light : scene.lights) {
    if (light.geometry_index >= 0) {
      light.geometry = &scene.geometries[light.geometry_index];
    }
  }

  return scene;
}

}  // namespace sh_baker
