#include "loader.h"

#include <algorithm>
#include <iostream>

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <glog/logging.h>

#include <Eigen/Geometry>

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
  geo.materialID = std::max(0, primitive.material);  // Default to 0 if -1

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

  // Get UV
  const float* uv_data = nullptr;
  int uv_stride = 0;
  std::string texcoord = "TEXCOORD_0";
  if (primitive.attributes.find("TEXCOORD_1") != primitive.attributes.end()) {
    // Prefer lightmap UVs if available? Or just check both?
    // Logic: If we are baking, we need the baking UVs.
    // Assuming TEXCOORD_1 is for lightmap if present.
    // But for scene geometry, we might want TEXCOORD_0 for albedo?
    // The task says "Input: glTF file (level geometry with lightmap UVs from
    // xatlas)". Let's assume TEXCOORD_0 is standard UV and TEXCOORD_1 might be
    // the lightmap UV. For now, let's just grab TEXCOORD_0 as "uvs" for the
    // geometry. If we need a second set, we should add uvs2 to Geometry. But
    // let's stick to what was there: check 1 then 0.
    if (primitive.attributes.find("TEXCOORD_0") != primitive.attributes.end()) {
      texcoord = "TEXCOORD_0";
    }
    // Actually, let's prioritize TEXCOORD_0 for material sampling.
    // The Baker phase will likely use a specific UV channel.
    // I'll stick to grabbing TEXCOORD_0 for now as the "Material UVs".
  }

  if (primitive.attributes.find(texcoord) != primitive.attributes.end()) {
    const tinygltf::Accessor& uv_accessor =
        model.accessors[primitive.attributes.at(texcoord)];
    const tinygltf::BufferView& uv_view =
        model.bufferViews[uv_accessor.bufferView];
    const tinygltf::Buffer& uv_buffer = model.buffers[uv_view.buffer];
    uv_data = reinterpret_cast<const float*>(
        &uv_buffer.data[uv_view.byteOffset + uv_accessor.byteOffset]);
    uv_stride = uv_accessor.ByteStride(uv_view)
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
  geo.uvs.reserve(vertex_count);

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

    if (uv_data) {
      geo.uvs.emplace_back(uv_data[i * uv_stride + 0],
                           uv_data[i * uv_stride + 1]);
    } else {
      geo.uvs.emplace_back(0, 0);
    }
  }

  scene->geometries.push_back(std::move(geo));
}

void ProcessMaterials(const tinygltf::Model& model, Scene* scene) {
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
    // Check KHR_materials_emissive_strength if needed, or just use
    // emissiveFactor
    if (gltf_mat.emissiveFactor.size() == 3) {
      float max_e = std::max({(float)gltf_mat.emissiveFactor[0],
                              (float)gltf_mat.emissiveFactor[1],
                              (float)gltf_mat.emissiveFactor[2]});
      mat.emission_intensity = max_e;
    }

    // Texture (Base Color)
    int tex_idx = gltf_mat.pbrMetallicRoughness.baseColorTexture.index;
    if (tex_idx >= 0 && tex_idx < model.textures.size()) {
      int img_idx = model.textures[tex_idx].source;
      if (img_idx >= 0 && img_idx < model.images.size()) {
        const auto& img = model.images[img_idx];
        mat.albedo.width = img.width;
        mat.albedo.height = img.height;
        mat.albedo.channels = img.component;
        mat.albedo.pixel_data = img.image;  // Copy data
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
    if (spot.Has("innerConeAngle"))
      l.inner_cone_angle =
          static_cast<float>(spot.Get("innerConeAngle").Get<double>());
    if (spot.Has("outerConeAngle"))
      l.outer_cone_angle =
          static_cast<float>(spot.Get("outerConeAngle").Get<double>());
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
  if (node.extensions.find("KHR_lights_punctual") != node.extensions.end()) {
    const auto& ext = node.extensions.at("KHR_lights_punctual");
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

  ProcessMaterials(model, &scene);

  const tinygltf::Scene& gltf_scene =
      model.scenes[model.defaultScene > -1 ? model.defaultScene : 0];
  for (int node_index : gltf_scene.nodes) {
    TraverseNodes(model, node_index, Eigen::Affine3f::Identity(), &scene);
  }

  return scene;
}

}  // namespace sh_baker
