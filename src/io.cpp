#include "io.h"

#include <iostream>

// Define these only in *one* .cpp file.
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "tiny_gltf.h"

#include <glog/logging.h>

namespace sh_baker {
namespace {

void ProcessPrimitive(const tinygltf::Model& model, const tinygltf::Primitive& primitive,
                      const Eigen::Affine3f& transform, Scene* scene) {
  // Get Position Accessor
  if (primitive.attributes.find("POSITION") == primitive.attributes.end()) {
    LOG(WARNING) << "Primitive missing POSITION attribute";
    return;
  }
  const tinygltf::Accessor& pos_accessor =
      model.accessors[primitive.attributes.at("POSITION")];
  const tinygltf::BufferView& pos_view = model.bufferViews[pos_accessor.bufferView];
  const tinygltf::Buffer& pos_buffer = model.buffers[pos_view.buffer];
  const float* pos_data = reinterpret_cast<const float*>(
      &pos_buffer.data[pos_view.byteOffset + pos_accessor.byteOffset]);
  int pos_stride = pos_accessor.ByteStride(pos_view) / sizeof(float);

  // Get Normal Accessor
  const float* norm_data = nullptr;
  int norm_stride = 0;
  if (primitive.attributes.find("NORMAL") != primitive.attributes.end()) {
    const tinygltf::Accessor& norm_accessor =
        model.accessors[primitive.attributes.at("NORMAL")];
    const tinygltf::BufferView& norm_view = model.bufferViews[norm_accessor.bufferView];
    const tinygltf::Buffer& norm_buffer = model.buffers[norm_view.buffer];
    norm_data = reinterpret_cast<const float*>(
        &norm_buffer.data[norm_view.byteOffset + norm_accessor.byteOffset]);
    norm_stride = norm_accessor.ByteStride(norm_view) / sizeof(float);
  }

  // Get UV Accessor
  const float* uv_data = nullptr;
  int uv_stride = 0;
  std::string texcoord = "TEXCOORD_0";
  if (primitive.attributes.find("TEXCOORD_1") != primitive.attributes.end()) {
    texcoord = "TEXCOORD_1";
  }
  if (primitive.attributes.find(texcoord) != primitive.attributes.end()) {
    const tinygltf::Accessor& uv_accessor =
        model.accessors[primitive.attributes.at(texcoord)];
    const tinygltf::BufferView& uv_view = model.bufferViews[uv_accessor.bufferView];
    const tinygltf::Buffer& uv_buffer = model.buffers[uv_view.buffer];
    uv_data = reinterpret_cast<const float*>(
        &uv_buffer.data[uv_view.byteOffset + uv_accessor.byteOffset]);
    uv_stride = uv_accessor.ByteStride(uv_view) / sizeof(float);
  }

  // Indices
  uint32_t index_offset = scene->vertices.size();
  if (primitive.indices > -1) {
      const tinygltf::Accessor& index_accessor = model.accessors[primitive.indices];
      const tinygltf::BufferView& index_view = model.bufferViews[index_accessor.bufferView];
      const tinygltf::Buffer& index_buffer = model.buffers[index_view.buffer];
      const uint8_t* index_data_ptr = &index_buffer.data[index_view.byteOffset + index_accessor.byteOffset];
      int stride = index_accessor.ByteStride(index_view);

      for (size_t i = 0; i < index_accessor.count; ++i) {
          uint32_t val = 0;
          if (index_accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE) {
              val = *(index_data_ptr + i * stride);
          } else if (index_accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
              val = *reinterpret_cast<const uint16_t*>(index_data_ptr + i * stride);
          } else if (index_accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
              val = *reinterpret_cast<const uint32_t*>(index_data_ptr + i * stride);
          }
          scene->indices.push_back(index_offset + val);
      }
  } else {
      // Non-indexed TODO
      LOG(WARNING) << "Non-indexed geometry not fully supported yet (assumed indexed)";
  }

  // Vertices
  for (size_t i = 0; i < pos_accessor.count; ++i) {
      Eigen::Vector3f pos(pos_data[i * pos_stride + 0],
                          pos_data[i * pos_stride + 1],
                          pos_data[i * pos_stride + 2]);
      // Apply transform
      pos = transform * pos;
      scene->vertices.push_back(pos);

      Eigen::Vector3f norm(0, 1, 0);
      if (norm_data) {
          norm = Eigen::Vector3f(norm_data[i * norm_stride + 0],
                                 norm_data[i * norm_stride + 1],
                                 norm_data[i * norm_stride + 2]);
          // Apply rotation transform to normal
          norm = transform.rotation() * norm; 
          norm.normalize();
      }
      scene->normals.push_back(norm);

      Eigen::Vector2f uv(0, 0);
      if (uv_data) {
          uv = Eigen::Vector2f(uv_data[i * uv_stride + 0],
                               uv_data[i * uv_stride + 1]);
      }
      scene->uvs.push_back(uv);
  }
}

void TraverseNodes(const tinygltf::Model& model, int node_index,
                   const Eigen::Affine3f& parent_transform, Scene* scene) {
  const tinygltf::Node& node = model.nodes[node_index];

  Eigen::Affine3f local_transform = Eigen::Affine3f::Identity();
  
  if (node.matrix.size() == 16) {
      Eigen::Matrix4f mat;
      for (int i = 0; i < 16; ++i) mat(i) = static_cast<float>(node.matrix[i]);
      local_transform = Eigen::Affine3f(mat); // tinygltf is column-major? yes gltf is col-major. Eigen map is col-major by default.
  } else {
      // Translation
      if (node.translation.size() == 3) {
          local_transform.translate(Eigen::Vector3f(node.translation[0], node.translation[1], node.translation[2]));
      }
      // Rotation (Quaternion)
      if (node.rotation.size() == 4) {
          Eigen::Quaternionf q(node.rotation[3], node.rotation[0], node.rotation[1], node.rotation[2]); // w, x, y, z
          local_transform.rotate(q);
      }
      // Scale
      if (node.scale.size() == 3) {
          local_transform.scale(Eigen::Vector3f(node.scale[0], node.scale[1], node.scale[2]));
      }
  }

  Eigen::Affine3f global_transform = parent_transform * local_transform;

  if (node.mesh >= 0) {
      const tinygltf::Mesh& mesh = model.meshes[node.mesh];
      for (const auto& primitive : mesh.primitives) {
          ProcessPrimitive(model, primitive, global_transform, scene);
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
    LOG(WARNING) << "TinyGLTF warning: " << warn;
  }
  if (!err.empty()) {
    LOG(ERROR) << "TinyGLTF error: " << err;
  }
  if (!ret) {
    return std::nullopt;
  }

  Scene scene;
  
  const tinygltf::Scene& gltf_scene = model.scenes[model.defaultScene > -1 ? model.defaultScene : 0];
  for (int node_index : gltf_scene.nodes) {
      TraverseNodes(model, node_index, Eigen::Affine3f::Identity(), &scene);
  }

  return scene;
}

}  // namespace sh_baker
