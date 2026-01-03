#include "saver.h"

#include <glog/logging.h>
#include <tiny_gltf.h>

#include <fstream>

#include "tinyexr.h"

namespace sh_baker {

namespace {

// Helpers for buffer management
void AddBufferView(tinygltf::Model& model, const void* data, size_t size,
                   size_t stride, int target, int& view_index) {
  if (model.buffers.empty()) {
    model.buffers.emplace_back();
  }
  tinygltf::Buffer& buffer = model.buffers[0];

  // Align to 4 bytes
  size_t padding = 0;
  if (buffer.data.size() % 4 != 0) {
    padding = 4 - (buffer.data.size() % 4);
  }
  for (size_t i = 0; i < padding; ++i) buffer.data.push_back(0);

  size_t byte_offset = buffer.data.size();
  const unsigned char* bytes = static_cast<const unsigned char*>(data);
  buffer.data.insert(buffer.data.end(), bytes, bytes + size);

  tinygltf::BufferView view;
  view.buffer = 0;
  view.byteOffset = byte_offset;
  view.byteLength = size;
  view.byteStride = stride;
  view.target = target;
  model.bufferViews.push_back(view);
  view_index = static_cast<int>(model.bufferViews.size() - 1);
}

int AddAccessor(tinygltf::Model& model, int buffer_view, int component_type,
                size_t count, int type, const std::vector<double>& min_vals,
                const std::vector<double>& max_vals) {
  tinygltf::Accessor acc;
  acc.bufferView = buffer_view;
  acc.byteOffset = 0;
  acc.componentType = component_type;
  acc.count = count;
  acc.type = type;
  acc.minValues = min_vals;
  acc.maxValues = max_vals;
  model.accessors.push_back(acc);
  return static_cast<int>(model.accessors.size() - 1);
}

// Coefficient names for file suffixes or channel prefixes
// Order: L0, L1m1, L10, L11, L2m2, L2m1, L20, L21, L22
const char* kCoeffNames[] = {"L0",   "L1m1", "L10", "L11", "L2m2",
                             "L2m1", "L20",  "L21", "L22"};

bool SaveCombined(const SHTexture& sh_texture,
                  const std::filesystem::path& path) {
  EXRHeader header;
  InitEXRHeader(&header);

  EXRImage image;
  InitEXRImage(&image);

  image.num_channels = 27;  // 9 coeffs * 3 (RGB)

  // Channel names
  std::vector<std::string> channel_names;
  // We need C-strings for header.channels[i].name, so keep the strings alive
  channel_names.reserve(27);

  // Band 0: L0
  // Band 1: L1m1, L10, L11
  // Band 2: L2m2, L2m1, L20, L21, L22
  const char* rgb_suffix[] = {".R", ".G", ".B"};

  for (int i = 0; i < 9; ++i) {
    for (int c = 0; c < 3; ++c) {
      channel_names.push_back(std::string(kCoeffNames[i]) + rgb_suffix[c]);
    }
  }

  std::vector<float> channels[27];
  float* image_ptr[27];
  header.channels =
      (EXRChannelInfo*)malloc(sizeof(EXRChannelInfo) * image.num_channels);

  // Allocate pixel types
  header.pixel_types = (int*)malloc(sizeof(int) * image.num_channels);
  header.requested_pixel_types = (int*)malloc(sizeof(int) * image.num_channels);

  int num_pixels = sh_texture.width * sh_texture.height;

  // Split SoA
  for (int i = 0; i < 27; ++i) {
    channels[i].resize(num_pixels);
    image_ptr[i] = channels[i].data();

    // Map i to (coeff_idx, rgb_idx)
    int coeff_idx = i / 3;
    int rgb_idx = i % 3;

    for (int p = 0; p < num_pixels; ++p) {
      if (rgb_idx == 0)
        channels[i][p] = sh_texture.pixels[p].coeffs[coeff_idx].x();
      else if (rgb_idx == 1)
        channels[i][p] = sh_texture.pixels[p].coeffs[coeff_idx].y();
      else
        channels[i][p] = sh_texture.pixels[p].coeffs[coeff_idx].z();
    }

    // Set header info
    strncpy(header.channels[i].name, channel_names[i].c_str(), 255);
    header.pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT;
    header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT;
  }

  image.images = (unsigned char**)image_ptr;
  image.width = sh_texture.width;
  image.height = sh_texture.height;

  header.num_channels = image.num_channels;
  header.compression_type = TINYEXR_COMPRESSIONTYPE_ZIP;

  const char* err = nullptr;
  int ret = SaveEXRImageToFile(&image, &header, path.string().c_str(), &err);

  free(header.channels);
  free(header.pixel_types);
  free(header.requested_pixel_types);

  if (ret != TINYEXR_SUCCESS) {
    LOG(ERROR) << "SaveEXRImageToFile failed: "
               << (err ? err : "Unknown error");
    FreeEXRErrorMessage(err);
    return false;
  }

  return true;
}

bool SaveSplit(const SHTexture& sh_texture, const std::filesystem::path& path) {
  // path is "dir/filename.exr". We want "dir/filename_L0.exr" etc.
  std::filesystem::path parent = path.parent_path();
  std::string stem = path.stem().string();
  std::string extension = path.extension().string();

  int num_pixels = sh_texture.width * sh_texture.height;

  for (int i = 0; i < 9; ++i) {
    std::string coeff_name = kCoeffNames[i];
    std::string filename = stem + "_" + coeff_name + extension;
    std::filesystem::path sub_path = parent / filename;

    // Save as standard RGB EXR
    EXRHeader header;
    InitEXRHeader(&header);
    EXRImage image;
    InitEXRImage(&image);

    image.num_channels = 3;

    std::vector<float> channels[3];
    float* image_ptr[3];
    header.channels = (EXRChannelInfo*)malloc(sizeof(EXRChannelInfo) * 3);
    header.pixel_types = (int*)malloc(sizeof(int) * 3);
    header.requested_pixel_types = (int*)malloc(sizeof(int) * 3);

    for (int c = 0; c < 3; ++c) {
      channels[c].resize(num_pixels);
      image_ptr[c] = channels[c].data();

      // Collect data
      for (int p = 0; p < num_pixels; ++p) {
        if (c == 0)
          channels[c][p] = sh_texture.pixels[p].coeffs[i].x();
        else if (c == 1)
          channels[c][p] = sh_texture.pixels[p].coeffs[i].y();
        else
          channels[c][p] = sh_texture.pixels[p].coeffs[i].z();
      }

      // Channel names: R, G, B
      const char* names[] = {"R", "G", "B"};
      strncpy(header.channels[c].name, names[c], 255);
      header.pixel_types[c] = TINYEXR_PIXELTYPE_FLOAT;
      header.requested_pixel_types[c] = TINYEXR_PIXELTYPE_FLOAT;
    }

    image.images = (unsigned char**)image_ptr;
    image.width = sh_texture.width;
    image.height = sh_texture.height;

    header.num_channels = 3;
    header.compression_type = TINYEXR_COMPRESSIONTYPE_ZIP;

    const char* err = nullptr;
    int ret =
        SaveEXRImageToFile(&image, &header, sub_path.string().c_str(), &err);

    free(header.channels);
    free(header.pixel_types);
    free(header.requested_pixel_types);

    if (ret != TINYEXR_SUCCESS) {
      LOG(ERROR) << "SaveEXRImageToFile failed for " << sub_path << ": "
                 << (err ? err : "Unknown error");
      FreeEXRErrorMessage(err);
      return false;
    }
    LOG(INFO) << "Saved: " << sub_path;
  }

  return true;
}

}  // namespace

bool SaveSHLightMap(const SHTexture& sh_texture,
                    const std::filesystem::path& path, SaveMode mode) {
  if (sh_texture.pixels.empty() || sh_texture.width <= 0 ||
      sh_texture.height <= 0) {
    LOG(ERROR) << "Invalid SHTexture dimensions or empty pixels.";
    return false;
  }

  if (mode == SaveMode::kCombined) {
    return SaveCombined(sh_texture, path);
  } else {
    return SaveSplit(sh_texture, path);
  }
}

bool SaveScene(const Scene& scene, const std::filesystem::path& path) {
  tinygltf::Model model;
  model.asset.generator = "sh_baker";
  model.asset.version = "2.0";

  // TODO: Export Materials (basic)
  // For now, we skip material export or create dummy ones to match indices
  for (const auto& mat : scene.materials) {
    tinygltf::Material gmat;
    gmat.name = mat.name;
    model.materials.push_back(gmat);
  }

  tinygltf::Scene gscene;

  for (size_t i = 0; i < scene.geometries.size(); ++i) {
    const auto& geo = scene.geometries[i];

    // Create Mesh
    tinygltf::Mesh mesh;
    tinygltf::Primitive prim;
    prim.mode = TINYGLTF_MODE_TRIANGLES;
    prim.material = geo.material_id;

    // Position
    {
      int view_idx;
      std::vector<float> buffer_data;
      buffer_data.reserve(geo.vertices.size() * 3);
      std::vector<double> min_v = {1e9, 1e9, 1e9};
      std::vector<double> max_v = {-1e9, -1e9, -1e9};

      for (const auto& v : geo.vertices) {
        buffer_data.push_back(v.x());
        buffer_data.push_back(v.y());
        buffer_data.push_back(v.z());

        if (v.x() < min_v[0]) min_v[0] = v.x();
        if (v.y() < min_v[1]) min_v[1] = v.y();
        if (v.z() < min_v[2]) min_v[2] = v.z();
        if (v.x() > max_v[0]) max_v[0] = v.x();
        if (v.y() > max_v[1]) max_v[1] = v.y();
        if (v.z() > max_v[2]) max_v[2] = v.z();
      }
      AddBufferView(model, buffer_data.data(),
                    buffer_data.size() * sizeof(float), 12,
                    TINYGLTF_TARGET_ARRAY_BUFFER, view_idx);
      prim.attributes["POSITION"] =
          AddAccessor(model, view_idx, TINYGLTF_COMPONENT_TYPE_FLOAT,
                      geo.vertices.size(), TINYGLTF_TYPE_VEC3, min_v, max_v);
    }

    // Normal
    if (!geo.normals.empty()) {
      int view_idx;
      std::vector<float> buffer_data;
      buffer_data.reserve(geo.normals.size() * 3);
      for (const auto& n : geo.normals) {
        buffer_data.push_back(n.x());
        buffer_data.push_back(n.y());
        buffer_data.push_back(n.z());
      }
      AddBufferView(model, buffer_data.data(),
                    buffer_data.size() * sizeof(float), 12,
                    TINYGLTF_TARGET_ARRAY_BUFFER, view_idx);
      prim.attributes["NORMAL"] =
          AddAccessor(model, view_idx, TINYGLTF_COMPONENT_TYPE_FLOAT,
                      geo.normals.size(), TINYGLTF_TYPE_VEC3, {}, {});
    }

    // Texcoord 0 (Texture UVs)
    if (!geo.texture_uvs.empty()) {
      int view_idx;
      std::vector<float> buffer_data;
      buffer_data.reserve(geo.texture_uvs.size() * 2);
      for (const auto& uv : geo.texture_uvs) {
        buffer_data.push_back(uv.x());
        buffer_data.push_back(uv.y());
      }
      AddBufferView(model, buffer_data.data(),
                    buffer_data.size() * sizeof(float), 8,
                    TINYGLTF_TARGET_ARRAY_BUFFER, view_idx);
      prim.attributes["TEXCOORD_0"] =
          AddAccessor(model, view_idx, TINYGLTF_COMPONENT_TYPE_FLOAT,
                      geo.texture_uvs.size(), TINYGLTF_TYPE_VEC2, {}, {});
    }

    // Texcoord 1 (Lightmap UVs)
    if (!geo.lightmap_uvs.empty()) {
      int view_idx;
      std::vector<float> buffer_data;
      buffer_data.reserve(geo.lightmap_uvs.size() * 2);
      for (const auto& uv : geo.lightmap_uvs) {
        buffer_data.push_back(uv.x());
        buffer_data.push_back(uv.y());
      }
      AddBufferView(model, buffer_data.data(),
                    buffer_data.size() * sizeof(float), 8,
                    TINYGLTF_TARGET_ARRAY_BUFFER, view_idx);
      prim.attributes["TEXCOORD_1"] =
          AddAccessor(model, view_idx, TINYGLTF_COMPONENT_TYPE_FLOAT,
                      geo.lightmap_uvs.size(), TINYGLTF_TYPE_VEC2, {}, {});
    }

    // Indices
    {
      int view_idx;
      AddBufferView(model, geo.indices.data(),
                    geo.indices.size() * sizeof(uint32_t), 0,
                    TINYGLTF_TARGET_ELEMENT_ARRAY_BUFFER, view_idx);
      prim.indices =
          AddAccessor(model, view_idx, TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT,
                      geo.indices.size(), TINYGLTF_TYPE_SCALAR, {}, {});
    }

    mesh.primitives.push_back(prim);
    model.meshes.push_back(mesh);

    // Node
    tinygltf::Node node;
    node.mesh = static_cast<int>(model.meshes.size() - 1);

    // Transform
    Eigen::Matrix4f mat = geo.transform.matrix();
    std::vector<double> matrix;
    for (int k = 0; k < 16; ++k) matrix.push_back(mat(k));
    node.matrix = matrix;

    model.nodes.push_back(node);
    gscene.nodes.push_back(static_cast<int>(model.nodes.size() - 1));
  }

  model.scenes.push_back(gscene);
  model.defaultScene = 0;

  tinygltf::TinyGLTF loader;
  return loader.WriteGltfSceneToFile(&model, path.string(), false, true, true,
                                     false);
}

}  // namespace sh_baker
