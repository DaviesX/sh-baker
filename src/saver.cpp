#include "saver.h"

#include <glog/logging.h>
#include <stb_image_write.h>
#include <tiny_gltf.h>

#include <cmath>
#include <filesystem>
#include <map>
#include <unordered_map>

#include "colorspace.h"
#include "tinyexr.h"

namespace sh_baker {

namespace {

// Helpers for buffer management
void AddBufferView(const void* data, size_t size, size_t stride, int target,
                   int& view_index, tinygltf::Model* model) {
  if (model->buffers.empty()) {
    model->buffers.emplace_back();
  }
  tinygltf::Buffer& buffer = model->buffers[0];

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
  model->bufferViews.push_back(view);
  view_index = static_cast<int>(model->bufferViews.size() - 1);
}

int AddAccessor(int buffer_view, int component_type, size_t count, int type,
                const std::vector<double>& min_vals,
                const std::vector<double>& max_vals, tinygltf::Model* model) {
  tinygltf::Accessor acc;
  acc.bufferView = buffer_view;
  acc.byteOffset = 0;
  acc.componentType = component_type;
  acc.count = count;
  acc.type = type;
  acc.minValues = min_vals;
  acc.maxValues = max_vals;
  model->accessors.push_back(acc);
  return static_cast<int>(model->accessors.size() - 1);
}

std::optional<int> AddOrReuseTexture(
    const std::filesystem::path& from_uri,
    const std::filesystem::path& output_dir, tinygltf::Model* model,
    std::unordered_map<std::string, int>* texture_allocations) {
  // Copy the file to the same directory as the output file.
  // We use the filename as the relative URI in the glTF.
  std::filesystem::path filename = from_uri.filename();
  std::filesystem::path destination = output_dir / filename;

  std::string uri_key = filename.string();
  auto texture_index_it = texture_allocations->find(uri_key);
  if (texture_index_it != texture_allocations->end()) {
    return texture_index_it->second;
  }

  try {
    // to_uri is usually unique, but just in case we have a collision we
    // will overwrite the file.
    std::filesystem::copy_file(
        from_uri, destination,
        std::filesystem::copy_options::overwrite_existing);
  } catch (const std::filesystem::filesystem_error& e) {
    LOG(ERROR) << "Failed to copy file from " << from_uri << " to "
               << destination << ". Cause: " << e.what();
    return std::nullopt;
  }

  tinygltf::Image img;
  img.uri = uri_key;
  model->images.push_back(img);

  tinygltf::Texture tex;
  tex.source = static_cast<int>(model->images.size() - 1);
  model->textures.push_back(tex);

  texture_index_it =
      texture_allocations->emplace(uri_key, model->textures.size() - 1).first;
  return texture_index_it->second;
}

// Coefficient names for file suffixes or channel prefixes
// Order: L0, L1m1, L10, L11, L2m2, L2m1, L20, L21, L22
const char* kCoeffNames[] = {"L0",   "L1m1", "L10", "L11", "L2m2",
                             "L2m1", "L20",  "L21", "L22"};

bool SaveCombined(const SHTexture& sh_texture,
                  const Texture32F& environment_visibility_texture,
                  const std::filesystem::path& path) {
  EXRHeader header;
  InitEXRHeader(&header);

  EXRImage image;
  InitEXRImage(&image);

  image.num_channels = 28;  // 9 coeffs * 3 (RGB) + 1 EnvOcclusion

  // Channel names
  std::vector<std::string> channel_names;
  // We need C-strings for header.channels[i].name, so keep the strings alive
  channel_names.reserve(28);

  // Band 0: L0
  // Band 1: L1m1, L10, L11
  // Band 2: L2m2, L2m1, L20, L21, L22
  const char* rgb_suffix[] = {".R", ".G", ".B"};

  for (int i = 0; i < 9; ++i) {
    for (int c = 0; c < 3; ++c) {
      channel_names.push_back(std::string(kCoeffNames[i]) + rgb_suffix[c]);
    }
  }
  channel_names.push_back("EnvVisibility");

  std::vector<float> channels[28];
  float* image_ptr[28];
  header.channels =
      (EXRChannelInfo*)malloc(sizeof(EXRChannelInfo) * image.num_channels);

  // Allocate pixel types
  header.pixel_types = (int*)malloc(sizeof(int) * image.num_channels);
  header.requested_pixel_types = (int*)malloc(sizeof(int) * image.num_channels);

  int num_pixels = sh_texture.width * sh_texture.height;

  // Split SoA
  for (int i = 0; i < 28; ++i) {
    channels[i].resize(num_pixels);
    image_ptr[i] = channels[i].data();

    // Map i to (coeff_idx, rgb_idx) or EnvOcclusion
    if (i < 27) {
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
    } else {
      // EnvVisibility (index 27)
      for (int p = 0; p < num_pixels; ++p) {
        // Handle mismatched sizes? Assuming aligned.
        if (p < environment_visibility_texture.pixel_data.size())
          channels[i][p] = environment_visibility_texture.pixel_data[p];
        else
          channels[i][p] = 0.0f;
      }
    }

    // Set header info
    strncpy(header.channels[i].name, channel_names[i].c_str(), 255);
    header.pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT;  // Input is float
    header.requested_pixel_types[i] =
        TINYEXR_PIXELTYPE_HALF;  // Storage is half
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

bool SaveSplit(const SHTexture& sh_texture,
               const Texture32F& environment_visibility_texture,
               const std::filesystem::path& path) {
  // path is "dir/filename.exr". We want "dir/filename_L0.exr" etc.
  std::filesystem::path parent = path.parent_path();
  std::string stem = path.stem().string();
  std::string extension = path.extension().string();

  int num_pixels = sh_texture.width * sh_texture.height;

  // Save SH Coefficients
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
      header.pixel_types[c] = TINYEXR_PIXELTYPE_FLOAT;  // Input is float
      header.requested_pixel_types[c] =
          TINYEXR_PIXELTYPE_HALF;  // Storage is half
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

  // FIXME(Gemini): Save the environment visibility to the alpha channel of the
  // first file.
  // Save Environment Visibility
  {
    std::string filename = stem + "_EnvVisibility" + extension;
    std::filesystem::path sub_path = parent / filename;

    EXRHeader header;
    InitEXRHeader(&header);
    EXRImage image;
    InitEXRImage(&image);

    image.num_channels = 1;

    std::vector<float> channel;
    channel.resize(num_pixels);
    // Copy data
    if (environment_visibility_texture.pixel_data.size() == num_pixels) {
      std::copy(environment_visibility_texture.pixel_data.begin(),
                environment_visibility_texture.pixel_data.end(),
                channel.begin());
    }

    float* image_ptr[1];
    image_ptr[0] = channel.data();

    header.channels = (EXRChannelInfo*)malloc(sizeof(EXRChannelInfo) * 1);
    header.pixel_types = (int*)malloc(sizeof(int) * 1);
    header.requested_pixel_types = (int*)malloc(sizeof(int) * 1);

    strncpy(header.channels[0].name, "Y", 255);
    header.pixel_types[0] = TINYEXR_PIXELTYPE_FLOAT;
    header.requested_pixel_types[0] = TINYEXR_PIXELTYPE_HALF;

    image.images = (unsigned char**)image_ptr;
    image.width = sh_texture.width;
    image.height = sh_texture.height;
    header.num_channels = 1;
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

bool SavePackedLuminance(const SHTexture& sh_texture,
                         const Texture32F& environment_visibility_texture,
                         const std::filesystem::path& path) {
  // We will save 3 RGBA files:
  // _packed_0.exr: L0.r, L0.g, L0.b, EnvVisibility
  // _packed_1.exr: L1m1.Y, L10.Y, L11.Y, L2m2.Y
  // _packed_2.exr: L2m1.Y, L20.Y, L21.Y, L22.Y

  std::filesystem::path parent = path.parent_path();
  std::string stem = path.stem().string();
  std::string extension = path.extension().string();

  int num_pixels = sh_texture.width * sh_texture.height;

  // L1/L2 Luminance weights (Rec. 709)
  const Eigen::Vector3f kLumWeights(0.2126f, 0.7152f, 0.0722f);

  // We have 3 output files.
  for (int file_idx = 0; file_idx < 3; ++file_idx) {
    std::string filename =
        stem + "_packed_" + std::to_string(file_idx) + extension;
    std::filesystem::path sub_path = parent / filename;

    EXRHeader header;
    InitEXRHeader(&header);
    EXRImage image;
    InitEXRImage(&image);

    image.num_channels = 4;

    std::vector<float> channels[4];
    float* image_ptr[4];
    header.channels = (EXRChannelInfo*)malloc(sizeof(EXRChannelInfo) * 4);
    header.pixel_types = (int*)malloc(sizeof(int) * 4);
    header.requested_pixel_types = (int*)malloc(sizeof(int) * 4);

    for (int c = 0; c < 4; ++c) {
      channels[c].resize(num_pixels);
      image_ptr[c] = channels[c].data();

      // Channel mapping logic
      for (int p = 0; p < num_pixels; ++p) {
        float val = 0.0f;
        const auto& coeffs = sh_texture.pixels[p].coeffs;

        if (file_idx == 0) {
          // File 0: L0.r, L0.g, L0.b, EnvOcclusion
          if (c < 3) {
            // L0 RGB (coeff 0)
            val = coeffs[0][c];  // x, y, z
          } else {
            // EnvVisibility
            if (p < environment_visibility_texture.pixel_data.size())
              val = environment_visibility_texture.pixel_data[p];
            else
              val = 0.0f;
          }
        } else if (file_idx == 1) {
          // File 1: L1m1.Y, L10.Y, L11.Y, L2m2.Y (coeffs 1, 2, 3, 4)
          int sh_idx = 1 + c;
          val = coeffs[sh_idx].dot(kLumWeights);
        } else if (file_idx == 2) {
          // File 2: L2m1.Y, L20.Y, L21.Y, L22.Y (coeffs 5, 6, 7, 8)
          int sh_idx = 5 + c;
          val = coeffs[sh_idx].dot(kLumWeights);
        }
        channels[c][p] = val;
      }

      // Channel names: R, G, B, A
      const char* names[] = {"R", "G", "B", "A"};
      strncpy(header.channels[c].name, names[c], 255);

      // Use HALF float for bandwidth optimization
      header.pixel_types[c] = TINYEXR_PIXELTYPE_FLOAT;  // Input is float
      header.requested_pixel_types[c] =
          TINYEXR_PIXELTYPE_HALF;  // Storage is half
    }

    image.images = (unsigned char**)image_ptr;
    image.width = sh_texture.width;
    image.height = sh_texture.height;

    header.num_channels = 4;
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
    LOG(INFO) << "Saved packed SH: " << sub_path;
  }
  return true;
}

}  // namespace

bool SaveSHLightMap(const SHTexture& sh_texture,
                    const Texture32F& environment_visibility_texture,
                    const std::filesystem::path& path, SaveMode mode) {
  if (sh_texture.pixels.empty() || sh_texture.width <= 0 ||
      sh_texture.height <= 0) {
    LOG(ERROR) << "Invalid SHTexture dimensions or empty pixels.";
    return false;
  }

  if (mode == SaveMode::kCombined) {
    return SaveCombined(sh_texture, environment_visibility_texture, path);
  } else if (mode == SaveMode::kLuminancePacked) {
    return SavePackedLuminance(sh_texture, environment_visibility_texture,
                               path);
  } else {
    return SaveSplit(sh_texture, environment_visibility_texture, path);
  }
}

bool SaveScene(const Scene& scene, const std::filesystem::path& path) {
  tinygltf::Model model;
  model.asset.generator = "sh_baker";
  model.asset.version = "2.0";

  // Export Materials
  // We need to allocate the textures a unique index as we push them into
  // the model. This is because glTF references the textures by index in the
  // material nodes.
  std::unordered_map<std::string, int> texture_allocations;

  for (const auto& mat : scene.materials) {
    tinygltf::Material gmat;
    gmat.name = mat.name;

    if (mat.albedo.file_path) {
      auto texture_index =
          AddOrReuseTexture(*mat.albedo.file_path, path.parent_path(), &model,
                            &texture_allocations);
      if (!texture_index.has_value()) {
        return false;
      }
      gmat.pbrMetallicRoughness.baseColorTexture.index = *texture_index;
    } else if (mat.albedo.width == 1 && mat.albedo.height == 1 &&
               mat.albedo.pixel_data.size() >= 4) {
      // 1x1 Texture -> baseColorFactor (Linear)
      std::vector<double> color(4);
      for (int i = 0; i < 3; ++i) {
        color[i] = SRGBToLinear(mat.albedo.pixel_data[i]);
      }
      color[3] = mat.albedo.pixel_data[3] / 255.0f;  // Alpha
      gmat.pbrMetallicRoughness.baseColorFactor = color;
    }

    // Normal Texture
    if (mat.normal_texture.file_path) {
      auto texture_index =
          AddOrReuseTexture(*mat.normal_texture.file_path, path.parent_path(),
                            &model, &texture_allocations);
      if (texture_index.has_value()) {
        gmat.normalTexture.index = *texture_index;
      }
    }
    // If 1x1 normal map (fallback), we skip it as it likely represents flat
    // normal

    // Metallic-Roughness Texture
    if (mat.metallic_roughness_texture.file_path) {
      auto texture_index =
          AddOrReuseTexture(*mat.metallic_roughness_texture.file_path,
                            path.parent_path(), &model, &texture_allocations);
      if (texture_index.has_value()) {
        gmat.pbrMetallicRoughness.metallicRoughnessTexture.index =
            *texture_index;
      }
    } else if (mat.metallic_roughness_texture.width == 1 &&
               mat.metallic_roughness_texture.height == 1 &&
               mat.metallic_roughness_texture.pixel_data.size() >= 3) {
      // Extract factors
      // B = Metallic, G = Roughness
      // We assume the stored pixel data is correct (PBR convention)
      unsigned char g = mat.metallic_roughness_texture.pixel_data[1];
      unsigned char b = mat.metallic_roughness_texture.pixel_data[2];

      gmat.pbrMetallicRoughness.roughnessFactor = g / 255.0;
      gmat.pbrMetallicRoughness.metallicFactor = b / 255.0;
    }

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
      AddBufferView(buffer_data.data(), buffer_data.size() * sizeof(float), 12,
                    TINYGLTF_TARGET_ARRAY_BUFFER, view_idx, &model);
      prim.attributes["POSITION"] = AddAccessor(
          view_idx, TINYGLTF_COMPONENT_TYPE_FLOAT, geo.vertices.size(),
          TINYGLTF_TYPE_VEC3, min_v, max_v, &model);
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
      AddBufferView(buffer_data.data(), buffer_data.size() * sizeof(float), 12,
                    TINYGLTF_TARGET_ARRAY_BUFFER, view_idx, &model);
      prim.attributes["NORMAL"] =
          AddAccessor(view_idx, TINYGLTF_COMPONENT_TYPE_FLOAT,
                      geo.normals.size(), TINYGLTF_TYPE_VEC3, {}, {}, &model);
    }

    // Tangent
    if (!geo.tangents.empty()) {
      int view_idx;
      std::vector<float> buffer_data;
      buffer_data.reserve(geo.tangents.size() * 4);
      for (const auto& t : geo.tangents) {
        buffer_data.push_back(t.x());
        buffer_data.push_back(t.y());
        buffer_data.push_back(t.z());
        buffer_data.push_back(t.w());
      }
      AddBufferView(buffer_data.data(), buffer_data.size() * sizeof(float), 16,
                    TINYGLTF_TARGET_ARRAY_BUFFER, view_idx, &model);
      prim.attributes["TANGENT"] =
          AddAccessor(view_idx, TINYGLTF_COMPONENT_TYPE_FLOAT,
                      geo.tangents.size(), TINYGLTF_TYPE_VEC4, {}, {}, &model);
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
      AddBufferView(buffer_data.data(), buffer_data.size() * sizeof(float), 8,
                    TINYGLTF_TARGET_ARRAY_BUFFER, view_idx, &model);
      prim.attributes["TEXCOORD_0"] = AddAccessor(
          view_idx, TINYGLTF_COMPONENT_TYPE_FLOAT, geo.texture_uvs.size(),
          TINYGLTF_TYPE_VEC2, {}, {}, &model);
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
      AddBufferView(buffer_data.data(), buffer_data.size() * sizeof(float), 8,
                    TINYGLTF_TARGET_ARRAY_BUFFER, view_idx, &model);
      prim.attributes["TEXCOORD_1"] = AddAccessor(
          view_idx, TINYGLTF_COMPONENT_TYPE_FLOAT, geo.lightmap_uvs.size(),
          TINYGLTF_TYPE_VEC2, {}, {}, &model);
    }

    // Indices
    {
      int view_idx;
      AddBufferView(geo.indices.data(), geo.indices.size() * sizeof(uint32_t),
                    0, TINYGLTF_TARGET_ELEMENT_ARRAY_BUFFER, view_idx, &model);
      prim.indices =
          AddAccessor(view_idx, TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT,
                      geo.indices.size(), TINYGLTF_TYPE_SCALAR, {}, {}, &model);
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

  // Export Environment (Skybox)
  if (scene.environment &&
      scene.environment->type == Environment::Type::Texture) {
    std::optional<std::filesystem::path> env_path =
        scene.environment->texture.file_path;

    if (env_path) {
      std::filesystem::path filename = env_path->filename();
      std::filesystem::path destination = path.parent_path() / filename;

      try {
        std::filesystem::copy_file(
            *env_path, destination,
            std::filesystem::copy_options::overwrite_existing);

        // Add "skybox" to extras
        tinygltf::Value::Object extras_obj;
        // If extras already exists and is object, use it?
        // tinygltf::Model::extras is Value.

        // Simpler: Just set "skybox" in extras value.
        // But tinygltf::Value is immutable-ish wrapper? No, it has Setters?
        // Actually model.extras is a tinygltf::Value.
        // We need to construct a Value that is an Object.

        // Current model.extras is empty (default).
        // Let's create an object.
        tinygltf::Value::Object extras_map;
        if (model.extras.IsObject()) {
          // Keep existing? Not implemented yet. Assuming new model.
        }
        extras_map["skybox"] = tinygltf::Value(filename.string());
        model.extras = tinygltf::Value(extras_map);

      } catch (const std::filesystem::filesystem_error& e) {
        LOG(ERROR) << "Failed to copy environment map from " << *env_path
                   << " to " << destination << ". Cause: " << e.what();
      }
    }
  }

  // Export Lights (KHR_lights_punctual)
  if (!scene.lights.empty()) {
    tinygltf::Value::Array light_array;
    std::vector<int> light_node_indices;

    int light_idx = 0;
    for (const auto& light : scene.lights) {
      if (light.type == Light::Type::Area) continue;

      tinygltf::Value::Object light_obj;

      // Color
      std::vector<tinygltf::Value> color_vec;
      color_vec.push_back(tinygltf::Value(double(light.color.x())));
      color_vec.push_back(tinygltf::Value(double(light.color.y())));
      color_vec.push_back(tinygltf::Value(double(light.color.z())));
      light_obj["color"] = tinygltf::Value(color_vec);

      light_obj["intensity"] = tinygltf::Value(double(light.intensity));

      std::string type_str;
      if (light.type == Light::Type::Directional) {
        type_str = "directional";
      } else if (light.type == Light::Type::Point) {
        type_str = "point";
      } else if (light.type == Light::Type::Spot) {
        type_str = "spot";

        tinygltf::Value::Object spot_obj;
        // Clamp to [-1, 1] to avoid NaN from std::acos with -ffast-math
        // Use manual clamping to avoid potential issues with -ffast-math
        // optimizations impacting std::clamp or std::acos behavior with edge
        // cases.
        auto safe_acos = [](float cos_val) -> double {
          if (cos_val >= 1.0f) return 0.0;
          if (cos_val <= -1.0f) return 3.14159265358979323846;
          return std::acos(cos_val);
        };

        spot_obj["innerConeAngle"] =
            tinygltf::Value(safe_acos(light.cos_inner_cone));
        spot_obj["outerConeAngle"] =
            tinygltf::Value(safe_acos(light.cos_outer_cone));
        light_obj["spot"] = tinygltf::Value(spot_obj);
      }
      light_obj["type"] = tinygltf::Value(type_str);
      light_obj["name"] = tinygltf::Value("Light_" + std::to_string(light_idx));

      light_array.push_back(tinygltf::Value(light_obj));

      // Create Node for this light
      tinygltf::Node node;
      node.name = "LightNode_" + std::to_string(light_idx);

      // Position (Translation)
      node.translation.push_back(light.position.x());
      node.translation.push_back(light.position.y());
      node.translation.push_back(light.position.z());

      // Orientation (Rotation)
      // glTF lights point down -Z. We need to align -Z with light.direction.
      if (light.type == Light::Type::Directional ||
          light.type == Light::Type::Spot) {
        // Construct Basis
        // Z = -direction
        Eigen::Vector3f Z = -light.direction.normalized();
        // Handle Up vector case
        Eigen::Vector3f up = Eigen::Vector3f::UnitY();
        if (std::abs(Z.dot(up)) > 0.99f) up = Eigen::Vector3f::UnitX();

        Eigen::Vector3f X = up.cross(Z).normalized();
        Eigen::Vector3f Y = Z.cross(X).normalized();

        // Convert to Quaternion
        // Matrix: [X Y Z]
        Eigen::Matrix3f rot;
        rot.col(0) = X;
        rot.col(1) = Y;
        rot.col(2) = Z;

        Eigen::Quaternionf q(rot);
        node.rotation.push_back(q.x());
        node.rotation.push_back(q.y());
        node.rotation.push_back(q.z());
        node.rotation.push_back(q.w());
      }

      // Extension on Node
      // Using node.light tells tinygltf to write the KHR_lights_punctual
      // extension
      node.light = light_idx;

      model.nodes.push_back(node);
      gscene.nodes.push_back(static_cast<int>(model.nodes.size() - 1));

      light_idx++;
    }

    if (light_idx > 0) {
      if (std::find(model.extensionsUsed.begin(), model.extensionsUsed.end(),
                    "KHR_lights_punctual") == model.extensionsUsed.end()) {
        model.extensionsUsed.push_back("KHR_lights_punctual");
      }

      tinygltf::Value::Object ext_container;
      ext_container["lights"] = tinygltf::Value(light_array);
      model.extensions["KHR_lights_punctual"] = tinygltf::Value(ext_container);
    }
  }

  model.scenes.push_back(gscene);
  model.defaultScene = 0;

  tinygltf::TinyGLTF loader;
  return loader.WriteGltfSceneToFile(&model, path.string(),
                                     /*embed_images=*/false,
                                     /*embed_textures=*/false,
                                     /*embed_buffers=*/false,
                                     /*embed_binary=*/false);
}

bool SaveTexture(const Texture& texture, const std::filesystem::path& path) {
  if (texture.pixel_data.empty() || texture.width == 0 || texture.height == 0) {
    return false;
  }

  // Use stbi_write_png
  int ret = stbi_write_png(path.string().c_str(), texture.width, texture.height,
                           texture.channels, texture.pixel_data.data(),
                           texture.width * texture.channels);
  return (ret != 0);
}

}  // namespace sh_baker
