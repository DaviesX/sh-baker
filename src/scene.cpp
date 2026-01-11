#include "scene.h"

#include <Eigen/src/Core/Matrix.h>
#include <embree4/rtcore_geometry.h>
#include <glog/logging.h>

#include <iostream>
#include <numeric>

namespace sh_baker {

void BuildEnvironmentCDF(Environment& env) {
  if (env.type != Environment::Type::Texture) return;

  const Texture& tex = env.texture;
  if (tex.width == 0 || tex.height == 0 || tex.pixel_data.empty()) return;

  uint32_t width = tex.width;
  uint32_t height = tex.height;
  int channels = tex.channels;

  env.conditional_cdfs.resize(height);
  env.marginal_cdf.resize(height);

  std::vector<float> marginal_pdf(height);

  for (uint32_t y = 0; y < height; ++y) {
    std::vector<float> conditional_pdf(width);
    float row_sum = 0.0f;

    float v = (y + 0.5f) / height;
    float theta = v * M_PI;
    float sin_theta = std::sin(theta);

    for (uint32_t x = 0; x < width; ++x) {
      // Calculate luminance
      float luminance = 0.0f;
      int idx = (y * width + x) * channels;

      // Handle different channel counts (e.g. 3 for RGB, 4 for RGBA)
      // Assuming data is float (HDR) or uint8?
      // Texture struct uses vector<uint8_t>. So need to handle that.
      // Usually HDR maps are float. If it's uint8_t, we convert.
      // Wait, tinygltf images are usually 8-bit unless float extension is used.
      // But for HDRi we expect float data.
      // The Texture struct definition: std::vector<uint8_t> pixel_data;
      // This implies 8-bit storage. If we use stbi_loadf, it returns floats.
      // We should probably update Texture to support float data or reinterpret
      // cast if we know it's float. But for now, let's assume if channels ==
      // 3/4 and we are doing environment, it might be 8-bit? Actually, standard
      // LDR skyboxes are fine too. IMPORTANT: If we want real HDR support, we
      // need to allow float storage. However, modifying Texture struct to be
      // variant or hold floats is a bigger change. Let's implement generic
      // luminance extraction from uint8 for now, but if the data is actually
      // float bytes (reinterpret), we need to know. Given the task mentions
      // "HDRi" and "tinyexr", it's likely we want float. But Texture struct is
      // fixed to uint8_t. Let's assume for this step we interpret based on
      // context or just process as LDR for now if uint8. If we used
      // tinyexr/stbi_loadf in loader, we would have put float bytes into
      // pixel_data vector? 4 bytes per float.

      // Let's look at `loader.cpp`. It uses `stbi_load` (implied by just
      // include, but actual calls in loader not shown for HDR yet). We will
      // handle float parsing in loader. For now, let's assume we can access
      // float data if we add a flag or checking size? Or just assume LDR for
      // the basic implementation if not specified. Let's check `LoadTexture` in
      // loader.cpp. It copies `img.image` which is `std::vector<unsigned
      // char>`. Generic approach: Compute luminance from whatever is there.

      float r, g, b;
      if (channels >= 3) {
        // If we are to support float textures, we need to know the component
        // type. Since the struct doesn't say, let's assume 8-bit for now as per
        // current codebase. Ideally we would add `component_type` to Texture.
        r = tex.pixel_data[idx + 0] / 255.0f;
        g = tex.pixel_data[idx + 1] / 255.0f;
        b = tex.pixel_data[idx + 2] / 255.0f;
      } else {
        r = g = b = tex.pixel_data[idx] / 255.0f;
      }

      luminance = 0.2126f * r + 0.7152f * g + 0.0722f * b;

      // Apply sin(theta) for equirectangular area correction
      float val = luminance * sin_theta;
      conditional_pdf[x] = val;
      row_sum += val;
    }

    // Build Conditional CDF for this row
    if (row_sum < 1e-9f) {
      // Uniform if black
      for (uint32_t x = 0; x < width; ++x)
        env.conditional_cdfs[y].push_back((float)(x + 1) / width);
    } else {
      env.conditional_cdfs[y].reserve(width);
      float accum = 0.0f;
      for (float val : conditional_pdf) {
        accum += val;
        env.conditional_cdfs[y].push_back(accum / row_sum);
      }
    }

    marginal_pdf[y] = row_sum;
  }

  // Build Marginal CDF
  float total_sum =
      std::accumulate(marginal_pdf.begin(), marginal_pdf.end(), 0.0f);
  if (total_sum < 1e-9f) {
    for (uint32_t y = 0; y < height; ++y)
      env.marginal_cdf[y] = (float)(y + 1) / height;
  } else {
    float accum = 0.0f;
    for (uint32_t y = 0; y < height; ++y) {
      accum += marginal_pdf[y];
      env.marginal_cdf[y] = accum / total_sum;
    }
  }
}

std::vector<Eigen::Vector3f> TransformedVertices(const Geometry& geometry) {
  std::vector<Eigen::Vector3f> vertices;
  vertices.reserve(geometry.vertices.size());
  for (const auto& v : geometry.vertices) {
    vertices.push_back(geometry.transform * v);
  }
  return vertices;
}

std::vector<Eigen::Vector3f> TransformedNormals(const Geometry& geometry) {
  std::vector<Eigen::Vector3f> normals;
  normals.reserve(geometry.normals.size());
  for (const auto& n : geometry.normals) {
    normals.push_back((geometry.transform.rotation() * n).normalized());
  }
  return normals;
}

std::vector<Eigen::Vector4f> TransformedTangents(const Geometry& geometry) {
  std::vector<Eigen::Vector4f> tangents;
  tangents.reserve(geometry.tangents.size());
  for (const auto& t : geometry.tangents) {
    Eigen::Vector3f tangent_vector = t.head<3>();
    float tangent_sign = t.w();
    Eigen::Vector3f transformed_tangent_vector =
        geometry.transform.rotation() * tangent_vector;

    Eigen::Vector4f transformed_tangent;
    transformed_tangent.head<3>() = transformed_tangent_vector.normalized();
    transformed_tangent.w() = tangent_sign;
    tangents.push_back(transformed_tangent);
  }
  return tangents;
}

RTCScene BuildBVH(const Scene& scene, RTCDevice device) {
  if (!device) {
    LOG(ERROR) << "Invalid RTCDevice provided to BuildBVH";
    return nullptr;
  }

  RTCScene rtc_scene = rtcNewScene(device);

  // Set scene build quality (optional, default is usually fine)
  rtcSetSceneBuildQuality(rtc_scene, RTC_BUILD_QUALITY_HIGH);

  for (const auto& geo : scene.geometries) {
    auto vertices = TransformedVertices(geo);

    RTCGeometry rtc_geo = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);

    // Vertices
    // Eigen::Vector3f is 3 floats.
    void* vertex_buffer =
        rtcSetNewGeometryBuffer(rtc_geo, /*type=*/RTC_BUFFER_TYPE_VERTEX,
                                /*slot=*/0,
                                /*format=*/RTC_FORMAT_FLOAT3,
                                /*byteStride=*/sizeof(Eigen::Vector3f),
                                /*itemCount=*/vertices.size());
    memcpy(vertex_buffer, vertices.data(),
           vertices.size() * sizeof(Eigen::Vector3f));

    // Indices
    // indices is vector of uint32_t. Embree expects triangles (3 indices).
    // Stride is 3 * sizeof(uint32_t).
    if (geo.indices.size() % 3 != 0) {
      LOG(WARNING)
          << "Geometry indices count is not a multiple of 3. Truncating.";
    }

    rtcSetSharedGeometryBuffer(rtc_geo, /*type=*/RTC_BUFFER_TYPE_INDEX,
                               /*slot=*/0,
                               /*format=*/RTC_FORMAT_UINT3,
                               /*ptr=*/geo.indices.data(),
                               /*byteOffset=*/0,
                               /*byteStride=*/3 * sizeof(uint32_t),
                               /*itemCount=*/geo.indices.size() / 3);

    rtcSetGeometryUserData(rtc_geo, (void*)&geo);

    rtcAttachGeometry(rtc_scene, rtc_geo);
    rtcCommitGeometry(rtc_geo);
    rtcReleaseGeometry(rtc_geo);
  }

  rtcCommitScene(rtc_scene);

  return rtc_scene;
}

void ReleaseBVH(RTCScene scene) { rtcReleaseScene(scene); }

}  // namespace sh_baker
