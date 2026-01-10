#include "scene.h"

#include <Eigen/src/Core/Matrix.h>
#include <embree4/rtcore_geometry.h>
#include <glog/logging.h>

namespace sh_baker {

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
