#include "scene.h"

#include <glog/logging.h>

#include <iostream>

namespace sh_baker {

RTCScene BuildBVH(const Scene& scene, RTCDevice device) {
  if (!device) {
    LOG(ERROR) << "Invalid RTCDevice provided to BuildBVH";
    return nullptr;
  }

  RTCScene rtc_scene = rtcNewScene(device);

  // Set scene build quality (optional, default is usually fine)
  rtcSetSceneBuildQuality(rtc_scene, RTC_BUILD_QUALITY_HIGH);

  for (auto& geo : scene.geometries) {
    RTCGeometry rtc_geo = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);

    // Vertices
    // Eigen::Vector3f is 3 floats.
    rtcSetSharedGeometryBuffer(rtc_geo, RTC_BUFFER_TYPE_VERTEX,
                               0,  // slot
                               RTC_FORMAT_FLOAT3, geo.vertices.data(),
                               0,                        // byte offset
                               sizeof(Eigen::Vector3f),  // stride
                               geo.vertices.size());

    // Indices
    // indices is vector of uint32_t. Embree expects triangles (3 indices).
    // Stride is 3 * sizeof(uint32_t).
    if (geo.indices.size() % 3 != 0) {
      LOG(WARNING)
          << "Geometry indices count is not a multiple of 3. Truncating.";
    }

    rtcSetSharedGeometryBuffer(
        rtc_geo, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, geo.indices.data(),
        0,
        3 * sizeof(uint32_t),  // Stride for one triangle (3 indices)
        geo.indices.size() / 3);

    rtcSetGeometryUserData(rtc_geo, (void*)&geo);

    rtcAttachGeometry(rtc_scene, rtc_geo);
    rtcCommitGeometry(rtc_geo);

    // rtcAttachGeometry increments the ref count, so we can release our local
    // handle? documentation says: "The application can release the geometry
    // handle after attaching it..."
    rtcReleaseGeometry(rtc_geo);
  }

  rtcCommitScene(rtc_scene);

  return rtc_scene;
}

}  // namespace sh_baker
