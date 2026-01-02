#include "occlusion.h"

#include <glog/logging.h>

#include "scene.h"

namespace sh_baker {

std::optional<Occlusion> FindOcclusion(RTCScene scene, const Ray& ray) {
  if (!scene) {
    LOG(ERROR) << "Invalid RTCScene provided to FindOcclusion";
    return std::nullopt;
  }

  alignas(16) RTCRayHit rtc_ray_hit;
  rtc_ray_hit.ray.org_x = ray.origin.x();
  rtc_ray_hit.ray.org_y = ray.origin.y();
  rtc_ray_hit.ray.org_z = ray.origin.z();
  rtc_ray_hit.ray.tnear = ray.tnear;

  rtc_ray_hit.ray.dir_x = ray.direction.x();
  rtc_ray_hit.ray.dir_y = ray.direction.y();
  rtc_ray_hit.ray.dir_z = ray.direction.z();
  rtc_ray_hit.ray.time = 0.0f;

  rtc_ray_hit.ray.tfar = ray.tfar;
  rtc_ray_hit.ray.mask = -1;
  rtc_ray_hit.ray.id = 0;
  rtc_ray_hit.ray.flags = 0;

  rtc_ray_hit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
  rtc_ray_hit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;

  RTCIntersectArguments args;
  rtcInitIntersectArguments(&args);

  rtcIntersect1(scene, &rtc_ray_hit, &args);

  if (rtc_ray_hit.hit.geomID != RTC_INVALID_GEOMETRY_ID) {
    Occlusion occ;

    // Position: origin + dir * t
    occ.position = ray.origin + ray.direction * rtc_ray_hit.ray.tfar;

    // Retrieve Geometry
    RTCGeometry geometry_handle = rtcGetGeometry(scene, rtc_ray_hit.hit.geomID);
    Geometry* geom = (Geometry*)rtcGetGeometryUserData(geometry_handle);

    if (geom) {
      occ.material_id = geom->material_id;

      // Barycentric coordinates
      float u = rtc_ray_hit.hit.u;
      float v = rtc_ray_hit.hit.v;
      float w = 1.0f - u - v;

      uint32_t primID = rtc_ray_hit.hit.primID;
      // Each triangle has 3 indices
      uint32_t idx0 = geom->indices[primID * 3 + 0];
      uint32_t idx1 = geom->indices[primID * 3 + 1];
      uint32_t idx2 = geom->indices[primID * 3 + 2];

      // Interpolate Normals
      if (!geom->normals.empty()) {
        Eigen::Vector3f n0 = geom->normals[idx0];
        Eigen::Vector3f n1 = geom->normals[idx1];
        Eigen::Vector3f n2 = geom->normals[idx2];
        occ.normal = (w * n0 + u * n1 + v * n2).normalized();
      } else {
        // Fallback if no vertex normals (use geometric normal from hit)
        occ.normal = Eigen::Vector3f(rtc_ray_hit.hit.Ng_x, rtc_ray_hit.hit.Ng_y,
                                     rtc_ray_hit.hit.Ng_z)
                         .normalized();
      }

      // Interpolate UVs
      if (!geom->texture_uvs.empty()) {
        Eigen::Vector2f uv0 = geom->texture_uvs[idx0];
        Eigen::Vector2f uv1 = geom->texture_uvs[idx1];
        Eigen::Vector2f uv2 = geom->texture_uvs[idx2];
        occ.uv = w * uv0 + u * uv1 + v * uv2;
      } else {
        occ.uv = Eigen::Vector2f::Zero();
      }
    } else {
      // Fallback if no user data (shouldn't happen with our loader)
      occ.material_id = -1;
      occ.normal = Eigen::Vector3f(rtc_ray_hit.hit.Ng_x, rtc_ray_hit.hit.Ng_y,
                                   rtc_ray_hit.hit.Ng_z)
                       .normalized();
      occ.uv = Eigen::Vector2f::Zero();
    }

    return occ;
  }

  return std::nullopt;
}

}  // namespace sh_baker
