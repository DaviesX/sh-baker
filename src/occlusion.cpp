#include "occlusion.h"

#include <glog/logging.h>

#include "material.h"
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
    occ.t = rtc_ray_hit.ray.tfar;
    occ.position = ray.origin + ray.direction * occ.t;

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

        // Interpolate Tangent
        CHECK(!geom->tangents.empty());
        Eigen::Vector4f t0 = geom->tangents[idx0];
        Eigen::Vector4f t1 = geom->tangents[idx1];
        Eigen::Vector4f t2 = geom->tangents[idx2];
        Eigen::Vector4f interpolated_tan = w * t0 + u * t1 + v * t2;

        Eigen::Vector3f t = interpolated_tan.head<3>();
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

float FindTransmittance(RTCScene scene, const std::vector<Material>& materials,
                        const Ray& ray) {
  // Step past each hit along the same ray (origin, direction and tfar kept),
  // by the tracer's transmission offset.
  constexpr float kStep = 0.005f;

  Ray walk = ray;
  float transmittance = 1.0f;
  for (int crossings = 0;; ++crossings) {
    std::optional<Occlusion> occ = FindOcclusion(scene, walk);
    if (!occ.has_value()) return transmittance;
    if (occ->material_id < 0) return 0.0f;
    CHECK_LT(static_cast<size_t>(occ->material_id), materials.size());
    float alpha = GetAlpha(materials[occ->material_id], occ->uv);
    if (alpha >= 1.0f) return 0.0f;
    if (crossings >= kMaxTransmittanceCrossings) return 0.0f;
    transmittance *= 1.0f - alpha;
    walk.tnear = occ->t + kStep;
    if (walk.tnear >= walk.tfar) return transmittance;
  }
}

}  // namespace sh_baker
