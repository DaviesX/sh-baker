#include "occlusion.h"

#include <glog/logging.h>

namespace sh_baker {

bool IsOccluded(RTCScene scene, const Ray& ray) {
  if (!scene) {
    LOG(ERROR) << "Invalid RTCScene provided to IsOccluded";
    return false;
  }

  // Embree 4 RTCRay
  // Standard layout: org_x, org_y, org_z, tnear, dir_x, dir_y, dir_z, time,
  // tfar, mask, id, flags But we use the structure provided by headers.

  // We need to use RTCRayHit if we wanted hit info, but for occlusion
  // rtcOccluded1 takes RTCRay. Actually, rtcOccluded1 takes (RTCScene scene,
  // RTCRay* ray, RTCIntersectArguments* args)

  // NOTE: Embree 4 uses a slightly different API structure than 3.
  // We should double check: rtcOccluded1(RTCScene scene, RTCRay* ray,
  // RTCOccludedArguments* args = NULL)

  alignas(16) RTCRay rtc_ray;
  rtc_ray.org_x = ray.origin.x();
  rtc_ray.org_y = ray.origin.y();
  rtc_ray.org_z = ray.origin.z();
  rtc_ray.tnear = ray.tnear;

  rtc_ray.dir_x = ray.direction.x();
  rtc_ray.dir_y = ray.direction.y();
  rtc_ray.dir_z = ray.direction.z();
  rtc_ray.time = 0.0f;

  rtc_ray.tfar = ray.tfar;
  rtc_ray.mask = -1;  // 0xFFFFFFFF
  rtc_ray.id = 0;
  rtc_ray.flags = 0;

  rtcOccluded1(scene, &rtc_ray, nullptr);

  // If tfar is set to -inf (neg_infinity), it hit something.
  // Embree documentation: "The ray.tfar value is set to -inf if a hit was
  // found."
  return (rtc_ray.tfar < 0.0f);
}

}  // namespace sh_baker
