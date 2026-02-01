#ifndef SH_BAKER_SRC_TRACER_H_
#define SH_BAKER_SRC_TRACER_H_

#include <embree4/rtcore.h>

#include <Eigen/Dense>
#include <functional>
#include <random>

#include "scene.h"

namespace sh_baker {

struct TraceConfig {
  TraceConfig(const RTCScene rtc_scene, const Scene& scene, int max_depth,
              int num_light_samples, std::function<void()> on_direct_hit_sky_fn)
      : rtc_scene(rtc_scene),
        scene(scene),
        max_depth(max_depth),
        num_light_samples(num_light_samples),
        on_direct_hit_sky_fn(on_direct_hit_sky_fn) {}

  const RTCScene rtc_scene;
  const Scene& scene;
  const int max_depth;
  const int num_light_samples;
  const std::function<void()> on_direct_hit_sky_fn;
};

Eigen::Vector3f SampleHemisphereUniform(std::mt19937& rng);

// Computes a Monte Carlo path and return a radiance sample.
// Rendering equation:
// L_o(x, \omega_o) = L_e(x, \omega_o) + \int_{\Omega} f_r(x, \omega_i,
// \omega_o) * L_i(x, \omega_i) * cos(\omega_i) d\omega_i
//
// where L_o is the radiance at the camera, f_r is the BRDF, L_i is the
// radiance from the light, and \omega_i is the direction from the light to the
// surface.
//
// To drastically reduce variance, we partition the paths into 2 disjoint sets:
// 1. Primary rays: see the sky/sun directly
// 2. Secondary rays: bounce off a surface
//
// Formally,
// L_o(x, \omega_o) = L_e(x, \omega_o) + \int_{A_e} ...Le(x, x')...dA_e(x') +
// \int_{\Omega \setminus A_e} ...L_i(x, \omega_i)...d\omega_i
//
// where Le(x, x') is the radiance from the light, L_i(x, \omega_i) is the
// radiance from the environment, and \omega_i is the direction from the light
// to the surface.
//
// This is also known as a technique called next event estimation (NEE).
Eigen::Vector3f Trace(const TraceConfig& config, const Eigen::Vector3f& origin,
                      const Eigen::Vector3f& dir, int depth, std::mt19937& rng);

}  // namespace sh_baker

#endif  // SH_BAKER_SRC_TRACER_H_
