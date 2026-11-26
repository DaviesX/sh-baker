#ifndef SH_BAKER_SRC_TRACER_H_
#define SH_BAKER_SRC_TRACER_H_

#include <embree4/rtcore.h>

#include <Eigen/Dense>
#include <functional>
#include <random>

#include "occlusion.h"
#include "scene.h"

namespace sh_baker {

class LightTree;

struct TraceConfig {
  TraceConfig(const RTCScene rtc_scene, const Scene& scene,
              const LightTree* light_tree, int max_depth, int num_light_samples,
              std::function<void()> on_direct_hit_sky_fn,
              float firefly_clamp = 0.0f,
              bool indirect_only = false)
      : rtc_scene(rtc_scene),
        scene(scene),
        light_tree(light_tree),
        max_depth(max_depth),
        num_light_samples(num_light_samples),
        on_direct_hit_sky_fn(on_direct_hit_sky_fn),
        firefly_clamp(firefly_clamp),
        indirect_only(indirect_only) {}

  const RTCScene rtc_scene;
  const Scene& scene;
  const LightTree* light_tree;
  const int max_depth;
  const int num_light_samples;
  const std::function<void()> on_direct_hit_sky_fn;
  // Max per-sample indirect luminance. Values above this are scaled down to
  // suppress fireflies. <= 0 disables clamping (unbiased). Safe to enable here
  // because the output is low-frequency 3rd-order diffuse SH.
  const float firefly_clamp;
  // Include sky radiance only after a surface reflection.
  const bool indirect_only;
};

// Computes a Monte Carlo path and return a radiance sample.
// Rendering equation:
// L_o(x, \omega_o) = L_e(x, \omega_o) + \int_{\Omega} f_r(x, \omega_i,
// \omega_o) * L_i(x, \omega_i) * cos(\omega_i) d\omega_i
//
// where L_o is the radiance at the camera, f_r is the BRDF, L_i is the
// radiance from the light, and \omega_i is the direction from the light to the
// surface.
//
// To drastically reduce variance, the scene's lights (punctual, directional
// and emissive area lights) are sampled explicitly at every hit with next
// event estimation (NEE), and the BSDF-sampled continuation ray gathers
// light reflected by other surfaces and sky radiance:
//
// L_o(x, \omega_o) = L_e(x, \omega_o) + \int_{A_e} ...Le(x, x')...dA_e(x') +
// \int_{\Omega} ...L_i(x, \omega_i)...d\omega_i
//
// where Le(x, x') is the radiance from the lights and L_i(x, \omega_i) the
// radiance arriving from other surfaces or, for a ray that escapes the scene,
// from the sky.
//
// The sky is not part of NEE. It is reached only by escape: a ray that hits
// nothing can return radiance from Scene::environment. In indirect_only mode,
// this requires a surface reflection. Passing through alpha coverage does
// not count as a reflection. An environment must therefore not
// contain a sun that a directional light already supplies, or it is counted
// twice.
Eigen::Vector3f Trace(const TraceConfig& config, const Ray& ray, int depth,
                      std::mt19937& rng,
                      Eigen::Vector3f throughput = Eigen::Vector3f::Ones());

}  // namespace sh_baker

#endif  // SH_BAKER_SRC_TRACER_H_
