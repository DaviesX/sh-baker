#include "baker.h"

#include <embree4/rtcore.h>
#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <random>

#include "light.h"
#include "material.h"
#include "occlusion.h"
#include "rasterizer.h"

namespace sh_baker {

namespace {

Eigen::Vector3f SampleHemisphereUniform(std::mt19937& rng) {
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  float u1 = dist(rng);
  float u2 = dist(rng);

  float r = std::sqrt(1.0f - u1 * u1);
  float phi = 2.0f * M_PI * u2;
  return Eigen::Vector3f(r * std::cos(phi), r * std::sin(phi), u1);
}

// Trace path using Occlusion helper
Eigen::Vector3f Trace(RTCScene rtc_scene, const Scene& scene,
                      const Eigen::Vector3f& origin, const Eigen::Vector3f& dir,
                      int depth, int max_depth, int num_light_samples,
                      std::mt19937& rng) {
  if (depth > max_depth) return Eigen::Vector3f::Zero();

  Ray ray;
  ray.origin = origin;
  ray.direction = dir;
  ray.tnear = 0.001f;

  std::optional<Occlusion> occ = FindOcclusion(rtc_scene, ray);

  if (!occ.has_value()) {
    // Sky
    float sun = std::max(0.0f, dir.dot(scene.sky.sun_direction));
    Eigen::Vector3f sky_radiance =
        scene.sky.sun_color * scene.sky.sun_intensity * sun +
        Eigen::Vector3f(0.05f, 0.05f, 0.05f);  // ambient

    if (depth == 0) {
      // Primary estimator rays must see the light sources directly
      return sky_radiance;
    } else {
      // Secondary rays (Indirect):
      // Sun is handled by NEE (EvaluateLights).
      // Ambient is handled here.
      return Eigen::Vector3f(0.05f, 0.05f, 0.05f);
    }
  }

  // Hit surface
  const Material& mat = scene.materials[occ->material_id];
  float alpha = GetAlpha(mat, occ->uv);

  Eigen::Vector3f color = Eigen::Vector3f::Zero();

  // If alpha < 1.0, continue ray
  if (alpha < 1.0f) {
    // Transmission
    Eigen::Vector3f hit_pos = occ->position + dir * 0.001f;
    Eigen::Vector3f transmission = Trace(rtc_scene, scene, hit_pos, dir, depth,
                                         max_depth, num_light_samples, rng);

    color += (1.0f - alpha) * transmission;
  }

  if (alpha == 0.0f) {
    return color;
  }

  // Evaluate surface
  // Emission (L_e)
  // If we hit an emissive surface, and we are using NEE, we usually ignore it
  // to avoid double counting, unless depth == 0 (primary hit? No, we bake
  // covariance from points). The rays called by BakeSHLightMap index loop are
  // "Camera rays" effectively (gathering L). NO. The rays in loop are "Sampling
  // rays" accumulating to SH. Essentially `Trace` computes Incoming Radiance
  // L_i. The loop computes: integral L_i * basis * ... So `Trace` returns
  // L(origin, -dir).

  // Actually, standard path tracer:
  // L = Le + integral ...
  // If we use NEE, we split integral into Direct + Indirect.
  // Direct samples lights.
  // Indirect samples BSDF.
  // If Indirect Ray hits Light, we behave as if it's black (to avoid double
  // count).

  // So:
  // 1. Emission from THIS surface is L_e. (Only if depth == 0? Or always?)
  // If we are looking at L_i reaching point P.
  // Trace(P, dir) finds point Q.
  // L_i = L_o(Q, -dir).
  // L_o(Q) = Le(Q) + ...
  // If Le(Q) is sampled by Light Sampling at P, then we ignore it here.
  // Light Sampling at P samples "Lights".
  // If Q is on a Light, we ignore Le(Q).

  // Check if material is emissive
  Eigen::Vector3f emission = GetEmission(mat, occ->uv);
  if (!emission.isZero()) {
    // If depth == 0, we see the emissive surface directly (Le).
    // If depth > 0, we are bouncing. In NEE, we sample direct light explicitly.
    // So we should NOT accumulate Le from random bounces to avoid double
    // counting.
    if (depth == 0) {
      color += alpha * emission;
    }
    // Else ignore (return 0 contribution from emission)
  }

  // Direct Lighting (NEE)
  {
    Eigen::Vector3f hit_normal = occ->normal;
    Eigen::Vector3f hit_pos = occ->position + hit_normal * 0.001f;
    // View direction is -dir
    Eigen::Vector3f wo = -dir;

    // 1. Sample Lights
    auto samples =
        SampleLights(scene, hit_pos, hit_normal, num_light_samples, rng);
    Eigen::Vector3f L_direct =
        EvaluateLights(scene.sky, samples, hit_pos, hit_normal, rtc_scene);

    // L_direct is the incoming radiance * geometric terms?
    // EvaluateLights returns: Sum ( Li * cos_theta / pdf )
    // We need to multiply by BRDF.
    // And EvaluateLights computes Li at hit_pos.

    // Wait, EvaluateLights returns "total_radiance += Li * (cos_theta /
    // sample.pdf);" It DOES include cos_theta. It does NOT include BRDF. We
    // need to pass BRDF or multiply here. Since EvaluateLights iterates lights
    // with different directions, we can't just multiply sum by BRDF if BRDF
    // depends on direction (it does).

    // EvaluateLights needs to know the BRDF or return the directions?
    // "EvaluateLights ... handles occlusion ... divides by PDF".
    // Usually it returns just L_i?
    // If EvaluateLights returns Sum(L_i * cos / pdf), that's Irradiance (if L_i
    // const).

    // Let's look at EvaluateLights implementation again.
    // "total_radiance += Li * (cos_theta / sample.pdf);"
    // It accumulates Irradiance-like terms.

    // Correct NEE: Sum ( Li * f_r * cos * V / pdf )
    // I should modify EvaluateLights to take the BSDF/Material or do the loop
    // here. The user said: "EvaluateLights ... Internally, it triages the
    // evaluation based on the light type." And "Please document what it does.
    // It should handle occlusion and dividing the PDFs."

    // If I cannot change EvaluateLights easily now (it's in light.cpp), I
    // should have made it return the components. But wait, the user provided
    // signature: EvaluateLights(..., hit_point, hit_point_normal, ...) ->
    // Vector3f If it doesn't take 'wo' (view dir) or Material, it calculates
    // Irradiance (assuming Lambertian 1.0?).

    // Assumption: The Baker is currently diffuse dominant or `EvaluateLights`
    // is determining Irradiance? But `Trace` returns Radiance (Color). If
    // `EvaluateLights` sums (Li * cos / pdf), that is basically Exitant
    // Radiance IF BRDF = 1/pi ? No. If BRDF is Diffuse (Albedo / pi), then Lo =
    // Albedo * (Sum Li cos / pdf).

    // Current `EvaluateLights` (Step 67) does:
    // total_radiance += Li * (cos_theta / sample.pdf);
    // It uses `hit_point_normal`.
    // This is effectively calculating Irradiance E.
    // For Lambertian: L_direct = (Albedo / PI) * E.

    // Let's assume Lambertian for Direct Light for now (since we don't have
    // BRDF in EvaluateLights). Or we should assume EvaluateLights returns the
    // "Un-albedo-ed" contribution? Yes.

    // So:
    // L_direct_out = (Albedo / PI) * EvaluateLights(...)
    // Note: 1/PI factor. Does current code use 1/PI?
    // `EvalMaterial` usually returns Albedo/PI.
    // In old Trace: incoming.cwiseProduct(brdf) * (cosine / pdf)
    // brdf = EvalMaterial -> Albedo/Pi.

    // So color += (Albedo / Pi) * EvaluateLights(...).
    // Note: `EvaluateLights` sums (Li * cos / pdf).
    // So yes.

    Eigen::Vector3f irradiance =
        EvaluateLights(scene.sky, samples, hit_pos, hit_normal, rtc_scene);
    Eigen::Vector3f albedo = GetAlbedo(mat, occ->uv);

    // Lambertian BRDF: f_r = rho / pi
    color += alpha * albedo.cwiseProduct(irradiance) * (1.0f / M_PI);
  }

  // Indirect Lighting (Recursive)
  {
    Eigen::Vector3f hit_normal = occ->normal;
    Eigen::Vector3f hit_pos = occ->position + hit_normal * 0.001f;
    ReflectionSample sample =
        SampleMaterial(mat, occ->uv, hit_normal, dir, rng);

    Eigen::Vector3f incoming =
        Trace(rtc_scene, scene, hit_pos, sample.direction, depth + 1, max_depth,
              num_light_samples, rng);

    Eigen::Vector3f brdf =
        EvalMaterial(mat, occ->uv, hit_normal, dir, sample.direction);

    float cosine_term = std::max(0.0f, hit_normal.dot(sample.direction));

    if (sample.pdf > 1e-6f) {
      color += alpha * incoming.cwiseProduct(brdf) * (cosine_term / sample.pdf);
    }
  }

  return color;
}

}  // namespace

SHTexture BakeSHLightMap(const Scene& scene,
                         const std::vector<SurfacePoint>& surface_points,
                         const RasterConfig& raster_config,
                         const BakeConfig& config) {
  SHTexture output;
  output.width = raster_config.width;
  output.height = raster_config.height;
  output.pixels.resize(raster_config.width * raster_config.height);

  RTCDevice device = rtcNewDevice(nullptr);
  RTCScene rtc_scene = BuildBVH(scene, device);

  // Bake
  std::mt19937 rng(12345);

  float inv_pdf_uniform = 2.0f * M_PI;

  for (int idx = 0; idx < surface_points.size(); ++idx) {
    const SurfacePoint& sp = surface_points[idx];
    if (!sp.valid) continue;

    SHCoeffs sh_accum;

    // Offset position to avoid self-intersection
    Eigen::Vector3f origin = sp.position + sp.normal * 0.001f;

    for (int s = 0; s < config.samples; ++s) {
      Eigen::Vector3f dir_local = SampleHemisphereUniform(rng);  // Z is up

      // Transform to World
      Eigen::Vector3f dir_world = sp.tangent * dir_local.x() +
                                  sp.bitangent * dir_local.y() +
                                  sp.normal * dir_local.z();

      Eigen::Vector3f Li = Trace(rtc_scene, scene, origin, dir_world, 0,
                                 config.bounces, config.num_light_samples, rng);

      AccumulateRadiance(Li * inv_pdf_uniform, dir_world, &sh_accum);
    }

    // Average
    output.pixels[idx] = sh_accum * (1.0f / config.samples);
  }

  // Clean up
  rtcReleaseScene(rtc_scene);
  rtcReleaseDevice(device);

  return output;
}

}  // namespace sh_baker
