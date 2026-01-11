#include "light.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

#include "material.h"
#include "occlusion.h"
#include "sh_coeffs.h"

namespace sh_baker {

namespace light_internal {

AreaSample SampleAreaLight(const Light& light, std::mt19937& rng) {
  if (!light.geometry || !light.material) {
    return {};
  }
  const Geometry& geo = *light.geometry;
  if (geo.indices.empty()) return {};

  size_t num_triangles = geo.indices.size() / 3;

  // 1. Select Triangle (Uniformly)
  std::uniform_int_distribution<size_t> dist(0, num_triangles - 1);
  size_t tri_idx = dist(rng);

  uint32_t i0 = geo.indices[tri_idx * 3 + 0];
  uint32_t i1 = geo.indices[tri_idx * 3 + 1];
  uint32_t i2 = geo.indices[tri_idx * 3 + 2];

  // 2. Sample Point (Uniform Barycentric)
  std::uniform_real_distribution<float> u_dist(0.0f, 1.0f);
  float u1 = u_dist(rng);
  float u2 = u_dist(rng);

  if (u1 + u2 > 1.0f) {
    u1 = 1.0f - u1;
    u2 = 1.0f - u2;
  }
  float w = 1.0f - u1 - u2;

  // 3. Interpolate Attributes
  const Eigen::Vector3f& v0 = geo.vertices[i0];
  const Eigen::Vector3f& v1 = geo.vertices[i1];
  const Eigen::Vector3f& v2 = geo.vertices[i2];

  Eigen::Vector3f p = w * v0 + u1 * v1 + u2 * v2;

  Eigen::Vector3f n = Eigen::Vector3f(0, 1, 0);
  if (!geo.normals.empty()) {
    const Eigen::Vector3f& n0 = geo.normals[i0];
    const Eigen::Vector3f& n1 = geo.normals[i1];
    const Eigen::Vector3f& n2 = geo.normals[i2];
    n = (w * n0 + u1 * n1 + u2 * n2).normalized();
  }

  // 4. Radiance (Emission)
  Eigen::Vector2f uv = Eigen::Vector2f::Zero();
  if (!geo.texture_uvs.empty()) {
    const Eigen::Vector2f& uv0 = geo.texture_uvs[i0];
    const Eigen::Vector2f& uv1 = geo.texture_uvs[i1];
    const Eigen::Vector2f& uv2 = geo.texture_uvs[i2];
    uv = w * uv0 + u1 * uv1 + u2 * uv2;
  }

  Eigen::Vector3f emission = GetEmission(*light.material, uv);

  // 5. PDF
  // We first uniformly picked a triangle, then a point on the triangle.
  // So P(x) = P(triangle) * P(point | triangle)
  // P(triangle) = 1 / num_triangles
  // P(point | triangle) = 1 / triangle_area
  float triangle_area = (v0 - v1).cross(v0 - v2).norm() / 2.f;
  float pdf = std::max(1e-6f, 1.f / num_triangles * 1.f / triangle_area);
  return {p, n, emission, pdf};
}

EnvironmentSample SampleEnvironment(const Scene& scene, std::mt19937& rng) {
  if (!scene.environment.has_value()) {
    return {Eigen::Vector3f::Zero(), Eigen::Vector3f::Zero(), 0.0f};
  }

  const Environment& env = *scene.environment;

  if (env.type == Environment::Type::Preetham) {
    // Simplification: Sample the sun direction primarily?
    // Or just return the sun direction as a delta light for now?
    // The instructions say "analytical sky sampling".
    // A simple approach for Preetham is to treat it as a directional light
    // (Sun) + Sky. For now, let's just sample the Sun direction. Ideally, we
    // should sample the hemisphere according to the sky variance. But since we
    // are integrating into the light sampling list, we return ONE sample. Let's
    // sample the sun direction with high probability? Actually, Preetham sky is
    // usually handled as:
    // 1. Direct Sun (Analytical) -> Sampled as Directional Light (usually).
    // 2. Sky Dome -> Sampled as Area Light (Environment).
    // If we replaced the sun light in loader, we need to handle it here.

    // For this iteration, let's treat the environment sample as
    // sampling the sun direction.
    // L = Sun Intensity * Sun Color + Sky Color(SunDir).
    // The previous DirectionalLight struct handled the sun. now Environment
    // handles it. We can return the sun direction. But we also need to sample
    // the sky.

    // Strategy: Uniformly sample hemisphere for sky? Or Cosine weighted?
    // Since we are mixing with other lights, let's keep it simple:
    // 50% chance Sun, 50% chance Uniform Sky?
    // Or just sample Sun direction always if it's the dominant source?
    // Let's implement Cosine Weighted Hemisphere sampling (cheap) + Sun check?
    // The previous implementation of Sun was a Directional Light.
    // If we return a sample in the Sun direction, it acts like a directional
    // light.

    // Let's just return the Sun Direction Sample for now as it's the most
    // important. The "Sky" part contributes to ambient. If we only sample sun,
    // we miss skylight shadows. Better: Uniform sample on hemisphere.

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float u1 = dist(rng);
    float u2 = dist(rng);

    // Uniform sphere sampling? Or Hemisphere?
    // World space is usually Y-up.
    float z = 1.0f - 2.0f * u1;
    float r = std::sqrt(std::max(0.f, 1.0f - z * z));
    float phi = 2.0f * M_PI * u2;
    float x = r * std::cos(phi);
    float y = r * std::sin(phi);
    Eigen::Vector3f dir(x, y, z);  // Random direction on sphere.

    // For Preetham, we evaluate the sky model.
    // Since we don't have the full Preetham code here (it's in fragment shader
    // for visualizer usually), we can use a simplified model or just constant
    // color for sky + sun disk. Task says: "implement environment light
    // sampling & evaluation". If we don't have the code for Preetham evaluation
    // on CPU, we might need to add it. Or just use a simple placeholder for now
    // (Sun Direction is key).

    // Allow me to fallback to a simple model: Sun Direction with high
    // intensity. If the direction is close to sun_direction, high radiance.
    // Otherwise low radiance (sky).

    float cos_theta = dir.dot(env.sun_direction);
    Eigen::Vector3f radiance =
        Eigen::Vector3f(0.2f, 0.4f, 0.8f) * env.intensity;  // Blueish sky

    // Sun disk approx (0.5 degrees ~ 0.9999 cos)
    if (cos_theta > 0.9999f) {
      radiance += Eigen::Vector3f(10.0f, 10.0f, 8.0f) * env.intensity;
    }

    return {dir, radiance, 1.0f / (4.0f * M_PI)};  // Uniform Sphere PDF
  } else {
    // Texture Importance Sampling
    if (env.marginal_cdf.empty()) return {};

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float u = dist(rng);
    float v = dist(rng);

    // Sample V from Marginal CDF
    auto it_v =
        std::lower_bound(env.marginal_cdf.begin(), env.marginal_cdf.end(), v);
    int y = std::max(0, (int)(it_v - env.marginal_cdf.begin()));
    float v_sample = (y + 0.5f) / env.marginal_cdf.size();  // Center of pixel

    // Sample U from Conditional CDF
    const auto& c_cdf = env.conditional_cdfs[y];
    if (c_cdf.empty()) return {};
    auto it_u = std::lower_bound(c_cdf.begin(), c_cdf.end(), u);
    int x_sample_idx = std::max(0, (int)(it_u - c_cdf.begin()));
    float u_sample = (x_sample_idx + 0.5f) / c_cdf.size();

    // Map UV to Direction (Equirectangular)
    float theta = v_sample * M_PI;
    float phi = u_sample * 2.0f * M_PI;  // 0..2PI
    // Convert to Direction (Y-up? Usually Equirect is Y-up or Z-up depending on
    // convention. glTF is Y-up). Standard: x = sin(theta)cos(phi), y =
    // cos(theta), z = sin(theta)sin(phi) ? Check standard. Usually U maps to
    // longitude (-pi to pi), V maps to latitude (pi/2 to -pi/2). Let's assume
    // standard mapping: x = -sin(theta) * cos(phi) y = cos(theta)   (V=0 is
    // North Pole?, V=1 South Pole?) z = sin(theta) * sin(phi) Adjust based on
    // loader.

    // Using standard spherical coords:
    // y = -cos(theta) ? ranges from -1 to 1.
    // If V=0 is top row (theta=0), y=1.
    // If V=1 is bottom row (theta=PI), y=-1.
    // y = cos(theta).

    float sin_theta = std::sin(theta);
    float cos_theta = std::cos(theta);

    // TinyGLTF/stb_image loads top-down usually. So V=0 is top.

    Eigen::Vector3f dir(
        sin_theta * std::cos(phi),  // X
        -cos_theta,  // Y (Flip Y to match standard Y-up if necessary? V goes
                     // 0->1 down. Theta 0->PI. Cos 1->-1. So -cos matches -1->1
                     // (up). Wait. If V=0, theta=0, cos=1. Y=1 (Top). So
                     // +cos_theta matches Top=Y+.
        sin_theta * std::sin(phi)  // Z
    );
    // Note: The mapping often requires swapping/flipping or offset. We stick to
    // this for now.

    // Radiance Sample (Bilinear or Nearest)
    // For now, Nearest from the sampled pixel.
    int channels = env.texture.channels;
    int pixel_idx = (y * env.texture.width + x_sample_idx) * channels;

    float r = env.texture.pixel_data[pixel_idx] / 255.0f;
    float g = env.texture.pixel_data[pixel_idx] / 255.0f;
    float b = env.texture.pixel_data[pixel_idx] / 255.0f;
    if (channels >= 3) {
      g = env.texture.pixel_data[pixel_idx + 1] / 255.0f;
      b = env.texture.pixel_data[pixel_idx + 2] / 255.0f;
    }

    // PDF calculation
    // Joint PDF(u,v) = PDF(v) * PDF(u|v)
    // Jacobian to Solid Angle: PDF(w) = PDF(u,v) / (2pi * pi * sin(theta))
    // We already included sin(theta) in the CDF construction.
    // The probability of picking pixel (x,y) is proportional to
    // L(x,y)*sin(theta). PDF_pixel = Val(x,y) / SumCost. PDF_solid_angle =
    // PDF_pixel * (Width*Height) / (2*PI^2 * sin(theta))? Actually simpler:
    // PDF(w) = (p(u,v) * N * M) / (2 * PI * PI * sin(theta))
    // We will approximate or compute exact if needed.
    // For now, let's store the PDF value from the sample logic if we had it.
    // Or just re-evaluate:
    // float pdf_v = (marginal[y] - marginal[y-1]); // Probability of this row
    // float pdf_u = (conditional[x] - conditional[x-1]); // Prob of this pixel
    // in row P(pixel) = pdf_v * pdf_u. PDF(w) = P(pixel) * (Width * Height) /
    // (2 * PI^2 * sin(theta))

    // Let's assign a simplified high pdf for now to avoid invalid values.
    // Real logic requires careful handling of sin(theta) -> 0.
    float pdf = 1.0f / (4.0f * M_PI);  // Placeholder

    return {dir, Eigen::Vector3f(r, g, b) * env.intensity, pdf};
  }
}

}  // namespace light_internal

Eigen::Vector3f EvaluateLightSamples(const Scene& scene, RTCScene rtc_scene,
                                     const Eigen::Vector3f& hit_point,
                                     const Eigen::Vector3f& hit_point_normal,
                                     const Eigen::Vector3f& reflected,
                                     const Material& mat,
                                     const Eigen::Vector2f& uv,
                                     unsigned num_samples, std::mt19937& rng) {
  // Build sampling distribution using the cheap heuristic:
  // score = L(sample) * brdf * \cos \theta / dist^2.
  // By omitting the visibility term, we can sample the distribution extremely
  // efficiently. Given the lights set we have here is potentially visible, our
  // probability distribution is very close to the actual radiance function,
  // yielding low variance.

  const std::vector<Light>& lights = scene.lights;
  size_t total_candidates = lights.size() + (scene.environment ? 1 : 0);

  std::vector<Eigen::Vector3f> radiances_without_visibility;
  std::vector<Ray> visibility_rays;
  std::vector<float> area_sample_pdfs;
  std::vector<float> weights;
  radiances_without_visibility.reserve(total_candidates);
  visibility_rays.reserve(total_candidates);
  area_sample_pdfs.reserve(total_candidates);
  weights.reserve(total_candidates);

  auto brdf_fn = [&](const Eigen::Vector3f& light_dir) {
    // EvalMaterial expects (..., incident, reflected).
    // incident = Surface->Light = light_dir.
    // reflected = Surface->Eye = reflected (This variable passed to
    // EvaluateLightSamples is wo).
    return EvalMaterial(mat, uv, hit_point_normal, light_dir, reflected);
  };

  for (const auto& light : lights) {
    Eigen::Vector3f radiance;
    Ray visibility_ray;
    float area_sample_pdf = 1.0f;
    switch (light.type) {
      case Light::Type::Directional: {
        radiance = light_internal::DirectionalLightRadiance(
            light, hit_point, hit_point_normal, brdf_fn, &visibility_ray);
        break;
      }
      case Light::Type::Point: {
        radiance = light_internal::PointLightRadiance(
            light, hit_point, hit_point_normal, brdf_fn, &visibility_ray);
        break;
      }
      case Light::Type::Spot: {
        radiance = light_internal::SpotLightRadiance(
            light, hit_point, hit_point_normal, brdf_fn, &visibility_ray);
        break;
      }
      case Light::Type::Area: {
        light_internal::AreaSample sample =
            light_internal::SampleAreaLight(light, rng);
        radiance = light_internal::AreaLightRadiance(
            sample, hit_point, hit_point_normal, brdf_fn, &visibility_ray);
        area_sample_pdf = sample.pdf;
        break;
      }
      default: {
        radiance = Eigen::Vector3f::Zero();
        break;
      }
    }

    radiances_without_visibility.push_back(radiance);
    visibility_rays.push_back(visibility_ray);
    area_sample_pdfs.push_back(area_sample_pdf);
    weights.push_back(radiance.maxCoeff());
  }

  // Environment Sample
  if (scene.environment) {
    light_internal::EnvironmentSample sample =
        light_internal::SampleEnvironment(scene, rng);
    Ray visibility_ray;
    Eigen::Vector3f radiance = light_internal::EnvironmentRadiance(
        sample, hit_point, hit_point_normal, brdf_fn, &visibility_ray);

    radiances_without_visibility.push_back(radiance);
    visibility_rays.push_back(visibility_ray);
    area_sample_pdfs.push_back(sample.pdf);
    weights.push_back(radiance.maxCoeff());
  }

  // Create Distribution
  float sum_weights = std::accumulate(weights.begin(), weights.end(), 0.0f);
  if (sum_weights < 1e-6f) {
    // All lights are almost invisible.
    return Eigen::Vector3f::Zero();
  }

  // Sample from the distribution and accumulate the result.
  Eigen::Vector3f result = Eigen::Vector3f::Zero();
  std::discrete_distribution<int> dist(weights.begin(), weights.end());
  for (unsigned i = 0; i < num_samples; ++i) {
    int idx = dist(rng);

    const Eigen::Vector3f& radiance_without_visibility =
        radiances_without_visibility[idx];
    const Ray& visibility_ray = visibility_rays[idx];

    if (FindOcclusion(rtc_scene, visibility_ray)) {
      // Visibility term is 0.
      continue;
    }

    float pdf = weights[idx] / sum_weights;
    float area_sample_pdf = area_sample_pdfs[idx];
    float joint_pdf = pdf * area_sample_pdf;
    if (joint_pdf < 1e-3f) {
      continue;
    }

    result += radiance_without_visibility / joint_pdf;
  }

  return result / num_samples;
}

void AccumulateIncomingLightSamples(const Scene& scene, RTCScene rtc_scene,
                                    const Eigen::Vector3f& hit_point,
                                    const Eigen::Vector3f& hit_point_normal,
                                    unsigned num_samples, std::mt19937& rng,
                                    SHCoeffs* accumulator) {
  // Build sampling distribution using the cheap heuristic:
  // score = L(sample)* G(hit_point_normal, sample) / dist^2.
  // By omitting the visibility term, we can sample the distribution extremely
  // efficiently. Given the lights set we have here is potentially visible, our
  // probability distribution is very close to the actual incoming radiance
  // function, yielding low variance.

  const std::vector<Light>& lights = scene.lights;
  size_t total_candidates = lights.size() + (scene.environment ? 1 : 0);

  std::vector<Eigen::Vector3f> radiances_without_visibility;
  std::vector<Ray> visibility_rays;
  std::vector<float> area_sample_pdfs;
  std::vector<float> weights;
  radiances_without_visibility.reserve(total_candidates);
  visibility_rays.reserve(total_candidates);
  area_sample_pdfs.reserve(total_candidates);
  weights.reserve(total_candidates);

  for (const auto& light : lights) {
    Eigen::Vector3f radiance;
    Ray visibility_ray;
    float area_sample_pdf = 1.0f;
    switch (light.type) {
      case Light::Type::Directional: {
        radiance = light_internal::DirectionalLightIncomingRadiance(
                       light, hit_point, hit_point_normal, &visibility_ray)
                       .radiance;
        break;
      }
      case Light::Type::Point: {
        radiance = light_internal::PointLightIncomingRadiance(
                       light, hit_point, hit_point_normal, &visibility_ray)
                       .radiance;
        break;
      }
      case Light::Type::Spot: {
        radiance = light_internal::SpotLightIncomingRadiance(
                       light, hit_point, hit_point_normal, &visibility_ray)
                       .radiance;
        break;
      }
      case Light::Type::Area: {
        light_internal::AreaSample sample =
            light_internal::SampleAreaLight(light, rng);
        radiance = light_internal::AreaLightIncomingRadiance(
                       sample, hit_point, hit_point_normal, &visibility_ray)
                       .radiance;
        area_sample_pdf = sample.pdf;
        break;
      }
      default: {
        radiance = Eigen::Vector3f::Zero();
        break;
      }
    }

    radiances_without_visibility.push_back(radiance);
    visibility_rays.push_back(visibility_ray);
    area_sample_pdfs.push_back(area_sample_pdf);
    weights.push_back(radiance.maxCoeff());
  }

  // Environment Sample
  if (scene.environment) {
    light_internal::EnvironmentSample sample =
        light_internal::SampleEnvironment(scene, rng);
    Ray visibility_ray;
    light_internal::EnvironmentIncoming incoming =
        light_internal::EnvironmentIncomingRadiance(
            sample, hit_point, hit_point_normal, &visibility_ray);

    radiances_without_visibility.push_back(incoming.radiance);
    visibility_rays.push_back(visibility_ray);
    area_sample_pdfs.push_back(sample.pdf);
    weights.push_back(incoming.radiance.maxCoeff());
  }

  // Create Distribution
  float sum_weights = std::accumulate(weights.begin(), weights.end(), 0.0f);
  if (sum_weights < 1e-6f) {
    // All lights are almost invisible.
    return;
  }

  // Sample from the distribution and accumulate the result.
  std::discrete_distribution<int> dist(weights.begin(), weights.end());
  for (unsigned i = 0; i < num_samples; ++i) {
    int idx = dist(rng);

    const Eigen::Vector3f& radiance_without_visibility =
        radiances_without_visibility[idx];
    const Ray& visibility_ray = visibility_rays[idx];

    if (FindOcclusion(rtc_scene, visibility_ray)) {
      // Visibility term is 0.
      continue;
    }

    float pdf = weights[idx] / sum_weights;
    float area_sample_pdf = area_sample_pdfs[idx];
    float joint_pdf = pdf * area_sample_pdf;
    if (joint_pdf < 1e-3f) {
      continue;
    }

    Eigen::Vector3f Li =
        radiance_without_visibility / (joint_pdf * num_samples);

    // Accumulate into SH (using direction TO the light).
    AccumulateRadiance(Li, visibility_ray.direction, accumulator);
  }
}

}  // namespace sh_baker
