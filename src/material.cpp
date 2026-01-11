#include "material.h"

#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <random>

#include "colorspace.h"

namespace sh_baker {
namespace {

// Trowbridge-Reitz GGX Normal Distribution Function
float DistributionGGX(const Eigen::Vector3f& N, const Eigen::Vector3f& H,
                      float roughness) {
  float a = roughness * roughness;
  float a2 = a * a;
  float NdotH = std::max(0.0f, N.dot(H));
  float NdotH2 = NdotH * NdotH;

  float nom = a2;
  float denom = (NdotH2 * (a2 - 1.0f) + 1.0f);
  denom = static_cast<float>(M_PI) * denom * denom;

  return nom / std::max(denom, 1e-3f);
}

// Smith's Schlick-GGX Geometry Shadowing Function
float GeometrySchlickGGX(float NdotV, float roughness) {
  float r = (roughness + 1.0f);
  float k = (r * r) / 8.0f;

  float nom = NdotV;
  float denom = NdotV * (1.0f - k) + k;

  return nom / denom;
}

float GeometrySmith(const Eigen::Vector3f& N, const Eigen::Vector3f& V,
                    const Eigen::Vector3f& L, float roughness) {
  float NdotV = std::max(0.0f, N.dot(V));
  float NdotL = std::max(0.0f, N.dot(L));
  float ggx1 = GeometrySchlickGGX(NdotV, roughness);
  float ggx2 = GeometrySchlickGGX(NdotL, roughness);

  return ggx1 * ggx2;
}

// Fresnel Schlick
Eigen::Vector3f FresnelSchlick(float cosTheta, const Eigen::Vector3f& F0) {
  return F0 + (Eigen::Vector3f::Ones() - F0) * std::pow(1.0f - cosTheta, 5.0f);
}

}  // namespace

float GetAlpha(const Material& mat, const Eigen::Vector2f& uv) {
  if (mat.albedo.pixel_data.empty()) return 1.0f;
  if (mat.albedo.channels < 4) return 1.0f;

  int tx = std::clamp((int)(uv.x() * mat.albedo.width), 0,
                      (int)mat.albedo.width - 1);
  int ty = std::clamp((int)(uv.y() * mat.albedo.height), 0,
                      (int)mat.albedo.height - 1);
  int idx = (ty * mat.albedo.width + tx) * mat.albedo.channels;
  return mat.albedo.pixel_data[idx + 3] / 255.0f;
}

Eigen::Vector3f GetAlbedo(const Material& mat, const Eigen::Vector2f& uv) {
  Eigen::Vector3f albedo(0.8f, 0.8f, 0.8f);  // Default gray
  if (!mat.albedo.pixel_data.empty()) {
    int tx = std::clamp((int)(uv.x() * mat.albedo.width), 0,
                        (int)mat.albedo.width - 1);
    int ty = std::clamp((int)(uv.y() * mat.albedo.height), 0,
                        (int)mat.albedo.height - 1);
    int idx = (ty * mat.albedo.width + tx) * mat.albedo.channels;

    // Convert sRGB -> Linear
    float r = SRGBToLinear(mat.albedo.pixel_data[idx]);
    float g = SRGBToLinear(mat.albedo.pixel_data[idx + 1]);
    float b = SRGBToLinear(mat.albedo.pixel_data[idx + 2]);
    albedo = Eigen::Vector3f(r, g, b);
  }
  return albedo;
}

Eigen::Vector3f GetEmission(const Material& mat, const Eigen::Vector2f& uv) {
  if (mat.emission_intensity > 0.0f) {
    return GetAlbedo(mat, uv) * mat.emission_intensity;
  }
  return Eigen::Vector3f::Zero();
}

void GetMetallicRoughness(const Material& mat, const Eigen::Vector2f& uv,
                          float& metallic, float& roughness) {
  CHECK(!mat.metallic_roughness_texture.pixel_data.empty());

  int tx = std::clamp((int)(uv.x() * mat.metallic_roughness_texture.width), 0,
                      (int)mat.metallic_roughness_texture.width - 1);
  int ty = std::clamp((int)(uv.y() * mat.metallic_roughness_texture.height), 0,
                      (int)mat.metallic_roughness_texture.height - 1);
  int idx = (ty * mat.metallic_roughness_texture.width + tx) *
            mat.metallic_roughness_texture.channels;

  // glTF: Metalness in B, Roughness in G.
  float b_metal = mat.metallic_roughness_texture.pixel_data[idx + 2] / 255.0f;
  float g_rough = mat.metallic_roughness_texture.pixel_data[idx + 1] / 255.0f;

  metallic = b_metal;
  roughness = std::max(g_rough, .01f);
}

ReflectionSample SampleMaterial(const Material& mat, const Eigen::Vector2f& uv,
                                const Eigen::Vector3f& normal,
                                const Eigen::Vector3f& reflected,
                                std::mt19937& rng) {
  // Compute basis vectors.
  Eigen::Vector3f t, b;
  if (std::abs(normal.x()) > std::abs(normal.z())) {
    t = Eigen::Vector3f(-normal.y(), normal.x(), 0.0f);
  } else {
    t = Eigen::Vector3f(0.0f, -normal.z(), normal.y());
  }
  t.normalize();
  b = normal.cross(t);

  // Diffuse Sample (Cosine)
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  float r1 = dist(rng);
  float r2 = dist(rng);
  float phi = 2.0f * M_PI * r1;
  float theta = std::acos(std::sqrt(r2));
  float sin_theta = std::sin(theta);

  Eigen::Vector3f local_dir(std::cos(phi) * sin_theta,
                            std::sin(phi) * sin_theta, std::cos(theta));

  // Transform to World
  Eigen::Vector3f world_dir =
      t * local_dir.x() + b * local_dir.y() + normal * local_dir.z();

  ReflectionSample sample;
  sample.direction = world_dir.normalized();
  sample.pdf = local_dir.z() / M_PI;
  return sample;
}

Eigen::Vector3f EvalMaterial(const Material& mat, const Eigen::Vector2f& uv,
                             const Eigen::Vector3f& normal,
                             const Eigen::Vector3f& incident,
                             const Eigen::Vector3f& reflected) {
  Eigen::Vector3f V = reflected;
  Eigen::Vector3f L = incident;
  Eigen::Vector3f N = normal;

  if (N.dot(V) <= 0.0f || N.dot(L) <= 0.0f) {
    return Eigen::Vector3f::Zero();
  }

  Eigen::Vector3f albedo = GetAlbedo(mat, uv);
  return albedo / M_PI;
}

ReflectionSample SampleMaterialAdvanced(const Material& mat,
                                        const Eigen::Vector2f& uv,
                                        const Eigen::Vector3f& normal,
                                        const Eigen::Vector3f& reflected,
                                        std::mt19937& rng) {
  // Compute basis vectors.
  Eigen::Vector3f t, b;
  if (std::abs(normal.x()) > std::abs(normal.z())) {
    t = Eigen::Vector3f(-normal.y(), normal.x(), 0.0f);
  } else {
    t = Eigen::Vector3f(0.0f, -normal.z(), normal.y());
  }
  t.normalize();
  b = normal.cross(t);

  float metallic, roughness;
  GetMetallicRoughness(mat, uv, metallic, roughness);

  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  float r0 = dist(rng);

  // F0 mix
  Eigen::Vector3f albedo = GetAlbedo(mat, uv);
  Eigen::Vector3f F0 = Eigen::Vector3f(0.04f, 0.04f, 0.04f);
  F0 = F0 * (1.0f - metallic) + albedo * metallic;

  float spec_prob = metallic;
  if (r0 < spec_prob) {
    // Specular Sample (GGX)
    float r1 = dist(rng);
    float r2 = dist(rng);

    float a = roughness * roughness;
    float phi = 2.0f * M_PI * r1;
    float cos_theta = std::sqrt((1.0f - r2) / (1.0f + (a * a - 1.0f) * r2));
    float sin_theta = std::sqrt(1.0f - cos_theta * cos_theta);

    Eigen::Vector3f H_local(sin_theta * std::cos(phi),
                            sin_theta * std::sin(phi), cos_theta);

    // Transform H to World
    Eigen::Vector3f H =
        (t * H_local.x() + b * H_local.y() + normal * H_local.z()).normalized();

    // L = 2(V.H)H - V
    Eigen::Vector3f V = reflected;
    Eigen::Vector3f L = (2.0f * V.dot(H) * H - V).normalized();

    if (L.dot(normal) <= 0.0f) {
      // Internal reflection. No sample is generated.
      return ReflectionSample();
    }

    ReflectionSample sample;
    sample.direction = L;

    float D = DistributionGGX(normal, H, roughness);
    float NdotH = std::max(0.0f, normal.dot(H));
    float HdotV = std::max(0.0f, H.dot(V));
    float pdf_desc = (D * NdotH) / (4.0f * HdotV + 0.0001f);

    sample.pdf = pdf_desc * spec_prob;
    return sample;
  } else {
    // Diffuse Sample (Cosine)
    float r1 = dist(rng);
    float r2 = dist(rng);
    float phi = 2.0f * M_PI * r1;
    float theta = std::acos(std::sqrt(r2));
    float sin_theta = std::sin(theta);

    Eigen::Vector3f local_dir(std::cos(phi) * sin_theta,
                              std::sin(phi) * sin_theta, std::cos(theta));

    // Transform to World
    Eigen::Vector3f world_dir =
        t * local_dir.x() + b * local_dir.y() + normal * local_dir.z();

    ReflectionSample sample;
    sample.direction = world_dir.normalized();
    sample.pdf = (local_dir.z() / M_PI) * (1.0f - spec_prob);
    return sample;
  }
}

Eigen::Vector3f EvalMaterialAdvanced(const Material& mat,
                                     const Eigen::Vector2f& uv,
                                     const Eigen::Vector3f& normal,
                                     const Eigen::Vector3f& incident,
                                     const Eigen::Vector3f& reflected) {
  Eigen::Vector3f V = reflected;
  Eigen::Vector3f L = incident;
  Eigen::Vector3f N = normal;

  if (N.dot(V) <= 0.0f || N.dot(L) <= 0.0f) {
    return Eigen::Vector3f::Zero();
  }

  float metallic, roughness;
  GetMetallicRoughness(mat, uv, metallic, roughness);
  Eigen::Vector3f albedo = GetAlbedo(mat, uv);

  Eigen::Vector3f H = (V + L).normalized();

  float D = DistributionGGX(N, H, roughness);
  float G = GeometrySmith(N, V, L, roughness);

  Eigen::Vector3f F0 = Eigen::Vector3f(0.04f, 0.04f, 0.04f);
  F0 = F0 * (1.0f - metallic) + albedo * metallic;

  Eigen::Vector3f F = FresnelSchlick(std::max(0.0f, H.dot(V)), F0);

  Eigen::Vector3f numerator = D * G * F;
  float denominator =
      4.0f * std::max(0.0f, N.dot(V)) * std::max(0.0f, N.dot(L)) + 0.001f;
  Eigen::Vector3f specular = numerator / denominator;

  Eigen::Vector3f kS = F;
  Eigen::Vector3f kD = Eigen::Vector3f::Ones() - kS;
  kD *= (1.0f - metallic);

  Eigen::Vector3f diffuse = kD.cwiseProduct(albedo) / M_PI;
  return diffuse + specular;
}

}  // namespace sh_baker
