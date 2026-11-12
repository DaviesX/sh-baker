#include "scene.h"

#include <embree4/rtcore_geometry.h>
#include <glog/logging.h>
#include <mikktspace.h>

#include <Eigen/Dense>
#include <cmath>
#include <cstdint>

#include "colorspace.h"
#include "sh_coeffs.h"

namespace sh_baker {
namespace {

constexpr float kPi = 3.14159265359f;
constexpr float kTwoPi = 6.28318530718f;

// Coefficients for a standard "Clear Sky" (Turbidity ~2.5)
const float perez_A = -1.0f;   // Darkening of horizon
const float perez_B = -0.32f;  // Luminance gradient near horizon
const float perez_C = 10.0f;   // Relative intensity of circumsolar region
const float perez_D = -3.0f;   // Width of circumsolar region
const float perez_E = 0.45f;   // Relative backscatter

// MikkTSpace Interface
struct MikkTSpaceUserData {
  const Geometry* geometry;
  std::vector<Eigen::Vector4f>* tangents_out;
};

int GetNumFaces(const SMikkTSpaceContext* pContext) {
  auto* data = static_cast<MikkTSpaceUserData*>(pContext->m_pUserData);
  return static_cast<int>(data->geometry->indices.size() / 3);
}

int GetNumVerticesOfFace(const SMikkTSpaceContext* ctx, const int iFace) {
  return 3;
}

void GetPosition(const SMikkTSpaceContext* pContext, float fvPosOut[],
                 const int iFace, const int iVert) {
  auto* data = static_cast<MikkTSpaceUserData*>(pContext->m_pUserData);
  uint32_t idx = data->geometry->indices[iFace * 3 + iVert];
  const auto& v = data->geometry->vertices[idx];
  fvPosOut[0] = v.x();
  fvPosOut[1] = v.y();
  fvPosOut[2] = v.z();
}

void GetNormal(const SMikkTSpaceContext* pContext, float fvNormOut[],
               const int iFace, const int iVert) {
  auto* data = static_cast<MikkTSpaceUserData*>(pContext->m_pUserData);
  uint32_t idx = data->geometry->indices[iFace * 3 + iVert];
  const auto& n = data->geometry->normals[idx];
  fvNormOut[0] = n.x();
  fvNormOut[1] = n.y();
  fvNormOut[2] = n.z();
}

void GetTexCoord(const SMikkTSpaceContext* pContext, float fvTexcOut[],
                 const int iFace, const int iVert) {
  auto* data = static_cast<MikkTSpaceUserData*>(pContext->m_pUserData);
  uint32_t idx = data->geometry->indices[iFace * 3 + iVert];

  CHECK(!data->geometry->texture_uvs.empty())
      << "MikkTSpace requires texture UVs.";
  const auto& uv = data->geometry->texture_uvs[idx];
  fvTexcOut[0] = uv.x();
  fvTexcOut[1] = uv.y();
}

void SetTSpaceBasic(const SMikkTSpaceContext* pContext, const float fvTangent[],
                    const float fSign, const int iFace, const int iVert) {
  auto* data = static_cast<MikkTSpaceUserData*>(pContext->m_pUserData);
  uint32_t idx = data->geometry->indices[iFace * 3 + iVert];
  // Since we might share vertices, this simple assignment might overwrite
  // if vertices are shared but have different tangent spaces (UV seams/hard
  // edges). However, tinygltf usually duplicates vertices for split attributes.
  // We assume vertices are already split correctly for UVs/Normals.
  if (idx >= data->tangents_out->size()) {
    data->tangents_out->resize(data->geometry->vertices.size());
  }
  (*data->tangents_out)[idx] =
      Eigen::Vector4f(fvTangent[0], fvTangent[1], fvTangent[2], fSign);
}

std::vector<Eigen::Vector4f> GenerateMikkTSpaceTangents(const Geometry& geo) {
  if (!geo.tangents.empty()) {
    return geo.tangents;
  }

  if (geo.texture_uvs.empty()) {
    // Can't generate without UVs.
    return {};
  }

  // Resize to match vertices, initial zero
  std::vector<Eigen::Vector4f> result(geo.vertices.size());
  MikkTSpaceUserData data;
  data.geometry = &geo;
  data.tangents_out = &result;

  SMikkTSpaceInterface iface = {};
  iface.m_getNumFaces = GetNumFaces;
  iface.m_getNumVerticesOfFace = GetNumVerticesOfFace;
  iface.m_getPosition = GetPosition;
  iface.m_getNormal = GetNormal;
  iface.m_getTexCoord = GetTexCoord;
  iface.m_setTSpaceBasic = SetTSpaceBasic;

  SMikkTSpaceContext context = {};
  context.m_pInterface = &iface;
  context.m_pUserData = &data;

  if (!genTangSpaceDefault(&context)) {
    return {};
  }

  return result;
}

Eigen::Vector3f GetPixel(const Texture32F& tex, int x, int y) {
  int idx = (y * tex.width + x) * tex.channels;
  float r = tex.pixel_data[idx + 0];
  float g = tex.pixel_data[idx + 1];
  float b = tex.pixel_data[idx + 2];
  return Eigen::Vector3f(r, g, b);
}

// Projects an Equirectangular (LatLong) map to SH
SHCoeffs ProjectLatLongMap(const Texture32F& tex) {
  SHCoeffs coeffs;
  int w = tex.width;
  int h = tex.height;

  // Solid angle of a pixel (approx constant per row y)
  // d_omega = (2pi / W) * (pi / H) * sin(theta)
  float d_phi = kTwoPi / w;
  float d_theta = kPi / h;

  Eigen::Vector3f up(0.0f, 1.0f, 0.0f);

  for (int y = 0; y < h; ++y) {
    // Theta from 0 (top) to pi (bottom)
    float theta = (y + 0.5f) * d_theta;
    float sin_theta = std::sin(theta);
    float d_omega = d_phi * d_theta * sin_theta;

    // Optimization: if sin_theta is near 0 (poles), weight is near 0.
    if (sin_theta < 1e-4f) continue;

    for (int x = 0; x < w; ++x) {
      // Phi from 0 to 2pi
      // Standard Equirectangular: Center (0.5) is usually +Z or -Z depending on
      // convention. Here assuming u=0 -> +Z (phi=0), u=0.25 -> +X. Note: glTF
      // usually assumes +Z is front.
      float phi = (x + 0.5f) * d_phi;

      // Convert Spherical to Cartesian (Y-up)
      // y = cos(theta)
      // x = sin(theta) * sin(phi)
      // z = sin(theta) * cos(phi)
      Eigen::Vector3f dir(std::sin(theta) * std::sin(phi), std::cos(theta),
                          std::sin(theta) * std::cos(phi));

      Eigen::Vector3f radiance = GetPixel(tex, x, y);

      AccumulateRadiance(radiance * d_omega, dir, up, &coeffs);
    }
  }
  return coeffs;
}

// Corrected Preetham / Perez Sky Evaluation
// Evaluates the Perez function for a given view direction
float EvaluatePerez(const Eigen::Vector3f& dir, const Eigen::Vector3f& sunDir) {
  // 1. Zenith Angle (gamma): Angle between View and Up (Y-axis)
  float cosThetaZenith = std::max(0.001f, dir.y());  // Clamp to horizon

  // 2. Sun Angle (zeta): Angle between View and Sun
  float cosThetaSun = dir.dot(sunDir);
  // Clamp for acos safety
  float gamma = std::acos(std::clamp(cosThetaSun, -1.0f, 1.0f));

  // Perez Function: F(theta, gamma) = (1 + A*exp(B/cos_theta)) * (1 +
  // C*exp(D*gamma) + E*cos^2(gamma))
  float f_zenith = (1.0f + perez_A * std::exp(perez_B / cosThetaZenith));
  float f_sun = (1.0f + perez_C * std::exp(perez_D * gamma) +
                 perez_E * cosThetaSun * cosThetaSun);

  return f_zenith * f_sun;
}

SHCoeffs ProjectPreethamToSH(const Environment& env) {
  SHCoeffs coeffs;

  // Integration resolution (higher = more accurate)
  const int steps_theta = 64;  // Latitude (0 to PI)
  const int steps_phi = 128;   // Longitude (0 to 2PI)

  float d_theta = kPi / steps_theta;
  float d_phi = kTwoPi / steps_phi;

  Eigen::Vector3f sun_dir = env.sun_direction.normalized();
  Eigen::Vector3f up(0.0f, 1.0f, 0.0f);

  // Zenith luminance for normalization
  float zenith_val = EvaluatePerez(up, sun_dir);
  if (zenith_val <= 1e-6f) zenith_val = 1.0f;

  for (int t = 0; t < steps_theta; ++t) {
    float theta = (t + 0.5f) * d_theta;  // 0 (North Pole/Up) to PI
    float sin_theta = std::sin(theta);

    for (int p = 0; p < steps_phi; ++p) {
      float phi = (p + 0.5f) * d_phi;

      // 1. Convert Spherical to Cartesian (Y-up)
      // Note: ProjectLatLongMap convention was:
      // y = cos(theta)
      // x = sin(theta) * sin(phi)
      // z = sin(theta) * cos(phi)
      // This makes theta=0 -> y=1 (Up). Consistent with Preetham.
      float y = std::cos(theta);
      float x = sin_theta * std::sin(phi);
      float z = sin_theta * std::cos(phi);
      Eigen::Vector3f dir(x, y, z);

      // Handle Horizon clipping (ground is black)
      if (y < 0.0f) continue;

      // 2. Evaluate Radiance using Preetham function
      float raw_luminance = EvaluatePerez(dir, sun_dir);

      // Apply the cap (e.g., 30x zenith) to prevent ringing from sun disk
      float cap = zenith_val * 30.0f;
      float luminance = std::min(raw_luminance, cap);

      // Convert luminance to RGB (Simplified: Blue sky color scaled by
      // luminance)
      Eigen::Vector3f radiance =
          Eigen::Vector3f(0.2f, 0.5f, 0.9f) * (luminance / zenith_val);

      // 3. Differential solid angle (dA = sin(theta) * dTheta * dPhi)
      float differential_solid_angle = sin_theta * d_theta * d_phi;

      // 4. Accumulate
      AccumulateRadiance(radiance * differential_solid_angle, dir, up, &coeffs);
    }
  }

  return coeffs;
}

float ComputeCDF(const std::vector<float>& weights, std::vector<float>* cdf) {
  cdf->resize(weights.size() + 1);
  (*cdf)[0] = 0.0f;
  for (size_t i = 0; i < weights.size(); ++i) {
    (*cdf)[i + 1] = (*cdf)[i] + weights[i];
  }
  float total_weight = (*cdf).back();
  if (total_weight > 1e-3f) {
    for (float& cdf_val : *cdf) {
      cdf_val /= total_weight;
    }
  } else {
    for (float& cdf_val : *cdf) {
      cdf_val = 0.0f;
    }
  }
  return total_weight;
}

}  // namespace

SHCoeffs ProjectEnvironmentToSH(const Environment& env) {
  if (env.type == Environment::Type::Texture) {
    return ProjectLatLongMap(env.texture);
  } else {
    return ProjectPreethamToSH(env);
  }
}

void LogSceneStats(const Scene& scene) {
  size_t shell_count = 0;
  for (const auto& geo : scene.geometries) {
    if (geo.material_id < 0) ++shell_count;
  }
  LOG(INFO) << "  Geometries: " << scene.geometries.size();
  LOG(INFO) << "  Materials: " << scene.materials.size();
  LOG(INFO) << "  Lights: " << scene.lights.size();
  LOG(INFO) << "  Occluder shells (material-less): " << shell_count;
}

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

std::vector<Eigen::Vector4f> ComputeTangents(const Geometry& geo) {
  std::vector<Eigen::Vector4f> mkts_result = GenerateMikkTSpaceTangents(geo);
  if (!mkts_result.empty()) {
    return mkts_result;
  }

  // Fallbacks to geometric tangents.
  std::vector<Eigen::Vector4f> result;
  const size_t vertex_count = geo.vertices.size();
  result.reserve(vertex_count);

  for (size_t i = 0; i < vertex_count; ++i) {
    const Eigen::Vector3f& n = geo.normals[i];
    // Arbitrary tangent basis
    Eigen::Vector3f t;
    if (std::abs(n.x()) < 0.9f) {
      t = Eigen::Vector3f(1, 0, 0);
    } else {
      t = Eigen::Vector3f(0, 1, 0);
    }
    Eigen::Vector3f tangent = (t - n * n.dot(t)).normalized();
    result.emplace_back(tangent.x(), tangent.y(), tangent.z(), 1.0f);
  }

  return result;
}

float SurfaceArea(const Geometry& geometry) {
  const auto& vertices = TransformedVertices(geometry);
  float total_area = 0.0f;
  for (size_t i = 0; i < geometry.indices.size(); i += 3) {
    Eigen::Vector3f v0 = vertices[geometry.indices[i]];
    Eigen::Vector3f v1 = vertices[geometry.indices[i + 1]];
    Eigen::Vector3f v2 = vertices[geometry.indices[i + 2]];
    float tri_area = 0.5f * (v1 - v0).cross(v2 - v0).norm();
    total_area += tri_area;
  }
  return total_area;
}

std::optional<EmissionCDF> ComputeTextureEmissionCDF(
    const Texture& texture, const Texture32F& uv_to_world_area_ratio) {
  if (texture.pixel_data.empty() || texture.width == 0 || texture.height == 0) {
    return std::nullopt;
  }

  int w = texture.width;
  int h = texture.height;

  // Step 1: Compute weighted luminance for each texel.
  std::vector<std::vector<float>> weights;
  weights.resize(h);
  for (int y = 0; y < h; ++y) {
    weights[y].resize(w);
    for (int x = 0; x < w; ++x) {
      int tex_idx = (y * w + x) * texture.channels;
      float r = SRGBToLinear(texture.pixel_data[tex_idx + 0]);
      float g = SRGBToLinear(texture.pixel_data[tex_idx + 1]);
      float b = SRGBToLinear(texture.pixel_data[tex_idx + 2]);
      float lum = Luminance(r, g, b);

      // Weight by Jacobian (UV-to-world area ratio).
      float jacobian = 0.0f;
      if (uv_to_world_area_ratio.width > 0) {
        int jx = std::clamp((int)(float(x) / w * uv_to_world_area_ratio.width),
                            0, (int)uv_to_world_area_ratio.width - 1);
        int jy = std::clamp((int)(float(y) / h * uv_to_world_area_ratio.height),
                            0, (int)uv_to_world_area_ratio.height - 1);
        jacobian = uv_to_world_area_ratio
                       .pixel_data[jy * uv_to_world_area_ratio.width + jx];
      }

      weights[y][x] = lum * jacobian;
    }
  }

  // Step 2: Build CDF using float types.
  EmissionCDF result;
  result.conditional_cdf.resize(h + 1);
  result.marginal_cdf.resize(h + 1);

  result.conditional_cdf[0].resize(w + 1);
  for (int y = 1; y <= h; ++y) {
    result.marginal_cdf[y] =
        result.marginal_cdf[y - 1] +
        ComputeCDF(weights[y - 1], &result.conditional_cdf[y]);
  }

  float total_marginal_sum = result.marginal_cdf.back();
  if (total_marginal_sum > 1e-3f) {
    for (float& marginal_cdf : result.marginal_cdf) {
      marginal_cdf /= total_marginal_sum;
    }
  } else {
    for (float& marginal_cdf : result.marginal_cdf) {
      marginal_cdf = 0.0f;
    }
  }

  return result;
}

std::pair<Eigen::Vector2f, float> SampleEmissionCDF(const EmissionCDF& cdf,
                                                    float u, float v) {
  // 1. Sample row v from marginal.
  auto marginal_it =
      std::upper_bound(cdf.marginal_cdf.begin(), cdf.marginal_cdf.end(), v);
  int v_idx = std::distance(cdf.marginal_cdf.begin(), marginal_it);
  v_idx = std::clamp(v_idx, 1, (int)cdf.marginal_cdf.size() - 1);
  float marginal_pdf = cdf.marginal_cdf[v_idx] - cdf.marginal_cdf[v_idx - 1];

  float row_total = cdf.conditional_cdf[v_idx].back();
  if (row_total == 0) {
    return {{0.f, 0.f}, 0.f};
  }

  // 2. Sample column u from conditional.
  auto row_it = std::upper_bound(cdf.conditional_cdf[v_idx].begin(),
                                 cdf.conditional_cdf[v_idx].end(), u);
  int u_idx = std::distance(cdf.conditional_cdf[v_idx].begin(), row_it);
  u_idx = std::clamp(u_idx, 1, (int)cdf.conditional_cdf[v_idx].size() - 1);
  float conditional_pdf =
      cdf.conditional_cdf[v_idx][u_idx] - cdf.conditional_cdf[v_idx][u_idx - 1];

  // Final PDF is the product of the marginal and conditional PDFs.
  float pdf = conditional_pdf * marginal_pdf;

  int w = cdf.conditional_cdf[v_idx].size() - 1;
  int h = cdf.marginal_cdf.size() - 1;

  return {{(u_idx - .5f) / w, (v_idx - .5f) / h}, pdf};
}

float Flux(const Light& light) {
  switch (light.type) {
    case Light::Type::Point: {
      return light.intensity * 4.0f * kPi;
    }
    case Light::Type::Spot: {
      return light.intensity * 2.0f * kPi * (1.0f - light.cos_outer_cone);
    }
    case Light::Type::Directional: {
      return light.intensity * kPi;
    }
    case Light::Type::Area: {
      const Material* mat = light.material;
      CHECK_NOTNULL(mat);

      // No emissive texture: flux = emissive_strength * max(emissive_factor) *
      // area.
      if (!mat->emissive_texture || mat->emissive_texture->pixel_data.empty()) {
        float max_factor = mat->emissive_factor.maxCoeff();
        return mat->emissive_strength * max_factor * light.area;
      }

      // With emissive texture: integrate luminance * Jacobian over texels.
      // Single-sided heuristic for light tree.
      const Texture& tex = *mat->emissive_texture;
      int w = tex.width;
      int h = tex.height;

      float max_emissive_factor = mat->emissive_factor.maxCoeff();
      float total_flux = 0.0f;

      for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
          int idx = (y * w + x) * tex.channels;
          float max_coeff = 0.f;
          for (int c = 0; c < tex.channels; ++c) {
            max_coeff =
                std::max(max_coeff, SRGBToLinear(tex.pixel_data[idx + c]));
          }

          // Get Jacobian if available.
          CHECK(light.uv_to_world_area_ratio);
          const auto& ratio = *light.uv_to_world_area_ratio;
          int jx = std::clamp((int)(float(x) / w * ratio.width), 0,
                              (int)ratio.width - 1);
          int jy = std::clamp((int)(float(y) / h * ratio.height), 0,
                              (int)ratio.height - 1);
          float jacobian = ratio.pixel_data[jy * ratio.width + jx];

          total_flux += max_coeff * max_emissive_factor * jacobian;
        }
      }

      return total_flux * mat->emissive_strength;
    }
    default:
      return 0.0f;
  }
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
    // Additive emitters (flames/glows) are transparent to light: keep them out
    // of the occluder scene so they cast no shadow. Their emission reaches the
    // bake through the area-light path (NEE) instead.
    if (geo.material_id >= 0 &&
        static_cast<size_t>(geo.material_id) < scene.materials.size() &&
        scene.materials[geo.material_id].additive) {
      continue;
    }

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
