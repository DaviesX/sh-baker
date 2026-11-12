#include "tangents.h"

#include <Eigen/Dense>
#include <cmath>
#include <cstdint>

#include "mikktspace.h"

namespace sh_baker {
namespace {

// MikkTSpace Interface
struct MikkTSpaceContext {
  Geometry* geometry;
};

int GetNumFaces(const SMikkTSpaceContext* pContext) {
  auto* ctx = static_cast<MikkTSpaceContext*>(pContext->m_pUserData);
  return static_cast<int>(ctx->geometry->indices.size() / 3);
}

int GetNumVerticesOfFace(const SMikkTSpaceContext* pContext, const int iFace) {
  return 3;
}

void GetPosition(const SMikkTSpaceContext* pContext, float fvPosOut[],
                 const int iFace, const int iVert) {
  auto* ctx = static_cast<MikkTSpaceContext*>(pContext->m_pUserData);
  uint32_t idx = ctx->geometry->indices[iFace * 3 + iVert];
  const auto& v = ctx->geometry->vertices[idx];
  fvPosOut[0] = v.x();
  fvPosOut[1] = v.y();
  fvPosOut[2] = v.z();
}

void GetNormal(const SMikkTSpaceContext* pContext, float fvNormOut[],
               const int iFace, const int iVert) {
  auto* ctx = static_cast<MikkTSpaceContext*>(pContext->m_pUserData);
  uint32_t idx = ctx->geometry->indices[iFace * 3 + iVert];
  const auto& n = ctx->geometry->normals[idx];
  fvNormOut[0] = n.x();
  fvNormOut[1] = n.y();
  fvNormOut[2] = n.z();
}

void GetTexCoord(const SMikkTSpaceContext* pContext, float fvTexcOut[],
                 const int iFace, const int iVert) {
  auto* ctx = static_cast<MikkTSpaceContext*>(pContext->m_pUserData);
  uint32_t idx = ctx->geometry->indices[iFace * 3 + iVert];
  // If no UVs, provide 0,0
  if (ctx->geometry->texture_uvs.empty()) {
    fvTexcOut[0] = 0.0f;
    fvTexcOut[1] = 0.0f;
  } else {
    const auto& uv = ctx->geometry->texture_uvs[idx];
    fvTexcOut[0] = uv.x();
    fvTexcOut[1] = uv.y();
  }
}

void SetTSpaceBasic(const SMikkTSpaceContext* pContext, const float fvTangent[],
                    const float fSign, const int iFace, const int iVert) {
  auto* ctx = static_cast<MikkTSpaceContext*>(pContext->m_pUserData);
  uint32_t idx = ctx->geometry->indices[iFace * 3 + iVert];
  // Since we might share vertices, this simple assignment might overwrite
  // if vertices are shared but have different tangent spaces (UV seams/hard
  // edges). However, tinygltf usually duplicates vertices for split attributes.
  // We assume vertices are already split correctly for UVs/Normals.
  if (idx >= ctx->geometry->tangents.size()) {
    ctx->geometry->tangents.resize(ctx->geometry->vertices.size());
  }
  ctx->geometry->tangents[idx] =
      Eigen::Vector4f(fvTangent[0], fvTangent[1], fvTangent[2], fSign);
}

void GenerateMikkTSpaceTangents(Geometry* geo) {
  if (!geo->tangents.empty()) return;

  // Resize to match vertices, initial zero
  geo->tangents.resize(geo->vertices.size(), Eigen::Vector4f::Zero());

  MikkTSpaceContext ctx;
  ctx.geometry = geo;

  SMikkTSpaceInterface iface = {};
  iface.m_getNumFaces = GetNumFaces;
  iface.m_getNumVerticesOfFace = GetNumVerticesOfFace;
  iface.m_getPosition = GetPosition;
  iface.m_getNormal = GetNormal;
  iface.m_getTexCoord = GetTexCoord;
  iface.m_setTSpaceBasic = SetTSpaceBasic;

  SMikkTSpaceContext context = {};
  context.m_pInterface = &iface;
  context.m_pUserData = &ctx;

  genTangSpaceDefault(&context);
}

}  // namespace

void GenerateTangents(Geometry* geo, bool use_texture_uvs) {
  if (!geo->tangents.empty()) return;

  if (use_texture_uvs) {
    // Has UVs -> Use MikkTSpace
    GenerateMikkTSpaceTangents(geo);
  }

  // If MikkTSpace failed or no UVs, fallback to geometric tangent
  if (geo->tangents.empty()) {
    const size_t vertex_count = geo->vertices.size();
    geo->tangents.resize(vertex_count);
    for (size_t i = 0; i < vertex_count; ++i) {
      const Eigen::Vector3f& n = geo->normals[i];
      // Arbitrary tangent basis
      Eigen::Vector3f t;
      if (std::abs(n.x()) < 0.9f) {
        t = Eigen::Vector3f(1, 0, 0);
      } else {
        t = Eigen::Vector3f(0, 1, 0);
      }
      Eigen::Vector3f tangent = (t - n * n.dot(t)).normalized();
      geo->tangents[i] =
          Eigen::Vector4f(tangent.x(), tangent.y(), tangent.z(), 1.0f);
    }
  }
}

}  // namespace sh_baker
