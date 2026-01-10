#version 410 core

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aTexCoord0;  // Albedo UV
layout(location = 3) in vec2 aTexCoord1;  // Lightmap UV
layout(location = 4) in vec4 aTangent;    // Tangent

out vec3 vWorldPos;
out vec3 vNormal;
out vec3 vTangent;
out vec3 vBitangent;
out vec2 vTexCoord0;
out vec2 vTexCoord1;

uniform mat4 u_MVP;
uniform mat4 u_Model;

void main() {
  gl_Position = u_MVP * vec4(aPosition, 1.0);

  // World Space Computations
  vec4 worldPos = u_Model * vec4(aPosition, 1.0);
  vWorldPos = worldPos.xyz;

  // Transform normal/tangent to world space
  // Note: If non-uniform scale, we need inverse-transpose of u_Model,
  // but assuming uniform scale for now or just rotation/translation.
  mat3 normalMatrix = mat3(u_Model);

  vNormal = normalize(normalMatrix * aNormal);
  vTangent = normalize(normalMatrix * aTangent.xyz);

  // Compute Bitangent (re-orthogonalize just in case, or trust mikkT)
  // MikkTSpace convention: B = cross(N, T) * sigma (tangent.w)
  vBitangent = cross(vNormal, vTangent) * aTangent.w;

  vTexCoord0 = aTexCoord0;
  vTexCoord1 = aTexCoord1;
}
