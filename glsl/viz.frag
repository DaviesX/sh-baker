#version 410 core

in vec3 vNormal;
in vec2 vTexCoord0;
in vec2 vTexCoord1;

out vec4 FragColor;

uniform sampler2D u_AlbedoTex;
uniform sampler2D u_NormalTex;
uniform sampler2D u_MRTex;  // Metallic (B), Roughness (G)

uniform vec3 u_CamPos;

// --- Options ---
uniform int u_UsePackedLuminance;
uniform bool u_ShowDirectional;

// --- SH Textures (Standard) ---
uniform sampler2D u_L0;
uniform sampler2D u_L1m1;
uniform sampler2D u_L10;
uniform sampler2D u_L11;
uniform sampler2D u_L2m2;
uniform sampler2D u_L2m1;
uniform sampler2D u_L20;
uniform sampler2D u_L21;
uniform sampler2D u_L22;

// --- SH Textures (Packed) ---
uniform sampler2D u_PackedTex0;
uniform sampler2D u_PackedTex1;
uniform sampler2D u_PackedTex2;
uniform sampler2D u_IrradianceTex;

// -- SH Sky ---
uniform vec3 u_SkySH[9];

// --- Attributes ---
in vec3 vWorldPos;
in vec4 vTangent;

// -- Helper: Evaluate SH Basis ---
// --- Helper: Evaluate SH Basis (Radiance) ---
// Reconstructs L(n) - The incident radiance from direction n
vec3 EvalSHRadiance(vec3 normal, vec3 sh_coeffs[9]) {
  float x = normal.x;
  float y = normal.y;
  float z = normal.z;

  float c1 = 0.282095;
  float c2 = 0.488603;
  float c3 = 1.092548;
  float c4 = 0.315392;
  float c5 = 0.546274;

  float b0 = c1;
  float b1 = c2 * y;
  float b2 = c2 * z;
  float b3 = c2 * x;
  float b4 = c3 * x * y;
  float b5 = c3 * y * z;
  float b6 = c4 * (3.0 * z * z - 1.0);
  float b7 = c3 * x * z;
  float b8 = c5 * (x * x - y * y);

  vec3 result = sh_coeffs[0] * b0 + sh_coeffs[1] * b1 + sh_coeffs[2] * b2 +
                sh_coeffs[3] * b3 + sh_coeffs[4] * b4 + sh_coeffs[5] * b5 +
                sh_coeffs[6] * b6 + sh_coeffs[7] * b7 + sh_coeffs[8] * b8;

  return max(result, 0.0);
}

// --- Helper: Evaluate SH Irradiance ---
// Reconstructs E(n) - The irradiance (cosine-weighted integral) at normal n
// Applies Cosine Lobe Convolution factors: A0=PI, A1=2PI/3, A2=PI/4
vec3 EvalSHIrradiance(vec3 normal, vec3 sh_coeffs[9]) {
  float x = normal.x;
  float y = normal.y;
  float z = normal.z;

  // Constants pre-multiplied by A_l convolution factors
  // Band 0: 0.282095 * 3.141593 = 0.886227
  float c1 = 0.886227;

  // Band 1: 0.488603 * 2.094395 = 1.023326
  float c2 = 1.023326;

  // Band 2: 1.092548 * 0.785398 = 0.858086
  // Band 2: 0.315392 * 0.785398 = 0.247708
  // Band 2: 0.546274 * 0.785398 = 0.429043
  float c3 = 0.858086;
  float c4 = 0.247708;
  float c5 = 0.429043;

  float b0 = c1;
  float b1 = c2 * y;
  float b2 = c2 * z;
  float b3 = c2 * x;
  float b4 = c3 * x * y;
  float b5 = c3 * y * z;
  float b6 = c4 * (3.0 * z * z - 1.0);
  float b7 = c3 * x * z;
  float b8 = c5 * (x * x - y * y);

  vec3 result = sh_coeffs[0] * b0 + sh_coeffs[1] * b1 + sh_coeffs[2] * b2 +
                sh_coeffs[3] * b3 + sh_coeffs[4] * b4 + sh_coeffs[5] * b5 +
                sh_coeffs[6] * b6 + sh_coeffs[7] * b7 + sh_coeffs[8] * b8;

  return max(result, 0.0);
}

// --- Helper: Fetch SH Coefficients ---
// Returns coefficients and visibility from textures
void GetSHCoeffs(vec2 uv, out vec3 sh_coeffs[9], out float visibility) {
  visibility = 1.0;

  if (u_UsePackedLuminance == 1) {
    vec4 p0 = texture(u_PackedTex0, uv);
    vec4 p1 = texture(u_PackedTex1, uv);
    vec4 p2 = texture(u_PackedTex2, uv);

    sh_coeffs[0] = p0.rgb;
    visibility = p0.a;

    // Chroma reconstruction for higher bands
    float L0_lum = dot(sh_coeffs[0], vec3(0.2126, 0.7152, 0.0722));
    vec3 chroma = vec3(1.0);
    if (L0_lum > 1e-6) {
      chroma = sh_coeffs[0] / L0_lum;
    }

    // File 1: L1m1, L10, L11, L2m2
    sh_coeffs[1] = vec3(p1.r) * chroma;
    sh_coeffs[2] = vec3(p1.g) * chroma;
    sh_coeffs[3] = vec3(p1.b) * chroma;
    sh_coeffs[4] = vec3(p1.a) * chroma;

    // File 2: L2m1, L20, L21, L22
    sh_coeffs[5] = vec3(p2.r) * chroma;
    sh_coeffs[6] = vec3(p2.g) * chroma;
    sh_coeffs[7] = vec3(p2.b) * chroma;
    sh_coeffs[8] = vec3(p2.a) * chroma;
  } else {
    vec4 l0 = texture(u_L0, uv);
    sh_coeffs[0] = l0.rgb;
    visibility = l0.a;
    sh_coeffs[1] = texture(u_L1m1, uv).rgb;
    sh_coeffs[2] = texture(u_L10, uv).rgb;
    sh_coeffs[3] = texture(u_L11, uv).rgb;
    sh_coeffs[4] = texture(u_L2m2, uv).rgb;
    sh_coeffs[5] = texture(u_L2m1, uv).rgb;
    sh_coeffs[6] = texture(u_L20, uv).rgb;
    sh_coeffs[7] = texture(u_L21, uv).rgb;
    sh_coeffs[8] = texture(u_L22, uv).rgb;
  }
}

// --- Fresnel ---
vec3 FresnelSchlick(float cosTheta, vec3 F0) {
  return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

void main() {
  // 1. PBR Parameters
  vec4 albedo_sample = texture(u_AlbedoTex, vTexCoord0);
  if (albedo_sample.a < 0.1) {
    discard;
  }

  vec3 albedo = albedo_sample.rgb;

  // Normal
  vec3 N = normalize(vNormal);
  vec3 T = normalize(vTangent.xyz - N * dot(vTangent.xyz, N));
  vec3 B = cross(N, T) * (vTangent.w > 0.0 ? 1.0 : -1.0);
  vec3 mapNormal = texture(u_NormalTex, vTexCoord0).rgb;
  mapNormal = mapNormal * 2.0 - 1.0;
  mat3 TBN = mat3(T, B, N);
  N = normalize(TBN * mapNormal);

  // Metallic/Roughness
  vec4 mrSample = texture(u_MRTex, vTexCoord0);
  float roughness = mrSample.g;
  float metallic = mrSample.b;

  // 2. View/Reflect
  vec3 V = normalize(u_CamPos - vWorldPos);
  vec3 R = reflect(-V, N);

  // 3. Shading
  vec3 sh_coeffs[9];
  float visibility;
  GetSHCoeffs(vTexCoord1, sh_coeffs, visibility);

  vec3 E_diffuse;
  if (u_ShowDirectional) {
    E_diffuse = EvalSHIrradiance(N, sh_coeffs);
  } else {
    // Sample Irradiance Map (which is already convoluted irradiance)
    E_diffuse = texture(u_IrradianceTex, vTexCoord1).rgb;
  }

  // Add Sky Ambient
  vec3 E_sky = visibility * EvalSHIrradiance(N, u_SkySH);
  vec3 E_total = E_diffuse + E_sky;

  // Specular
  vec3 specularRadiance = EvalSHRadiance(R, sh_coeffs);

  // Fresnel
  vec3 F0 = vec3(0.04);
  F0 = mix(F0, albedo, metallic);
  vec3 F = FresnelSchlick(max(dot(N, V), 0.0), F0);

  vec3 kS = F;
  vec3 kD = vec3(1.0) - kS;
  kD *= (1.0 - metallic);

  const float PI = 3.14159265359;
  vec3 diffuse = kD * (E_total * (1.0 / PI)) * albedo;
  vec3 specular = specularRadiance * F;

  vec3 color =
      u_ShowDirectional ? diffuse + specular : (E_total * (1.0 / PI)) * albedo;
  FragColor = vec4(color, 1.0);
}
