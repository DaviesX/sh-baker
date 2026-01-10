#version 410 core

in vec3 vNormal;
in vec2 vTexCoord0;
in vec2 vTexCoord1;

out vec4 FragColor;

uniform sampler2D u_AlbedoTex;
uniform int u_HasAlbedo;
uniform vec3 u_AlbedoColor;
// --- PBR Inputs ---
uniform sampler2D u_NormalTex;
uniform int u_HasNormal;

uniform sampler2D u_MRTex;  // Metallic (B), Roughness (G)
uniform int u_HasMR;
uniform float u_Metallic;
uniform float u_Roughness;

uniform vec3 u_CamPos;

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
uniform int u_UsePackedLuminance;
uniform sampler2D u_PackedTex0;
uniform sampler2D u_PackedTex1;
uniform sampler2D u_PackedTex2;

// --- Attributes ---
in vec3 vWorldPos;
in vec3 vTangent;
in vec3 vBitangent;

// --- Helper: Sample SH ---
vec3 SampleSH(vec3 normal, vec2 uv) {
  // Basis functions:
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

  vec3 L0, L1m1, L10, L11, L2m2, L2m1, L20, L21, L22;

  if (u_UsePackedLuminance == 1) {
    vec4 p0 = texture(u_PackedTex0, uv);
    vec4 p1 = texture(u_PackedTex1, uv);
    vec4 p2 = texture(u_PackedTex2, uv);

    L0 = p0.rgb;
    float L0_lum = dot(L0, vec3(0.2126, 0.7152, 0.0722));
    vec3 chroma = vec3(1.0);
    if (L0_lum > 1e-6) {
      chroma = L0 / L0_lum;
    }

    L1m1 = vec3(p0.a) * chroma;
    L10 = vec3(p1.r) * chroma;
    L11 = vec3(p1.g) * chroma;
    L2m2 = vec3(p1.b) * chroma;
    L2m1 = vec3(p1.a) * chroma;
    L20 = vec3(p2.r) * chroma;
    L21 = vec3(p2.g) * chroma;
    L22 = vec3(p2.b) * chroma;
  } else {
    L0 = texture(u_L0, uv).rgb;
    L1m1 = texture(u_L1m1, uv).rgb;
    L10 = texture(u_L10, uv).rgb;
    L11 = texture(u_L11, uv).rgb;
    L2m2 = texture(u_L2m2, uv).rgb;
    L2m1 = texture(u_L2m1, uv).rgb;
    L20 = texture(u_L20, uv).rgb;
    L21 = texture(u_L21, uv).rgb;
    L22 = texture(u_L22, uv).rgb;
  }

  vec3 result = L0 * b0 + L1m1 * b1 + L10 * b2 + L11 * b3 + L2m2 * b4 +
                L2m1 * b5 + L20 * b6 + L21 * b7 + L22 * b8;
  return max(result, 0.0);
}

// --- Fresnel ---
vec3 FresnelSchlick(float cosTheta, vec3 F0) {
  return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

void main() {
  // 1. PBR Parameters
  // Albedo
  vec3 albedo = u_AlbedoColor;
  if (u_HasAlbedo > 0) {
    vec4 texColor = texture(u_AlbedoTex, vTexCoord0);
    albedo *= texColor.rgb;
  }

  // Normal
  vec3 N = normalize(vNormal);
  if (u_HasNormal > 0) {
    vec3 mapNormal = texture(u_NormalTex, vTexCoord0).rgb;
    mapNormal = mapNormal * 2.0 - 1.0;
    mat3 TBN = mat3(normalize(vTangent), normalize(vBitangent), N);
    N = normalize(TBN * mapNormal);
  }

  // Metallic/Roughness
  float metallic = u_Metallic;
  float roughness = u_Roughness;
  if (u_HasMR > 0) {
    vec4 mrSample = texture(u_MRTex, vTexCoord0);
    // glTF: G = Roughness, B = Metallic
    roughness = mrSample.g;
    metallic = mrSample.b;
  }

  // 2. View/Reflect vectors
  vec3 V = normalize(u_CamPos - vWorldPos);
  vec3 R = reflect(-V, N);  // Reflection vector

  // 3. Shading
  // Diffuse Irradiance (SH along Normal)
  vec3 irradiance = SampleSH(N, vTexCoord1);

  // Specular Radiance (SH along Reflection) -> Rough approximation
  // Ideally we convolve SH with Specular Lobe, but cheap way is sampling along
  // R. We can blur the lookup based on roughness if we had MIPs of coeffs? For
  // now, just sample along R.
  vec3 specularIrradiance = SampleSH(R, vTexCoord1);

  // Compute F0
  vec3 F0 = vec3(0.04);
  F0 = mix(F0, albedo, metallic);

  vec3 F = FresnelSchlick(max(dot(N, V), 0.0), F0);

  vec3 kS = F;
  vec3 kD = vec3(1.0) - kS;
  kD *= (1.0 - metallic);

  vec3 diffuse = kD * irradiance * albedo;

  // Simple Specular term
  // (Usually use Split Sum approx with Pre-filtered env map + BRDF LUT)
  // Here we use SH as Pre-filtered env map (very low freq).
  // It works okay for rough surfaces. For shiny, it will look blurry (which is
  // SH limitation).
  vec3 specular = specularIrradiance * F;

  // Combine
  vec3 color = diffuse + specular;

  // Tone mapping (Reinhard)
  color = color / (color + vec3(1.0));
  // Gamma
  color = pow(color, vec3(1.0 / 2.2));

  FragColor = vec4(color, 1.0);
}
