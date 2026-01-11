#version 410 core

out vec4 FragColor;

uniform sampler2D u_SkyboxTex;  // Equirectangular
uniform int u_UsePreetham;
uniform vec3 u_SunDir;
uniform float u_SkyIntensity;

in vec3 vWorldPos;

// --- Preetham Sky ---
// Simplified Perez model or similar analytical sky
// Based on typical GLSL implementations
vec3 PreethamSky(vec3 viewDir, vec3 sunDir) {
  float cosTheta = max(viewDir.y, 0.0);
  float cosGamma = dot(viewDir, sunDir);

  // Rayleigh
  vec3 rayleigh = vec3(0.18867780436772762, 0.4978442963618773,
                       0.6616065586417131);  // Blue sky color
  // Gradient based on zenith
  rayleigh *= (1.0 + 2.0 * cosTheta);

  // Mie (Sun halo)
  float mie = pow(max(0.0, cosGamma), 100.0) * 0.5;

  // Sun disk
  float sunUtils = step(0.9995, cosGamma);  // Small disk

  vec3 skyColor = rayleigh * 0.5 + vec3(mie) + vec3(sunUtils) * 20.0;
  return skyColor * u_SkyIntensity;
}

vec3 SampleSkybox(vec3 dir) {
  if (u_UsePreetham == 1) {
    return PreethamSky(normalize(dir), normalize(u_SunDir));
  } else {
    // Equirectangular mapping
    // dir is normalized
    vec2 uv = vec2(atan(dir.z, dir.x), asin(dir.y));
    uv *= vec2(0.1591, 0.3183);  // inv(2PI), inv(PI)
    uv += 0.5;
    // Flip UV
    uv.y = 1.0 - uv.y;

    return texture(u_SkyboxTex, uv).rgb * u_SkyIntensity;
  }
}

void main() {
  vec3 dir = normalize(vWorldPos);  // Cube vertex pos -> direction
  vec3 color = SampleSkybox(dir);

  // Output Linear HDR Color
  FragColor = vec4(color, 1.0);
}
