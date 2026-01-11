#version 410 core

out vec4 FragColor;
in vec2 vTexCoord;

uniform sampler2D u_ScreenTexture;
uniform sampler2D u_LumTexture;

// --- Tonemapping Helper ---
const float A = 0.15;
const float B = 0.50;
const float C = 0.10;
const float D = 0.20;
const float E = 0.02;
const float F = 0.30;
const float W = 11.2;

const float exposureBias = 1.0f;

vec3 Uncharted2Tonemap(vec3 x) {
  return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

void main() {
  vec3 color = texture(u_ScreenTexture, vTexCoord).rgb;

  // Auto Exposure
  // Sample 1x1 mip level (max LOD)
  float logLumAvg = textureLod(u_LumTexture, vec2(0.5), 10.0).r;
  // Assuming 1024x1024 or smaller. 10.0 covers up to 1024.
  float lumAvg = exp(logLumAvg);

  // Key value (Middle Gray)
  float key = 0.18;
  // Exposure Formula: Key / GeomMean
  float exposure = key / max(lumAvg, 0.0001);

  color *= exposure;

  // Tone mapping
  vec3 curr = Uncharted2Tonemap(exposureBias * color);
  vec3 whiteScale = 1.0f / Uncharted2Tonemap(vec3(W));
  color = curr * whiteScale;

  FragColor = vec4(color, 1.0);
}
