#version 410 core

out vec4 FragColor;
in vec2 vTexCoord;

uniform sampler2D u_ScreenTexture;

// --- Tonemapping Helper ---
float A = 0.15;
float B = 0.50;
float C = 0.10;
float D = 0.20;
float E = 0.02;
float F = 0.30;
float W = 11.2;

vec3 Uncharted2Tonemap(vec3 x) {
  return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

void main() {
  vec3 color = texture(u_ScreenTexture, vTexCoord).rgb;

  // Tone mapping
  float exposureBias = 4.0f;
  vec3 curr = Uncharted2Tonemap(exposureBias * color);
  vec3 whiteScale = 1.0f / Uncharted2Tonemap(vec3(W));
  color = curr * whiteScale;

  // Gamma Correction
  color = pow(color, vec3(1.0 / 2.2));

  FragColor = vec4(color, 1.0);
}
