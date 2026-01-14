#version 410 core
out vec4 FragColor;
in vec2 vTexCoord;

uniform sampler2D u_HdrTex;
uniform sampler2D u_LumTexture;

void main() {
  vec3 color = texture(u_HdrTex, vTexCoord).rgb;

  // Get exposure
  float logLumAvg = textureLod(u_LumTexture, vec2(0.5), 10.0).r;
  float lumAvg = exp(logLumAvg);
  float key = 0.18;
  float exposure = key / max(lumAvg, 0.1);

  // Apply exposure temporarily for thresholding
  // We want bloom for parts that will be "white" or brighter
  vec3 exposedColor = color * exposure;

  // Threshold
  float threshold = 1.0;
  // Soft knee? or Hard cut?
  // Let's do simple: keep values > 1.0
  vec3 brightColor = max(exposedColor - vec3(threshold), vec3(0.0));

  // We store the *exposed* bright color, or unexposed?
  // Often getting the exposed bloom is easier for compositing later if we
  // composite *before* tonemap but *after* exposure. In post.frag: `color *=
  // exposure; color += bloom; tonemap(color);` So yes, store exposed bloom.

  FragColor = vec4(brightColor, 1.0);
}
