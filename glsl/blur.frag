#version 410 core
out vec4 FragColor;
in vec2 vTexCoord;

uniform sampler2D u_Image;
uniform bool u_Horizontal;

// Gaussian weights for 2-tap (linear interpolation optimization can be used but
// this is explicit) 0.227027, 0.1945946 We'll use a simple 2-sample loop on
// each side + center = 5 samples total: Center, +/- 2.
const float weight[2] = float[](0.44198, 0.27901);

void main() {
  vec2 tex_offset = 1.0 / textureSize(u_Image, 0);  // gets size of single texel
  vec3 result = texture(u_Image, vTexCoord).rgb * weight[0];

  if (u_Horizontal) {
    for (int i = 1; i < 2; ++i) {
      result += texture(u_Image, vTexCoord + vec2(tex_offset.x * i, 0.0)).rgb *
                weight[i];
      result += texture(u_Image, vTexCoord - vec2(tex_offset.x * i, 0.0)).rgb *
                weight[i];
    }
  } else {
    for (int i = 1; i < 2; ++i) {
      result += texture(u_Image, vTexCoord + vec2(0.0, tex_offset.y * i)).rgb *
                weight[i];
      result += texture(u_Image, vTexCoord - vec2(0.0, tex_offset.y * i)).rgb *
                weight[i];
    }
  }
  FragColor = vec4(result, 1.0);
}
