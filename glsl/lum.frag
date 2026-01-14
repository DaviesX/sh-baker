#version 410 core
out vec4 FragColor;
in vec2 vTexCoord;
uniform sampler2D u_HdrTex;

void main() {
  vec3 color = texture(u_HdrTex, vTexCoord).rgb;
  // Rec. 709 / sRGB luminance coeffs
  float lum = dot(color, vec3(0.2126, 0.7152, 0.0722));
  float logLum = log(lum + 0.0001);  // Epsilon to avoid log(0)
  FragColor = vec4(logLum, 0.0, 0.0, 1.0);
}
