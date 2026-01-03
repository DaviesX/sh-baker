#version 410 core

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aTexCoord0; // Albedo UV
layout(location = 3) in vec2 aTexCoord1; // Lightmap UV

out vec3 vNormal;
out vec2 vTexCoord0;
out vec2 vTexCoord1;

uniform mat4 u_MVP;

void main() {
    gl_Position = u_MVP * vec4(aPosition, 1.0);
    vNormal = aNormal;
    vTexCoord0 = aTexCoord0;
    vTexCoord1 = aTexCoord1;
}
