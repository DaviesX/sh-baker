#version 330 core

layout(location = 0) in vec3 aPos;
// We don't need other attributes for depth only

uniform mat4 u_MVP;

void main() { gl_Position = u_MVP * vec4(aPos, 1.0); }
