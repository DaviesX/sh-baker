#version 410 core

layout(location = 0) in vec3 aPosition;
// Using aPosition from the cube (location 0)

out vec3 vWorldPos;

uniform mat4 u_MVP;

void main() {
  // Skybox mode: Position is local cube vertex.
  vec4 pos = u_MVP * vec4(aPosition, 1.0);

  // Force z to w so it ends up at far plane (z/w = 1.0)
  gl_Position = pos.xyww;

  // Pass local position as "WorldPos" to use as direction vector in frag shader
  vWorldPos = aPosition;
}
