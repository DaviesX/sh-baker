#ifndef VISUALIZER_CONTROL_H
#define VISUALIZER_CONTROL_H

#include <GLFW/glfw3.h>

#include "visualizer_camera.h"

namespace sh_baker {

class InputController {
 public:
  explicit InputController(Camera& camera);

  void ProcessInput(GLFWwindow* window, float deltaTime);
  void MouseButtonCallback(GLFWwindow* window, int button, int action,
                           int mods);
  void CursorPosCallback(GLFWwindow* window, double xpos, double ypos);
  void ScrollCallback(GLFWwindow* window, double xoffset, double yoffset);

 private:
  Camera& camera_;
  bool mouse_pressed_ = false;
  bool first_mouse_ = true;
  double last_mouse_x_ = 0.0;
  double last_mouse_y_ = 0.0;
};

}  // namespace sh_baker

#endif  // VISUALIZER_CONTROL_H
