#include "visualizer_control.h"

#include <iostream>

namespace sh_baker {

InputController::InputController(Camera& camera) : camera_(camera) {}

void InputController::ProcessInput(GLFWwindow* window, float deltaTime) {
  if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    glfwSetWindowShouldClose(window, true);

  if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
    camera_.Translate(CameraMovement::FORWARD, deltaTime);
  if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
    camera_.Translate(CameraMovement::BACKWARD, deltaTime);
  if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
    camera_.Translate(CameraMovement::LEFT, deltaTime);
  if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
    camera_.Translate(CameraMovement::RIGHT, deltaTime);
  if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
    camera_.Translate(CameraMovement::UP, deltaTime);
  if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
    camera_.Translate(CameraMovement::DOWN, deltaTime);
}

void InputController::MouseButtonCallback(GLFWwindow* window, int button,
                                          int action, int mods) {
  if (button == GLFW_MOUSE_BUTTON_LEFT) {
    if (action == GLFW_PRESS) {
      mouse_pressed_ = true;
      first_mouse_ = true;
    } else {
      mouse_pressed_ = false;
    }
  }
}

void InputController::CursorPosCallback(GLFWwindow* window, double xpos,
                                        double ypos) {
  if (mouse_pressed_) {
    if (first_mouse_) {
      last_mouse_x_ = xpos;
      last_mouse_y_ = ypos;
      first_mouse_ = false;
    }

    float xoffset = static_cast<float>(xpos - last_mouse_x_);
    float yoffset = static_cast<float>(last_mouse_y_ - ypos);  // Reversed

    last_mouse_x_ = xpos;
    last_mouse_y_ = ypos;

    camera_.Rotate(xoffset, yoffset);
  }
}

void InputController::ScrollCallback(GLFWwindow* window, double xoffset,
                                     double yoffset) {
  camera_.Zoom(static_cast<float>(yoffset));
}

}  // namespace sh_baker
