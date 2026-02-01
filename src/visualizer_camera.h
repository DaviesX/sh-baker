#ifndef VISUALIZER_CAMERA_H
#define VISUALIZER_CAMERA_H

#include <Eigen/Dense>

namespace sh_baker {

enum class CameraMovement { FORWARD, BACKWARD, LEFT, RIGHT, UP, DOWN };

class Camera {
 public:
  Camera(Eigen::Vector3f position = Eigen::Vector3f(0.0f, 0.0f, 3.0f),
         Eigen::Vector3f up = Eigen::Vector3f(0.0f, 1.0f, 0.0f),
         float yaw = -90.0f, float pitch = 0.0f);

  Eigen::Matrix4f GetViewMatrix() const;

  void Translate(CameraMovement direction, float deltaTime);
  void Rotate(float xoffset, float yoffset, bool constrainPitch = true);
  void Zoom(float yoffset);

  Eigen::Vector3f Position() const { return position_; }
  Eigen::Vector3f Front() const { return front_; }
  Eigen::Vector3f Up() const { return up_; }
  Eigen::Vector3f Right() const { return right_; }

 private:
  void UpdateCameraVectors();

  Eigen::Vector3f position_;
  Eigen::Vector3f front_;
  Eigen::Vector3f up_;
  Eigen::Vector3f right_;
  Eigen::Vector3f world_up_;

  float yaw_;
  float pitch_;

  float movement_speed_;
  float mouse_sensitivity_;
};

}  // namespace sh_baker

#endif  // VISUALIZER_CAMERA_H
