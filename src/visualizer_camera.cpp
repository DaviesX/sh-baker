#include "visualizer_camera.h"

#include <iostream>

namespace sh_baker {

Camera::Camera(Eigen::Vector3f position, Eigen::Vector3f up, float yaw,
               float pitch)
    : position_(position),
      world_up_(up),
      yaw_(yaw),
      pitch_(pitch),
      movement_speed_(2.5f),
      mouse_sensitivity_(0.1f) {
  UpdateCameraVectors();
}

Eigen::Matrix4f Camera::GetViewMatrix() const {
  // LookAt
  Eigen::Vector3f f = front_.normalized();
  Eigen::Vector3f s = f.cross(world_up_).normalized();
  Eigen::Vector3f u = s.cross(f);

  Eigen::Matrix4f view = Eigen::Matrix4f::Identity();
  view(0, 0) = s.x();
  view(0, 1) = s.y();
  view(0, 2) = s.z();
  view(1, 0) = u.x();
  view(1, 1) = u.y();
  view(1, 2) = u.z();
  view(2, 0) = -f.x();
  view(2, 1) = -f.y();
  view(2, 2) = -f.z();
  view(0, 3) = -s.dot(position_);
  view(1, 3) = -u.dot(position_);
  view(2, 3) = f.dot(position_);

  return view;
}

void Camera::Translate(CameraMovement direction, float deltaTime) {
  float velocity = movement_speed_ * deltaTime;
  if (direction == CameraMovement::FORWARD) position_ += front_ * velocity;
  if (direction == CameraMovement::BACKWARD) position_ -= front_ * velocity;
  if (direction == CameraMovement::LEFT) position_ -= right_ * velocity;
  if (direction == CameraMovement::RIGHT) position_ += right_ * velocity;
  if (direction == CameraMovement::UP)
    position_ += world_up_ * velocity;  // Global Up
  if (direction == CameraMovement::DOWN) position_ -= world_up_ * velocity;
}

void Camera::Rotate(float xoffset, float yoffset, bool constrainPitch) {
  xoffset *= mouse_sensitivity_;
  yoffset *= mouse_sensitivity_;

  yaw_ += xoffset;
  pitch_ += yoffset;

  if (constrainPitch) {
    if (pitch_ > 89.0f) pitch_ = 89.0f;
    if (pitch_ < -89.0f) pitch_ = -89.0f;
  }

  UpdateCameraVectors();
}

void Camera::Zoom(float yoffset) {
  movement_speed_ += yoffset * 0.5f;
  if (movement_speed_ < 0.1f) movement_speed_ = 0.1f;
  if (movement_speed_ > 50.0f) movement_speed_ = 50.0f;
}

void Camera::UpdateCameraVectors() {
  Eigen::Vector3f front;
  front.x() = cos(yaw_ * (M_PI / 180.0f)) * cos(pitch_ * (M_PI / 180.0f));
  front.y() = sin(pitch_ * (M_PI / 180.0f));
  front.z() = sin(yaw_ * (M_PI / 180.0f)) * cos(pitch_ * (M_PI / 180.0f));
  front_ = front.normalized();

  right_ = front_.cross(world_up_).normalized();
  up_ = right_.cross(front_).normalized();
}

}  // namespace sh_baker
