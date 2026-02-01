#pragma once

#include <Eigen/Dense>

#include "visualizer_radiance.h"

namespace sh_baker {

class DepthPrepass {
 public:
  DepthPrepass() = default;
  ~DepthPrepass();

  bool Init();

  // Draws the scene geometry strictly for depth pre-pass.
  // Writes to depth buffer, disables color writes.
  // Leaves state with ColorMask(True) and DepthMask(True) usually (restores
  // default).
  void Draw(const Scene& scene, const RadianceRenderer& radiance_renderer,
            const Eigen::Matrix4f& vp);

 private:
  GLuint program_ = 0;
};

}  // namespace sh_baker
