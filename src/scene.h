#ifndef SH_BAKER_SRC_SCENE_H_
#define SH_BAKER_SRC_SCENE_H_

#include <vector>
#include <cstdint>
#include <Eigen/Dense>

namespace sh_baker {

struct Scene {
  std::vector<Eigen::Vector3f> vertices;
  std::vector<Eigen::Vector3f> normals;
  std::vector<Eigen::Vector2f> uvs;
  std::vector<uint32_t> indices;
};

}  // namespace sh_baker

#endif  // SH_BAKER_SRC_SCENE_H_
