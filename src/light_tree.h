#ifndef SH_BAKER_SRC_LIGHT_TREE_H_
#define SH_BAKER_SRC_LIGHT_TREE_H_

#include <Eigen/Dense>
#include <vector>

#include "scene.h"

namespace sh_baker {

// Forward declarations.
namespace light_tree_internal {

// LightBounds stores the bounding information for a light or a set of lights.
// - bounds: The axis-aligned bounding box.
// - axis: The principal direction of the light(s).
// - cos_theta_o: Cosine of the half-angle of the bounding cone for the light's
//   principal directions.
// - cos_theta_e: Cosine of the half-angle of the emission cone (how wide the
//   light emits).
// - phi: Total power (flux) of the light(s) in the node.
// - two_sided: True if the light emits from both sides.
struct LightBounds {
  Eigen::AlignedBox3f bounds;
  Eigen::Vector3f axis = Eigen::Vector3f::UnitY();
  float cos_theta_o = 1.0f;  // Spread of principal directions.
  float cos_theta_e = 1.0f;  // Spread of emission cone.
  float phi = 0.0f;          // Total flux.
  bool two_sided = false;

  Eigen::Vector3f Centroid() const { return bounds.center(); }
};

// Union two LightBounds.
LightBounds Union(const LightBounds& a, const LightBounds& b);

// Compute the importance of a LightBounds from a reference point.
// p: the reference point.
// n: the normal at the reference point (can be zero for volumetric).
float Importance(const LightBounds& lb, const Eigen::Vector3f& p,
                 const Eigen::Vector3f& n);

// Compute LightBounds for a single Light.
LightBounds ComputeLightBounds(const Light& light);

// Cost function for SAH-based splitting.
float EvaluateCost(const LightBounds& lb, const Eigen::AlignedBox3f& bounds,
                   int dim);

}  // namespace light_tree_internal

// A node in the Light BVH.
struct LightBVHNode {
  light_tree_internal::LightBounds light_bounds;
  int child_or_light_index = -1;  // Leaf: index into `lights`. Interior: index
                                  // of right child.
  bool is_leaf = true;
};

// Result of sampling a light from the tree.
struct SampledLight {
  int light_index = -1;  // Index into original lights vector.
  float pdf = 0.0f;
};

// LightTree: A BVH for efficient light sampling.
class LightTree {
 public:
  LightTree() = default;

  // Builds the light tree from a list of lights.
  // After construction, the tree is ready for sampling.
  void Build(const std::vector<Light>& lights);

  // Samples a light given a reference point and normal.
  // p: the shading point.
  // n: the shading normal.
  // u: a uniform random number in [0, 1).
  // Returns SampledLight with light_index and pdf. If no light is sampled,
  // light_index is -1.
  SampledLight Sample(const Eigen::Vector3f& p, const Eigen::Vector3f& n,
                      float u) const;

  // Computes the PDF for sampling a specific light.
  // p: the shading point.
  // n: the shading normal.
  // light_index: the index of the light.
  float Pdf(const Eigen::Vector3f& p, const Eigen::Vector3f& n,
            int light_index) const;

  // Returns true if the tree is empty (no lights).
  bool Empty() const { return nodes_.empty(); }

  // Returns the number of lights in the tree.
  size_t NumLights() const { return lights_.size(); }

  // Access nodes for testing.
  const std::vector<LightBVHNode>& Nodes() const { return nodes_; }

  // Access the bit trail map for testing.
  const std::vector<uint32_t>& BitTrails() const { return light_to_bit_trail_; }

 private:
  // Recursively builds the BVH.
  // Returns (node_index, LightBounds).
  std::pair<int, light_tree_internal::LightBounds> BuildBVH(
      std::vector<std::pair<int, light_tree_internal::LightBounds>>& bvh_lights,
      int start, int end, uint32_t bit_trail, int depth);

  std::vector<LightBVHNode> nodes_;
  std::vector<const Light*> lights_;
  std::vector<uint32_t> light_to_bit_trail_;  // Maps light index -> bit trail.
  Eigen::AlignedBox3f all_light_bounds_;
};

}  // namespace sh_baker

#endif  // SH_BAKER_SRC_LIGHT_TREE_H_
