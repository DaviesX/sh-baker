#ifndef SH_BAKER_SRC_LIGHT_TREE_H_
#define SH_BAKER_SRC_LIGHT_TREE_H_

#include <Eigen/Dense>
#include <optional>
#include <unordered_map>
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
// - max_range: The largest Light::range of the light(s), or 0 if any of them is
//   unranged. A shaded point farther than max_range from `bounds` gets zero
//   importance.
struct LightBounds {
  Eigen::AlignedBox3f bounds;
  Eigen::Vector3f axis = Eigen::Vector3f::UnitY();
  float cos_theta_o = 1.0f;  // Spread of principal directions.
  float cos_theta_e = 1.0f;  // Spread of emission cone.
  float phi = 0.0f;          // Total flux.
  bool two_sided = false;
  float max_range = 0.0f;  // 0: unbounded.

  Eigen::Vector3f Centroid() const { return bounds.center(); }
};

// Union two LightBounds. An empty side (phi 0) leaves the other side's range;
// otherwise max_range is 0 if either side is 0, else the larger range.
LightBounds Union(const LightBounds& a, const LightBounds& b);

// True when lb is ranged (max_range > 0) and p is farther than max_range from
// its bounds, with a small relative slack that keeps the test conservative.
bool BeyondRange(const LightBounds& lb, const Eigen::Vector3f& p);

// Compute the importance of a LightBounds from a reference point. It is 0
// when max_range > 0 and p is farther than max_range from the bounds.
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
  // For leaf nodes: pointer to the light.
  // For interior nodes: this is unused (use right_child_index).
  const Light* light = nullptr;
  // For interior nodes: index of the right child node.
  // Left child is always at node_index + 1.
  int right_child_index = -1;
  bool is_leaf = true;
};

// Result of sampling a light from the tree.
struct SampledLight {
  const Light* light = nullptr;
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
  //
  // Directional lights get a fixed share of the choice, n_dir / (n_dir + 1),
  // and the BVH the rest, except where p is beyond the range of the BVH's
  // root (light_tree_internal::BeyondRange): there the whole choice goes to
  // the directional lights, or nothing is sampled if there are none. Pdf()
  // applies the same per-point rule.
  //
  // p: the shading point.
  // n: the shading normal.
  // u: a uniform random number in [0, 1).
  // Returns SampledLight with light pointer and pdf. If no light can be
  // sampled, returns std::nullopt.
  std::optional<SampledLight> Sample(const Eigen::Vector3f& p,
                                     const Eigen::Vector3f& n, float u) const;

  // Samples a light uniformly at random, ignoring importance.
  // Useful for debugging and comparison against importance-based sampling.
  // u: a uniform random number in [0, 1).
  // Returns SampledLight with light pointer and pdf = 1 / NumLights().
  // If no lights exist, returns std::nullopt.
  std::optional<SampledLight> SampleUniform(float u) const;

  // Computes the PDF for sampling a specific light.
  // p: the shading point.
  // n: the shading normal.
  // light: pointer to the light.
  float Pdf(const Eigen::Vector3f& p, const Eigen::Vector3f& n,
            const Light* light) const;

  // Deprecated overload using light index.
  float Pdf(const Eigen::Vector3f& p, const Eigen::Vector3f& n,
            int light_index) const;

  // Returns true if the tree has no lights (neither BVH nor directional).
  bool Empty() const { return nodes_.empty() && directional_lights_.empty(); }

  // Returns the number of lights in the tree (including directional).
  size_t NumLights() const {
    return lights_.size() + directional_lights_.size();
  }

  // Returns the number of directional lights.
  size_t NumDirectionalLights() const { return directional_lights_.size(); }

  // Access nodes for testing.
  const std::vector<LightBVHNode>& Nodes() const { return nodes_; }

  // Access the bit trail map for testing.
  const std::unordered_map<const Light*, uint32_t>& BitTrails() const {
    return light_to_bit_trail_;
  }

 private:
  // The probability of choosing the directional set at p, by the rule in
  // Sample(). 0 with no directional lights.
  float DirectionalShare(const Eigen::Vector3f& p) const;

  // Recursively builds the BVH.
  // Returns (node_index, LightBounds).
  std::pair<int, light_tree_internal::LightBounds> BuildBVH(
      std::vector<std::pair<const Light*, light_tree_internal::LightBounds>>&
          bvh_lights,
      int start, int end, uint32_t bit_trail, int depth);

  std::vector<LightBVHNode> nodes_;
  std::vector<const Light*> lights_;  // BVH lights (non-directional).
  std::vector<const Light*>
      directional_lights_;  // Unbounded directional lights.
  std::unordered_map<const Light*, uint32_t>
      light_to_bit_trail_;  // Maps light pointer -> bit trail.
  Eigen::AlignedBox3f all_light_bounds_;
};

}  // namespace sh_baker

#endif  // SH_BAKER_SRC_LIGHT_TREE_H_
