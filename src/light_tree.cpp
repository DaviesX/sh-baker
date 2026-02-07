#include "light_tree.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace sh_baker {
namespace light_tree_internal {

// Helper: Safe square root that clamps negative values to 0.
inline float SafeSqrt(float x) { return std::sqrt(std::max(0.0f, x)); }

// Helper: Cosine subtraction clamped. cos(a - b) with handling for a > b.
inline float CosSubClamped(float sin_a, float cos_a, float sin_b, float cos_b) {
  if (cos_a > cos_b) return 1.0f;
  return cos_a * cos_b + sin_a * sin_b;
}

// Helper: Sine subtraction clamped. sin(a - b) with handling for a > b.
inline float SinSubClamped(float sin_a, float cos_a, float sin_b, float cos_b) {
  if (cos_a > cos_b) return 0.0f;
  return sin_a * cos_b - cos_a * sin_b;
}

LightBounds Union(const LightBounds& a, const LightBounds& b) {
  LightBounds result;

  // Union bounding boxes.
  result.bounds = a.bounds.merged(b.bounds);

  // Union flux.
  result.phi = a.phi + b.phi;

  // Union two-sided.
  result.two_sided = a.two_sided || b.two_sided;

  // Union orientation cones.
  // We need to find a cone that bounds both cones.
  // This is a simplified approach: we compute the average axis and widen the
  // cone to include both.
  if (a.phi == 0.0f) {
    result.axis = b.axis;
    result.cos_theta_o = b.cos_theta_o;
    result.cos_theta_e = b.cos_theta_e;
    return result;
  }
  if (b.phi == 0.0f) {
    result.axis = a.axis;
    result.cos_theta_o = a.cos_theta_o;
    result.cos_theta_e = a.cos_theta_e;
    return result;
  }

  // DirectionCone union (simplified from PBRT).
  // Compute angle between axes.
  float cos_angle = a.axis.dot(b.axis);

  // Compute half-angles.
  float theta_a = std::acos(std::clamp(a.cos_theta_o, -1.0f, 1.0f));
  float theta_b = std::acos(std::clamp(b.cos_theta_o, -1.0f, 1.0f));
  float angle_between = std::acos(std::clamp(cos_angle, -1.0f, 1.0f));

  // Check if one cone contains the other.
  if (std::min(angle_between + theta_b, static_cast<float>(M_PI)) <= theta_a) {
    // a contains b.
    result.axis = a.axis;
    result.cos_theta_o = a.cos_theta_o;
  } else if (std::min(angle_between + theta_a, static_cast<float>(M_PI)) <=
             theta_b) {
    // b contains a.
    result.axis = b.axis;
    result.cos_theta_o = b.cos_theta_o;
  } else {
    // Need to compute a new cone.
    float theta_o =
        (theta_a + angle_between + theta_b) / 2.0f;  // New half-angle.
    if (theta_o >= static_cast<float>(M_PI)) {
      // Full sphere.
      result.axis = a.axis;
      result.cos_theta_o = -1.0f;
    } else {
      // Rotate a.axis towards b.axis.
      float theta_r = theta_o - theta_a;
      Eigen::Vector3f cross = a.axis.cross(b.axis);
      float cross_norm = cross.norm();
      if (cross_norm < 1e-6f) {
        // Axes are parallel or anti-parallel.
        result.axis = a.axis;
      } else {
        // Rotate axis.
        Eigen::Vector3f rotation_axis = cross.normalized();
        Eigen::AngleAxisf rotation(theta_r, rotation_axis);
        result.axis = (rotation * a.axis).normalized();
      }
      result.cos_theta_o = std::cos(theta_o);
    }
  }

  // Union emission cones (simplified: take the wider one).
  result.cos_theta_e = std::min(a.cos_theta_e, b.cos_theta_e);

  return result;
}

float Importance(const LightBounds& lb, const Eigen::Vector3f& p,
                 const Eigen::Vector3f& n) {
  // Compute clamped squared distance to bounding box center.
  Eigen::Vector3f pc = lb.bounds.center();
  float d2 = (p - pc).squaredNorm();
  float diag_len = lb.bounds.diagonal().norm();
  d2 = std::max(d2, diag_len * diag_len / 4.0f);

  // Compute direction from p to center.
  Eigen::Vector3f wi = (pc - p).normalized();

  // Compute cos(theta_w): angle between the light's axis and the direction
  // from the light TO the point. This checks if the light is "pointing toward"
  // the point.
  Eigen::Vector3f dir_to_p = -wi;  // Direction from light center to p.
  float cos_theta_w = lb.axis.dot(dir_to_p);
  if (lb.two_sided) {
    cos_theta_w = std::abs(cos_theta_w);
  }
  float sin_theta_w = SafeSqrt(1.0f - cos_theta_w * cos_theta_w);

  // Compute cos(theta_b): the angle subtended by the bounding box.
  // Approximation: sin(theta_b) ~ diag_len / (2 * d).
  float d = std::sqrt(d2);
  float sin_theta_b = std::min(diag_len / (2.0f * d), 1.0f);
  float cos_theta_b = SafeSqrt(1.0f - sin_theta_b * sin_theta_b);

  // Compute cos(theta') = cos(theta_w - theta_o - theta_b).
  float sin_theta_o = SafeSqrt(1.0f - lb.cos_theta_o * lb.cos_theta_o);
  float cos_theta_x =
      CosSubClamped(sin_theta_w, cos_theta_w, sin_theta_o, lb.cos_theta_o);
  float sin_theta_x =
      SinSubClamped(sin_theta_w, cos_theta_w, sin_theta_o, lb.cos_theta_o);
  float cos_theta_p =
      CosSubClamped(sin_theta_x, cos_theta_x, sin_theta_b, cos_theta_b);

  if (cos_theta_p <= lb.cos_theta_e) {
    return 0.0f;
  }

  float importance = lb.phi * cos_theta_p / d2;

  // Account for incident angle at surface.
  if (n.squaredNorm() > 0.5f) {
    float cos_theta_i = std::abs(wi.dot(n));
    float sin_theta_i = SafeSqrt(1.0f - cos_theta_i * cos_theta_i);
    float cos_theta_p_i =
        CosSubClamped(sin_theta_i, cos_theta_i, sin_theta_b, cos_theta_b);
    importance *= cos_theta_p_i;
  }

  return std::max(importance, 0.0f);
}

LightBounds ComputeLightBounds(const Light& light) {
  LightBounds lb;

  switch (light.type) {
    case Light::Type::Point: {
      // Point light: bounds is a small sphere around position.
      lb.bounds = Eigen::AlignedBox3f(light.position, light.position);
      lb.axis = Eigen::Vector3f::UnitY();  // Arbitrary.
      lb.cos_theta_o = -1.0f;              // Emits in all directions.
      lb.cos_theta_e = -1.0f;              // Full sphere emission.
      lb.phi = light.intensity * light.color.maxCoeff();
      lb.two_sided = true;
      break;
    }
    case Light::Type::Spot: {
      // Spot light.
      lb.bounds = Eigen::AlignedBox3f(light.position, light.position);
      lb.axis = light.direction;
      lb.cos_theta_o = 1.0f;  // Single direction.
      lb.cos_theta_e = light.cos_outer_cone;
      lb.phi = light.intensity * light.color.maxCoeff();
      lb.two_sided = false;
      break;
    }
    case Light::Type::Directional: {
      // Directional light: infinite light, cannot be bounded.
      // Return empty bounds to indicate it should be handled separately.
      lb.phi = 0.0f;
      break;
    }
    case Light::Type::Area: {
      // Area light: bounds from geometry.
      if (light.geometry) {
        const auto& geo = *light.geometry;
        for (const auto& v : geo.vertices) {
          lb.bounds.extend(v);
        }
        // Compute average normal as axis.
        Eigen::Vector3f avg_normal = Eigen::Vector3f::Zero();
        for (const auto& n : geo.normals) {
          avg_normal += n;
        }
        if (avg_normal.squaredNorm() > 1e-6f) {
          lb.axis = avg_normal.normalized();
        } else {
          lb.axis = Eigen::Vector3f::UnitY();
        }
        lb.cos_theta_o = 0.0f;   // Hemisphere.
        lb.cos_theta_e = -1.0f;  // Full sphere emission (two-sided).
        lb.phi = light.intensity * light.color.maxCoeff() * light.area;
        lb.two_sided = true;
      }
      break;
    }
    default:
      break;
  }

  return lb;
}

float EvaluateCost(const LightBounds& lb, const Eigen::AlignedBox3f& bounds,
                   int dim) {
  // Evaluate direction bounds measure for LightBounds.
  float theta_o = std::acos(std::clamp(lb.cos_theta_o, -1.0f, 1.0f));
  float theta_e = std::acos(std::clamp(lb.cos_theta_e, -1.0f, 1.0f));
  float theta_w = std::min(theta_o + theta_e, static_cast<float>(M_PI));
  float sin_theta_o = SafeSqrt(1.0f - lb.cos_theta_o * lb.cos_theta_o);

  // M_omega: measure of the solid angle of the bounding cone.
  float M_omega =
      2.0f * static_cast<float>(M_PI) * (1.0f - lb.cos_theta_o) +
      static_cast<float>(M_PI) / 2.0f *
          (2.0f * theta_w * sin_theta_o - std::cos(theta_o - 2.0f * theta_w) -
           2.0f * theta_o * sin_theta_o + lb.cos_theta_o);

  // Kr: anisotropy factor.
  Eigen::Vector3f diagonal = bounds.diagonal();
  float max_diag = diagonal.maxCoeff();
  float Kr = (diagonal[dim] > 1e-6f) ? max_diag / diagonal[dim] : 1.0f;

  // Surface area of the light bounds.
  Eigen::Vector3f lb_diag = lb.bounds.diagonal();
  float surface_area =
      2.0f * (lb_diag.x() * lb_diag.y() + lb_diag.y() * lb_diag.z() +
              lb_diag.z() * lb_diag.x());

  return lb.phi * M_omega * Kr * surface_area;
}

}  // namespace light_tree_internal

void LightTree::Build(const std::vector<Light>& lights) {
  nodes_.clear();
  lights_.clear();
  light_to_bit_trail_.clear();
  all_light_bounds_ = Eigen::AlignedBox3f();

  if (lights.empty()) {
    return;
  }

  // Collect bounded lights and compute their LightBounds.
  std::vector<std::pair<int, light_tree_internal::LightBounds>> bvh_lights;
  for (size_t i = 0; i < lights.size(); ++i) {
    light_tree_internal::LightBounds lb =
        light_tree_internal::ComputeLightBounds(lights[i]);
    // Note: We check if bounds are valid (min <= max) rather than !isEmpty()
    // because point/spot lights have degenerate bounds (single point), which
    // isEmpty() considers empty (zero volume).
    if (lb.phi > 0.0f &&
        (lb.bounds.min().array() <= lb.bounds.max().array()).all()) {
      bvh_lights.push_back({static_cast<int>(i), lb});
      all_light_bounds_.extend(lb.bounds);
    }
  }

  if (bvh_lights.empty()) {
    return;
  }

  // Store pointers to lights.
  lights_.resize(lights.size());
  for (size_t i = 0; i < lights.size(); ++i) {
    lights_[i] = &lights[i];
  }

  // Initialize bit trail map.
  light_to_bit_trail_.resize(lights.size(), 0);

  // Build BVH.
  BuildBVH(bvh_lights, 0, static_cast<int>(bvh_lights.size()), 0, 0);
}

std::pair<int, light_tree_internal::LightBounds> LightTree::BuildBVH(
    std::vector<std::pair<int, light_tree_internal::LightBounds>>& bvh_lights,
    int start, int end, uint32_t bit_trail, int depth) {
  // Base case: single light -> leaf node.
  if (end - start == 1) {
    int node_index = static_cast<int>(nodes_.size());
    LightBVHNode leaf;
    leaf.light_bounds = bvh_lights[start].second;
    leaf.child_or_light_index = bvh_lights[start].first;
    leaf.is_leaf = true;
    nodes_.push_back(leaf);
    light_to_bit_trail_[bvh_lights[start].first] = bit_trail;
    return {node_index, bvh_lights[start].second};
  }

  // Compute bounds and centroid bounds for partition.
  Eigen::AlignedBox3f bounds;
  Eigen::AlignedBox3f centroid_bounds;
  for (int i = start; i < end; ++i) {
    bounds.extend(bvh_lights[i].second.bounds);
    centroid_bounds.extend(bvh_lights[i].second.Centroid());
  }

  // Choose split dimension and bucket using SAH.
  constexpr int kNumBuckets = 12;
  float min_cost = std::numeric_limits<float>::infinity();
  int min_cost_split_bucket = -1;
  int min_cost_split_dim = -1;

  for (int dim = 0; dim < 3; ++dim) {
    if (centroid_bounds.max()[dim] == centroid_bounds.min()[dim]) {
      continue;
    }

    // Initialize buckets.
    light_tree_internal::LightBounds bucket_bounds[kNumBuckets];

    for (int i = start; i < end; ++i) {
      Eigen::Vector3f pc = bvh_lights[i].second.Centroid();
      float offset = (pc[dim] - centroid_bounds.min()[dim]) /
                     (centroid_bounds.max()[dim] - centroid_bounds.min()[dim]);
      int b = static_cast<int>(kNumBuckets * offset);
      b = std::clamp(b, 0, kNumBuckets - 1);
      bucket_bounds[b] =
          light_tree_internal::Union(bucket_bounds[b], bvh_lights[i].second);
    }

    // Compute costs for each split.
    for (int i = 0; i < kNumBuckets - 1; ++i) {
      light_tree_internal::LightBounds b0, b1;
      for (int j = 0; j <= i; ++j) {
        b0 = light_tree_internal::Union(b0, bucket_bounds[j]);
      }
      for (int j = i + 1; j < kNumBuckets; ++j) {
        b1 = light_tree_internal::Union(b1, bucket_bounds[j]);
      }

      float cost = light_tree_internal::EvaluateCost(b0, bounds, dim) +
                   light_tree_internal::EvaluateCost(b1, bounds, dim);
      if (cost > 0.0f && cost < min_cost) {
        min_cost = cost;
        min_cost_split_bucket = i;
        min_cost_split_dim = dim;
      }
    }
  }

  // Partition lights.
  int mid;
  if (min_cost_split_dim == -1) {
    // No valid split found, just split in half.
    mid = (start + end) / 2;
  } else {
    auto* pmid = std::partition(
        bvh_lights.data() + start, bvh_lights.data() + end,
        [&](const std::pair<int, light_tree_internal::LightBounds>& l) {
          Eigen::Vector3f pc = l.second.Centroid();
          float offset = (pc[min_cost_split_dim] -
                          centroid_bounds.min()[min_cost_split_dim]) /
                         (centroid_bounds.max()[min_cost_split_dim] -
                          centroid_bounds.min()[min_cost_split_dim]);
          int b = static_cast<int>(kNumBuckets * offset);
          b = std::clamp(b, 0, kNumBuckets - 1);
          return b <= min_cost_split_bucket;
        });
    mid = static_cast<int>(pmid - bvh_lights.data());
    if (mid == start || mid == end) {
      mid = (start + end) / 2;
    }
  }

  // Allocate interior node and recursively build children.
  int node_index = static_cast<int>(nodes_.size());
  nodes_.push_back(LightBVHNode());  // Placeholder.

  auto [child0_idx, child0_bounds] =
      BuildBVH(bvh_lights, start, mid, bit_trail, depth + 1);
  auto [child1_idx, child1_bounds] =
      BuildBVH(bvh_lights, mid, end, bit_trail | (1u << depth), depth + 1);

  // Initialize interior node.
  light_tree_internal::LightBounds lb =
      light_tree_internal::Union(child0_bounds, child1_bounds);
  nodes_[node_index].light_bounds = lb;
  nodes_[node_index].child_or_light_index = child1_idx;
  nodes_[node_index].is_leaf = false;

  return {node_index, lb};
}

SampledLight LightTree::Sample(const Eigen::Vector3f& p,
                               const Eigen::Vector3f& n, float u) const {
  if (nodes_.empty()) {
    return {-1, 0.0f};
  }

  int node_index = 0;
  float pmf = 1.0f;

  while (true) {
    const LightBVHNode& node = nodes_[node_index];
    if (node.is_leaf) {
      return {node.child_or_light_index, pmf};
    }

    // Compute importances of children.
    const LightBVHNode& child0 = nodes_[node_index + 1];
    const LightBVHNode& child1 = nodes_[node.child_or_light_index];

    float ci0 = light_tree_internal::Importance(child0.light_bounds, p, n);
    float ci1 = light_tree_internal::Importance(child1.light_bounds, p, n);

    if (ci0 == 0.0f && ci1 == 0.0f) {
      return {-1, 0.0f};
    }

    // Sample child based on importance.
    float p0 = ci0 / (ci0 + ci1);
    if (u < p0) {
      // Go to child 0.
      pmf *= p0;
      u = u / p0;
      node_index = node_index + 1;
    } else {
      // Go to child 1.
      pmf *= (1.0f - p0);
      u = (u - p0) / (1.0f - p0);
      node_index = node.child_or_light_index;
    }
  }
}

float LightTree::Pdf(const Eigen::Vector3f& p, const Eigen::Vector3f& n,
                     int light_index) const {
  if (nodes_.empty() || light_index < 0 ||
      light_index >= static_cast<int>(light_to_bit_trail_.size())) {
    return 0.0f;
  }

  uint32_t bit_trail = light_to_bit_trail_[light_index];
  int node_index = 0;
  float pmf = 1.0f;

  while (true) {
    const LightBVHNode& node = nodes_[node_index];
    if (node.is_leaf) {
      return pmf;
    }

    // Compute importances of children.
    const LightBVHNode& child0 = nodes_[node_index + 1];
    const LightBVHNode& child1 = nodes_[node.child_or_light_index];

    float ci0 = light_tree_internal::Importance(child0.light_bounds, p, n);
    float ci1 = light_tree_internal::Importance(child1.light_bounds, p, n);

    if (ci0 + ci1 == 0.0f) {
      return 0.0f;
    }

    // Follow bit trail.
    if (bit_trail & 1) {
      // Go to child 1.
      pmf *= ci1 / (ci0 + ci1);
      node_index = node.child_or_light_index;
    } else {
      // Go to child 0.
      pmf *= ci0 / (ci0 + ci1);
      node_index = node_index + 1;
    }
    bit_trail >>= 1;
  }
}

}  // namespace sh_baker
