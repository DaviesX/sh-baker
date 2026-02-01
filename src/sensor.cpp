#include "sensor.h"

#include <Eigen/Eigenvalues>
#include <cmath>
#include <optional>
#include <random>

#include "occlusion.h"
#include "sh_coeffs.h"

namespace sh_baker {
namespace sensor_internal {

Eigen::Vector3f SampleHemisphereUniform(std::mt19937& rng) {
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  float u1 = dist(rng);
  float u2 = dist(rng);

  float r = std::sqrt(1.0f - u1 * u1);
  float phi = 2.0f * M_PI * u2;
  return Eigen::Vector3f(r * std::cos(phi), r * std::sin(phi), u1);
}

}  // namespace sensor_internal

namespace sensor_internal {

float ComputeLuminance(const SHCoeffs& coeffs) {
  // Extract DC component (band 0) and convert to luminance.
  // Band 0 is the average color * constant.
  // We use the RGB vector of the 0-th coefficient directly.
  const Eigen::Vector3f& rgb = coeffs.coeffs[0];
  // Rec. 709 constants
  return rgb.x() * 0.2126f + rgb.y() * 0.7152f + rgb.z() * 0.0722f;
}

Eigen::Matrix3f ComputeCovariance(const SHCoeffs& coeffs) {
  // Convert SH coefficients to luminance first
  // We want a scalar field to define the probability, so we take luminance of
  // the RGB coefficients.
  // Note: This is an approximation. We construct the covariance of the
  // "Intensity" distribution.

  float L0 = ComputeLuminance(coeffs);

  float L2_2 = coeffs.coeffs[8].dot(
      Eigen::Vector3f(0.2126f, 0.7152f, 0.0722f));  // x^2 - y^2
  float L2_1 =
      coeffs.coeffs[7].dot(Eigen::Vector3f(0.2126f, 0.7152f, 0.0722f));  // zx
  float L2_0 = coeffs.coeffs[6].dot(
      Eigen::Vector3f(0.2126f, 0.7152f, 0.0722f));  // 3z^2 - 1
  float L2_n1 =
      coeffs.coeffs[5].dot(Eigen::Vector3f(0.2126f, 0.7152f, 0.0722f));  // yz
  float L2_n2 =
      coeffs.coeffs[4].dot(Eigen::Vector3f(0.2126f, 0.7152f, 0.0722f));  // xy

  // Construct Covariance Matrix.
  // We use the heuristic mapping:
  // L0 provides the isotropic scale (base variance).
  // L2 terms provide the deviatoric part.
  // We want the covariance to be proportional to the intensity along axes.
  //
  // M = Identity * L0 + Scale * Tensor(L2)
  //
  // tensor_xx = -L2_0/sqrt(3) + L2_2
  // tensor_yy = -L2_0/sqrt(3) - L2_2
  // tensor_zz = 2*L2_0/sqrt(3)
  // tensor_xy = L2_n2
  // tensor_yz = L2_n1
  // tensor_zx = L2_1
  //
  // Scaling factors are technically important but generic weighting works for
  // guiding. We use a constant to tune the strength of anisotropy.

  Eigen::Matrix3f cov;
  float c0 = L2_0;  // ~ 3z^2-1
  float c2 = L2_2;  // ~ x^2-y^2

  // Coefficients for the diagonal elements derived from SH L2 definition
  // Y2,0 ~ 3z^2 - 1
  // Y2,2 ~ x^2 - y^2
  // We assume a linear mapping add to identity.
  // The L0 term ensures positivity (hopefully).

  float k = 1.0f;  // Tuning constant for L2 strength

  // Diagonal
  // To match 3z^2 - 1: -1, -1, 2
  // To match x^2 - y^2: 1, -1, 0
  cov(0, 0) = L0 - k * c0 + k * c2;
  cov(1, 1) = L0 - k * c0 - k * c2;
  cov(2, 2) = L0 + 2.0f * k * c0;

  // Off-diagonal
  cov(0, 1) = cov(1, 0) = k * L2_n2;  // xy
  cov(1, 2) = cov(2, 1) = k * L2_n1;  // yz
  cov(2, 0) = cov(0, 2) = k * L2_1;   // zx

  return cov;
}

Eigen::Vector3f SampleAngularGaussian(const Eigen::Matrix3f& cov,
                                      std::mt19937& rng) {
  // 1. Eigen Decomposition
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigensolver(cov);
  Eigen::Vector3f eigenvalues = eigensolver.eigenvalues();
  Eigen::Matrix3f eigenvectors = eigensolver.eigenvectors();

  // 2. Clamp eigenvalues to epsilon to prevent collapse
  float kEpsilon = 1e-4f;
  eigenvalues = eigenvalues.cwiseMax(kEpsilon);

  // 3. Sample standard normal vector
  std::normal_distribution<float> dist(0.0f, 1.0f);
  Eigen::Vector3f u(dist(rng), dist(rng), dist(rng));

  // 4. Scale by sqrt(eigenvalues) (Standard Deviation)
  // We want to stretch the distribution along the principal axes.
  Eigen::Vector3f v;
  v.x() = u.x() * std::sqrt(eigenvalues.x());
  v.y() = u.y() * std::sqrt(eigenvalues.y());
  v.z() = u.z() * std::sqrt(eigenvalues.z());

  // 5. Rotate to align with covariance
  Eigen::Vector3f d = eigenvectors * v;

  // 6. Normalize to get direction
  return d.normalized();
}

float PdfAngularGaussian(const Eigen::Matrix3f& cov,
                         const Eigen::Vector3f& dir) {
  // 1. Reconstruct eigenvalues/vectors for the inverse/det calculation.
  //    (Alternatively we could inverse `cov` directly if well-conditioned).
  //    Let's using SelfAdjointEigenSolver again to be consistent with sampling
  //    and safe against non-pd.
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigensolver(cov);
  Eigen::Vector3f eigenvalues = eigensolver.eigenvalues();
  Eigen::Matrix3f eigenvectors = eigensolver.eigenvectors();

  float kEpsilon = 1e-4f;
  eigenvalues = eigenvalues.cwiseMax(kEpsilon);

  // 2. Compute Determinant and Inverse from eigens
  float det = eigenvalues.prod();
  Eigen::Matrix3f inv_cov = eigenvectors *
                            eigenvalues.cwiseInverse().asDiagonal() *
                            eigenvectors.transpose();

  // 3. Evaluate PDF
  // p(w) = 1 / (4pi * sqrt(det) * (w^T * inv_cov * w)^1.5)

  float term = dir.dot(inv_cov * dir);
  if (term <= 0.0f) return 0.0f;  // Should not happen for PD matrix

  float denominator = 4.0f * M_PI * std::sqrt(det) * std::pow(term, 1.5f);
  if (denominator <= 1e-6f) return 0.0f;

  return 1.0f / denominator;
}

}  // namespace sensor_internal

void AddSample(const SHCoeffs& direct_sh, const SHCoeffs& indirect_sh,
               Sensor* sensor) {
  sensor->sample_count++;
  sensor->sh_direct_coeffs_sum += direct_sh;
  sensor->sh_indirect_coeffs_sum += indirect_sh;

  // L0 luminance (DC component brightness).
  // SH basis 0 is (1/sqrt(4pi)) * Y00. Y00 is constant.
  // We can just use the norm of the first coefficient vector (RGB).
  // Or simpler: average RGB.
  // Let's stick to previous implementation or update to use consistent
  // Luminance? Previous: float lum = sample.coeffs[0].norm(); Using Rec 709
  // luminance is consistent with our new code.
  float lum = sensor_internal::ComputeLuminance(direct_sh + indirect_sh);

  // Welford's algorithm
  float delta = lum - sensor->mean_luminance;
  sensor->mean_luminance += delta / sensor->sample_count;
  float delta2 = lum - sensor->mean_luminance;
  sensor->m2_luminance += delta * delta2;
}

SHCoeffs GetEstimation(const Sensor& sensor) {
  if (sensor.sample_count == 0) {
    return SHCoeffs(0);
  }
  return (sensor.sh_direct_coeffs_sum + sensor.sh_indirect_coeffs_sum) /
         static_cast<float>(sensor.sample_count);
}

std::optional<Ray> SampleRay(const Sensor& sensor, std::mt19937& rng,
                             float* pdf) {
  // 1. Check max samples
  if (sensor.sample_count >= sensor.max_samples) {
    return std::nullopt;
  }

  // 2. Check convergence (Adaptive Sampling)
  constexpr int kMinSamples = 16;
  if (sensor.sample_count >= kMinSamples) {
    // Calculate Standard Error of the Mean
    if (sensor.mean_luminance > 1e-3f) {
      float variance = sensor.m2_luminance / (sensor.sample_count - 1);
      float std_dev = std::sqrt(variance);
      float sem = std_dev / std::sqrt((float)sensor.sample_count);

      // Coefficient of Variation of the Mean = SEM / Mean
      // If error is small enough relative to the signal, we stop.
      if (sem < sensor.confidence_threshold * sensor.mean_luminance) {
        return std::nullopt;
      }
    } else {
      // If it's pitch black (or very close), we can stop early too.
      return std::nullopt;
    }
  }

  // 3. MIS Logic
  // Calculate temperature
  // Decay rate schedule: T = exp(-sample_count / decay_rate)
  // At sample_count = max_samples / 2, T = 0.05
  // 0.05 = exp(-(max/2) / decay)
  // ln(0.05) = -(max/2) / decay
  // decay = -(max/2) / ln(0.05)
  float decay_rate =
      -static_cast<float>(sensor.max_samples) / (2.0f * std::log(0.05f));
  float temperature =
      std::exp(-static_cast<float>(sensor.sample_count) / decay_rate);

  // Use uniform sampling initially to build up SH estimate
  if (sensor.sample_count < kMinSamples) {
    temperature = 1.0f;
  }

  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  bool use_uniform = dist(rng) < temperature;

  Eigen::Vector3f dir_local;

  // Need to compute Covariance if guided or for PDF evaluation
  Eigen::Matrix3f cov;
  if (sensor.sample_count >= kMinSamples) {
    SHCoeffs indirect_estimate =
        sensor.sh_indirect_coeffs_sum / static_cast<float>(sensor.sample_count);
    cov = sensor_internal::ComputeCovariance(indirect_estimate);
  } else {
    // Identity covariance for uniform-ish behavior if we forced it,
    // but we only use cov if we are in the guided branch or evaluating PDF.
    cov = Eigen::Matrix3f::Identity();
  }

  if (use_uniform) {
    dir_local = sensor_internal::SampleHemisphereUniform(rng);
  } else {
    dir_local = sensor_internal::SampleAngularGaussian(cov, rng);
    // If the sampled direction is below the horizon (z < 0), flip it or reject?
    // Our sensor is on a surface, we care about the hemisphere.
    // The Guided distribution is spherical.
    // Simple fix: abs(z) ensures upper hemisphere.
    if (dir_local.z() < 0.0f) {
      dir_local = -dir_local;
    }
  }

  // 4. Calculate Combined PDF (Balance Heuristic)
  // PDF = temperature * PDF_Uniform + (1 - temperature) * PDF_Guided

  // PDF Uniform = 1 / (2 * PI) for hemisphere
  float pdf_uniform = 1.0f / (2.0f * M_PI);

  // PDF Guided
  // Note: PdfAngularGaussian is defined on the Sphere.
  // The integral over sphere is 1.
  // If we flipped the direction, the density on the hemisphere is roughly
  // double? Or rather, p_hemi(w) = p_sphere(w) + p_sphere(-w) if we map -w to
  // w. Let's assume symmetric covariance (which it is) -> p(w) = p(-w). So
  // p_hemi(w) = 2 * p_sphere(w).
  float pdf_guided = 0.0f;
  if (sensor.sample_count >= kMinSamples) {
    pdf_guided = sensor_internal::PdfAngularGaussian(cov, dir_local);
    pdf_guided *= 2.0f;  // Correction for hemisphere folding
  } else {
    // If we are in warmup, guided pdf is effectively uniform or we just don't
    // use it. But strictly speaking, if we *could* sample guided, we should
    // evaluate its pdf for the weight. However, temperature is 1.0, so the 2nd
    // term is 0.
    pdf_guided = 0.0f;
  }

  float final_pdf =
      temperature * pdf_uniform + (1.0f - temperature) * pdf_guided;

  if (pdf) {
    *pdf = final_pdf;
  }

  // 5. Generate Ray
  Ray ray;
  ray.tnear = 0.005f;
  ray.tfar = 1e10f;
  ray.origin = sensor.sp.position;

  // Transform to World
  // Calculate bitangent (using w for handedness)
  Eigen::Vector3f bitangent =
      (sensor.sp.normal.cross(sensor.sp.tangent.head<3>()) *
       sensor.sp.tangent.w())
          .normalized();

  ray.direction = sensor.sp.tangent.head<3>() * dir_local.x() +
                  bitangent * dir_local.y() + sensor.sp.normal * dir_local.z();

  return ray;
}

}  // namespace sh_baker
