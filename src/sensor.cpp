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

Eigen::Vector3f ComputeMean(const SHCoeffs& coeffs) {
  // L1 bands encode the mean direction scaled by brightness.
  // Y1,1  (index 3) ~ x
  // Y1,-1 (index 1) ~ y
  // Y1,0  (index 2) ~ z
  // SH basis factor: sqrt(3/4pi)

  // Actually ComputeLuminance takes SHCoeffs which has 9 bands.
  // We want luminance of the specific band vector.

  auto Lum = [](const Eigen::Vector3f& v) {
    return v.x() * 0.2126f + v.y() * 0.7152f + v.z() * 0.0722f;
  };

  // Indices:
  // 1: Y1,-1 (y)
  // 2: Y1,0  (z)
  // 3: Y1,1  (x)

  float My = Lum(coeffs.coeffs[1]);
  float Mz = Lum(coeffs.coeffs[2]);
  float Mx = Lum(coeffs.coeffs[3]);

  // Note: The SH coefficients are integral(L * Y).
  // Y_1x = sqrt(3/4pi) * x
  // So coeff_x = integral(L * sqrt(3/4pi) * x)
  // Mean vector = first moment.
  // We just return the vector composed of these luminances.
  // The scale matters relative to Covariance (L0/L2).
  // L0 ~ 1/sqrt(4pi).
  // If we just use them as is, they are consistent moments.

  return Eigen::Vector3f(Mx, My, Mz);
}

Eigen::Vector3f SampleAngularGaussian(const Eigen::Matrix3f& cov,
                                      const Eigen::Vector3f& mean,
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

  // 6. Add Mean and Normalize (Projected Normal)
  Eigen::Vector3f sample_unnormalized = mean + d;

  if (sample_unnormalized.squaredNorm() < 1e-8f) {
    return sensor_internal::SampleHemisphereUniform(rng);
  }

  return sample_unnormalized.normalized();
}

float PdfAngularGaussian(const Eigen::Matrix3f& cov,
                         const Eigen::Vector3f& mean,
                         const Eigen::Vector3f& dir) {
  // Projected Normal Distribution PDF
  // P(w) = ... integral over r ...
  // Ref: https://en.wikipedia.org/wiki/Projected_normal_distribution (usually
  // 2D on circle, generalized to 3D sphere)

  // Let X ~ N(mean, cov).
  // We want density of W = X / |X|.

  // Inverse covariance (Precision matrix)
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigensolver(cov);
  Eigen::Vector3f eigenvalues = eigensolver.eigenvalues().cwiseMax(1e-4f);
  Eigen::Matrix3f eigenvectors = eigensolver.eigenvectors();
  float det_cov = eigenvalues.prod();
  Eigen::Matrix3f prec = eigenvectors *
                         eigenvalues.cwiseInverse().asDiagonal() *
                         eigenvectors.transpose();

  // Constants
  // f(x) = (2pi)^-1.5 * det(cov)^-0.5 * exp(-0.5 * (x-mean)' prec (x-mean))
  // x = r * dir
  // Exponent: -0.5 * (r^2 * dir' prec dir - 2 * r * dir' prec mean + mean' prec
  // mean) Let a = dir' prec dir Let b = dir' prec mean  (Note: factor 2 cancels
  // with 0.5) Let c = mean' prec mean Arg = -0.5 * a * r^2 + b * r - 0.5 * c

  float a = dir.dot(prec * dir);
  float b = dir.dot(prec * mean);
  float c = mean.dot(prec * mean);

  if (a <= 1e-6f) return 0.0f;  // Degenerate

  // Helper for integral \int_0^inf r^2 exp(-0.5*a*r^2 + b*r) dr
  // This doesn't have a trivial closed form without Phi (CDF).
  // Standard approximation or exact form is needed.
  // Exact form involves Normal CDF.

  // Alternative: Numerical integration? Too slow.

  // Implementation of exact analytic integral:
  // I = \int_0^\infty r^2 e^{-\frac{1}{2}a r^2 + b r} dr
  // Let s = b / sqrt(a)
  // I = \frac{1}{a^{1.5}} \frac{\sqrt{2\pi}}{a} e^{s^2/2} \dots wait.

  // Let's rely on standard formulation for Projected Gaussian PDF using shadow
  // boundaries approx? No, let's implement the analytic integral
  // component-wise. Or just assume b is large positive (dir aligned with mean)
  // -> Gaussian-ish?

  // Let's use a robust approximation:
  // If b > 0 and b^2 >> a, roughly Gaussian around b/a.
  //
  // Let's try to implement the exact formula carefully.
  // integral_{0}^{\infty} r^2 e^{-0.5 a r^2 + b r} dr
  // = d^2/db^2 integral_{0}^{\infty} e^{-0.5 a r^2 + b r} dr
  //
  // Let I0(b) = integral_{0}^{\infty} e^{-0.5 a r^2 + b r} dr
  // = sqrt(2*pi/a) * exp(b^2 / (2a)) * Phi(b / sqrt(a))
  // where Phi is standard normal CDF.
  //
  // We need I2(b) = d^2/db^2 I0(b).

  float inv_sqrt_a = 1.0f / std::sqrt(a);
  float s = b * inv_sqrt_a;  // "signal to noise"

  // Phi(s)
  auto Phi = [](float x) { return 0.5f * std::erfc(-x * M_SQRT1_2); };

  // pdf_normal(s)
  // auto pdf_n = [](float x) {
  //     return M_2_SQRTPI * M_SQRT1_2 * 0.5f * std::exp(-0.5f * x * x);
  // };

  float phi_s = Phi(s);
  // float n_s = pdf_n(s);

  // Terms from derivation of I2
  // exp term E = exp(0.5 * s^2)
  // But wait, the full expression has exp(-0.5 * c).
  // Combined exponent: exp(-0.5 * c + 0.5 * s^2) = exp(-0.5 * (c - b^2/a))

  float exponent_factor = -0.5f * (c - b * b / a);
  // If exponent is too small, 0.
  float base_exp_shifted = std::exp(exponent_factor);

  // The polynomial part from derivatives:
  // I2 = ...
  // Let's look up the result for Projected Normal (Isotropic) and generalize.
  // Actually, standard reference:
  // p(w) = constant * ( ... )
  // Constant K = 1 / ( (2pi)^1.5 * sqrt(det) ) * exp(-0.5*c)
  //
  // Let's use the code structure from similar implementations (e.g.
  // Mitsuba/PBRT if available? No).
  //
  // Let's trust the logic:
  // I0 = sqrt(2pi/a) * exp(s^2/2) * Phi(s)
  // I1 = d/db I0 = s/sqrt(a) * I0 + 1/a * exp(s^2/2) * n_s * sqrt(2pi/a)?
  // Easier: I1 = (b/a) * I0 + (1/a) * exp(b^2/2a - 0.5 * a * 0) wait.
  //
  // Result for I2:
  // I2 = ( (s^2 + 1)/a ) * I0 + ( s / (a * sqrt(a)) ) * exp(s^2/2) * sqrt(2*pi)
  // * n_s ... No.

  // Let's simplify.
  // p(w) ~ 1/C^1.5 approx?
  // User asked for Mean Direction support.
  // If we can't do exact PDF, maybe we rely on MIS to balance it?
  // But we need a plausible PDF.
  // Let's implement the "Approximate" Projected Gaussian which treats it as
  // angular gaussian aligned to mean? No, let's try the formula: p(w) =
  // \frac{e^{-0.5 c}}{2 \pi \sqrt{det} a^{1.5}} \left( s \cdot \sqrt{2\pi}
  // e^{s^2/2} \Phi(s) (s^2+1) + (s^2+1)? No... \right)

  // Let's stick to a basic approximation if exact is hard:
  // Treat as Gaussian in tangent plane?
  //
  // RE-DERIVATION:
  // I2 = Integral r^2 exp(-0.5 a r^2 + b r) dr
  //    = (1/a) * Integral r * (a r - b + b) * exp(...) dr
  //    = (1/a) * [ -r exp ] + (1/a) * Integral exp(...) dr + (b/a) * Integral r
  //    exp(...) dr = 0 + (1/a) I0 + (b/a) I1
  //
  // I1 = Integral r exp(-0.5 a r^2 + b r) dr
  //    = (1/a) * Integral (ar - b + b) exp files
  //    = (1/a) * [-exp] + (b/a) I0
  //    = (1/a) + (b/a) I0
  //
  // So I2 = (1/a) I0 + (b/a) [ 1/a + (b/a) I0 ]
  //       = (1/a + b^2/a^2) I0 + b/a^2
  //       = ( (a + b^2)/a^2 ) I0 + b/a^2
  //
  // I0 = sqrt(2pi/a) * exp(b^2/2a) * Phi(b/sqrt(a))
  //
  // Total PDF factor = (2pi)^-1.5 * det^-0.5 * exp(-0.5 c) * I2
  //
  // Let's implement this!

  float term1 = (a + b * b) / (a * a);
  float term2 = b / (a * a);

  float I0_factor = std::sqrt(2.0f * M_PI / a) * phi_s;
  // Note: we grouped exp(b^2/2a) with exp(-0.5c) earlier into base_exp_shifted.

  // Combined I2 with the exp(-0.5 c)
  // Result = K * ( term1 * I0_factor * base_exp(with b^2/2a) + term2 *
  // exp(-0.5c) )

  float const_K = 1.0f / (std::pow(2.0f * M_PI, 1.5f) * std::sqrt(det_cov));
  float exp_c = std::exp(-0.5f * c);

  float val = const_K * (term1 * I0_factor * base_exp_shifted + term2 * exp_c);
  return std::max(0.0f, val);
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
  Eigen::Vector3f mean(0, 0, 0);

  if (sensor.sample_count >= kMinSamples) {
    SHCoeffs indirect_estimate =
        sensor.sh_indirect_coeffs_sum / static_cast<float>(sensor.sample_count);
    cov = sensor_internal::ComputeCovariance(indirect_estimate);
    mean = sensor_internal::ComputeMean(indirect_estimate);
  } else {
    // Identity covariance for uniform-ish behavior if we forced it,
    // but we only use cov if we are in the guided branch or evaluating PDF.
    cov = Eigen::Matrix3f::Identity();
    // Mean 0.
  }

  if (use_uniform) {
    dir_local = sensor_internal::SampleHemisphereUniform(rng);
  } else {
    dir_local = sensor_internal::SampleAngularGaussian(cov, mean, rng);
    // If the sampled direction is below the horizon (z < 0), flip it?
    // With mean shifting, it might be heavily biased down if light is below?
    // But we are on a surface. Light should be from above?
    // If we flip, we need to account for it in PDF.
    // Let's assume reflection logic:
    if (dir_local.z() < 0.0f) {
      dir_local.z() *= -1.0f;  // Mirror at horizon
      dir_local.normalize();
    }
  }

  // 4. Calculate Combined PDF (Balance Heuristic)
  // PDF = temperature * PDF_Uniform + (1 - temperature) * PDF_Guided

  // PDF Uniform = 1 / (2 * PI) for hemisphere
  float pdf_uniform = 1.0f / (2.0f * M_PI);

  // PDF Guided
  float pdf_guided = 0.0f;
  if (sensor.sample_count >= kMinSamples) {
    pdf_guided = sensor_internal::PdfAngularGaussian(cov, mean, dir_local);

    // Correction for hemisphere folding (Mirroring)
    // p_hemi(w) = p(w) + p(mirror(w))
    // mirror(w) = (x, y, -z) if we flipped z.
    // Since we forced z > 0, we sum densities.
    Eigen::Vector3f dir_mirror = dir_local;
    dir_mirror.z() *= -1.0f;
    pdf_guided += sensor_internal::PdfAngularGaussian(cov, mean, dir_mirror);
  } else {
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
