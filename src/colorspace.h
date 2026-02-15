#ifndef SH_BAKER_SRC_COLORSPACE_H
#define SH_BAKER_SRC_COLORSPACE_H

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace sh_baker {

inline float SRGBToLinear(uint8_t x) {
  float v = x / 255.0f;
  if (v <= 0.04045f) return v / 12.92f;
  return std::pow((v + 0.055f) / 1.055f, 2.4f);
}

inline uint8_t LinearToSRGB(float x) {
  if (x <= 0.0031308f)
    return static_cast<uint8_t>(std::rint(x * 12.92f * 255.0f));
  return static_cast<uint8_t>(
      std::rint((1.055f * std::pow(x, 1.0f / 2.4f) - 0.055f) * 255.0f));
}

inline float Luminance(float r, float g, float b) {
  return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

}  // namespace sh_baker

#endif  // SH_BAKER_SRC_COLORSPACE_H