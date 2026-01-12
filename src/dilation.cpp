#include "dilation.h"

#include "sh_coeffs.h"

namespace sh_baker {

template <typename T>
void Dilate(int width, int height, std::vector<T>& pixels,
            std::vector<uint8_t>& mask, int passes) {
  if (pixels.empty() || mask.empty() || pixels.size() != mask.size()) return;

  std::vector<T> next_pixels = pixels;
  std::vector<uint8_t> next_mask = mask;

  for (int p = 0; p < passes; ++p) {
    // Current state is pixels/mask. Write to next_pixels/next_mask.

    // Reset next buffers to current state
    // Actually we can accumulate.
    // Ideally we want to fill 'invalid' pixels that have 'valid' neighbors.
    // And mark them as valid for the *next* pass (if we want propagation).

    bool changed = false;

    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        int idx = y * width + x;
        if (mask[idx]) {
          // Already valid, keep it.
          continue;
        }

        // It is invalid. Check neighbors.
        T accum = T();
        // Initialize zero? SHCoeffs constructor does zero.

        int count = 0;

        // 3x3 window
        for (int dy = -1; dy <= 1; ++dy) {
          for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;

            int nx = x + dx;
            int ny = y + dy;

            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
              int nidx = ny * width + nx;
              if (mask[nidx]) {
                accum += pixels[nidx];
                count++;
              }
            }
          }
        }

        if (count > 0) {
          next_pixels[idx] = accum * (1.0f / count);
          next_mask[idx] = 1;  // Mark as valid for next pass/iteration?
          // If we update 'mask' immediately, order of traversal matters
          // (directional bias). Standard way is read from 'mask', write to
          // 'next_mask'.
          changed = true;
        }
      }
    }

    pixels = next_pixels;
    mask = next_mask;

    if (!changed) break;
  }
}

template void Dilate<SHCoeffs>(int width, int height,
                               std::vector<SHCoeffs>& pixels,
                               std::vector<uint8_t>& mask, int passes);
template void Dilate<float>(int width, int height, std::vector<float>& pixels,
                            std::vector<uint8_t>& mask, int passes);
template void Dilate<uint8_t>(int width, int height,
                              std::vector<uint8_t>& pixels,
                              std::vector<uint8_t>& mask, int passes);

}  // namespace sh_baker
