#ifndef SH_BAKER_SRC_DILATION_H_
#define SH_BAKER_SRC_DILATION_H_

#include <cstdint>
#include <vector>

#include "sh_coeffs.h"

namespace sh_baker {

// Dilates the valid pixels into invalid pixels to fill gaps/seams.
// width, height: dimensions of the texture.
// pixels: The SH coefficient data (modified in place).
// mask: Validity mask (1 = valid, 0 = invalid).
// passes: Number of dilation passes.
void Dilate(int width, int height, std::vector<SHCoeffs>& pixels,
            std::vector<uint8_t>& mask, int passes = 1);

}  // namespace sh_baker

#endif  // SH_BAKER_SRC_DILATION_H_
