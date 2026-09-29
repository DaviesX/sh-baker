# sh-baker

Offline Path-Tracer for Baking Spherical Harmonic (SH) Lightmaps.

## Prerequisites

- C++20 compatible compiler
- CMake 3.10+
- Dependencies:
  - Intel Embree 4
  - glog
  - gflags
  - tinygltf (included or fetched)
  - GoogleTest (fetched)

## Building

This project uses CMake. To build:

```bash
cmake -DCMAKE_BUILD_TYPE=Release -B build -S .
cmake --build build --parallel 12
```

Options:
- `-DSH_BAKER_BUILD_TESTS=OFF`: skip the GoogleTest unit tests (GTest is then not required).
- `-DSH_BAKER_BUILD_VISUALIZER=OFF`: skip the OpenGL visualizer (GLFW and OpenGL are then not required).
- `-DCMAKE_PREFIX_PATH=<dir>`: point at an unpacked Embree release (https://github.com/RenderKit/embree/releases) when it is not installed system-wide.

To run the main application:

```bash
./build/sh_baker_main --input data/Sponza/scene.gltf --output out --width 1024 --height 1024 --samples 128 --supersample_scale 3 --luminance_only
```

## Testing

To run the unit tests:

```bash
./build/sh_baker_test
```

## Usage

### Baking

To bake a scene:

```bash
./build/sh_baker --input scene.gltf --output output_dir --width 1024 --height 1024 --samples 128
```

Arguments:
- `--input`: Path to the input glTF file.
- `--output`: Output directory.
- `--width`, `--height`: Lightmap resolution.
- `--samples`: Rays per texel.
- `--bounces`: Light bounces (default 3).
- `--dilation`: Dilation passes (default 0).
- `--split_channels`: If set, outputs 9 separate EXR files for SH coefficients (for Blender Viz).
- `--sky_scale`: If > 0, bake the skybox (or the Preetham sky) into the SH, direct and bounced, scaled by this factor (pre-scale the HDR itself for per-scene levels). By default (0) rays that see the sky only feed the environment-visibility texture and the renderer adds sky colour times visibility.

Scene conventions:
- Lights come from `KHR_lights_punctual`. Intensities are divided by 200 on load (lux to the baker's radiance units), so a directional light of intensity 200 gives an irradiance of 1 at normal incidence; the diffuse BRDF is albedo / pi.
- The sky is an equirectangular image named by the glTF's top-level `extras.skybox` (`.hdr` for HDR, otherwise 8-bit), resolved relative to the glTF. Direction `(sin t sin p, cos t, sin t cos p)` maps to `u = p / 2pi` (u = 0 is +Z, u = 0.25 is +X) and `v = t / pi` from the top. Without one, a Preetham sky is built from the brightest directional light. Rays that see the sky only feed the environment-visibility texture; the sky term is applied at render time as sky colour times visibility.
- A primitive without a material is a pure occluder: it blocks light but receives no lightmap chart.

### Blender Visualization

To visualize the baked lightmap in Blender:

1. Bake with `--split_channels` enabled.
   ```bash
   ./build/sh_baker --input scene.gltf --output out --split_channels
   ```
   This generates `out/lightmap_L0.exr`, `out/lightmap_L1m1.exr`, ...

2. Open Blender and go to the Scripting tab.
3. Open `tools/blender_viz.py`.
4. At the bottom of the script, uncomment and modify the usage line:
   ```python
   create_sh_shader("TargetMaterialName", "/absolute/path/to/out/lightmap.exr")
   ```
   Note: Point to the base name (e.g., `lightmap.exr`), the script will automatically append `_L0.exr` etc.
5. Run the script. It will create a shader node tree in the specified material that reconstructs the SH lighting.

### OpenGL Visualization

To use the standalone OpenGL visualizer:

```bash
./build/visualizer --input output_dir
```

- `--input`: Path to the input folder containing `scene.gltf` and `lightmap_*.exr` files.

Controls:
- **W, A, S, D**: Move Forward, Left, Backward, Right
- **Q, E**: Move Down, Up
- **Left Mouse Drag**: Rotate Camera
- **Scroll**: Adjust Movement Speed

### EXR to PNG Converter

Tool to extract the L0 component (irradiance/diffuse color) from EXR lightmaps and save it as a PNG. Useful for quick debugging or previewing.

Usage:

```bash
./build/sh_cvt --input <directory> [--reinhard]
```

Arguments:

- `--input`: Input directory containing EXR files.
- `--reinhard`: Apply Reinhard tone mapping (`x / (1+x)`) to the output.

This tool will process all `.exr` files in the directory and save `<filename>_L0.png`.
