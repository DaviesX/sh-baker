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
cmake -B build -S .
cmake --build build
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
- **Left Mouse Drag**: Orbit camera.
- **Scroll**: Zoom.
