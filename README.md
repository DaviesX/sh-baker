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

Scene conventions:
- Lights come from `KHR_lights_punctual`. Intensities are divided by 200 on load (lux to the baker's radiance units), so a directional light of intensity 200 gives an irradiance of 1 at normal incidence; the diffuse BRDF is albedo / pi.
- The sky is an equirectangular image named by the glTF's top-level `extras.skybox` (`.hdr` for HDR, otherwise 8-bit), resolved relative to the glTF. Direction `(sin t sin p, cos t, sin t cos p)` maps to `u = p / 2pi` (u = 0 is +Z, u = 0.25 is +X) and `v = t / pi` from the top. Without one, a Preetham sky is built from the brightest directional light. Rays that see the sky only feed the environment-visibility texture; the sky term is applied at render time as sky colour times visibility.
- A primitive without a material is a pure occluder: it blocks light but receives no lightmap chart.
- Alpha is coverage: the alpha channel of the material's albedo, which is the base colour texture or, with `SH_material_layers`, the stack's composited coverage (nearest texel; opaque without an alpha channel). Indirect rays and direct-light shadow rays both pass through a surface with weight 1 - alpha, so fences and leaf cards cast shadows with holes. A shadow ray's light is scaled by the product of 1 - alpha over the surfaces between the point and the light. Opaque surfaces and pure occluders block it, as do more than 64 partially covered crossings. `alphaMode` and `alphaCutoff` are ignored.

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

## Library: q3map2 input

q3map2 links `libsh_baker.so` and includes headers from `src/` to bake its own luxels. Its contract with the library:

- **Materials.** `MaterialLayers` in `material.h` is sh-baker's own form of a Quake 3 stage stack. Its parts are the surface blend, the cull mode, the base layer and one `MaterialLayer` per stage, holding the texture and animMap frame sources, the blend factors, the rgbGen and the tcMods. The glTF loader and saver translate it to and from the `SH_material_layers` extension, and other loaders fill it directly.
- **Surfaces.** `AddQ3Map2Surface` (`loader_q3map2.h`) appends one `Q3Map2Surface` (positions, normals, texture UVs, indices in q3map2's clockwise winding, and a material, negative for a pure occluder) as tracing geometry.
  - It aborts with a `CHECK` on input only a caller bug can produce: missing positions, counts that do not match, indices or a material out of range, or non-finite values.
  - It repairs what real content produces: it flips the winding to counter-clockwise, normalizes normals and rebuilds zero ones from the faces, and drops degenerate triangles.
  - It gives a pure occluder without texture UVs UVs from its world-space XY.
  - It generates tangents the way the glTF loader does: MikkTSpace from the given texture UVs, and the geometric fallback basis for a surface without them.
  - Add every surface before creating lights and building the BVH; a surface added after an area light aborts.
- **Headers.** `loader_q3map2.h`, `baker.h` and `side_file.h` need only `src` and Eigen on the include path, and they compile under `-fno-exceptions -fno-rtti`. `scene.h` declares Embree's two handle types itself.
- **Baking.** `BakeSHLightMap` bakes any list of points without rasterizing. Lay N points out as an N x 1 buffer (`RasterConfig` width N, height 1). Each point needs a position, a unit normal, a unit tangent perpendicular to it with `w` of +1 or -1, and `material_id >= 0`. Result i belongs to point i.
- **Progress and cancel.** `BakeConfig` takes optional callbacks. Progress and cancel callbacks run on bake worker threads, so keep them short: they must not call back into sh-baker, run TBB work, or block waiting on the bake. `done_fn` runs once on the calling thread after the bake, including empty or cancelled bakes. The library writes no progress to stdout itself; `sh_baker_main` prints `[Baking] N%` through `progress_fn`.
  - `progress_fn(done, total)` reports points finished, counting points skipped as invalid. Calls never overlap, `done` strictly increases, and there is at most one call per whole percent. A bake of at least one point that is not cancelled ends with `(total, total)`.
  - `BakeConfig` and `BakeResult` are shared between q3map2 and the library by layout, and there is no ABI versioning. A change to either struct, such as adding these callbacks, changes that layout, so rebuild q3map2 against the same headers whenever `libsh_baker.so` moves to a new commit.
  - `cancel_fn()` is polled before each point in the bake loop. Polls never overlap, so it need not be thread-safe, and after one returns true there are no more. No further point starts, points in progress finish, and the call returns with `BakeResult::cancelled` set. Every baked point then holds exactly what a full bake gives it, and the rest keep the not-baked marker (`SHCoeffs(-1)`, environment visibility -1).

None of this changes the baked output: on the same scene and arguments, `sh_baker_main` writes the same lightmaps. The saved glTF differs in one case: a material-less primitive read without `TEXCOORD_0` keeps no texture UVs, so it is written without `TEXCOORD_0` instead of with zero UVs.

## Library: SH side file

`side_file.h` defines the SH side file, which carries a level's baked SH lightmap pages and its solid hull mesh from q3map2 to `renderer_sh`. q3map2 chooses the pages, lays out its luxels in them and builds the hulls. sh-baker only checks, encodes and decodes them, and stores everything as given apart from the value encoding.

- **Contents.** An ordered list of pages, each with its own width and height (both at least 1), and one hull mesh. Either may be empty. Each luxel holds the nine SH coefficients as RGB, in the order L0, L1m1, L10, L11, L2m2, L2m1, L20, L21, L22, and the environment visibility in [0, 1]. The basis is real SH with no Condon–Shortley phase, for a unit direction (x, y, z) in the caller's world frame pointing towards the incoming light: L0 = 0.282095; L1m1, L10, L11 = 0.488603 · (y, z, x); L2m2 = 1.092548 xy; L2m1 = 1.092548 yz; L20 = 0.315392 (3z² − 1); L21 = 1.092548 xz; L22 = 0.546274 (x² − y²). Each coefficient is the projection of the incident radiance, ∫ L_i(ω) Y(ω) dω over the hemisphere above the surface, not convolved with the cosine lobe. `SHCoeffs::irradiance` is not stored, and reads back as zero.
- **No "not baked" value.** The writer refuses the bake's marker (`SHCoeffs(-1)`, visibility -1), so fill luxels the bake skipped (with zeros or by dilation) before writing.
- **Byte layout, version 1.** Little-endian, with no padding beyond what is listed, ending exactly after the last section:

  | Section | Contents |
  | --- | --- |
  | Header, 24 bytes | `S`,`H`,`S`,`F`; then `uint32` version (1), page count, hull count, hull vertex count, hull index count |
  | Each page | `uint32` width, `uint32` height; 9 coefficient planes of width × height RGB `binary16` triples; 1 visibility plane of width × height `binary16` values. Luxels run x fastest, then y, from row 0. A page takes 8 + 56 × width × height bytes |
  | Hull ranges | per hull, `uint32` first index and `uint32` index count |
  | Hull vertices | per vertex, `float32` x, y, z |
  | Hull indices | one `uint32` per index |

- **Version rule.** The writer always writes version 1, and the reader accepts only version 1. Any change to the layout or the meaning of the values takes a new version.
- **Values.** Coefficients and visibility are IEEE `binary16`, rounded to nearest with ties to even. Finite values beyond ±65504 are clamped to ±65504, so very bright luxels clip. Positions are `float32`, stored exactly.
- **Hulls.** The un-shrunk closed hulls of the casting brushes, in the caller's world space and units. Triangles face outward, with front faces clockwise when seen from outside (q3map2's winding). Each hull is a contiguous index range, and the ranges cover the index list in order. The file neither checks nor changes closure, convexity or winding.
- **Writing.** `SaveSHSideFile` aborts with a `CHECK` on input only a caller bug can produce: page sizes below 1, luxel counts that differ from width × height, non-finite values, a visibility outside [0, 1] or a negative L0 component, hull ranges that are empty, not multiples of 3 or not contiguous, an index out of range, or a count beyond `uint32`. It writes `<path>.tmp` and renames it over `path`. An I/O failure returns false, removes the temporary file and leaves an existing file unchanged. Two writers to one path at once are not supported.
- **Reading.** `LoadSHSideFile` returns `std::nullopt`, logging the reason, for a file it cannot open, a bad magic, an unknown version, a size other than its header and pages imply, a page side of 0 or above 2^31 - 1, or values the writer would refuse. It checks the sizes against the file before allocating any page or hull storage, and never aborts on file contents.
- **Inspection.** `SaveSHSidePageL0Hdr` writes one page's L0 RGB as a Radiance `.hdr`, with row 0 first and negative components written as 0. It refuses file names starting with `lm_` in any letter case, because rend2 loads `maps/<name>/lm_*.hdr` on its own.
