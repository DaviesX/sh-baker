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
- `--samples`: Rays per texel (the maximum while adaptive sampling is on).
- `--confidence_threshold`: Adaptive-sampling stop tolerance (default 0.01). After a 64-sample warm-up, a texel stops once the 3-sigma standard error of its L0 luminance falls below this fraction of its mean (or an absolute floor of 1e-4). `<= 0` disables adaptive stopping, so every texel takes exactly `--samples` rays.
- `--bounces`: Light bounces (default 3).
- `--indirect_only`: Skip direct lighting from scene lights and the sky at baked points. Both still contribute through surface bounces.
- `--dilation`: Dilation passes (default 0).
- `--split_channels`: If set, outputs 9 separate EXR files for SH coefficients (for Blender Viz).

Scene conventions:
- Lights come from `KHR_lights_punctual`. Intensities are divided by 200 on load (lux to the baker's radiance units), so a directional light of intensity 200 gives an irradiance of 1 at normal incidence; the diffuse BRDF is albedo / pi.
- The sky is an equirectangular image named by the glTF's top-level `extras.skybox` (`.hdr` for HDR, otherwise 8-bit), resolved relative to the glTF. Direction `(sin t sin p, cos t, sin t cos p)` maps to `u = p / 2pi` (u = 0 is +Z, u = 0.25 is +X) and `v = t / pi` from the top. Without one, a Preetham sky is built from the brightest directional light. The CLI includes direct and bounced sky in a full bake. With `--indirect_only`, it includes only bounced sky. Environment visibility is the cosine-weighted fraction of first rays that escape for surface points and the uniform fraction for full-sphere points. Alpha cut-outs count as blocking for visibility.
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
- **Headers.** `loader_q3map2.h` and `baker.h` need only `src` and Eigen on the include path, and they compile under `-fno-exceptions -fno-rtti`. `scene.h` declares Embree's two handle types itself.
- **Baking.** `BakeSHLightMap` bakes any list of points without rasterizing. Lay N points out as an N x 1 buffer (`RasterConfig` width N, height 1). Each point needs a position, a unit normal, a unit tangent perpendicular to it with `w` of +1 or -1, and `material_id >= 0`. Result i belongs to point i.
  - **Sample count.** `BakeConfig::samples` is a maximum: with the default `confidence_threshold` (0.01) a point may stop adaptively after 64 rays, and a point whose per-sample L0 does not vary (one lit only by unshadowed point lights, say) always stops there. Set `confidence_threshold <= 0` to take exactly `samples` rays at every valid point, for a fixed cost per quality preset.
  - **Full-sphere points.** `SurfacePoint::full_sphere` (default false) makes a point a probe in open space, such as one of q3map2's light-grid points. Its SH is the projection of the radiance arriving from the whole sphere, ∫ L_i(ω) Y(ω) dω, with no cosine. Its rays are uniform over the sphere in world space, weighted 4π instead of 2π, and next-event estimation counts light from every direction, giving the light tree a zero normal. Its normal and tangent are not used, so they may be zero; only `position` and `material_id >= 0` matter. Only the baked point changes: the bounces of its paths shade surfaces as usual. Its `irradiance` has no meaning and should not be used, and its environment visibility is the fraction of rays over the whole sphere that escape. The flag works the same with `indirect_only`, and points of both kinds bake in one call without changing each other's results.
- **Sky light.** When `Scene::environment` is present, a full bake includes direct and bounced sky radiance. An `indirect_only` bake includes sky radiance after at least one surface reflection. Paths that reach the sky through alpha cut-outs alone are direct sky light.
  - The radiance is the function `ProjectEnvironmentToSH` integrates: the equirect texel, bilinearly filtered (wrapping in longitude, clamped in latitude, channels 0-2, values as stored), or the Preetham sky's capped Perez value, black below the horizon.
  - `Environment::to_environment` rotates world directions into the environment's Y-up frame before the lookup (identity by default). A Z-up scene such as q3map2's passes a rotation mapping world +Z to +Y. It doesn't change `sh_coeffs`.
  - The sky is gathered by escape only, never by next-event estimation. An environment must not contain a sun that a directional light already supplies, or that sun is counted twice.
  - Environment visibility is recorded the same way with or without sky light. With `confidence_threshold <= 0` it is bit-identical; with adaptive stopping, sky radiance can change how many rays a point takes.
- **Progress and cancel.** `BakeConfig` takes optional callbacks. Progress and cancel callbacks run on bake worker threads, so keep them short: they must not call back into sh-baker, run TBB work, or block waiting on the bake. `done_fn` runs once on the calling thread after the bake, including empty or cancelled bakes. The library writes no progress to stdout itself; `sh_baker_main` prints `[Baking] N%` through `progress_fn`.
  - `progress_fn(done, total)` reports points finished, counting points skipped as invalid. Calls never overlap, `done` strictly increases, and there is at most one call per whole percent. A bake of at least one point that is not cancelled ends with `(total, total)`.
  - `BakeConfig` and `BakeResult` are shared between q3map2 and the library by layout, and there is no ABI versioning. A change to either struct, such as adding these callbacks, changes that layout, so rebuild q3map2 against the same headers whenever `libsh_baker.so` moves to a new commit.
  - `cancel_fn()` is polled before each point in the bake loop. Polls never overlap, so it need not be thread-safe, and after one returns true there are no more. No further point starts, points in progress finish, and the call returns with `BakeResult::cancelled` set. Every baked point then holds exactly what a full bake gives it, and the rest keep the not-baked marker (`SHCoeffs(-1)`, environment visibility -1).
- **Light model.** `Light` carries the editor's physical light model. Every field defaults to the earlier behaviour.
  - `range` (point and spot): a shaded point farther than the range gets nothing from the light, at the baked point and at every bounce, and the light tree gives zero importance to any node whose bounds are farther than its largest range from the point, so a lone ranged light is never picked there (lights sharing a node can still be picked out of range, and then add nothing). 0 means no cutoff.
  - `cone_falloff`: `kLinear` (t) or `kSmoothstep` (t² (3 − 2t)), with t = clamp((cos θ − cos outer) / (cos inner − cos outer), 0, 1). Smoothstep gives a hard edge when cos inner ≤ cos outer.
  - Set `flux = Flux(light)` after the other fields, as the glTF loader does: the light tree drops a bounded light whose flux is 0.

For library bakes without a scene environment, `BakerDefaultsTest` pins the default SH and irradiance values and the cosine-weighted surface visibility. A material-less primitive read without `TEXCOORD_0` is written without `TEXCOORD_0`.
