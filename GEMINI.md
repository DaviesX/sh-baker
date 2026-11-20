# Project Goal: Offline Path-Tracer for Baking Spherical Harmonic (SH) Lightmaps. Tech Stack: C++20, Intel Embree (Ray-tracing), xatlas (UV Unwrapping), tinygltf (I/O). Input: glTF file (level geometry with lightmap UVs from xatlas). Output: glTF file with a custom extension (e.g., EXT_sh_lightmap) or a secondary binary file containing 9 SH coefficients per lightmap texel. Key Logic: Monte Carlo sampling on the hemisphere of each lightmap texel; project result into 3rd-order SH basis functions.

## Tips
- Build and test the project using the following commands:
```bash
cmake -DCMAKE_BUILD_TYPE=Release -B build -S . && cmake --build build --parallel 12 && ./build/sh_baker_test
```

# Phase 1: The SkeletonLoader
1. Implement a loader using tinygltf to read the input mesh.
2. BVH Setup: Use Embree to build a Bounding Volume Hierarchy of the triangles.
3. Test Ray: Write a simple "Ambient Occlusion" pass. If the corners of a box are dark, the BVH and ray-casting are working.

# Phase 2: The SH Baker
1. UV Mapping: For each texel in the $1024 \times 1024$ lightmap, calculate the corresponding 3D world position and normal.
2. Sampling: Fire 128–512 rays from that point.
3. Projection: Convert the color of those hits into SH coefficients.
4. Dilation: Use a simple push-pull or flood-fill to prevent black edges at UV seams.

# Phase 3: The Blender Visualizer
1. You need a way to "see" if the coefficients are right without running the game.
2. The Plugin: Ask Gemini to write a Python script for Blender that adds a Custom Shader Node.
3. The Logic: The shader should take the 9 SH coefficients (stored in an Image Texture or Vertex Colors) and perform the reconstruction: $Color = \sum (coeff_i \cdot basis_i(normal))$.

# Phase 4: Optimization - low-hanging fruit
1. Implement the next-event estimation (NEE) algorithm to reduce the number of samples needed to bake the lightmap.
    - Exclude the back-facing directional lights and spot lights that are out of cone.
    - Sample the sunlight if it is front-facing.
    - Sample the punctual light sources based on a heuristic derived from their radiance and inverse-square law. The score can be used to form a PDF for sampling.
    - Add a new light type: Area Light. Add corresponding parameters that describes the area light (i.e. center, normal, area, flux, geometry index) to the Light struct. Change the loader to add emissive geometry to the scene and add corresponding light parameters to the Light struct.
    - Sample the area light based on a heuristic derived from their radiance and inverse-square law. Merge the score with those in c.
    - Add a light module (light.h, light.cpp and light_test.cpp) to the project. It should contain the light sampling and transport functions needed for the next-event estimation.
    - Change the SH Baker to use the next-event estimation algorithm.
2. Jittered Sampling
    - We can use a jittered sampling pattern on each lightmap texel to avoid aliasing (a type of supersampling).
3. Parallelization
    - We know that Embree depends on TBB for task scheduling. We can use TBB to parallelize the lightmap baking process.

# Phase 5: Accurate PBR Material Handling
1. Load the normal map and tangent space from the glTF file. If the tangent vertex attribute is not present, we will compute it through MikkTSpace https://github.com/mmikk/MikkTSpace.
2. Load the metalic roughness map from the glTF file.
3. Update the SH Baker's material module to implement the glTF PBR BRDF (Reference: BRDF Implementation: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#appendix-b-brdf-implementation).
4. Update the saver module to copy the normal map and metalic roughness map to the output directory.
5. Luminance only L1 and L2 SH Coefficients
    - We can add an option to save the luminance only L1 and L2 SH coefficients. This should save a lot of space.
    - Pack the coefficients in few textures?
    - Update the visualizer to support this mode (additional fragment shader?).


# Phase 6 Better Visualizer
1. Implement normal mapping with SH coefficients.
2. Implement metalic roughness mapping with SH coefficients.
hint:
```glsl
// 1. Get Diffuse SH (sampled along Normal)
vec3 diffuse = SampleSH(v_Normal, u_L0, u_L1, u_L2);

// 2. Get Specular SH (sampled along Reflection)
vec3 specular = SampleSH(R, u_L0, u_L1, u_L2);

// 3. Combine using Fresnel (Schlick's approximation)
vec3 F = FresnelSchlick(max(dot(v_Normal, V), 0.0), F0);
vec3 ks = F;
vec3 kd = 1.0 - ks;
kd *= 1.0 - u_Metallic;	  

vec3 finalColor = (kd * diffuse * albedo) + (ks * specular);
```

# Phase 7: Atlas Allocation Optimization (Resolution-Aware Scaling)
Context: We need to move from a uniform lightmap density to a Resolution-Aware Allocation system in sh-baker. The goal is to ensure the SH lightmap density matches the visual frequency of the original albedo textures.
Task: Please implement a scaling heuristic for xatlas mesh declarations based on the following logic:

1. Albedo-Relative Scaling (Non-Tiling Case)
* For each mesh, determine the dimensions of its primary diffuse/albedo texture ($W_{tex}, H_{tex}$).
* Calculate the base area of the texture in pixels:
    $Area_{tex} = W_{tex} \times H_{tex}$.
* Set the meshRelativeScaling for xatlas such that the lightmap texel density is proportional to this area.
* Heuristic: Use a global "Density Multiplier" ($k$) to allow us to scale the entire atlas quality:
    $$\text{Target Scale} = k \times \sqrt{Area_{tex}}$$

2. Tiling Estimation (The Tiling Case)
* For surfaces that tile, calculate the Effective UV Area.
* Find the UV0 bounding box ($U_{max} - U_{min}$ and $V_{max} - V_{min}$).Estimate the Tile Count ($TC$): $TC = (U_{range}) \times (V_{range})$.
* Adjusted Effective Resolution: If a surface tiles, it covers more physical world space with the same texture. To maintain detail, scale the allotment by the square root of the tile count:
    $$\text{Effective Scale} = \text{Target Scale} \times \sqrt{TC}$$

3. Constraint & Safety Limits
* Max Clamp: To prevent a single massive tiled floor from eating the entire atlas, implement a max_scale cap (no single mesh can exceed k times the median scale, in other words, an outlier).

4. Integration with xatlas
* Pass these calculated scales into the xatlas::MeshDeclaration::meshRelativeScaling before the xatlas::PackCharts call.
* Padding Enforcement: Ensure the 16px padding rule from Phase 6 is maintained regardless of the individual mesh scale.

Goal: Provide the C++ logic to calculate these scales during the mesh processing loop and apply them to the xatlas setup.

# Phase 8: Decoupled Environment & Occlusion Masking
Context: We are moving to a decoupled environment model for `sh-baker`. The sky will not be "baked" into the SH coefficients. Instead, we will store a **Sky Visibility** mask and resolve the sky contribution at runtime using global SH uniforms.

Task: Please implement the following architecture:

1. Runtime Environment Initialization (`loader.cpp` / `scene.h`)
    * **Dual-Path Initialization:**
        - **Cubemap Path:** If a glTF environment texture is present, implement `ProjectCubeMapToSH()` to generate 9 RGB SH coefficients at load-time.
        - **Analytical Path:** If no texture is found, use the Sun's direction and intensity to run the **Preetham Sky** projection (with a 30x Zenith intensity cap to prevent ringing).
    * **Storage:** Store these 9 coefficients in the `Scene` struct (as SHCoeffs) for transmission to the visualizer as **Uniforms** (`u_SkySH[9]`).

2. The Occlusion Baker (`integrator.cpp`)
    * **The "Hole Punch" Integrator:** During the lightmap bake, modify the integration loop. When a ray is fired:
        - **If it hits geometry:** Accumulate radiance as usual (Indirect Sun Bounces).
        - **If it hits the "Miss" (Sky):** Do not accumulate radiance. Instead, increment a **Visibility Counter**.
    * **Monte Carlo Estimator:** Calculate the probability .
    * **Storage:** Store this scalar value ( to ) in the **Alpha Channel** of the third SH Lightmap texture (Texture C).

3. Visualizer Reconstruction (`visualizer_main.cpp` / `viz.frag`)
    * **The Shader Bridge:** Update the fragment shader to incorporate the ambient:
        - **Sky Ambient:** Evaluate the 9 Global SH Uniforms () and multiply the result by the **Sky Visibility** fetched from the lightmap's Alpha channel.

# Phase 9: Many-light Optimization
Context: Current light sampling strategy is O(N) where N is the number of lights. We need to optimize this to O(log N) by implementing the light tree (a.k.a. BVH light sampling). We will take the implementation from pbrt v4 as a reference 
* https://pbr-book.org/4ed/Light_Sources/Light_Sampling#x3-LightBoundingVolumeHierarchies
* https://github.com/mmp/pbrt-v4/blob/master/src/pbrt/lightsamplers.h#L259 (BVHLightSampler)
* https://github.com/mmp/pbrt-v4/blob/master/src/pbrt/lightsamplers.cpp#L103 (BVHLightSampler)
* Implement in `light_tree.h` / `light_tree.cpp`
* Test with the `light_tree_test.cpp`

Task:
1. Light Tree Construction 
    * **Node Structure:** Define a `LightNode` struct to store the bounding box with orientation spread.
    * **Tree Construction:** Implement a recursive function `BuildLightTree` that takes a list of lights and builds a binary tree of `LightNode`s. Pay extra attention to the cost function used to split the nodes.

2. Light Tree Sampling
    * **Sampling Strategy:** Implement a function `SampleLightTree` that takes a position and a `LightNode` and returns the index of the light.

3. Refactor helper functions to the namespace `light_tree_internal` and properly test them in `light_tree_test.cpp`.

# Phase 10: Reusable Path Guiding for Indirect-Only Bakes

Train a directional guide in a separate tool and let the bake consume it as optional input. The sensor ray is the ray launched from a bake point. In `indirect_only`, it finds the surfaces that supply bounce light. The same incident-radiance guide can propose continuation directions at diffuse hits. [Müller, Gross, and Novák, Practical Path Guiding](https://jannovak.info/publications/PathGuide/PathGuide.pdf)

1. **Add a standalone trainer.** Use `sh_guide_main` as the proposed executable name. Share scene preparation, rasterization, the light tree, and tracing with `sh_baker_main`. The trainer takes a separate ray budget, starts from the baseline sampler or reads a previous guide as a frozen proposal, and writes a new guide after successful training. Use an independent random stream for each training run and for production. When refining a guide, build fresh histograms from the new samples. Weight each observation by the inverse PDF that actually generated it. The production baker reads the guide without updating it.
2. **Train a surface guide at output resolution.** Pool observations from the actual valid rasterized points in each supersample block. Start with 32 rays per fully covered output texel, eight per point at scale 2. Use 16 equal-solid-angle bins with four azimuth sectors and four bands uniform in cosine elevation. Store incident radiance mass from the luminance returned by `Trace` before SH projection and before applying the local cosine. Define a reproducible directional frame and record enough surface identity to validate lookups. Omit mixed-chart blocks and incompatible surfaces from the initial guide. Full-sphere probes use their uniform sphere sampler until a separate probe guide is available.
3. **Save a versioned guide atlas.** Train in 32×32 output tiles with reusable scratch. Publish normalized directional bin masses in an atlas available to every bake worker, since continuation rays can hit another tile. Sixteen 16-bit masses per texel take 128 MiB at 2048×2048 before metadata. Decode and renormalize the masses before sampling and PDF evaluation. Store the grid dimensions, bin layout, frame convention, and a fingerprint of the scene geometry and lightmap mapping. Reject incompatible mappings. Record lighting, material and transport settings as provenance so reusable guides can be flagged when stale. Output packing, denoising and production sample count do not affect mapping compatibility. Derive lookups from lightmap UVs using the guide grid dimensions, allowing different bake resolutions when the UV layout is shared.
4. **Load the guide optionally in the bake.** With a valid guide, mix its proposal with the existing uniform sensor sampler, starting at equal weights. Evaluate both branch densities in solid angle. Each branch has zero density outside its own domain. A guide direction outside the current surface hemisphere contributes zero and still counts as a draw. Accumulate incident radiance divided by the mixture PDF, project it into SH, and keep the separate cosine weighting for irradiance. Weight an environment escape by the uniform PDF divided by the mixture PDF. With no guide or no valid entry, use the existing sampler. Preserve production supersampling and downsampling.
5. **Reuse the guide at diffuse hits.** Start with the first diffuse continuation. Derive lightmap UV from the hit primitive and barycentrics and validate chart or surface identity before using the entry. Weight its bin masses by the average positive cosine against the hit normal, then mix the resulting proposal with the existing cosine sampler. Draw uniformly within the selected bin and evaluate the actual mixture PDF in throughput and Russian roulette. Preserve light-tree next-event estimation and use cosine sampling at uncharted or incompatible hits. Handle all valid positive PDFs in the guided path. Measure the effect of the shorter remaining path budget before enabling reuse at deeper hits.
6. **Validate separate and repeated runs.** Compare unguided baking, sensor ray guiding, and sensor ray guiding with first-continuation reuse on both bottlenecked indoor and open maps. Report training plus bake time for a cold run and bake-only time for guide reuse. Compare SH bands, reconstructed lighting, and environment visibility against high-sample references. Begin with fixed production sample counts and firefly clamping off, then check production settings. Test guide round-tripping, atlas mismatch handling, and fallback at chart boundaries.

# Phase 11: Indirect SH Lightmap Denoising

Prototype an opt-in denoising pass for diffuse Quake 3 bakes in `indirect_only` mode. Use Intel Open Image Denoise (OIDN) on the CPU. Its `RTLightmap` filter supports HDR lightmaps and signed, normalized directional coefficients. Run on linear floating-point data after supersample downsampling and before final dilation and EXR packing. [OIDN RTLightmap documentation](https://www.openimagedenoise.org/documentation.html#rtlightmap)

1. **Isolate chart crops.** Carry chart identity and validity from rasterization to output resolution. Process each chart in an independently padded crop because `RTLightmap` has no chart mask or geometry inputs. Extend the chart's own valid values into crop padding and holes for filtering, then copy back only valid, unambiguous texels belonging to that chart. Leave mixed-chart output texels and full-sphere probes out of the pass.
2. **Filter the SH representation.** Denoise the original RGB L0 in HDR mode. For each RGB coefficient in L1 and L2, divide channelwise by the corresponding original L0 with a positive floor for near-zero values. Choose reversible scales that map the signed ratios into OIDN's expected range. Filter the RGB ratio images in directional mode, undo the scales, and multiply by the denoised L0 channelwise. Define the dark-channel behavior so normalization stays finite and a fully black crop stays black. Keep original L0 available throughout the directional passes.
3. **Preserve the output pipeline.** Filter the separately accumulated irradiance in HDR mode using the same chart isolation. Pass environment visibility through the existing downsample, dilation, and export path. Make OIDN an optional build dependency and reuse its device and scratch buffers across chart and coefficient passes. Surface filter errors before saving a partially denoised result.
4. **Measure quality and memory.** Reuse RGB float scratch images across coefficient passes. One full 2048×2048 RGB float image occupies 48 MiB. Measure peak process memory, including OIDN and the scale-2 bake buffers, on a representative map against the 32 GB RAM budget. Compare noisy and denoised bakes with high-sample references, with sensor ray guiding both off and on when available. Check signed L1/L2 coefficients and relighting under several normals, especially around dark rooms, doorway edges, and adjacent atlas islands. Verify constant and black charts, finite outputs, and chart isolation. Evaluate a geometry-aware filter if the OIDN prototype loses directional detail or blurs lighting boundaries.
