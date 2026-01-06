Project Goal: Offline Path-Tracer for Baking Spherical Harmonic (SH) Lightmaps. Tech Stack: C++20, Intel Embree (Ray-tracing), xatlas (UV Unwrapping), tinygltf (I/O). Input: glTF file (level geometry with lightmap UVs from xatlas). Output: glTF file with a custom extension (e.g., EXT_sh_lightmap) or a secondary binary file containing 9 SH coefficients per lightmap texel. Key Logic: Monte Carlo sampling on the hemisphere of each lightmap texel; project result into 3rd-order SH basis functions.

Phase 1: The SkeletonLoader
1. Implement a loader using tinygltf to read the input mesh.
2. BVH Setup: Use Embree to build a Bounding Volume Hierarchy of the triangles.
3. Test Ray: Write a simple "Ambient Occlusion" pass. If the corners of a box are dark, the BVH and ray-casting are working.

Phase 2: The SH Baker
1. UV Mapping: For each texel in the $1024 \times 1024$ lightmap, calculate the corresponding 3D world position and normal.
2. Sampling: Fire 128–512 rays from that point.
3. Projection: Convert the color of those hits into SH coefficients.
4. Dilation: Use a simple push-pull or flood-fill to prevent black edges at UV seams.

Phase 3: The Blender Visualizer
1. You need a way to "see" if the coefficients are right without running the game.
2. The Plugin: Ask Gemini to write a Python script for Blender that adds a Custom Shader Node.
3. The Logic: The shader should take the 9 SH coefficients (stored in an Image Texture or Vertex Colors) and perform the reconstruction: $Color = \sum (coeff_i \cdot basis_i(normal))$.

Phase 4: Optimization - low-hanging fruit
1. Implement the next-event estimation (NEE) algorithm to reduce the number of samples needed to bake the lightmap.
    a. Exclude the back-facing directional lights and spot lights that are out of cone.
    b. Sample the sunlight if it is front-facing.
    c. Sample the punctual light sources based on a heuristic derived from their radiance and inverse-square law. The score can be used to form a PDF for sampling.
    d. Add a new light type: Area Light. Add corresponding parameters that describes the area light (i.e. center, normal, area, flux, geometry index) to the Light struct. Change the loader to add emissive geometry to the scene and add corresponding light parameters to the Light struct.
    e. Sample the area light based on a heuristic derived from their radiance and inverse-square law. Merge the score with those in c.
    f. Add a light module (light.h, light.cpp and light_test.cpp) to the project. It should contain the light sampling and transport functions needed for the next-event estimation.
    g. Change the SH Baker to use the next-event estimation algorithm.
2. Jittered Sampling
    We can use a jittered sampling pattern on each lightmap texel to avoid aliasing (a type of supersampling).
3. Parallelization
    We know that Embree depends on TBB for task scheduling. We can use TBB to parallelize the lightmap baking process.

Phase 5: Accurate PBR Material Handling
1. Load the normal map and tangent space from the glTF file. If the tangent vertex attribute is not present, we will compute it through MikkTSpace https://github.com/mmikk/MikkTSpace.
2. Load the metalic roughness map from the glTF file.
3. Update the SH Baker's material module to implement the glTF PBR BRDF (Reference: BRDF Implementation: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#appendix-b-brdf-implementation).
4. Update the saver module to copy the normal map and metalic roughness map to the output directory.
5. Luminance only L1 and L2 SH Coefficients
    - We can add an option to save the luminance only L1 and L2 SH coefficients. This should save a lot of space.
    - Pack the coefficients in few textures?
    - Update the visualizer to support this mode (additional fragment shader?).


Phase 6 Better Visualizer
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

Phase 7: Atlas Allocation Optimization (Resolution-Aware Scaling)
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

Phase 8 Visibility-Aware Importance Sampling
Implement a visibility-aware importance sampling system using a 3D Voxel Grid. Follow these technical requirements:
    a. Data Structure: The Light Grid.Create a LightGrid struct that partitions the world-space bounding box into a 3D grid (e.g., $16 \times 16 \times 16$ or $32 \times 32 \times 32$ cells). Each cell should store a std::vector<const Light*> (or std::bitset, which may be more efficient because we are possibly managing at most 512 lights in total) pointing to "potentially visible" lights.
    b. Pre-pass: Stochastic Visibility Casting.
        - Before the main bake, iterate through every cell in the grid.
        - From the center of each cell, fire a fixed number of rays (e.g., 64) toward random lights in the scene.
        - If a ray hits a light without occlusion (use rtcOccluded1), mark that light index as Visible for that cell.
    c. Visibility-Weighted PDF.
        During the main path tracing loop, for every hit point $P$ with normal $N$:
        - Identify the grid cell containing $P$.
        - For every light $i$ in the scene, calculate its geometric importance weight $W_i$ (using intensity, distance, and $cos \theta$ factors).
        - Apply a Visibility Multiplier $V_i$:If the light is in the cell's "Visible" list: $V_i = 1.0$.If not: $V_i = 0.05$ (this is the Russian Roulette factor to maintain unbiased results).
        - Build a Cumulative Distribution Function (CDF) from $W_{final} = W_i \times V_i$.
        - Sample one light index $i$ from this CDF and trace a shadow ray.
        - Crucial: Divide the contribution by the correct probability $P_i$ to ensure the estimator remains unbiased.
    Note: 
    * Ensure the grid lookup is an $O(1)$ operation using a simple index calculation: (x - min.x) / cell_size.
    * Combine with 1. to further cull and importance sample the potentially visible lights.
    