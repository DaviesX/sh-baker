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
    c. Sample the punctual light sources based on a heuristic derived from their flux and inverse-square law. The score can be used to form a PDF for sampling.
    d. Add a new light type: Area Light. Add corresponding parameters that describes the area light (i.e. center, normal, area, flux, geometry index) to the Light struct. Change the loader to add emissive geometry to the scene and add corresponding light parameters to the Light struct.
    e. Sample the area light based on a heuristic derived from their flux and inverse-square law. Merge the score with those in c.
    f. Add a light module (light.h, light.cpp and light_test.cpp) to the project. It should contain the light sampling and transport functions needed for the next-event estimation.
    e. Change the SH Baker to use the next-event estimation algorithm.
2. Implement a visibility-aware importance sampling system using a 3D Voxel Grid. Follow these technical requirements:
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
    