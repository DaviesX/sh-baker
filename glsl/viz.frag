#version 410 core

in vec3 vNormal;
in vec2 vTexCoord0;
in vec2 vTexCoord1;

out vec4 FragColor;

uniform sampler2D u_AlbedoTex;
uniform int u_HasAlbedo;
uniform vec3 u_AlbedoColor;

// SH Textures (9 coefficients)
uniform sampler2D u_L0;
uniform sampler2D u_L1m1;
uniform sampler2D u_L10;
uniform sampler2D u_L11;
uniform sampler2D u_L2m2;
uniform sampler2D u_L2m1;
uniform sampler2D u_L20;
uniform sampler2D u_L21;
uniform sampler2D u_L22;

uniform int u_UsePackedLuminance;
uniform sampler2D u_PackedTex0;
uniform sampler2D u_PackedTex1;
uniform sampler2D u_PackedTex2;

void main() {
    // 1. Albedo
    vec3 albedo = u_AlbedoColor;
    if (u_HasAlbedo > 0) {
        vec4 texColor = texture(u_AlbedoTex, vTexCoord0);
        albedo *= texColor.rgb; // Ignore alpha for now
    }

    // 2. SH Basis Functions (Zonal harmonics usually Z-up? Check baker)
    // Baker code in material.cpp: Z is up for local sampling.
    // Trace code:
    // basis t, b from hit_normal.
    // bounce direction constructed.
    // AccumulateRadiance uses dir_world.
    // If bake was in world space, we reconstruct in world space.
    // vNormal is interpolated world normal (if model matrix was Identity).
    
    // Basis functions:
    // Y00 = 0.282095
    // Y1m1 = 0.488603 * y
    // Y10  = 0.488603 * z
    // Y11  = 0.488603 * x
    // Y2m2 = 1.092548 * x * y
    // Y2m1 = 1.092548 * y * z
    // Y20  = 0.315392 * (3z^2 - 1)
    // Y21  = 1.092548 * x * z
    // Y22  = 0.546274 * (x^2 - y^2)

    vec3 n = normalize(vNormal);
    float x = n.x;
    float y = n.y;
    float z = n.z;

    float c1 = 0.282095;
    float c2 = 0.488603;
    float c3 = 1.092548;
    float c4 = 0.315392;
    float c5 = 0.546274;

    float b0 = c1;
    float b1 = c2 * y;
    float b2 = c2 * z;
    float b3 = c2 * x;
    float b4 = c3 * x * y;
    float b5 = c3 * y * z;
    float b6 = c4 * (3.0 * z * z - 1.0);
    float b7 = c3 * x * z;
    float b8 = c5 * (x * x - y * y);

    // 3. Sample SH Coefficients
    vec3 L0, L1m1, L10, L11, L2m2, L2m1, L20, L21, L22;

    if (u_UsePackedLuminance == 1) {
        vec4 p0 = texture(u_PackedTex0, vTexCoord1); // L0.rgb, L1m1.Y
        vec4 p1 = texture(u_PackedTex1, vTexCoord1); // L10, L11, L2m2, L2m1
        vec4 p2 = texture(u_PackedTex2, vTexCoord1); // L20, L21, L22, pad

        L0 = p0.rgb;
        
        // Compute Chromaticity from L0
        float L0_lum = dot(L0, vec3(0.2126, 0.7152, 0.0722));
        vec3 chroma = vec3(1.0);
        if (L0_lum > 0.000001) {
            chroma = L0 / L0_lum;
        }

        // Expand luminance coefficients to RGB
        L1m1 = vec3(p0.a) * chroma;
        L10  = vec3(p1.r) * chroma;
        L11  = vec3(p1.g) * chroma;
        L2m2 = vec3(p1.b) * chroma;
        L2m1 = vec3(p1.a) * chroma;
        L20  = vec3(p2.r) * chroma;
        L21  = vec3(p2.g) * chroma;
        L22  = vec3(p2.b) * chroma;

    } else {
        L0 = texture(u_L0, vTexCoord1).rgb;
        L1m1 = texture(u_L1m1, vTexCoord1).rgb;
        L10 = texture(u_L10, vTexCoord1).rgb;
        L11 = texture(u_L11, vTexCoord1).rgb;
        L2m2 = texture(u_L2m2, vTexCoord1).rgb;
        L2m1 = texture(u_L2m1, vTexCoord1).rgb;
        L20 = texture(u_L20, vTexCoord1).rgb;
        L21 = texture(u_L21, vTexCoord1).rgb;
        L22 = texture(u_L22, vTexCoord1).rgb;
    }

    // 4. Reconstruct Irradiance/Radiance
    vec3 irradiance = L0 * b0 +
                      L1m1 * b1 + L10 * b2 + L11 * b3 +
                      L2m2 * b4 + L2m1 * b5 + L20 * b6 + L21 * b7 + L22 * b8;
    
    // Clamp negative values (SH ringing)
    irradiance = max(irradiance, 0.0);

    // 5. Combine
    vec3 result = albedo * irradiance;

    // Tone mapping (simple reinhard)
    result = result / (result + vec3(1.0));
    // Gamma correction
    result = pow(result, vec3(1.0/2.2));

    FragColor = vec4(result, 1.0);
}
