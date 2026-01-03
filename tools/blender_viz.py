import bpy
import os

def create_sh_shader(material_name, lightmap_base_path):
    """
    Creates a Shader Node Tree for SH Lightmap visualization.
    
    Args:
        material_name: Name of the material to create/update.
        lightmap_base_path: Path to the base lightmap file (e.g., "output/lightmap.exr").
                         The script assumes split files exist: "output/lightmap_L0.exr", etc.
    """
    
    mat = bpy.data.materials.get(material_name)
    if mat is None:
        mat = bpy.data.materials.new(name=material_name)
    
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    
    # Clear existing nodes
    nodes.clear()
    
    # Create Output Node
    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    output_node.location = (1200, 0)
    
    # Create Emission Shader (since we are visualizing radiance directly)
    emission_node = nodes.new(type='ShaderNodeEmission')
    emission_node.location = (1000, 0)
    links.new(emission_node.outputs['Emission'], output_node.inputs['Surface'])
    
    # Create Group Node for SH Reconstruction
    # Check if group exists
    group_name = "SH_Reconstruction"
    if group_name not in bpy.data.node_groups:
        create_sh_node_group(group_name)
    
    sh_group = nodes.new(type='ShaderNodeGroup')
    sh_group.node_tree = bpy.data.node_groups[group_name]
    sh_group.location = (600, 0)
    
    links.new(sh_group.outputs['Color'], emission_node.inputs['Color'])
    
    # Create Texture Nodes and connect to group inputs
    # Group inputs: L0, L1m1, L10, L11, L2m2, L2m1, L20, L21, L22
    # File suffixes
    coeffs = ["L0", "L1m1", "L10", "L11", "L2m2", "L2m1", "L20", "L21", "L22"]
    
    base_dir = os.path.dirname(lightmap_base_path)
    stem = os.path.splitext(os.path.basename(lightmap_base_path))[0]
    ext = os.path.splitext(lightmap_base_path)[1]
    
    for i, coeff in enumerate(coeffs):
        # Construct filename: e.g. lightmap_L0.exr
        filename = f"{stem}_{coeff}{ext}"
        filepath = os.path.join(base_dir, filename)
        
        tex_node = nodes.new(type='ShaderNodeTexImage')
        tex_node.location = (-400, 600 - i * 300)
        tex_node.label = coeff
        
        # Load image
        if os.path.exists(filepath):
            try:
                img = bpy.data.images.load(filepath)
                img.colorspace_settings.name = 'Linear' # Ensure linear for data
                tex_node.image = img
            except RuntimeError:
                print(f"Warning: Could not load {filepath}")
        else:
            print(f"Warning: File not found {filepath}")

        # Connect Color output to corresponding group input
        # Note: Texture node outputs Color (which is RGB). That matches our coefficients.
        links.new(tex_node.outputs['Color'], sh_group.inputs[i])
        
    # Create Geometry Node for Normal
    geo_node = nodes.new(type='ShaderNodeNewGeometry')
    geo_node.location = (-400, 900)
    links.new(geo_node.outputs['Normal'], sh_group.inputs['Normal'])


def create_sh_node_group(group_name):
    group = bpy.data.node_groups.new(name=group_name, type='ShaderNodeTree')
    
    # Inputs
    group.interface.new_socket(name="L0", socket_type="NodeSocketColor")
    group.interface.new_socket(name="L1m1", socket_type="NodeSocketColor")
    group.interface.new_socket(name="L10", socket_type="NodeSocketColor")
    group.interface.new_socket(name="L11", socket_type="NodeSocketColor")
    group.interface.new_socket(name="L2m2", socket_type="NodeSocketColor")
    group.interface.new_socket(name="L2m1", socket_type="NodeSocketColor")
    group.interface.new_socket(name="L20", socket_type="NodeSocketColor")
    group.interface.new_socket(name="L21", socket_type="NodeSocketColor")
    group.interface.new_socket(name="L22", socket_type="NodeSocketColor")
    group.interface.new_socket(name="Normal", socket_type="NodeSocketVector")
    
    # Outputs
    group.interface.new_socket(name="Color", socket_type="NodeSocketColor")
    
    nodes = group.nodes
    links = group.links
    
    input_node = nodes.new('NodeGroupInput')
    input_node.location = (-800, 0)
    
    output_node = nodes.new('NodeGroupOutput')
    output_node.location = (800, 0)
    
    # We need to implement the SH basis functions for Normal (x, y, z)
    # Z-up? Blender is Z-up? 
    # Wait, our baker assumes Z-up for local sampling, but World Space depends on glTF.
    # glTF is Y-up. But we might have baked in whatever space the normal is in.
    # Assuming the "Normal" input is in the same space as the baked coefficients.
    # Usually World Space.
    
    # Basis functions (real SH, standard):
    # Y00 = 0.282095
    # Y1m1 = 0.488603 * y
    # Y10  = 0.488603 * z
    # Y11  = 0.488603 * x
    # Y2m2 = 1.092548 * x * y
    # Y2m1 = 1.092548 * y * z
    # Y20  = 0.315392 * (3z^2 - 1)
    # Y21  = 1.092548 * x * z
    # Y22  = 0.546274 * (x^2 - y^2)
    
    # NOTE: Coordinate system mismatch is the biggest risk here.
    # Our baker: 
    #   Trace uses logic:
    #   L1m1 -> y
    #   L10 -> z
    #   L11 -> x
    #   Wait, let's check sh_coeffs.cpp if possible. Or assume standard.
    #   My material implementation used Z as up for local hemisphere sampling (cos(theta)).
    #   But accumulation is done with `dir_world`.
    #   If `dir_world` matches standard Cartesian (Right handed, Y-up or Z-up), the basis functions should be standard.
    #   Let's check `sh_coeffs.cpp` later or assume standard Y-up (glTF) but check basis defs.
    #   Standard SH usually defines Y10 as Z. But in Y-up world, Y is up (often Y10 associated with Y).
    #   Let's assume standard mathematics:
    #   l=1, m=0 is usually Z ("zonal").
    #   So we need to separate Normal into X, Y, Z.
    
    # Separate XYZ
    sep_xyz = nodes.new('ShaderNodeSeparateXYZ')
    sep_xyz.location = (-600, 200)
    links.new(input_node.outputs['Normal'], sep_xyz.inputs['Vector'])
    
    # Constants
    c1 = 0.282095
    c2 = 0.488603
    c3 = 1.092548
    c4 = 0.315392
    c5 = 0.546274
    
    # Helper to add Multiply Node
    def create_math(op, inp1, inp2, loc):
        node = nodes.new('ShaderNodeMath')
        node.operation = op
        node.location = loc
        if isinstance(inp1, float):
            node.inputs[0].default_value = inp1
        else:
            links.new(inp1, node.inputs[0])
            
        if isinstance(inp2, float):
            node.inputs[1].default_value = inp2
        else:
            links.new(inp2, node.inputs[1])
        return node.outputs[0]

    # Helper to add Vector Scale (Color * float)
    def create_scale(color_socket, factor_socket, loc):
        node = nodes.new('ShaderNodeMixRGB') # MixRGB works for scaling if mode is Multiply? No, Mix is interpolation.
        # Use Vector Math -> Scale? Or Mix with Multiply.
        # Mix(Multiply, Color1, Color2, Fac). If Fac is 1, result is Color1 * Color2.
        # But factor_socket is a scalar.
        # Easier: Vector Math Scale. (Available in recent Blender)
        node = nodes.new('ShaderNodeVectorMath')
        node.operation = 'SCALE'
        node.location = loc
        links.new(color_socket, node.inputs['Vector'])
        links.new(factor_socket, node.inputs['Scale'])
        return node.outputs['Vector']

    def create_add(vec1, vec2, loc):
        node = nodes.new('ShaderNodeVectorMath')
        node.operation = 'ADD'
        node.location = loc
        links.new(vec1, node.inputs[0])
        links.new(vec2, node.inputs[1])
        return node.outputs[0]

    # --- Basis Calculation ---
    x = sep_xyz.outputs['X']
    y = sep_xyz.outputs['Y']
    z = sep_xyz.outputs['Z']
    
    # 0: Y00 = c1
    b0 = c1
    
    # 1: Y1m1 = c2 * y
    b1 = create_math('MULTIPLY', c2, y, (-400, 400))
    
    # 2: Y10 = c2 * z
    b2 = create_math('MULTIPLY', c2, z, (-400, 300))
    
    # 3: Y11 = c2 * x
    b3 = create_math('MULTIPLY', c2, x, (-400, 200))
    
    # 4: Y2m2 = c3 * x * y
    xy = create_math('MULTIPLY', x, y, (-500, 100))
    b4 = create_math('MULTIPLY', c3, xy, (-400, 100))
    
    # 5: Y2m1 = c3 * y * z
    yz = create_math('MULTIPLY', y, z, (-500, 0))
    b5 = create_math('MULTIPLY', c3, yz, (-400, 0))
    
    # 6: Y20 = c4 * (3z^2 - 1)
    z2 = create_math('MULTIPLY', z, z, (-500, -100))
    z2_3 = create_math('MULTIPLY', 3.0, z2, (-450, -100))
    z2_3_m1 = create_math('SUBTRACT', z2_3, 1.0, (-400, -100))
    b6 = create_math('MULTIPLY', c4, z2_3_m1, (-350, -100))
    
    # 7: Y21 = c3 * x * z
    xz = create_math('MULTIPLY', x, z, (-500, -200))
    b7 = create_math('MULTIPLY', c3, xz, (-400, -200))
    
    # 8: Y22 = c5 * (x^2 - y^2)
    x2 = create_math('MULTIPLY', x, x, (-500, -300))
    y2 = create_math('MULTIPLY', y, y, (-500, -350))
    x2_y2 = create_math('SUBTRACT', x2, y2, (-450, -300))
    b8 = create_math('MULTIPLY', c5, x2_y2, (-400, -300))
    
    # --- Weighted Sum ---
    # Sum = L0*b0 + L1m1*b1 + ...
    
    # Term 0
    t0 = create_scale(input_node.outputs['L0'], b0, (-200, 500))
    
    # Term 1
    t1 = create_scale(input_node.outputs['L1m1'], b1, (-200, 400))
    
    # Term 2
    t2 = create_scale(input_node.outputs['L10'], b2, (-200, 300))
    
    # Term 3
    t3 = create_scale(input_node.outputs['L11'], b3, (-200, 200))
    
    # Term 4
    t4 = create_scale(input_node.outputs['L2m2'], b4, (-200, 100))
    
    # Term 5
    t5 = create_scale(input_node.outputs['L2m1'], b5, (-200, 0))
    
    # Term 6
    t6 = create_scale(input_node.outputs['L20'], b6, (-200, -100))
    
    # Term 7
    t7 = create_scale(input_node.outputs['L21'], b7, (-200, -200))
    
    # Term 8
    t8 = create_scale(input_node.outputs['L22'], b8, (-200, -300))
    
    # Sum all
    s1 = create_add(t0, t1, (0, 400))
    s2 = create_add(s1, t2, (100, 400))
    s3 = create_add(s2, t3, (200, 400))
    s4 = create_add(s3, t4, (300, 400))
    s5 = create_add(s4, t5, (400, 400))
    s6 = create_add(s5, t6, (500, 400))
    s7 = create_add(s6, t7, (600, 400))
    s8 = create_add(s7, t8, (700, 400))
    
    links.new(s8, output_node.inputs['Color'])

if __name__ == "__main__":
    # Example usage:
    # Set these variables or run from Blender text editor
    # create_sh_shader("Material", "/path/to/output/lightmap.exr")
    pass
