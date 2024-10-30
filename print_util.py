import numpy as np
def print_patch_positions(patch_positions):
    """
    Prints the patch positions in a tabular format for each micro-layer.

    Args:
        patch_positions (dict): Dictionary containing patch information with keys:
                                'x_coords', 'y_coords', 'radii', 'angles', 'densities'.
    """
    x_coords = patch_positions['x_coords']
    y_coords = patch_positions['y_coords']
    radii = patch_positions['radii']
    angles = patch_positions['angles']
    densities = patch_positions['densities']
    # diist2parentCenter= patch_positions['distance_2_parent_center']

    
    num_micro_layers, num_patches = x_coords.shape
    
    for micro_idx in range(num_micro_layers):
        print(f"{'='*50}")
        print(f"Micro Layer {micro_idx }")
        print(f"{'='*50}")
        header = f"{'Patch':<6} | {'X':>10} | {'Y':>10} | {'Radius':>10} | {'Angle (rad)':>8} | {'Density':>8}"
        print(header)
        print("-" * len(header))
        
        for patch_idx in range(num_patches):
            x = x_coords[micro_idx, patch_idx]
            y = y_coords[micro_idx, patch_idx]
            radius = radii[micro_idx]
            angle = angles[patch_idx]
            density = densities[micro_idx, patch_idx]
            # dist2perent = diist2parentCenter[micro_idx, patch_idx]
            
            row = f"{patch_idx :<6} | {x:10.3f} | {y:10.3f} | {radius:10.3f} | {angle*182/np.pi:8.2f} | {density:8.2f} "
            print(row)
        
        print(f"{'='*50}\n")

# def print_patch_mappings(child_patch_positions, parent_patch_positions, child_patch_mappings, at_angle=0):
#     """
#     Prints the patch mappings in a tabular format showing child and corresponding parent patch information,
#     including radial distances and micro-layer boundaries.
    
#     Args:
#         child_patch_positions (dict): Dictionary containing child patch information
#         parent_patch_positions (dict): Dictionary containing parent patch information
#         child_patch_mappings (dict): Dictionary mapping child patches to parent patches
#         at_angle (float): Current angle for reference
#     """
#     print(f"\n{'='*120}")
#     print(f"Child-Parent Patch Mappings Angle: {at_angle}")
#     print(f"{'='*120}")
    
#     header = (
#         f"{'Ch P':<5} | "
#         f"{'P ML':>5} | "
#         f"{'P P':>5} | "
#         f"{'P °':>6} | "
#         f"{'C °':>6} | "
#         f"{'Ch R':>5} | "
#         f"{'P Inner':>7} | "
#         f"{'P Outer':>7} | "
#         f"{'Ch X':>5} | "
#         f"{'Ch Y':>5} | "
#         f"{'P X':>5} | "
#         f"{'P Y':>5} | "
#         f"{'Ch D':>5} | "
#         f"{'P D':>5} |"
#     )
#     print(header)
#     print("-" * 120)
    
#     # Calculate micro-layer boundaries
#     parent_radii = parent_patch_positions['radii']
#     num_parent_micro_layers = len(parent_radii)
#     layer_boundaries = []
    
#     for i in range(num_parent_micro_layers):
#         current_radius = parent_radii[i]
#         # Assuming uniform thickness within the layer
#         if i < num_parent_micro_layers - 1:
#             thickness = abs(parent_radii[i] - parent_radii[i + 1])
#         else:
#             thickness = abs(parent_radii[i] - parent_radii[i - 1])
            
#         outer_radius = current_radius + thickness/2
#         inner_radius = current_radius - thickness/2
#         layer_boundaries.append((inner_radius, outer_radius))
    
#     for child_idx in sorted(child_patch_mappings.keys()):
#         # Get child patch info
#         child_x = child_patch_positions['x_coords'].flatten()[child_idx]
#         child_y = child_patch_positions['y_coords'].flatten()[child_idx]
#         child_radius = np.sqrt(child_x**2 + child_y**2)
#         child_density = child_patch_positions['densities'].flatten()[child_idx]
#         child_angle = child_patch_positions['angles'][child_idx] * 180 / np.pi
        
#         # Get parent patch info from mapping
#         parent_info = child_patch_mappings[child_idx]
#         parent_micro_layer = parent_info[1]
#         parent_patch = parent_info[2]
        
#         # Get parent layer boundaries
#         parent_inner_r, parent_outer_r = layer_boundaries[parent_micro_layer]
        
#         # Get corresponding parent patch info
#         parent_x = parent_patch_positions['x_coords'][parent_micro_layer, parent_patch]
#         parent_y = parent_patch_positions['y_coords'][parent_micro_layer, parent_patch]
#         parent_density = parent_patch_positions['densities'][parent_micro_layer, parent_patch]
#         parent_angle = parent_patch_positions['angles'][parent_patch] * 180 / np.pi
        
#         row = (
#             f"{child_idx:<5} | "
#             f"{parent_micro_layer:5d} | "
#             f"{parent_patch:5d} | "
#             f"{parent_angle:6.2f} | "
#             f"{child_angle:6.2f} | "
#             f"{child_radius:5.2f} | "
#             f"{parent_inner_r:7.2f} | "
#             f"{parent_outer_r:7.2f} | "
#             f"{child_x:5.2f} | "
#             f"{child_y:5.2f} | "
#             f"{parent_x:5.2f} | "
#             f"{parent_y:5.2f} | "
#             f"{child_density:5.2f} | "    
#             f"{parent_density:5.2f} |"
#         )
#         print(row)
    
#     print(f"{'='*120}\n")
def print_patch_mappings(
    child_patch_positions, 
    parent_patch_positions, 
    child_patch_mappings, 
    parent_center= (0.0, 0.0),
    at_angle= 0
    ):
    """
    Prints the patch mappings in a tabular format showing child and corresponding parent patch information,
    including radial distances and micro-layer boundaries.
    
    Args:
        child_patch_positions (dict): Dictionary containing child patch information
        parent_patch_positions (dict): Dictionary containing parent patch information
        child_patch_mappings (dict): Dictionary mapping child patches to parent patches
        parent_center (tuple): Center coordinates of parent body (x, y)
        at_angle (float): Current angle for reference
    """
    print(f"\n{'='*120}")
    print(f"Child-Parent Patch Mappings Angle: {at_angle}")
    print(f"Parent Center: ({parent_center[0]:.2f}, {parent_center[1]:.2f})")
    print(f"{'='*120}")
    
    header = (
        f"{'Ch P':<5} | "
        f"{'P ML':>5} | "
        f"{'P P':>5} | "
        f"{'P °':>6} | "
        f"{'C °':>6} | "
        f"{'Ch RD':>7} | "  # Changed from Ch R to Ch RD for Radial Distance
        f"{'P Inner':>7} | "
        f"{'P Outer':>7} | "
        f"{'Ch X':>5} | "
        f"{'Ch Y':>5} | "
        f"{'P X':>5} | "
        f"{'P Y':>5} | "
        f"{'Ch D':>5} | "
        f"{'P D':>5} |"
    )
    print(header)
    print("-" * 120)
    
    # Calculate micro-layer boundaries
    parent_radii = parent_patch_positions['radii']
    num_parent_micro_layers = len(parent_radii)
    layer_boundaries = []
    
    for i in range(num_parent_micro_layers):
        current_radius = parent_radii[i]
        # Assuming uniform thickness within the layer
        if i < num_parent_micro_layers - 1:
            thickness = abs(parent_radii[i] - parent_radii[i + 1])
        else:
            thickness = abs(parent_radii[i] - parent_radii[i - 1])
            
        outer_radius = current_radius + thickness/2
        inner_radius = current_radius - thickness/2
        layer_boundaries.append((inner_radius, outer_radius))
    
    for child_idx in sorted(child_patch_mappings.keys()):
        # Get child patch info
        child_x = child_patch_positions['x_coords'].flatten()[child_idx]
        child_y = child_patch_positions['y_coords'].flatten()[child_idx]
        # Calculate radial distance from parent center
        child_radial_distance = np.sqrt(
            (child_x - parent_center[0])**2 + 
            (child_y - parent_center[1])**2
        )
        child_density = child_patch_positions['densities'].flatten()[child_idx]
        child_angle = child_patch_positions['angles'][child_idx] * 180 / np.pi
        
        # Get parent patch info from mapping
        parent_info = child_patch_mappings[child_idx]
        parent_micro_layer = parent_info[1]
        parent_patch = parent_info[2]
        
        # Get parent layer boundaries
        parent_inner_r, parent_outer_r = layer_boundaries[parent_micro_layer]
        
        # Get corresponding parent patch info
        parent_x = parent_patch_positions['x_coords'][parent_micro_layer, parent_patch]
        parent_y = parent_patch_positions['y_coords'][parent_micro_layer, parent_patch]
        parent_density = parent_patch_positions['densities'][parent_micro_layer, parent_patch]
        parent_angle = parent_patch_positions['angles'][parent_patch] * 180 / np.pi
        
        row = (
            f"{child_idx:<5} | "
            f"{parent_micro_layer:5d} | "
            f"{parent_patch:5d} | "
            f"{parent_angle:6.2f} | "
            f"{child_angle:6.2f} | "
            f"{child_radial_distance:7.2f} | "  # Using child_radial_distance instead of child_radius
            f"{parent_inner_r:7.2f} | "
            f"{parent_outer_r:7.2f} | "
            f"{child_x:5.2f} | "
            f"{child_y:5.2f} | "
            f"{parent_x:5.2f} | "
            f"{parent_y:5.2f} | "
            f"{child_density:5.2f} | "
            f"{parent_density:5.2f} |"
        )
        print(row)
    
    print(f"{'='*120}\n")
def print_material_object(material_object, print_child_only=False, indent=""):
    if not print_child_only:
        print(f"{indent}Body: {material_object['name']}")
        print(f"{indent}Layers:")
        for i, layer in enumerate(material_object['layers']):
            print(f"{indent}  Layer {i}:")
            print(f"{indent}    Thickness: {layer['thickness']}")
            print(f"{indent}    Density: {layer['density']}")
            print(f"{indent}    Number of micro-layers: {layer.get('num_micro_layers', 1)}")
            print(f"{indent}    Center: {layer.get('center', (1,1))}")
            print(f"{indent}    radius: {layer.get('radius', 0.0)}")
            print(f"{indent}    Density profile:")
            density_profile = layer['density_profile']
            for j, micro_layer in enumerate(density_profile):
                print(f"{indent}      Micro-layer {j}: {micro_layer}")

    if 'child_bodies' in material_object:
        if not print_child_only:
            print(f"{indent}Child Bodies:")
        for child in material_object['child_bodies']:
            print_material_object(child, False, indent + "  ")



