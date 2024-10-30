#material_body_simulator.py
import sys
import pickle
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

import numpy as np


from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QToolBar, QColorDialog,
    QComboBox, QPushButton, QDialog, QFormLayout, QLabel,
    QSpinBox, QDoubleSpinBox, QDialogButtonBox, QHBoxLayout, QCheckBox,
    QFileDialog, QSizePolicy,  QMenu
)
from PyQt6.QtGui import QAction, QColor
from PyQt6.QtCore import QTimer, Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Circle, Wedge
from matplotlib.collections import PatchCollection
from dialogs import *
from print_util import *

# Utility functions and classes

@dataclass
class RotationResult:
    angles: List[float]
    min_diff: float

def find_optimal_rotation(
    test_angles: List[float],
    calculate_density_diff_fn,
    tolerance: float = 1e-6
    ) -> RotationResult:
    min_diff = float('inf')
    best_angles = []

    for angle in test_angles:
        current_diff = calculate_density_diff_fn(angle)
        if abs(current_diff - min_diff) < tolerance:
            best_angles.append(angle)
        elif current_diff < min_diff:
            min_diff = current_diff
            best_angles = [angle]

    return RotationResult(angles=best_angles, min_diff=min_diff)

def initialize_densities(
    body: Dict[str, Any],
    num_patches: int,
    compulsory_increase: float,
    parent_body: Dict[str, Any] = None,
    parent_radius: float = 1.0,
    parent_center: Tuple[float, float] = (0.0, 0.0)
    ) -> None:
    """
    Initialize densities properly handling micro-layers for each layer.
    """
    for layer_index, layer in enumerate(body['layers']):
        num_micro_layers = layer.get('num_micro_layers', 1)
        layer['density_profile'] = np.zeros((num_micro_layers, num_patches))
        dl1 = layer['density']
        dl2 = body['layers'][layer_index + 1]['density'] if layer_index < len(body['layers']) - 1 else dl1
        increment = (dl2 - dl1) / max(num_micro_layers - 1, 1)
        for micro_layer_index in range(num_micro_layers):
            density = dl1 + micro_layer_index * increment
            layer['density_profile'][micro_layer_index, :] = density

    # Recursively initialize child bodies
    for child in body.get('child_bodies', []):
        child_outer_radius = parent_radius * (
            1 - sum(l['thickness'] for l in body['layers'][:child['parent_layer']])
        )
        child_inner_radius = parent_radius * (
            1 - sum(l['thickness'] for l in body['layers'][:child['parent_layer'] + 1])
        )
        child_radius = (child_outer_radius + child_inner_radius) / 2
        child_center_radius = (child_outer_radius + child_inner_radius) / 2
        child_placement_angle = np.radians(child.get('placement_angle', 0))
        child_center = (
            parent_center[0] + child_center_radius * np.cos(child_placement_angle),
            parent_center[1] + child_center_radius * np.sin(child_placement_angle)
        )
        initialize_densities(
            child, num_patches, compulsory_increase,
            parent_body=body, parent_radius=child_radius, parent_center=child_center
        )

def calculate_density_mismatch(
    body: Dict[str, Any],
    external_density_info: Tuple[np.ndarray, int, int]
    ) -> Tuple[float, int, int]:
    """
    Compute the density mismatch between a body and its external environment.

    Args:
        body: The material body.
        external_density_info: Tuple containing external density profile and indices.

    Returns:
        Tuple containing the density difference and indices.
    """
    parent_density_profile, parent_micro_layer_index, parent_patch_index = external_density_info
    parent_density = parent_density_profile[parent_micro_layer_index, parent_patch_index]
    body_micro_layer_index = 0  # Outermost micro-layer
    body_patch_index = parent_patch_index  # Same angular position
    sigma_surface = body['layers'][0]['density_profile'][body_micro_layer_index, body_patch_index]
    delta_rho = parent_density - sigma_surface
    return delta_rho, body_micro_layer_index, body_patch_index

def calculate_all_patch_centers(
    body: Dict[str, Any],
    body_center: Tuple[float, float],
    outer_radius: float,
    layer_thicknesses: List[float],
    num_patches: int,
    layer_indices: List[int] = [0],
    num_micro_layers: int = 1, #num_micro_layers to be returned
    rotation_angle: float = 0.0
    ) -> Dict[int, Dict[str, Any]]:
    base_angles = np.linspace(0, 2 * np.pi, num_patches, endpoint=False)
    angles = base_angles + rotation_angle + (np.pi / num_patches)
    patch_info = {}
    current_radius = outer_radius

    for layer_idx in layer_indices:
        if layer_idx >= len(layer_thicknesses):
            continue
        layer_thickness = layer_thicknesses[layer_idx]
        # Get the number of micro-layers for this layer
        layer_num_micro_layers = body['layers'][layer_idx].get('num_micro_layers', num_micro_layers)
        micro_layer_thickness = layer_thickness / layer_num_micro_layers

        print(f"body: {body['name']} layer_num_micro_layers {layer_num_micro_layers} micro_layer_thickness {micro_layer_thickness}")
        layer_x_coords = np.zeros((num_micro_layers, num_patches)) #num_micro_layers to be returned
        layer_y_coords = np.zeros((num_micro_layers, num_patches))
        layer_radii = np.zeros(num_micro_layers)
        for micro_idx in range(num_micro_layers):#num_micro_layers to be returned
            micro_layer_radius = current_radius - micro_idx * micro_layer_thickness
            mid_radius = micro_layer_radius - micro_layer_thickness / 2
            layer_radii[micro_idx] = mid_radius
            layer_x_coords[micro_idx] = body_center[0] + mid_radius * np.cos(angles)
            layer_y_coords[micro_idx] = body_center[1] + mid_radius * np.sin(angles)
        patch_info[layer_idx] = {
            'x_coords': layer_x_coords,
            'y_coords': layer_y_coords,
            'radii': layer_radii,
            'angles': angles,
            'densities': body['layers'][layer_idx]['density_profile']
        }
        current_radius -= layer_thickness
    return patch_info

def find_nearest_patches_vectorized(
    child_patch_mappings: Dict[int, List[int]],
    parent_patch_positions: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find nearest parent patches and their densities for all child patches.

    Args:
        child_patch_mappings: Dictionary mapping child patches to parent patches.
        parent_patch_positions: Dictionary containing parent patch information.

    Returns:
        Tuple containing nearest densities, micro-layer indices, and patch indices.
    """
    micro_layer_indices = []
    patch_indices = []
    for child_idx in sorted(child_patch_mappings.keys()):
        parent_info = child_patch_mappings[child_idx]
        micro_layer_indices.append(parent_info[1])
        patch_indices.append(parent_info[2])
    micro_layer_indices = np.array(micro_layer_indices)
    patch_indices = np.array(patch_indices)
    nearest_densities = parent_patch_positions['densities'][
        micro_layer_indices, patch_indices
    ]
    return nearest_densities, micro_layer_indices, patch_indices

def patch_mappings(
    child_patch_positions: Dict[str, Any],
    parent_patch_positions: Dict[str, Any]
    ) -> Dict[int, List[int]]:
    """
    Maps child patches to the nearest parent patches.

    Args:
        child_patch_positions: Dictionary containing child patch coordinates and info.
        parent_patch_positions: Dictionary containing parent patch coordinates and info.

    Returns:
        Dictionary mapping child patch indices to parent patch information.
    """
    child_x = child_patch_positions['x_coords'].flatten()
    child_y = child_patch_positions['y_coords'].flatten()
    child_angles = np.arctan2(child_y, child_x)
    parent_x = parent_patch_positions['x_coords']
    parent_y = parent_patch_positions['y_coords']
    parent_angles = np.arctan2(parent_y, parent_x)
    parent_radii = parent_patch_positions['radii']
    num_child_patches = len(child_x)
    num_parent_micro_layers = len(parent_radii)
    layer_boundaries = []
    for i in range(num_parent_micro_layers):
        current_radius = parent_radii[i]
        if i < num_parent_micro_layers - 1:
            thickness = abs(parent_radii[i] - parent_radii[i + 1])
        else:
            thickness = abs(parent_radii[i] - parent_radii[i - 1])
        outer_radius = current_radius + thickness / 2
        inner_radius = current_radius - thickness / 2
        layer_boundaries.append((inner_radius, outer_radius))
    mappings = {}
    for child_idx in range(num_child_patches):
        child_angle = child_angles[child_idx]
        child_radius = np.sqrt(child_x[child_idx] ** 2 + child_y[child_idx] ** 2)
        micro_layer_idx = None
        for idx, (inner_r, outer_r) in enumerate(layer_boundaries):
            if inner_r <= child_radius <= outer_r:
                micro_layer_idx = idx
                break
        if micro_layer_idx is None:
            distances = [
                min(abs(child_radius - inner_r), abs(child_radius - outer_r))
                for inner_r, outer_r in layer_boundaries
            ]
            micro_layer_idx = np.argmin(distances)
        angular_diff = np.abs(np.angle(
            np.exp(1j * (parent_angles[micro_layer_idx] - child_angle))
        ))
        patch_idx = np.argmin(angular_diff)
        mappings[child_idx] = [0, micro_layer_idx, patch_idx]
    return mappings

def calculate_child_patch_positions(
    child_body: Dict[str, Any],
    parent_radius: float,
    parent_center: Tuple[float, float],
    rotation_angle: float = 0.0
    ) -> Dict[str, Any]:
    """
    Calculate patch positions for child body's outer layer.

    Args:
        child_body: The child body.
        parent_radius: Radius of the parent body.
        parent_center: Center of the parent body.
        rotation_angle: Rotation angle in radians.

    Returns:
        Dictionary containing patch positions for the child body.
    """
    child_center = child_body['center']
    child_radius = child_body['radius']
    num_patches = len(child_body['layers'][0]['density_profile'][0])
    layer_thicknesses = [layer['thickness'] * child_radius for layer in child_body['layers']]
    num_micro_layers = 1  # Only outermost micro-layer
    layer_indices = [0]
    patch_positions = calculate_all_patch_centers(
        child_body,
        child_center,
        child_radius,
        layer_thicknesses,
        num_patches,
        layer_indices=layer_indices,
        num_micro_layers=num_micro_layers,
        rotation_angle=rotation_angle
    )
    return patch_positions[0]

def calculate_parent_patch_positions(
    parent_body: Dict[str, Any],
    parent_layer_index: int,
    parent_center: Tuple[float, float],
    parent_radius: float,
    rotation_angle: float = 0.0
    ) -> Dict[str, Any]:
    """
    Calculate parent patch positions for a given layer.

    Args:
        parent_body: The parent body.
        parent_layer_index: Index of the parent layer.
        parent_center: Center of the parent body.
        parent_radius: Radius of the parent body.
        rotation_angle: Rotation angle in radians.

    Returns:
        Dictionary containing patch positions for the parent layer.
    """
    parent_layer = parent_body['layers'][parent_layer_index]
    num_micro_layers = parent_layer.get('num_micro_layers', 1)
    layer_thicknesses = [
        layer['thickness'] * parent_radius
        for layer in parent_body['layers'][:parent_layer_index + 1]
    ]
    layer_outer_radius = parent_radius * (
        1 - sum(layer_thicknesses[:-1])
    )
    num_patches = len(parent_layer['density_profile'][0])
    patch_positions = calculate_all_patch_centers(
        parent_body,
        parent_center,
        layer_outer_radius,
        layer_thicknesses,
        num_patches,
        layer_indices=[parent_layer_index],
        num_micro_layers=num_micro_layers,
        rotation_angle=rotation_angle
    )
    return patch_positions[parent_layer_index]

# def update_child_body_density(
#     parent_body: Dict[str, Any],
#     child_body: Dict[str, Any],
#     parent_center: Tuple[float, float] = (0.0, 0.0),
#     parent_radius: float = 1.0
#     ) -> None:
#     """
#     Update the density of a child body based on its parent.

#     Args:
#         parent_body: The parent body.
#         child_body: The child body.
#         parent_center: Center of the parent body.
#         parent_radius: Radius of the parent body.
#     """
#     parent_layer = parent_body['layers'][child_body['parent_layer']]
#     num_patches = len(parent_layer['density_profile'][0])
#     child_center_radius = parent_radius * (
#         1 - sum(l['thickness'] for l in parent_body['layers'][:child_body['parent_layer']]) -
#         parent_layer['thickness'] / 2
#     )
#     child_placement_angle = np.radians(child_body['placement_angle'])
#     child_center = (
#         parent_center[0] + child_center_radius * np.cos(child_placement_angle),
#         parent_center[1] + child_center_radius * np.sin(child_placement_angle)
#     )
#     child_body['center'] = child_center
#     child_body['radius'] = parent_radius * parent_layer['thickness'] / 2
#     parent_patch_positions = calculate_parent_patch_positions(
#         parent_body,
#         child_body['parent_layer'],
#         parent_center,
#         parent_radius
#     )
#     child_patch_positions = calculate_child_patch_positions(
#         child_body,
#         parent_radius,
#         parent_center
#     )
#     child_patch_mappings = patch_mappings(child_patch_positions, parent_patch_positions)
#     nearest_densities, _, _ = find_nearest_patches_vectorized(
#         child_patch_mappings,
#         parent_patch_positions
#     )
#     # print(f"\nchild_patch_mappings: \n")
#     # print(f"child_patch_positions {child_patch_positions}")
#     # print_patch_mappings(child_patch_positions, parent_patch_positions, child_patch_mappings)
#     for layer_index, child_layer in enumerate(child_body['layers']):
#         num_micro_layers = child_layer.get('num_micro_layers', 1)
#         if layer_index == 0:
#             if 'original_density_profile' not in child_layer:
#                 child_layer['original_density_profile'] = child_layer['density_profile'][0:1].copy()
#             child_layer['density_profile'][0] = nearest_densities + 0.01
#             outer_layer_increase = child_layer['density_profile'][0] - child_layer['original_density_profile'][0]
#             if num_micro_layers > 1:
#                 child_layer['density_profile'][1:] += outer_layer_increase
#         else:
#             outer_layer_increase = (
#                 child_body['layers'][0]['density_profile'][0] -
#                 child_body['layers'][0]['original_density_profile'][0]
#             )
#             child_layer['density_profile'] += outer_layer_increase[np.newaxis, :]

def update_child_body_density(
    parent_body: Dict[str, Any],
    child_body: Dict[str, Any],
    parent_center: Tuple[float, float] = (0.0, 0.0),
    parent_radius: float = 1.0
    ) -> None:
    """
    Recursively update the density of a child body and all its descendants based on their parents.

    Args:
        parent_body: The parent body.
        child_body: The child body whose density needs to be updated.
        parent_center: Center coordinates of the parent body.
        parent_radius: Radius of the parent body.
    """
    # Update the current child body's density based on its parent
    parent_layer = parent_body['layers'][child_body['parent_layer']]
    num_patches = len(parent_layer['density_profile'][0])
    
    # Calculate child body's position relative to parent
    child_center_radius = parent_radius * (
        1 - sum(l['thickness'] for l in parent_body['layers'][:child_body['parent_layer']]) -
        parent_layer['thickness'] / 2
    )
    child_placement_angle = np.radians(child_body['placement_angle'])
    child_center = (
        parent_center[0] + child_center_radius * np.cos(child_placement_angle),
        parent_center[1] + child_center_radius * np.sin(child_placement_angle)
    )
    child_body['center'] = child_center
    child_body['radius'] = parent_radius * parent_layer['thickness'] / 2

    # Calculate densities
    parent_patch_positions = calculate_parent_patch_positions(
        parent_body,
        child_body['parent_layer'],
        parent_center,
        parent_radius
    )
    child_patch_positions = calculate_child_patch_positions(
        child_body,
        parent_radius,
        parent_center
    )
    child_patch_mappings = patch_mappings(child_patch_positions, parent_patch_positions)
    nearest_densities, _, _ = find_nearest_patches_vectorized(
        child_patch_mappings,
        parent_patch_positions
    )

    # Update densities for all layers of the current child body
    for layer_index, child_layer in enumerate(child_body['layers']):
        num_micro_layers = child_layer.get('num_micro_layers', 1)
        if layer_index == 0:
            if 'original_density_profile' not in child_layer:
                child_layer['original_density_profile'] = child_layer['density_profile'][0:1].copy()
            child_layer['density_profile'][0] = nearest_densities + 0.01
            outer_layer_increase = child_layer['density_profile'][0] - child_layer['original_density_profile'][0]
            if num_micro_layers > 1:
                child_layer['density_profile'][1:] += outer_layer_increase
        else:
            outer_layer_increase = (
                child_body['layers'][0]['density_profile'][0] -
                child_body['layers'][0]['original_density_profile'][0]
            )
            child_layer['density_profile'] += outer_layer_increase[np.newaxis, :]

    # Recursively update all descendants
    for grandchild in child_body.get('child_bodies', []):
        update_child_body_density(
            child_body,  # This child becomes the parent for its own children
            grandchild,
            child_center,  # Pass the updated center
            child_body['radius']  # Pass the updated radius
        )

def find_parent_body(
    current_body: Dict[str, Any],
    target_body: Dict[str, Any],
    parent: Dict[str, Any] = None
    ) -> Dict[str, Any]:
    if current_body == target_body:
        return parent
    for child in current_body.get('child_bodies', []):
        result = find_parent_body(child, target_body, current_body)
        if result:
            return result
    return None

# added on 29-10-2024
def check_and_shed_higher_order_bodies(body, external_density):
    if body.get('can_shed', False) and external_density > body.get('density_threshold', float('inf')):
        if body.get('child_bodies'):
            shed_body = body['child_bodies'].pop(0)
            print(f"{body['name']} has shed higher-order body {shed_body['name']}")
            update_body_properties_after_shedding(body)
        else:
            print(f"{body['name']} cannot shed further; no child bodies to shed")

def update_body_properties_after_shedding(body):
    if body['layers']:
        outermost_layer = body['layers'][0]
        reduction_factor = 0.1  # Adjust as needed
        new_thickness = outermost_layer['thickness'] - reduction_factor * outermost_layer['thickness']
        if new_thickness <= 0:
            body['layers'].pop(0)
            print(f"{body['name']} has reduced its layers by removing an outer layer")
        else:
            outermost_layer['thickness'] = new_thickness
            print(f"{body['name']}'s outer layer thickness reduced to {new_thickness}")
    else:
        print(f"{body['name']} has no layers to adjust after shedding")

def calculate_external_density(body):
    parent_body = find_parent_body(material_object, body)
    if parent_body:
        parent_center = parent_body.get('center', (0, 0))
        parent_radius = parent_body.get('radius', 1)
        parent_layer_index = body['parent_layer']
        parent_layer = parent_body['layers'][parent_layer_index]
        parent_rotation_angle = parent_body.get('rotation_angle', 0.0)
        parent_patch_positions = calculate_parent_patch_positions(
            parent_body,
            parent_layer_index,
            parent_center,
            parent_radius,
            rotation_angle=np.deg2rad(parent_rotation_angle)
        )
        num_patches = len(parent_layer['density_profile'][0])
        placement_angle = (body['placement_angle'] + parent_rotation_angle) % 360
        patch_index = int((placement_angle / 360.0) * num_patches) % num_patches
        micro_layer_index = 0
        external_density = parent_layer['density_profile'][micro_layer_index, patch_index]
        return external_density
    else:
        return 0.0


# GUI Classes


class MaterialBodyCanvas(FigureCanvas):
    """
    Canvas for rendering material bodies using Matplotlib within a PyQt6 widget.
    """

    def __init__(self, material_object, parent=None, width=5, height=4, dpi=100):
        self.fig = plt.Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        self.material_object = material_object
        self.selected_body = material_object
        self.current_zoom = 1.0
        self.compulsory_increase = 0.01
        # Load saved colorbars and set default
        self.load_saved_colorbars()
        self.set_default_colorbar()

        # self.cmap = cm.get_cmap('viridis_r')
        self.show_labels = False  # Added for label visibility
        self.pan_active = False
        self.pan_start = None
        self.zoom_limits = None
        self.num_patches = 32  # Number of patches per layer
        self.arrest_revolutions = False
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.updateGeometry()
        initialize_densities(self.material_object, self.num_patches, self.compulsory_increase)
        for child in self.material_object.get('child_bodies', []):
            update_child_body_density(self.material_object, child)
        self.min_density, self.max_density = self.get_density_range(self.material_object)
        self.mpl_connect('scroll_event', self.on_scroll)
        self.mpl_connect('button_press_event', self.start_pan)
        self.mpl_connect('button_release_event', self.end_pan)
        self.mpl_connect('motion_notify_event', self.on_motion)
        self.plot_material_body()
    def load_saved_colorbars(self):
        """Load saved colorbars from the colorbars directory"""
        self.custom_colorbars = {}
        colorbar_folder = "colorbars"
        if os.path.exists(colorbar_folder):
            for filename in os.listdir(colorbar_folder):
                if filename.endswith('.json'):
                    full_path = os.path.join(colorbar_folder, filename)
                    colorbar = ColorBar.load(full_path)
                    cmap = colorbar.get_matplotlib_colormap()
                    self.custom_colorbars[colorbar.name] = cmap

    def set_default_colorbar(self):
        """Set the default colorbar to 'ab' if available"""
        if 'ab' in self.custom_colorbars:
            self.cmap = self.custom_colorbars['ab']
        else:
            self.cmap = cm.get_cmap('viridis_r')
    # In MaterialBodyCanvas, add method to zoom to specific body:
    # Add this utility function at the class level if not already present
    def find_parent_body(self, current_body: Dict[str, Any], target_body: Dict[str, Any]) -> Dict[str, Any]:
        """Find the parent body of a given target body"""
        if current_body == target_body:
            return None
        for child in current_body.get('child_bodies', []):
            if child == target_body:
                return current_body
            result = self.find_parent_body(child, target_body)
            if result:
                return result
        return None

    def zoom_to_body(self, body_name: str) -> None:
        """Zoom to the extent of the specified body"""
        def find_body(current_body, target_name):
            if current_body['name'] == target_name:
                return current_body
            for child in current_body.get('child_bodies', []):
                result = find_body(child, target_name)
                if result:
                    return result
            return None

        def get_parent_chain(current_body, target_name, chain=None):
            """Get the chain of parent bodies up to the target"""
            if chain is None:
                chain = []
                
            if current_body['name'] == target_name:
                return chain
                
            for child in current_body.get('child_bodies', []):
                if child['name'] == target_name:
                    return chain + [current_body]
                result = get_parent_chain(child, target_name, chain + [current_body])
                if result is not None:
                    return result
            return None

        def calculate_body_position(body, parent_chain):
            """Calculate body position using parent chain"""
            if not parent_chain:  # Root body
                return (0, 0), 1.0, 0.0
                
            center = (0, 0)
            radius = 1.0
            total_rotation = 0.0
            
            # Start from the root and work down to the immediate parent
            for i in range(len(parent_chain)):
                parent = parent_chain[i]
                current = parent_chain[i + 1] if i < len(parent_chain) - 1 else body
                
                parent_layer_idx = current['parent_layer']
                previous_layers_thickness = sum(
                    layer['thickness'] for layer in parent['layers'][:parent_layer_idx]
                )
                current_layer_thickness = parent['layers'][parent_layer_idx]['thickness']
                
                # Update radius
                child_center_radius = radius * (
                    1 - previous_layers_thickness - current_layer_thickness / 2
                )
                radius = radius * current_layer_thickness / 2
                
                # Update rotation
                total_rotation = (total_rotation + parent.get('rotation_angle', 0.0)) % 360
                angle = (current['placement_angle'] + total_rotation) % 360
                angle_rad = np.radians(angle)
                
                # Update center
                new_x = center[0] + child_center_radius * np.cos(angle_rad)
                new_y = center[1] + child_center_radius * np.sin(angle_rad)
                center = (new_x, new_y)
                
            return center, radius, total_rotation

        def find_body_bounds(body, parent_chain):
            """Calculate bounds for the specified body"""
            center, radius, _ = calculate_body_position(body, parent_chain)
            
            return {
                'xmin': center[0] - radius,
                'xmax': center[0] + radius,
                'ymin': center[1] - radius,
                'ymax': center[1] + radius
            }

        # Find target body
        target_body = find_body(self.material_object, body_name)
        if not target_body:
            return

        # Get chain of parent bodies
        parent_chain = get_parent_chain(self.material_object, body_name)
        if parent_chain is None and body_name != self.material_object['name']:
            return
        
        # Calculate bounds for the target body only
        bounds = find_body_bounds(target_body, parent_chain or [])
        
        if bounds:
            # Add padding around the bounds
            width = bounds['xmax'] - bounds['xmin']
            height = bounds['ymax'] - bounds['ymin']
            padding_x = width * 0.2
            padding_y = height * 0.2
            
            self.ax.set_xlim(bounds['xmin'] - padding_x, bounds['xmax'] + padding_x)
            self.ax.set_ylim(bounds['ymin'] - padding_y, bounds['ymax'] + padding_y)
            self.update_limits()
            self.draw()
    def plot_material_body(self):
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        self.plot_body(self.ax, self.material_object)
        self.ax.set_aspect('equal', adjustable='datalim')
        self.ax.axis('off')
        self.fig.tight_layout(pad=0)
        self.update_plot_area()

    def update_plot_area(self):
        self.ax.set_position([0, 0, 1, 1])
        fig_width, fig_height = self.get_width_height()
        aspect_ratio = fig_width / fig_height
        if aspect_ratio > 1:
            self.ax.set_xlim(-aspect_ratio * 1.5, aspect_ratio * 1.5)
            self.ax.set_ylim(-1.5, 1.5)
        else:
            self.ax.set_xlim(-1.5, 1.5)
            self.ax.set_ylim(-1.5 / aspect_ratio, 1.5 / aspect_ratio)
        self.draw_idle()

    def get_layer_color(self, density):
        return self.cmap((density - self.min_density) / (self.max_density - self.min_density))

    def get_density_range(self, body, min_density=float('inf'), max_density=float('-inf')):
        for layer in body['layers']:
            min_density = min(min_density, np.min(layer['density_profile']))
            max_density = max(max_density, np.max(layer['density_profile']))
        for child in body.get('child_bodies', []):
            min_density, max_density = self.get_density_range(child, min_density, max_density)
        return min_density, max_density

    def on_scroll(self, event):
        zoom_factor = 1.15
        xdata, ydata = event.xdata, event.ydata
        if xdata is None or ydata is None:
            return
        if event.button == 'up':
            scale_factor = 1 / zoom_factor
        elif event.button == 'down':
            scale_factor = zoom_factor
        else:
            return
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        new_width = (xlim[1] - xlim[0]) * scale_factor
        new_height = (ylim[1] - ylim[0]) * scale_factor
        relx = (xlim[1] - xdata) / (xlim[1] - xlim[0])
        rely = (ylim[1] - ydata) / (ylim[1] - ylim[0])
        new_xlim = [xdata - new_width * (1 - relx), xdata + new_width * relx]
        new_ylim = [ydata - new_height * (1 - rely), ydata + new_height * rely]
        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        self.update_limits()
        self.current_zoom *= scale_factor
        self.draw()

    def start_pan(self, event):
        if event.button == 3:
            self.pan_active = True
            self.pan_start = (event.x, event.y)

    def end_pan(self, event):
        if event.button == 3:
            self.pan_active = False
        self.update_limits()

    def on_motion(self, event):
        if self.pan_active:
            dx = event.x - self.pan_start[0]
            dy = event.y - self.pan_start[1]
            self.pan_start = (event.x, event.y)
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            scale_x = (xlim[1] - xlim[0]) / self.figure.get_size_inches()[0] / self.figure.dpi
            scale_y = (ylim[1] - ylim[0]) / self.figure.get_size_inches()[1] / self.figure.dpi
            self.ax.set_xlim(xlim[0] - dx * scale_x, xlim[1] - dx * scale_x)
            self.ax.set_ylim(ylim[0] - dy * scale_y, ylim[1] - dy * scale_y)
            self.update_limits()
            self.draw()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.figure.tight_layout()
        self.draw()

    def update_limits(self):
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        self.zoom_limits = (xlim, ylim)

    def save_state(self) -> Dict[str, Any]:
        state = {}
        self._save_body_state(self.material_object, state)
        return state

    def _save_body_state(self, body, state):
        body_state = {
            'placement_angle': body.get('placement_angle'),
            'rotation_angle': body.get('rotation_angle'),
            'layers': [{'density_profile': layer['density_profile'].copy()} for layer in body['layers']],
        }
        state[body['name']] = body_state
        for child in body.get('child_bodies', []):
            self._save_body_state(child, state)

    def restore_state(self, state: Dict[str, Any]) -> None:
        self._restore_body_state(self.material_object, state)
        self.plot_material_body()

    def _restore_body_state(self, body, state):
        if body['name'] in state:
            body_state = state[body['name']]
            if 'placement_angle' in body:
                body['placement_angle'] = body_state['placement_angle']
            if 'rotation_angle' in body:
                body['rotation_angle'] = body_state['rotation_angle']
            for layer, saved_layer in zip(body['layers'], body_state['layers']):
                layer['density_profile'] = saved_layer['density_profile'].copy()
        for child in body.get('child_bodies', []):
            self._restore_body_state(child, state)

    def adjust_child_rotation(self, child, parent, time_step):
        parent_layer_index = child['parent_layer']
        parent_layer = parent['layers'][parent_layer_index]
        num_patches = len(parent_layer['density_profile'][0])
        parent_center = parent.get('center', (0, 0))
        current_placement_angle = child['placement_angle']
        parent_patch_positions = calculate_parent_patch_positions(
            parent,
            parent_layer_index,
            parent_center,
            parent.get('radius', 1),
            rotation_angle=np.deg2rad(parent.get('rotation_angle', 0))
        )
        child_rot_angle = child['rotation_angle']
        rotation_increment = 360.0 / (num_patches * 2)
        test_angles = np.arange(
            child_rot_angle - 3 * rotation_increment,
            child_rot_angle + 5 * rotation_increment,
            rotation_increment
        )

        def calculate_density_diff(angle: float) -> float:
            child_patch_positions = calculate_child_patch_positions(
                child,
                parent.get('radius', 1),
                parent_center,
                rotation_angle=np.deg2rad(angle)
            )
            mappings = patch_mappings(child_patch_positions, parent_patch_positions)
            child_densities = child['layers'][0]['density_profile'][0]
            parent_densities, _, _ = find_nearest_patches_vectorized(
                mappings,
                parent_patch_positions
            )
            return np.sum(np.abs(child_densities - parent_densities))

        result = find_optimal_rotation(test_angles, calculate_density_diff)
        optimal_rotation = sum(result.angles) / len(result.angles)
        child['rotation_angle'] = optimal_rotation % 360
        return optimal_rotation

    # def update_body_and_children(self, body, time_step):
    #     if not self.arrest_revolutions and 'placement_angle' in body:
    #         angular_speed = body.get('angular_speed', 0)
    #         current_placement_angle = body.get('placement_angle', 0)
    #         new_placement_angle = (current_placement_angle + angular_speed * time_step) % 360
    #         body['placement_angle'] = new_placement_angle
    #     if body.get('parent_layer') is not None:
    #         parent_body = find_parent_body(self.material_object, body)
    #         if parent_body:
    #             parent_layer = parent_body['layers'][body['parent_layer']]
    #             parent_radius = parent_body.get('radius', 1)
    #             previous_layers_thickness = sum(
    #                 layer['thickness'] for layer in parent_body['layers'][:body['parent_layer']]
    #             )
    #             current_layer_thickness = parent_layer['thickness']
    #             child_center_radius = parent_radius * (
    #                 1 - previous_layers_thickness - current_layer_thickness / 2
    #             )
    #             placement_angle_rad = np.radians(body['placement_angle'])
    #             parent_center = parent_body.get('center', (0, 0))
    #             body['center'] = (
    #                 parent_center[0] + child_center_radius * np.cos(placement_angle_rad),
    #                 parent_center[1] + child_center_radius * np.sin(placement_angle_rad)
    #             )
    #             body['radius'] = parent_radius * current_layer_thickness / 2
    #             self.adjust_child_rotation(body, parent_body, time_step)
    #     for child in body.get('child_bodies', []):
    #         self.update_body_and_children(child, time_step)
    def update_body_and_children(self, body, time_step):
        if not self.arrest_revolutions and 'placement_angle' in body:
            angular_speed = body.get('angular_speed', 0)
            current_placement_angle = body.get('placement_angle', 0)
            new_placement_angle = (current_placement_angle + angular_speed * time_step) % 360
            body['placement_angle'] = new_placement_angle
        if body.get('parent_layer') is not None:
            parent_body = find_parent_body(self.material_object, body)
            if parent_body:
                parent_layer = parent_body['layers'][body['parent_layer']]
                parent_radius = parent_body.get('radius', 1)
                previous_layers_thickness = sum(
                    layer['thickness'] for layer in parent_body['layers'][:body['parent_layer']]
                )
                current_layer_thickness = parent_layer['thickness']
                child_center_radius = parent_radius * (
                    1 - previous_layers_thickness - current_layer_thickness / 2
                )
                placement_angle_rad = np.radians(body['placement_angle'])
                parent_center = parent_body.get('center', (0, 0))
                body['center'] = (
                    parent_center[0] + child_center_radius * np.cos(placement_angle_rad),
                    parent_center[1] + child_center_radius * np.sin(placement_angle_rad)
                )
                body['radius'] = parent_radius * current_layer_thickness / 2
                self.adjust_child_rotation(body, parent_body, time_step)
                # Calculate external density
                external_density = calculate_external_density(body)
                # Check for shedding
                check_and_shed_higher_order_bodies(body, external_density)
                # Update densities if necessary
                update_child_body_density(parent_body, body, parent_center, parent_radius)
        for child in body.get('child_bodies', []):
            self.update_body_and_children(child, time_step)
    def plot_body(
        self,
        ax,
        material: Dict[str, Any],
        center: Tuple[float, float] = (0, 0),
        radius: float = 1.0,
        is_child: bool = False,
        parent_rotation: float = 0.0
        ) -> None:
        layers = material['layers']
        rotation_angle = material.get('rotation_angle', 0.0)
        total_rotation = (rotation_angle + parent_rotation) % 360
        current_radius = radius
        # Always plot the name at center, regardless of show_labels setting
        ax.text(
            center[0], center[1], material['name'], 
            ha='center', va='center',
            fontsize=10, fontweight='bold', 
            color='black', 
            rotation=total_rotation
        )
        for i, layer in enumerate(layers):
            layer_thickness = layer['thickness'] * radius
            next_radius = current_radius - layer_thickness
            num_micro_layers = layer.get('num_micro_layers', 1)
            num_patches = self.num_patches
            micro_layer_thickness = layer_thickness / num_micro_layers
            for j in range(num_micro_layers):
                r = current_radius - j * micro_layer_thickness
                patches = []
                for k in range(num_patches):
                    start_angle = (k * 360 / num_patches + total_rotation) % 360
                    end_angle = ((k + 1) * 360 / num_patches + total_rotation) % 360
                    density = layer['density_profile'][j, k]
                    color = self.get_layer_color(density)
                    wedge = Wedge(
                        center, r, start_angle, end_angle,
                        width=micro_layer_thickness, facecolor=color, edgecolor='none'
                    )
                    patches.append(wedge)
                    # Add text labels if enabled
                    if self.show_labels:
                        mid_angle = np.radians((start_angle + end_angle) / 2)
                        text_r = r - micro_layer_thickness / 2
                        text_x = center[0] + text_r * np.cos(mid_angle)
                        text_y = center[1] + text_r * np.sin(mid_angle)
                        ax.text(
                            text_x, text_y, f"{j},{k}\n{density:.2f}",
                            ha='center', va='center', fontsize=8
                        )
                collection = PatchCollection(patches, match_original=True)
                ax.add_collection(collection)
            current_radius = next_radius
        outer_circle = Circle(center, radius, fill=False, edgecolor='black', linewidth=2)
        ax.add_artist(outer_circle)
        if self.show_labels:
            ax.text(
                center[0], center[1], material['name'], ha='center', va='center',
                fontsize=10, fontweight='bold', color='black', rotation=total_rotation
            )
        if 'rotation_angle' in material:
            marker_angle = np.radians(total_rotation)
            marker_radius = radius * 0.8
            marker_x = center[0] + marker_radius * np.cos(marker_angle)
            marker_y = center[1] + marker_radius * np.sin(marker_angle)
            ax.plot([center[0], marker_x], [center[1], marker_y], color='red', linewidth=2)
        for child in material.get('child_bodies', []):
            parent_layer = child['parent_layer']
            placement_angle = (child['placement_angle'] + total_rotation) % 360
            parent_layers = layers[:parent_layer + 1]
            parent_outer_radius = radius - sum(layer['thickness'] for layer in layers[:parent_layer]) * radius
            parent_inner_radius = radius - sum(layer['thickness'] for layer in parent_layers) * radius
            child_radius = (parent_outer_radius - parent_inner_radius) / 2
            child_center_radius = (parent_outer_radius + parent_inner_radius) / 2
            child_center = (
                center[0] + child_center_radius * np.cos(np.radians(placement_angle)),
                center[1] + child_center_radius * np.sin(np.radians(placement_angle))
            )
            self.plot_body(
                ax, child, child_center, child_radius, is_child=True, parent_rotation=total_rotation
            )

class MaterialBodySimulator(QMainWindow):
    """
    The main application window that orchestrates the simulation.
    """

    def __init__(self, material_object):
        super().__init__()
        self.material_object = material_object
        self.selected_body = material_object
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.animate_placement)
        self.is_playing = False
        self.time_step = 1  # Time step for each animation frame
        self.state_history = []
        self.current_state_index = -1
        self.initial_state = None
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Material Body Simulator')
        self.setGeometry(100, 100, 800, 600)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.canvas = MaterialBodyCanvas(self.material_object, self)
        self.canvas.arrest_revolutions = False
        layout.addWidget(self.canvas)
        self.create_toolbar()

    def create_toolbar(self):
        toolbar = QToolBar()
        self.addToolBar(toolbar)
        self.body_selector = QComboBox()
        self.populate_body_selector(self.material_object)
        self.body_selector.currentTextChanged.connect(self.on_body_selected)
        toolbar.addWidget(self.body_selector)

        # Add zoom body selector
        self.zoom_selector = QComboBox()
        self.zoom_selector.addItem("Full View")  # Add option for full view
        self.populate_body_selector(self.material_object, combobox=self.zoom_selector)
        self.zoom_selector.currentTextChanged.connect(self.on_zoom_body_selected)
        toolbar.addWidget(QLabel("Zoom to:"))
        toolbar.addWidget(self.zoom_selector)


        self.arrest_checkbox = QCheckBox("Arrest Revolutions")
        self.arrest_checkbox.stateChanged.connect(self.toggle_arrest_revolutions)
        toolbar.addWidget(self.arrest_checkbox)
        self.play_pause_action = QAction('Play', self)
        self.play_pause_action.triggered.connect(self.toggle_animation)
        toolbar.addAction(self.play_pause_action)
        self.play_step_action = QAction('Step Forward', self)
        self.play_step_action.triggered.connect(self.step_animation)
        toolbar.addAction(self.play_step_action)
        self.step_back_action = QAction('Step Back', self)
        self.step_back_action.triggered.connect(self.step_back)
        toolbar.addAction(self.step_back_action)
        self.restart_action = QAction('Restart', self)
        self.restart_action.triggered.connect(self.restart_simulation)
        toolbar.addAction(self.restart_action)
        # Visualization Settings
        visualization_action = QAction('Visualization Settings', self)
        visualization_action.triggered.connect(self.open_visualization_settings)
        toolbar.addAction(visualization_action)
        # Edit Bodies and Layers
        edit_bodies_action = QAction('Edit Bodies/Layers', self)
        edit_bodies_action.triggered.connect(self.open_edit_bodies_dialog)
        toolbar.addAction(edit_bodies_action)
        change_properties_action = QAction('Change Properties', self)
        change_properties_action.triggered.connect(self.open_properties_dialog)
        toolbar.addAction(change_properties_action)
        save_project_action = QAction('Save Project', self)
        save_project_action.triggered.connect(self.save_project)
        toolbar.addAction(save_project_action)
        load_project_action = QAction('Load Project', self)
        load_project_action.triggered.connect(self.load_project)
        toolbar.addAction(load_project_action)
    def on_zoom_body_selected(self, selected_name):
        """Handle zoom body selection"""
        if selected_name == "Full View":
            # Reset to full view
            self.canvas.ax.set_xlim(-1.5, 1.5)
            self.canvas.ax.set_ylim(-1.5, 1.5)
            self.canvas.update_limits()
            self.canvas.draw()
        else:
            # Zoom to selected body
            self.canvas.zoom_to_body(selected_name.strip())
    # Modify populate_body_selector to work with both selectors
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.canvas.fig.tight_layout(pad=0)
        self.canvas.draw()

    def toggle_arrest_revolutions(self, state):
        self.canvas.arrest_revolutions = state == Qt.CheckState.Checked

    def step_animation(self):
        if self.selected_body:
            current_state = self.canvas.save_state()
            self.state_history = self.state_history[:self.current_state_index + 1]
            self.state_history.append(current_state)
            self.current_state_index += 1
            self.canvas.update_body_and_children(self.selected_body, self.time_step)
            self.mindful_plot()

    def mindful_plot(self):
        current_xlim = self.canvas.ax.get_xlim()
        current_ylim = self.canvas.ax.get_ylim()
        self.canvas.plot_material_body()
        self.canvas.ax.set_xlim(current_xlim)
        self.canvas.ax.set_ylim(current_ylim)
        self.canvas.draw()

    def step_back(self):
        if self.current_state_index > 0:
            current_xlim = self.canvas.ax.get_xlim()
            current_ylim = self.canvas.ax.get_ylim()
            self.current_state_index -= 1
            previous_state = self.state_history[self.current_state_index]
            self.canvas.restore_state(previous_state)
            self.canvas.ax.set_xlim(current_xlim)
            self.canvas.ax.set_ylim(current_ylim)
            self.canvas.draw()

    def restart_simulation(self):
        if self.initial_state:
            current_xlim = self.canvas.ax.get_xlim()
            current_ylim = self.canvas.ax.get_ylim()
            self.canvas.restore_state(self.initial_state)
            self.state_history = [self.initial_state]
            self.current_state_index = 0
            self.canvas.ax.set_xlim(current_xlim)
            self.canvas.ax.set_ylim(current_ylim)
            self.canvas.draw()

    def animate_placement(self):
        if self.selected_body:
            current_state = self.canvas.save_state()
            self.state_history = self.state_history[:self.current_state_index + 1]
            self.state_history.append(current_state)
            self.current_state_index += 1
            self.canvas.update_body_and_children(self.selected_body, self.time_step)
            self.mindful_plot()

    def toggle_animation(self):
        if self.is_playing:
            self.animation_timer.stop()
            self.play_pause_action.setText('Play')
        else:
            self.animation_timer.start(100)  # 0.1 second interval
            self.play_pause_action.setText('Pause')
        self.is_playing = not self.is_playing

    def open_properties_dialog(self):
        all_bodies = self.get_all_bodies(self.material_object)
        dialog = PropertiesDialog(all_bodies, self)
        if dialog.exec():
            updated_bodies = dialog.get_updated_properties()
            self.update_bodies(self.material_object, updated_bodies)
            self.mindful_plot()

    def open_edit_bodies_dialog(self):
        dialog = EditBodiesDialog(self.material_object, self)
        if dialog.exec():
            # You would need to implement updating the material_object based on the dialog's output.
            # For now, we'll just refresh the plot.
            self.mindful_plot()

    def open_visualization_settings(self):
        dialog = VisualizationSettingsDialog(self.canvas, self)
        dialog.exec()

    def get_all_bodies(self, body):
        bodies = [body]
        for child in body.get('child_bodies', []):
            bodies.extend(self.get_all_bodies(child))
        return bodies

    def update_bodies(self, body, updated_bodies):
        for updated_body in updated_bodies:
            if body['name'] == updated_body['name']:
                body.update(updated_body)
                break
        for child in body.get('child_bodies', []):
            self.update_bodies(child, updated_bodies)

    def save_project(self):
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Project", "", "Pickle Files (*.pkl)")
        if file_name:
            project_data = {
                'material_object': self.material_object,
                'selected_body': self.selected_body,
                'zoom_limits': self.canvas.zoom_limits
            }
            with open(file_name, 'wb') as f:
                pickle.dump(project_data, f)

    def load_project(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Project", "", "Pickle Files (*.pkl)")
        if file_name:
            with open(file_name, 'rb') as f:
                project_data = pickle.load(f)
            self.material_object = project_data['material_object']
            self.selected_body = project_data['selected_body']
            self.canvas.zoom_limits = project_data['zoom_limits']
            self.canvas.material_object = self.material_object
            self.canvas.selected_body = self.selected_body
            self.body_selector.clear()
            self.populate_body_selector(self.material_object)
            self.canvas.plot_material_body()

    def populate_body_selector(self, body, prefix='', combobox=None):
        if combobox is None:
            combobox = self.body_selector
        combobox.addItem(f"{prefix}{body['name']}")
        for child in body.get('child_bodies', []):
            self.populate_body_selector(child, prefix + '  ', combobox)

    def on_body_selected(self, selected_name):
        self.selected_body = self.find_body_by_name(self.material_object, selected_name.strip())
        self.canvas.selected_body = self.selected_body
        self.mindful_plot()

    def find_body_by_name(self, body, name):
        if body['name'] == name.strip():
            return body
        for child in body.get('child_bodies', []):
            result = self.find_body_by_name(child, name)
            if result:
                return result
        return None


if __name__ == '__main__':
    # Define your material object with bodies A, B, C
    material_object = {
        "name": "A",
        "color": "#00dd00",
        "placement_angle": 90,
        "angular_speed": 0.0,
        "rotation_angle": 0.0,
        "rotation_speed": 0.0,
        "show_label": False,
        # "use_micro_layers": True,
        # "num_micro_layers": 3,
        "can_shed": False,
        "layers": [
            {"thickness": 0.4, "density": 0.2, "num_micro_layers": 6},
            {"thickness": 0.2, "density": 0.3, "num_micro_layers": 2},
            {"thickness": 0.2, "density": 0.5, "num_micro_layers": 2},
            {"thickness": 0.2, "density": 0.8, "num_micro_layers": 1},
        ],
        "child_bodies": [
            {
                "parent_layer": 0,
                "placement_angle": 90,
                "angular_speed": 10,
                "rotation_angle": 0.0,
                "rotation_speed": 0.0,
                "name": "B",
                "color": "#0000dd",
                "show_label": False,
                "can_shed": False,
                # "use_micro_layers": True,
                # "num_micro_layers": 2,
                "density_tolerance": 0.001,
                "layers": [
                    {"thickness": 0.4, "density": 0.2, "num_micro_layers": 6},
                    {"thickness": 0.2, "density": 0.4, "num_micro_layers": 2},
                    {"thickness": 0.2, "density": 0.6, "num_micro_layers": 1},
                ],
                "child_bodies": [
                    {
                        "parent_layer": 0,
                        "placement_angle": 90,
                        "angular_speed": 10,
                        "rotation_angle": 0.0,
                        "rotation_speed": 0.0,
                        "name": "C",
                        "color": "#0000dd",
                        "show_label": False,
                        "can_shed": True,
                        # "use_micro_layers": True,
                        # "num_micro_layers": 2,
                        "density_tolerance": 0.001,
                        "layers": [
                            {"thickness": 0.4, "density": 0.2, "num_micro_layers": 6},
                            {"thickness": 0.2, "density": 0.4, "num_micro_layers": 1},
                            # {"thickness": 0.2, "density": 0.6, "num_micro_layers": 1},
                        ],
                        "child_bodies": [
                            {
                                "parent_layer": 0,
                                "placement_angle": 90,
                                "angular_speed": 10,
                                "rotation_angle": 0.0,
                                "rotation_speed": 0.0,
                                "name": "D",
                                "color": "#0000dd",
                                "show_label": False,
                                "can_shed": True,
                                # "use_micro_layers": True,
                                # "num_micro_layers": 2,
                                "density_tolerance": 0.001,
                                "layers": [
                                    {"thickness": 0.4, "density": 0.2, "num_micro_layers": 6},
                                    {"thickness": 0.2, "density": 0.4, "num_micro_layers": 1},
                                    # {"thickness": 0.2, "density": 0.6, "num_micro_layers": 1},
                                ],
                                "child_bodies": [
                                    {
                                        "parent_layer": 0,
                                        "placement_angle": 90,
                                        "angular_speed": 10,
                                        "rotation_angle": 0.0,
                                        "rotation_speed": 0.0,
                                        "name": "E",
                                        "color": "#0000dd",
                                        "show_label": False,
                                        "can_shed": False,
                                        # "use_micro_layers": True,
                                        # "num_micro_layers": 2,
                                        "density_tolerance": 0.001,
                                        "layers": [
                                            {"thickness": 0.4, "density": 0.2, "num_micro_layers": 2},
                                            {"thickness": 0.2, "density": 0.4, "num_micro_layers": 1},
                                            # {"thickness": 0.2, "density": 0.6, "num_micro_layers": 1},
                                        ],
                                        "child_bodies": []
                                    }

                                ]
                            }
                        ]
                    }
                ]
            }
        ]
    }
    app = QApplication(sys.argv)
    simulator = MaterialBodySimulator(material_object)
    simulator.show()
    sys.exit(app.exec())
