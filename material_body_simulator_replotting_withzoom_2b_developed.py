#Preivous version
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
        # Ensure that dl2 is greater than dl1 by at least compulsory_increase
        if layer_index < len(body['layers']) - 1:
            dl2 = body['layers'][layer_index + 1]['density']
        else:
            dl2 = dl1 + compulsory_increase  # If it's the innermost layer, set dl2 accordingly

        increment = (dl2 - dl1) / (num_micro_layers + 1)  # Adjusted increment calculation
        for micro_layer_index in range(num_micro_layers):
            density = dl1 + (micro_layer_index + 1) * increment  # Start from dl1 + increment
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
    parent_patch_positions: Dict[str, Any],
    parent_center: Tuple[float, float]
    ) -> Dict[int, List[int]]:
    """
    Maps child patches to the nearest parent patches using direct distance between patch centers.

    Args:
        child_patch_positions: Dictionary containing child patch coordinates and info.
        parent_patch_positions: Dictionary containing parent patch coordinates and info.
        parent_center: Center coordinates of parent body (x, y)

    Returns:
        Dictionary mapping child patch indices to parent patch information.
    """
    # Get child patch coordinates
    child_x = child_patch_positions['x_coords'].flatten()
    child_y = child_patch_positions['y_coords'].flatten()
    
    # Get parent patch coordinates for all micro-layers
    parent_x = parent_patch_positions['x_coords']  # shape: (num_micro_layers, num_patches)
    parent_y = parent_patch_positions['y_coords']
    parent_radii = parent_patch_positions['radii']
    
    num_child_patches = len(child_x)
    num_parent_micro_layers = len(parent_radii)
    
    # Calculate layer boundaries for radial distance check
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
        # Calculate child patch's radial distance from parent center
        child_radial_distance = np.sqrt(
            (child_x[child_idx] - parent_center[0]) ** 2 + 
            (child_y[child_idx] - parent_center[1]) ** 2
        )
        
        # Find appropriate micro-layer based on radial distance
        micro_layer_idx = None
        for idx, (inner_r, outer_r) in enumerate(layer_boundaries):
            if inner_r <= child_radial_distance <= outer_r:
                micro_layer_idx = idx
                break
        
        if micro_layer_idx is None:
            distances = [
                min(abs(child_radial_distance - inner_r), abs(child_radial_distance - outer_r))
                for inner_r, outer_r in layer_boundaries
            ]
            micro_layer_idx = np.argmin(distances)
        
        # Calculate distances to all patches in the selected micro-layer
        distances_to_parent_patches = np.sqrt(
            (parent_x[micro_layer_idx] - child_x[child_idx]) ** 2 +
            (parent_y[micro_layer_idx] - child_y[child_idx]) ** 2
        )
        
        # Find the closest parent patch
        patch_idx = np.argmin(distances_to_parent_patches)
        
        mappings[child_idx] = [0, micro_layer_idx, patch_idx]
    
    return mappings

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
    child_placement_angle = np.radians(child_body['placement_angle'] + parent_body.get('rotation_angle', 0))
    child_center = (
        parent_center[0] + child_center_radius * np.cos(child_placement_angle),
        parent_center[1] + child_center_radius * np.sin(child_placement_angle)
    )
    child_body['center'] = child_center
    child_body['radius'] = parent_radius * parent_layer['thickness'] / 2

    # Calculate parent patch positions with respect to parent center
    parent_patch_positions = calculate_parent_patch_positions(
        parent_body,
        child_body['parent_layer'],
        parent_center,
        parent_radius
    )

    # Calculate child patch positions with respect to parent center
    child_patch_positions = calculate_child_patch_positions(
        child_body,
        parent_radius,
        parent_center
    )

    # Map patches using the correct parent center
    child_patch_mappings = patch_mappings(
        child_patch_positions, 
        parent_patch_positions,
        parent_center
    )
    print(f"\nchild_patch_mappings for Body {child_body['name']}: \n")
    print_patch_mappings(child_patch_positions, parent_patch_positions, child_patch_mappings,parent_center,)
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

    # Print debug information
    print(f"\nUpdating {child_body['name']} (child of {parent_body['name']})")
    print(f"Parent center: {parent_center}")
    print(f"Child center: {child_center}")
    print(f"Child radius: {child_body['radius']}")
    print(f"Number of patches mapped: {len(child_patch_mappings)}")
    print(f"First few density updates: {nearest_densities[:5]}")

    # Recursively update all descendants with the new child center and radius
    for grandchild in child_body.get('child_bodies', []):
        update_child_body_density(
            child_body,  # This child becomes the parent for its own children
            grandchild,
            child_center,  # Pass the child's center as the new parent center
            child_body['radius']  # Pass the child's radius as the new parent radius
        )

# Also update the calculate_child_patch_positions function to properly handle parent center
def calculate_child_patch_positions(
    child_body: Dict[str, Any],
    parent_radius: float,
    parent_center: Tuple[float, float],
    rotation_angle: float = 0.0
    ) -> Dict[str, Any]:
    """
    Calculate patch positions for child body's outer layer with respect to parent center.

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
    
    # Calculate patch positions relative to parent center
    patch_positions = calculate_all_patch_centers(
        child_body,
        child_center,  # Use child's center, which is already relative to parent
        child_radius,
        layer_thicknesses,
        num_patches,
        layer_indices=layer_indices,
        num_micro_layers=num_micro_layers,
        rotation_angle=rotation_angle
    )
    return patch_positions[0]
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
        self.previous_plot_detail = {}  # For caching
        self.body_extents = {}  # To track body positions and sizes

        # Load saved colorbars and set default
        self.load_saved_colorbars()
        self.set_default_colorbar()
        self.label_texts = []  # List to store label Text objects

        self.pan_active = False
        self.pan_start = None
        self.zoom_limits = None
        self.num_patches = 32  # Number of patches per layer
        self.arrest_revolutions = False

        self.show_labels = True
        self.label_settings = {
            'show_density': True,
            'show_layer_index': True,
            'show_micro_layer_index': True,
            'show_patch_index': True,
            'font_size': 10,
            'min_patch_width_density': 50,     # minimum pixel width to show density
            'min_patch_width_indices': 80,     # minimum pixel width to show indices
            'min_patch_width_all': 120,        # minimum pixel width to show all labels
        }

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
        self.mpl_connect('draw_event', self.on_draw)  # Connect draw event for caching

        self.ax = self.fig.add_subplot(111)
        self.ax.callbacks.connect('xlim_changed', self.update_labels)
        self.ax.callbacks.connect('ylim_changed', self.update_labels)
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

    def on_draw(self, event):
        """Handle the draw event to update the previous plot detail."""
        self.previous_plot_detail.clear()

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

        def calculate_body_bounds(body, center, radius, parent_rotation):
            """Calculate bounds for the specified body including its children"""
            bounds = {
                'xmin': center[0] - radius,
                'xmax': center[0] + radius,
                'ymin': center[1] - radius,
                'ymax': center[1] + radius
            }
            rotation_angle = body.get('rotation_angle', 0.0)
            total_rotation = (rotation_angle + parent_rotation) % 360
            layers = body['layers']
            current_radius = radius
            for layer in layers:
                layer_thickness = layer['thickness'] * radius
                current_radius -= layer_thickness

            # Handle child bodies
            for child in body.get('child_bodies', []):
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
                child_bounds = calculate_body_bounds(child, child_center, child_radius, total_rotation)
                bounds['xmin'] = min(bounds['xmin'], child_bounds['xmin'])
                bounds['xmax'] = max(bounds['xmax'], child_bounds['xmax'])
                bounds['ymin'] = min(bounds['ymin'], child_bounds['ymin'])
                bounds['ymax'] = max(bounds['ymax'], child_bounds['ymax'])
            return bounds

        # Find target body
        target_body = find_body(self.material_object, body_name)
        if not target_body:
            return

        # Calculate bounds for the target body including its children
        bounds = calculate_body_bounds(target_body, (0, 0), 1.0, 0.0)

        if bounds:
            # Add padding around the bounds
            width = bounds['xmax'] - bounds['xmin']
            height = bounds['ymax'] - bounds['ymin']
            padding_x = width * 0.1
            padding_y = height * 0.1

            self.ax.set_xlim(bounds['xmin'] - padding_x, bounds['xmax'] + padding_x)
            self.ax.set_ylim(bounds['ymin'] - padding_y, bounds['ymax'] + padding_y)
            self.update_limits()
            self.redraw_cached_plot()
            self.draw_idle()

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

    def start_pan(self, event):
        if event.button == 3:
            self.pan_active = True
            self.pan_start = (event.x, event.y)

    def end_pan(self, event):
        if event.button == 3:
            self.pan_active = False
        self.update_limits()

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
            mappings = patch_mappings(
                child_patch_positions, 
                parent_patch_positions,
                parent_center  # Pass the parent_center here
            )
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
        relx = (xdata - xlim[0]) / (xlim[1] - xlim[0])
        rely = (ydata - ylim[0]) / (ylim[1] - ylim[0])
        new_xlim = [xdata - new_width * relx, xdata + new_width * (1 - relx)]
        new_ylim = [ydata - new_height * rely, ydata + new_height * (1 - rely)]
        self.ax.set_xlim(new_xlim)
        self.ax.set_ylim(new_ylim)
        self.update_limits()
        self.current_zoom *= scale_factor
        self.redraw_cached_plot()  # Use efficient redraw
        self.draw_idle()

    def on_motion(self, event):
        if self.pan_active:
            dx = event.x - self.pan_start[0]
            dy = event.y - self.pan_start[1]
            self.pan_start = (event.x, event.y)
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            scale_x = (xlim[1] - xlim[0]) / self.figure.get_size_inches()[0] / self.figure.dpi
            scale_y = (ylim[1] - ylim[0]) / self.figure.get_size_inches()[1] / self.figure.dpi
            new_xlim = (xlim[0] - dx * scale_x, xlim[1] - dx * scale_x)
            new_ylim = (ylim[0] - dy * scale_y, ylim[1] - dy * scale_y)
            self.ax.set_xlim(new_xlim)
            self.ax.set_ylim(new_ylim)
            self.update_limits()
            self.redraw_cached_plot()  # Use efficient redraw
            self.draw_idle()

    def redraw_cached_plot(self):
        """Efficiently redraw the plot by leveraging caching."""
        self.ax.cla()
        xlim, ylim = self.zoom_limits
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.axis('off')

        # Reset plot state
        self.label_texts = []
        self.label_data = []
        if not hasattr(self, 'body_extents'):
            self.body_extents = {}

        # Plot visible bodies with appropriate detail
        self.plot_material_body_efficient(self.material_object)

        self.ax.set_aspect('equal', adjustable='datalim')
        self.fig.tight_layout(pad=0)
        self.update_labels()

    def plot_material_body_efficient(self, material, center=(0, 0), radius=1.0, parent_rotation=0.0):
        """Efficient version of plot_body that only draws visible elements"""
        if not self.is_body_visible(center, radius):
            return

        detail_level = self.get_plot_detail(material, radius)
        self._plot_body_details(material, center, radius, parent_rotation, detail_level)

        # Handle child bodies
        for child in material.get('child_bodies', []):
            parent_layer = child['parent_layer']
            placement_angle = (child['placement_angle'] + parent_rotation + material.get('rotation_angle', 0.0)) % 360
            layers = material['layers']
            parent_layers = layers[:parent_layer + 1]
            parent_outer_radius = radius - sum(layer['thickness'] for layer in layers[:parent_layer]) * radius
            parent_inner_radius = radius - sum(layer['thickness'] for layer in parent_layers) * radius
            child_radius = (parent_outer_radius - parent_inner_radius) / 2
            child_center_radius = (parent_outer_radius + parent_inner_radius) / 2
            child_center = (
                center[0] + child_center_radius * np.cos(np.radians(placement_angle)),
                center[1] + child_center_radius * np.sin(np.radians(placement_angle))
            )
            self.plot_material_body_efficient(child, child_center, child_radius, (parent_rotation + material.get('rotation_angle', 0.0)) % 360)

    def _plot_body_details(self, material, center, radius, parent_rotation, detail_level):
        layers = material['layers']
        rotation_angle = material.get('rotation_angle', 0.0)
        total_rotation = (rotation_angle + parent_rotation) % 360
        current_radius = radius

        # Plot body name only if it fits within the view
        if self.is_point_visible(center):
            self.label_texts.append(
                self.ax.text(
                    center[0], center[1], material['name'],
                    ha='center', va='center',
                    fontsize=10, fontweight='bold',
                    color='black',
                    rotation=total_rotation
                )
            )
        self.body_extents[material['name']] = {'center': center, 'radius': radius}

        # Plot layers based on detail level
        for i, layer in enumerate(layers):
            layer_thickness = layer['thickness'] * radius
            next_radius = current_radius - layer_thickness

            if detail_level == 'outline':
                # Only plot the outline
                pass  # We will draw the outer_circle at the end
            elif detail_level == 'layers':
                # Plot layers without micro-layers or patches
                circle = Circle(center, current_radius, fill=False, edgecolor='black', linewidth=1)
                self.ax.add_artist(circle)
            elif detail_level == 'micro_layers':
                num_micro_layers = layer.get('num_micro_layers', 1)
                micro_layer_thickness = layer_thickness / num_micro_layers
                for j in range(num_micro_layers):
                    r = current_radius - j * micro_layer_thickness
                    circle = Circle(center, r, fill=False, edgecolor='black', linewidth=0.5)
                    self.ax.add_artist(circle)
            elif detail_level == 'patches':
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
                        # Collect label data
                        self.collect_label_data(
                            center, r, micro_layer_thickness,
                            start_angle, end_angle,
                            density, i, j, k, total_rotation,
                            body_name=material['name']  # Pass body name
                        )
                    collection = PatchCollection(patches, match_original=True)
                    self.ax.add_collection(collection)
            current_radius = next_radius

        outer_circle = Circle(center, radius, fill=False, edgecolor='black', linewidth=2)
        self.ax.add_artist(outer_circle)

    def is_body_visible(self, center, radius):
        """Check if the body is visible within the current view limits"""
        x0, y0 = center
        xmin, xmax = self.ax.get_xlim()
        ymin, ymax = self.ax.get_ylim()
        return not (x0 + radius < xmin or x0 - radius > xmax or y0 + radius < ymin or y0 - radius > ymax)

    def is_point_visible(self, point):
        """Check if a point is within the current view limits"""
        x0, y0 = point
        xmin, xmax = self.ax.get_xlim()
        ymin, ymax = self.ax.get_ylim()
        return xmin <= x0 <= xmax and ymin <= y0 <= ymax

    def get_plot_detail(self, material, radius):
        """Determine the level of detail to plot for the given material body."""
        # Compute zoom scale and decide detail level
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        data_width = xlim[1] - xlim[0]
        data_height = ylim[1] - ylim[0]
        bbox = self.ax.get_window_extent()
        view_scale = bbox.width / data_width  # pixels per data unit

        body_pixel_size = radius * view_scale

        if body_pixel_size < 20:
            return 'outline'
        elif body_pixel_size < 50:
            return 'layers'
        elif body_pixel_size < 100:
            return 'micro_layers'
        else:
            return 'patches'

    def collect_label_data(
        self, center, r, micro_layer_thickness,
        start_angle, end_angle,
        density, layer_idx, micro_layer_idx, patch_idx, total_rotation,
        body_name
    ):
        if self.show_labels:
            # Calculate patch center
            mid_angle = np.radians((start_angle + end_angle) / 2)
            text_r = r - micro_layer_thickness / 2
            text_x = center[0] + text_r * np.cos(mid_angle)
            text_y = center[1] + text_r * np.sin(mid_angle)
            # Store label data for later use
            self.label_data.append({
                'text_x': text_x,
                'text_y': text_y,
                'mid_angle': mid_angle,
                'density': density,
                'layer_idx': layer_idx,
                'micro_layer_idx': micro_layer_idx,
                'patch_idx': patch_idx,
                'rotation': total_rotation,
                'start_angle': start_angle,
                'end_angle': end_angle,
                'r': r,
                'body_name': body_name  # Store body name
            })

    def update_labels(self, event=None):
        # Clear existing labels
        for text in self.label_texts:
            text.remove()
        self.label_texts = []
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        bbox = self.ax.get_window_extent()
        data_width = xlim[1] - xlim[0]
        data_height = ylim[1] - ylim[0]
        view_scale = bbox.width / data_width  # pixels per data unit

        min_pixel_size = 20  # Minimum pixel size for labels
        max_step = 8  # Maximum step size
        # Create a dictionary to track if labels are displayed for each body
        body_labels_displayed = {body_name: False for body_name in self.body_extents.keys()}

        def is_body_partially_visible(center, radius, xlim, ylim):
            x0, y0 = center
            xmin, xmax = xlim
            ymin, ymax = ylim
            return not (x0 + radius < xmin or x0 - radius > xmax or y0 + radius < ymin or y0 - radius > ymax)

        for data in self.label_data:
            body_name = data['body_name']
            center = self.body_extents[body_name]['center']
            radius = self.body_extents[body_name]['radius']

            # Check if the body is at least partially visible
            if not is_body_partially_visible(center, radius, xlim, ylim):
                continue  # Skip labels for this body

            text_x = data['text_x']
            text_y = data['text_y']
            mid_angle = data['mid_angle']
            density = data['density']
            layer_idx = data['layer_idx']
            micro_layer_idx = data['micro_layer_idx']
            patch_idx = data['patch_idx']
            start_angle = data['start_angle']
            end_angle = data['end_angle']
            r = data['r']

            # Compute angular width in radians
            delta_theta = (end_angle - start_angle) * np.pi / 180

            # Compute arc length in data units
            s = r * delta_theta

            # Compute arc length in pixels
            s_pixels = s * view_scale

            # Compute step size
            step = max(1, int(self.label_settings['min_patch_width_density'] / s_pixels))
            step = min(step, max_step)

            # Decide whether to show label
            if patch_idx % step == 0:
                # Get label content based on patch width
                label_content = self.get_visible_label_content(
                    density=density,
                    layer_idx=layer_idx,
                    micro_layer_idx=micro_layer_idx,
                    patch_idx=patch_idx,
                    patch_width_pixels=s_pixels
                )

                if label_content:
                    text = self.ax.text(
                        text_x, text_y, label_content,
                        ha='center', va='center',
                        fontsize=self.label_settings['font_size'],
                        rotation=np.degrees(mid_angle) + 90
                    )
                    self.label_texts.append(text)
                    body_labels_displayed[body_name] = True  # Mark that labels are displayed for this body

        self.draw_idle()

    def plot_material_body(self):
        self.fig.clear()
        self.ax = self.fig.add_subplot(111)
        self.label_texts = []
        self.label_data = []
        self.body_extents = {}  # Initialize body extents
        self.body_parent_map = {}  # Initialize body parent map
        self.build_body_parent_map(self.material_object)
        self.plot_material_body_efficient(self.material_object)
        self.ax.set_aspect('equal', adjustable='datalim')
        self.ax.axis('off')
        self.fig.tight_layout(pad=0)
        self.update_plot_area()
        self.update_labels()

    def build_body_parent_map(self, body, parent_name=None):
        """Build a map of child body names to their parent body names."""
        if not hasattr(self, 'body_parent_map'):
            self.body_parent_map = {}
        for child in body.get('child_bodies', []):
            self.body_parent_map[child['name']] = body['name']
            self.build_body_parent_map(child, body['name'])

    def get_visible_label_content(self, density, layer_idx, micro_layer_idx, patch_idx, patch_width_pixels):
        """Determine what labels should be visible based on patch width"""
        if patch_width_pixels < self.label_settings['min_patch_width_density']:
            return ""

        parts = []
        if patch_width_pixels >= self.label_settings['min_patch_width_all']:
            # Show all enabled labels
            if self.label_settings['show_density']:
                parts.append(f"{density:.2f}")
            if self.label_settings['show_layer_index']:
                parts.append(f"L{layer_idx}, M{micro_layer_idx}")
            if self.label_settings['show_patch_index']:
                parts.append(f"P{patch_idx}")
        elif patch_width_pixels >= self.label_settings['min_patch_width_indices']:
            # Show only indices if enabled
            if self.label_settings['show_layer_index']:
                parts.append(f"L{layer_idx}, M{micro_layer_idx}")
            if self.label_settings['show_density']:
                parts.append(f"{density:.2f}")
        else:
            # Show only density if enabled
            if self.label_settings['show_density']:
                parts.append(f"{density:.2f}")

        return "\n".join(parts)

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
