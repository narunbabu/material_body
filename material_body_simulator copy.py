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
def update_all_child_densities(parent_body, parent_center=(0.0, 0.0), parent_radius=1.0):
    for child in parent_body.get('child_bodies', []):
        update_child_body_density(parent_body, child, parent_center, parent_radius)
        # Now recursively call for this child's children
        update_all_child_densities(child, child['center'], child['radius'])


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
    num_micro_layers: int = 1,
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
        micro_layer_thickness = layer_thickness / num_micro_layers
        layer_x_coords = np.zeros((num_micro_layers, num_patches))
        layer_y_coords = np.zeros((num_micro_layers, num_patches))
        layer_radii = np.zeros(num_micro_layers)
        for micro_idx in range(num_micro_layers):
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
    parent_layer_index = child_body['parent_layer']
    parent_layer = parent_body['layers'][parent_layer_index]
    num_patches = len(parent_layer['density_profile'][0])
    # Calculate child center and radius
    previous_layers_thickness = sum(
        layer['thickness'] for layer in parent_body['layers'][:parent_layer_index]
    )
    current_layer_thickness = parent_layer['thickness']
    child_center_radius = parent_radius * (
        1 - previous_layers_thickness - current_layer_thickness / 2
    )
    placement_angle_rad = np.radians(child_body['placement_angle'])
    parent_center = parent_body.get('center', (0, 0))
    child_center = (
        parent_center[0] + child_center_radius * np.cos(placement_angle_rad),
        parent_center[1] + child_center_radius * np.sin(placement_angle_rad)
    )
    child_body['center'] = child_center
    child_body['radius'] = parent_radius * current_layer_thickness / 2
    # Adjust child outer layer densities
    child_outer_layer = child_body['layers'][0]
    child_num_micro_layers = child_outer_layer.get('num_micro_layers', 1)
    parent_densities = parent_layer['density_profile'][-1, :]  # Innermost micro-layer of parent layer
    child_outer_densities = child_outer_layer['density_profile'][0, :]
    min_increase = 0.01  # Tolerance
    adjusted_densities = np.maximum(child_outer_densities, parent_densities + min_increase)
    child_outer_layer['density_profile'][0, :] = adjusted_densities
    # Ensure densities increase towards inner micro-layers in child body
    for micro_layer_index in range(1, child_num_micro_layers):
        outer_densities = child_outer_layer['density_profile'][micro_layer_index - 1, :]
        current_densities = child_outer_layer['density_profile'][micro_layer_index, :]
        # Ensure current densities are greater than outer densities plus a tolerance
        adjusted_densities = np.maximum(current_densities, outer_densities + min_increase)
        child_outer_layer['density_profile'][micro_layer_index, :] = adjusted_densities
    # Recursively update child bodies
    for child in child_body.get('child_bodies', []):
        update_child_body_density(child_body, child, parent_center=child_center, parent_radius=child_body['radius'])

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
        self.cmap = cm.get_cmap('viridis_r')
        self.show_labels = True  # Added for label visibility
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
                            ha='center', va='center', fontsize=6
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
    # def plot_body(
    #     self,
    #     ax,
    #     material: Dict[str, Any],
    #     center: Tuple[float, float] = (0, 0),
    #     radius: float = 1.0,
    #     is_child: bool = False,
    #     parent_rotation: float = 0.0
    #     ) -> None:
    #     """
    #     Plots a material body on the given axes.

    #     Args:
    #         ax: Matplotlib axes to plot on.
    #         material: The material body to plot.
    #         center: Center coordinates of the body.
    #         radius: Radius of the body.
    #         is_child: Whether the body is a child body.
    #         parent_rotation: Rotation angle inherited from the parent.
    #     """
    #     layers = material['layers']
    #     rotation_angle = material.get('rotation_angle', 0.0)
    #     total_rotation = (rotation_angle + parent_rotation) % 360
    #     current_radius = radius
    #     for i, layer in enumerate(layers):
    #         layer_thickness = layer['thickness'] * radius
    #         next_radius = current_radius - layer_thickness
    #         num_micro_layers = layer.get('num_micro_layers', 1)
    #         num_patches = self.num_patches
    #         micro_layer_thickness = layer_thickness / num_micro_layers
    #         for j in range(num_micro_layers):
    #             r = current_radius - j * micro_layer_thickness
    #             patches = []
    #             for k in range(num_patches):
    #                 start_angle = (k * 360 / num_patches + total_rotation) % 360
    #                 end_angle = ((k + 1) * 360 / num_patches + total_rotation) % 360
    #                 density = layer['density_profile'][j, k]
    #                 color = self.get_layer_color(density)
    #                 wedge = Wedge(
    #                     center, r, start_angle, end_angle,
    #                     width=micro_layer_thickness, facecolor=color, edgecolor='none'
    #                 )
    #                 patches.append(wedge)
    #             collection = PatchCollection(patches, match_original=True)
    #             ax.add_collection(collection)
    #         current_radius = next_radius
    #     outer_circle = Circle(center, radius, fill=False, edgecolor='black', linewidth=2)
    #     ax.add_artist(outer_circle)
    #     ax.text(
    #         center[0], center[1], material['name'], ha='center', va='center',
    #         fontsize=10, fontweight='bold', color='black', rotation=total_rotation
    #     )
    #     if 'rotation_angle' in material:
    #         marker_angle = np.radians(total_rotation)
    #         marker_radius = radius * 0.8
    #         marker_x = center[0] + marker_radius * np.cos(marker_angle)
    #         marker_y = center[1] + marker_radius * np.sin(marker_angle)
    #         ax.plot([center[0], marker_x], [center[1], marker_y], color='red', linewidth=2)
    #     for child in material.get('child_bodies', []):
    #         parent_layer = child['parent_layer']
    #         placement_angle = (child['placement_angle'] + total_rotation) % 360
    #         parent_layers = layers[:parent_layer + 1]
    #         parent_outer_radius = radius - sum(layer['thickness'] for layer in layers[:parent_layer]) * radius
    #         parent_inner_radius = radius - sum(layer['thickness'] for layer in parent_layers) * radius
    #         child_radius = (parent_outer_radius - parent_inner_radius) / 2
    #         child_center_radius = (parent_outer_radius + parent_inner_radius) / 2
    #         child_center = (
    #             center[0] + child_center_radius * np.cos(np.radians(placement_angle)),
    #             center[1] + child_center_radius * np.sin(np.radians(placement_angle))
    #         )
    #         self.plot_body(
    #             ax, child, child_center, child_radius, is_child=True, parent_rotation=total_rotation
    #         )

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
            # Re-initialize densities after changes
            initialize_densities(self.material_object, self.canvas.num_patches, self.canvas.compulsory_increase)
            # Update densities for child bodies
            for child in self.material_object.get('child_bodies', []):
                update_child_body_density(self.material_object, child)
            # Update density range and re-plot
            self.canvas.min_density, self.canvas.max_density = self.canvas.get_density_range(self.material_object)
            self.canvas.plot_material_body()
            # Re-populate the body selector
            self.body_selector.clear()
            self.populate_body_selector(self.material_object)

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

    def populate_body_selector(self, body, prefix=''):
        self.body_selector.addItem(f"{prefix}{body['name']}")
        for child in body.get('child_bodies', []):
            self.populate_body_selector(child, prefix + '  ')

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
        "show_label": True,
        "use_micro_layers": True,
        "num_micro_layers": 3,
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
                "show_label": True,
                "use_micro_layers": True,
                "num_micro_layers": 2,
                "density_tolerance": 0.001,
                "layers": [
                    {"thickness": 0.4, "density": 0.2, "num_micro_layers": 2},
                    {"thickness": 0.2, "density": 0.4, "num_micro_layers": 2},
                    {"thickness": 0.2, "density": 0.6, "num_micro_layers": 1},
                ],
                "child_bodies": []
            }
        ]
    }
    app = QApplication(sys.argv)
    simulator = MaterialBodySimulator(material_object)
    simulator.show()
    sys.exit(app.exec())
