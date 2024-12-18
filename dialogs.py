#dialogs.py

import numpy as np


from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QToolBar, QColorDialog,
    QComboBox, QPushButton, QDialog, QFormLayout, QLabel,QListWidget,
    QSpinBox, QDoubleSpinBox, QDialogButtonBox, QHBoxLayout, QCheckBox,QMessageBox,
    QFileDialog, QSizePolicy,  QMenu,     QInputDialog,
    QStyledItemDelegate, QStyle
)
from PyQt6.QtGui import QAction, QColor, QPainter, QPen, QLinearGradient
from PyQt6.QtCore import QTimer, QRect, Qt, QSize, QRectF, QPointF, pyqtSignal
from PyQt6.QtCore import pyqtSignal

from PyQt6.QtWidgets import (     QLineEdit,QCompleter, QCheckBox,   QTreeWidget, QTreeWidgetItem, QGroupBox)


import matplotlib.pyplot as plt
from matplotlib import cm
from copy import deepcopy
import os, sys
import json
from matplotlib.colors import ListedColormap  # Add this import at the top
import matplotlib



class VisualizationSettingsDialog(QDialog):
    def __init__(self, canvas, parent=None):
        super().__init__(parent)
        self.canvas = canvas
        self.color_editor = None
        self.custom_colorbars = {}  # Dictionary to store custom color bars
        self.colormap_usage_counts = {}  # To keep track of colormap usage
        self.init_ui()
        self.load_custom_colorbars()
        self.populate_colormap_combo()
        self.filter_timer = QTimer()
        self.filter_timer.setSingleShot(True)
        self.filter_timer.timeout.connect(self.filter_colormaps)

    def init_ui(self):
        self.setWindowTitle("Visualization Settings")
        self.setMinimumWidth(500)
        layout = QVBoxLayout()

        # Color Map Selection with Preview and Search
        color_map_layout = QVBoxLayout()
        search_layout = QHBoxLayout()
        self.search_line_edit = QLineEdit()
        self.search_line_edit.setPlaceholderText("Search color maps...")
        self.search_line_edit.textChanged.connect(self.on_search_text_changed)
        search_layout.addWidget(QLabel("Search:"))
        search_layout.addWidget(self.search_line_edit)
        color_map_layout.addLayout(search_layout)

        self.cmap_combo = QComboBox()
        self.cmap_combo.setEditable(False)
        self.cmap_combo.setItemDelegate(ColorBarItemDelegate(self.cmap_combo))
        self.cmap_combo.setMinimumHeight(30)
        self.cmap_combo.currentTextChanged.connect(self.on_cmap_changed)

        edit_cmap_button = QPushButton("Edit Color Maps")
        edit_cmap_button.clicked.connect(self.open_color_editor)

        cmap_layout = QHBoxLayout()
        cmap_layout.addWidget(QLabel("Color Map:"))
        cmap_layout.addWidget(self.cmap_combo)
        cmap_layout.addWidget(edit_cmap_button)

        color_map_layout.addLayout(cmap_layout)
        layout.addLayout(color_map_layout)

        # # Show Labels Checkbox
        # self.show_labels_checkbox = QCheckBox("Show Labels")
        # self.show_labels_checkbox.setChecked(self.canvas.show_labels)
        # layout.addWidget(self.show_labels_checkbox)
        # Label Settings Group
        label_group = QGroupBox("Label Settings")
        label_layout = QVBoxLayout()
        
        # Show Labels Master Switch
        self.show_labels_checkbox = QCheckBox("Show Labels")
        self.show_labels_checkbox.setChecked(self.canvas.show_labels)
        label_layout.addWidget(self.show_labels_checkbox)
        
        # Label Content Options
        content_group = QGroupBox("Label Content")
        content_layout = QVBoxLayout()
        
        self.density_checkbox = QCheckBox("Show Density")
        self.density_checkbox.setChecked(self.canvas.label_settings['show_density'])
        
        self.layer_idx_checkbox = QCheckBox("Show Layer Index")
        self.layer_idx_checkbox.setChecked(self.canvas.label_settings['show_layer_index'])
        
        self.micro_layer_checkbox = QCheckBox("Show Micro-layer Index")
        self.micro_layer_checkbox.setChecked(self.canvas.label_settings['show_micro_layer_index'])
        
        self.patch_idx_checkbox = QCheckBox("Show Patch Index")
        self.patch_idx_checkbox.setChecked(self.canvas.label_settings['show_patch_index'])
        
        content_layout.addWidget(self.density_checkbox)
        content_layout.addWidget(self.layer_idx_checkbox)
        content_layout.addWidget(self.micro_layer_checkbox)
        content_layout.addWidget(self.patch_idx_checkbox)
        
        content_group.setLayout(content_layout)
        label_layout.addWidget(content_group)
        
        # Font Size Control
        font_layout = QHBoxLayout()
        font_layout.addWidget(QLabel("Font Size:"))
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(6, 24)
        self.font_size_spin.setValue(self.canvas.label_settings['font_size'])
        font_layout.addWidget(self.font_size_spin)
        label_layout.addLayout(font_layout)
        
        # Minimum Width Controls
        width_layout = QFormLayout()
        
        self.min_width_density = QSpinBox()
        self.min_width_density.setRange(10, 100)
        self.min_width_density.setValue(self.canvas.label_settings['min_patch_width_density'])
        width_layout.addRow("Min Width for Density:", self.min_width_density)
        
        self.min_width_indices = QSpinBox()
        self.min_width_indices.setRange(10, 100)
        self.min_width_indices.setValue(self.canvas.label_settings['min_patch_width_indices'])
        width_layout.addRow("Min Width for Indices:", self.min_width_indices)
        
        self.min_width_all = QSpinBox()
        self.min_width_all.setRange(10, 100)
        self.min_width_all.setValue(self.canvas.label_settings['min_patch_width_all'])
        width_layout.addRow("Min Width for All Labels:", self.min_width_all)
        
        label_layout.addLayout(width_layout)
        label_group.setLayout(label_layout)
        layout.addWidget(label_group)

        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Apply |
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        button_box.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self.apply_changes)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def load_custom_colorbars(self):
        self.custom_colorbars = {}
        colorbar_folder = "colorbars"
        if os.path.exists(colorbar_folder):
            for filename in os.listdir(colorbar_folder):
                if filename.endswith('.json'):
                    full_path = os.path.join(colorbar_folder, filename)
                    colorbar = ColorBar.load(full_path)
                    cmap = colorbar.get_matplotlib_colormap()
                    self.custom_colorbars[colorbar.name] = cmap

    def populate_colormap_combo(self):
        # Combine built-in and custom colormaps
        self.all_colormaps = list(matplotlib.colormaps.keys()) + list(self.custom_colorbars.keys())

        # Sort colormaps, placing most used at the top
        sorted_colormaps = sorted(self.all_colormaps, key=lambda x: -self.colormap_usage_counts.get(x, 0))

        # Populate the combo box
        self.cmap_combo.clear()
        self.cmap_combo.addItems(sorted_colormaps)

        # Set current selection
        current_cmap_name = self.canvas.cmap.name if hasattr(self.canvas.cmap, 'name') else 'viridis'
        index = self.cmap_combo.findText(current_cmap_name)
        if index >= 0:
            self.cmap_combo.setCurrentIndex(index)

    def on_search_text_changed(self, text):
        self.filter_timer.start(300)  # Delay filtering to avoid too many updates

    def filter_colormaps(self):
        text = self.search_line_edit.text().strip().lower()
        current_cmap_name = self.canvas.cmap.name if hasattr(self.canvas.cmap, 'name') else 'viridis'
        
        try:
            if not text:  # Show all if search is empty
                filtered_colormaps = self.all_colormaps
            else:
                # More sophisticated filtering
                filtered_colormaps = []
                for cmap in self.all_colormaps:
                    cmap_lower = cmap.lower()
                    # Exact match gets highest priority
                    if cmap_lower == text:
                        filtered_colormaps.insert(0, cmap)
                    # Starts with search term gets second priority
                    elif cmap_lower.startswith(text):
                        filtered_colormaps.append(cmap)
                    # Contains search term gets lowest priority
                    elif text in cmap_lower:
                        filtered_colormaps.append(cmap)
            
            # Sort by usage count within each priority group
            filtered_colormaps.sort(key=lambda x: -self.colormap_usage_counts.get(x, 0))
            
            self.cmap_combo.clear()
            self.cmap_combo.addItems(filtered_colormaps)
            
            # Try to maintain current selection if it's in filtered results
            index = self.cmap_combo.findText(current_cmap_name)
            if index >= 0:
                self.cmap_combo.setCurrentIndex(index)
            elif self.cmap_combo.count() > 0:
                self.cmap_combo.setCurrentIndex(0)
                
        except Exception as e:
            print(f"Error in filtering colormaps: {str(e)}")
            # Fallback to showing all colormaps
            self.cmap_combo.clear()
            self.cmap_combo.addItems(self.all_colormaps)

    def on_cmap_changed(self, cmap_name):
        # Update usage count
        self.colormap_usage_counts[cmap_name] = self.colormap_usage_counts.get(cmap_name, 0) + 1

    def update_color_maps(self):
        # Refresh the combo box with any new colormaps, including custom ones
        current_cmap = self.cmap_combo.currentText()
        self.load_custom_colorbars()
        self.populate_colormap_combo()
        items = [self.cmap_combo.itemText(i) for i in range(self.cmap_combo.count())]
        if current_cmap in items:
            self.cmap_combo.setCurrentText(current_cmap)

    # def apply_changes(self):
    #     try:
    #         # Store current view limits
    #         current_xlim = self.canvas.ax.get_xlim()
    #         current_ylim = self.canvas.ax.get_ylim()
            
    #         # Apply colormap change
    #         cmap_name = self.cmap_combo.currentText()
    #         if cmap_name in self.custom_colorbars:
    #             self.canvas.cmap = self.custom_colorbars[cmap_name]
    #         else:
    #             self.canvas.cmap = matplotlib.colormaps[cmap_name]
            
    #         # Apply label visibility change
    #         self.canvas.show_labels = self.show_labels_checkbox.isChecked()
            
    #         # Redraw with preserved limits
    #         self.canvas.plot_material_body()
    #         self.canvas.ax.set_xlim(current_xlim)
    #         self.canvas.ax.set_ylim(current_ylim)
    #         self.canvas.draw()
            
    #     except Exception as e:
    #         QMessageBox.critical(self, "Error", f"Failed to apply changes: {str(e)}")
    def apply_changes(self):
        try:
            # Store current view limits
            current_xlim = self.canvas.ax.get_xlim()
            current_ylim = self.canvas.ax.get_ylim()
            
            # Apply colormap change
            cmap_name = self.cmap_combo.currentText()
            if cmap_name in self.custom_colorbars:
                self.canvas.cmap = self.custom_colorbars[cmap_name]
            else:
                self.canvas.cmap = matplotlib.colormaps[cmap_name]
            
            # Apply label visibility changes
            self.canvas.show_labels = self.show_labels_checkbox.isChecked()
            self.canvas.label_settings.update({
                'show_density': self.density_checkbox.isChecked(),
                'show_layer_index': self.layer_idx_checkbox.isChecked(),
                'show_micro_layer_index': self.micro_layer_checkbox.isChecked(),
                'show_patch_index': self.patch_idx_checkbox.isChecked(),
                'font_size': self.font_size_spin.value(),
                'min_patch_width_density': self.min_width_density.value(),
                'min_patch_width_indices': self.min_width_indices.value(),
                'min_patch_width_all': self.min_width_all.value()
            })
            
            # Redraw with preserved limits
            self.canvas.plot_material_body()
            self.canvas.ax.set_xlim(current_xlim)
            self.canvas.ax.set_ylim(current_ylim)
            self.canvas.draw()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply changes: {str(e)}")
            
    def accept(self):
        self.apply_changes()
        super().accept()

    def open_color_editor(self):
        if not hasattr(self, 'color_editor') or self.color_editor is None:
            self.color_editor = ColorBarEditor(self)
            self.color_editor.colorbar_saved.connect(self.update_color_maps)

        current_cmap_name = self.cmap_combo.currentText()
        if current_cmap_name in self.custom_colorbars:
            current_cmap = self.custom_colorbars[current_cmap_name]
        else:
            current_cmap = matplotlib.colormaps[current_cmap_name]

        color_stops = []
        num_stops = 5  # Number of color stops to sample

        for i in range(num_stops):
            pos = i / (num_stops - 1)
            rgba = current_cmap(pos)
            color = QColor.fromRgbF(rgba[0], rgba[1], rgba[2])
            color_stops.append(ColorStop(pos, color))

        self.color_editor.colorbar.color_stops = color_stops
        self.color_editor.colorbar.name = current_cmap_name
        self.color_editor.colorbar_name_label.setText(current_cmap_name)
        self.color_editor.colorbar_widget.update()

        self.color_editor.show()
        self.color_editor.raise_()
        self.color_editor.activateWindow()


class PropertiesDialog(QDialog):
    """
    Dialog window for changing properties of material bodies.
    """

    def __init__(self, bodies, parent=None):
        super().__init__(parent)
        self.bodies = bodies
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Change Properties")
        self.setMinimumWidth(800)
        self.setMinimumHeight(600)
        layout = QFormLayout()
        self.color_buttons = {}
        self.parent_layer_spins = {}
        self.placement_angle_spins = {}
        self.label_checkboxes = {}
        # Removed angular_speed_spins, tolerance_spins, micro_layer_checkboxes, micro_layer_spins

        for body in self.bodies:
            name = body['name']
            color = QColor(body['color'])
            parent_layer = body.get('parent_layer')
            placement_angle = body.get('placement_angle')
            show_label = body.get('show_label', True)

            color_button = QPushButton()
            color_button.setStyleSheet(f"background-color: {color.name()}")
            color_button.clicked.connect(lambda _, b=body: self.change_color(b))

            parent_layer_spin = QSpinBox()
            parent_layer_spin.setRange(0, 100)
            parent_layer_spin.setValue(parent_layer if parent_layer is not None else 0)
            parent_layer_spin.setEnabled(parent_layer is not None)

            placement_angle_spin = QDoubleSpinBox()
            placement_angle_spin.setRange(0, 360)
            placement_angle_spin.setValue(placement_angle if placement_angle is not None else 0)
            placement_angle_spin.setEnabled(placement_angle is not None)

            label_checkbox = QCheckBox("Layer label on")
            label_checkbox.setChecked(show_label)

            body_layout = QHBoxLayout()
            body_layout.addWidget(QLabel("Color:"))
            body_layout.addWidget(color_button)
            body_layout.addWidget(QLabel("Parent Layer:"))
            body_layout.addWidget(parent_layer_spin)
            body_layout.addWidget(QLabel("Placement Angle:"))
            body_layout.addWidget(placement_angle_spin)
            body_layout.addWidget(label_checkbox)

            layout.addRow(QLabel(name), body_layout)

            self.color_buttons[name] = color_button
            self.parent_layer_spins[name] = parent_layer_spin
            self.placement_angle_spins[name] = placement_angle_spin
            self.label_checkboxes[name] = label_checkbox

        # Add Apply button
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Apply |
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        button_box.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(self.apply_changes)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addRow(button_box)
        self.setLayout(layout)

    def apply_changes(self):
        self.get_updated_properties()
        self.parent().canvas.plot_material_body()

    def change_color(self, body):
        color = QColorDialog.getColor()
        if color.isValid():
            body['color'] = color.name()
            self.color_buttons[body['name']].setStyleSheet(f"background-color: {color.name()}")

    def get_updated_properties(self):
        for body in self.bodies:
            name = body['name']
            body['color'] = self.color_buttons[name].palette().button().color().name()
            if 'parent_layer' in body:
                body['parent_layer'] = self.parent_layer_spins[name].value()
            if 'placement_angle' in body:
                body['placement_angle'] = self.placement_angle_spins[name].value()
            body['show_label'] = self.label_checkboxes[name].isChecked()
        return self.bodies



class EditBodiesDialog(QDialog):
    def __init__(self, material_object, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Bodies and Layers")
        self.material_object = material_object
        self.init_ui()

    def init_ui(self):
        self.layout = QVBoxLayout(self)
        # Create the tree widget to display bodies and layers
        self.tree_widget = QTreeWidget()
        self.tree_widget.setHeaderLabels(["Name", "Type"])
        self.layout.addWidget(self.tree_widget)
        # Populate the tree with the material object
        self.populate_tree()
        # Create buttons for adding/deleting bodies and layers
        button_layout = QHBoxLayout()
        self.add_body_button = QPushButton("Add Body")
        self.add_layer_button = QPushButton("Add Layer")
        self.delete_button = QPushButton("Delete")
        self.save_button = QPushButton("Save Changes")
        self.cancel_button = QPushButton("Cancel")
        button_layout.addWidget(self.add_body_button)
        button_layout.addWidget(self.add_layer_button)
        button_layout.addWidget(self.delete_button)
        button_layout.addStretch()
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.cancel_button)
        self.layout.addLayout(button_layout)
        # Connect signals to slots
        self.tree_widget.itemClicked.connect(self.on_item_clicked)
        self.add_body_button.clicked.connect(self.add_body)
        self.add_layer_button.clicked.connect(self.add_layer)
        self.delete_button.clicked.connect(self.delete_item)
        self.save_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)
        # Create the property editor area
        self.property_editor = QGroupBox("Properties")
        self.property_layout = QFormLayout()
        self.property_editor.setLayout(self.property_layout)
        self.layout.addWidget(self.property_editor)
        self.current_item = None

    def populate_tree(self):
        self.tree_widget.clear()
        self.add_body_to_tree(self.material_object, None)

    def add_body_to_tree(self, body, parent_item):
        body_item = QTreeWidgetItem()
        body_item.setText(0, body['name'])
        body_item.setText(1, 'Body')
        body_item.setData(0, Qt.ItemDataRole.UserRole, body)
        if parent_item:
            parent_item.addChild(body_item)
        else:
            self.tree_widget.addTopLevelItem(body_item)
        # Add layers to the body in the tree
        for idx, layer in enumerate(body['layers']):
            layer_item = QTreeWidgetItem()
            layer_item.setText(0, f"Layer {idx}")
            layer_item.setText(1, 'Layer')
            layer_item.setData(0, Qt.ItemDataRole.UserRole, (layer, body))
            body_item.addChild(layer_item)
        # Recursively add child bodies
        for child_body in body.get('child_bodies', []):
            self.add_body_to_tree(child_body, body_item)

    def on_item_clicked(self, item, column):
        self.current_item = item
        data = item.data(0, Qt.ItemDataRole.UserRole)
        # Clear the property editor
        while self.property_layout.count():
            child = self.property_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        # Determine whether a body or layer is selected and display properties
        if item.text(1) == 'Body':
            body = data
            self.edit_body_properties(body)
        elif item.text(1) == 'Layer':
            layer, parent_body = data
            self.edit_layer_properties(layer, parent_body)

    def edit_body_properties(self, body):
        self.current_body = body
        # Create input fields for body properties
        self.name_edit = QLineEdit(body.get('name', ''))
        self.color_button = QPushButton()
        self.color_button.setStyleSheet(f"background-color: {body.get('color', '#FFFFFF')}")
        self.color_button.clicked.connect(self.choose_color)
        self.placement_angle_spin = QDoubleSpinBox()
        self.placement_angle_spin.setRange(0, 360)
        self.placement_angle_spin.setValue(body.get('placement_angle', 0.0))
        self.angular_speed_spin = QDoubleSpinBox()
        self.angular_speed_spin.setRange(-360, 360)
        self.angular_speed_spin.setValue(body.get('angular_speed', 0.0))
        self.rotation_angle_spin = QDoubleSpinBox()
        self.rotation_angle_spin.setRange(0, 360)
        self.rotation_angle_spin.setValue(body.get('rotation_angle', 0.0))
        self.rotation_speed_spin = QDoubleSpinBox()
        self.rotation_speed_spin.setRange(-360, 360)
        self.rotation_speed_spin.setValue(body.get('rotation_speed', 0.0))
        self.show_label_checkbox = QCheckBox()
        self.show_label_checkbox.setChecked(body.get('show_label', True))
        self.use_micro_layers_checkbox = QCheckBox()
        self.use_micro_layers_checkbox.setChecked(body.get('use_micro_layers', True))
        self.num_micro_layers_spin = QSpinBox()
        self.num_micro_layers_spin.setValue(body.get('num_micro_layers', 1))
        # Add widgets to property layout
        self.property_layout.addRow("Name:", self.name_edit)
        self.property_layout.addRow("Color:", self.color_button)
        self.property_layout.addRow("Placement Angle:", self.placement_angle_spin)
        self.property_layout.addRow("Angular Speed:", self.angular_speed_spin)
        self.property_layout.addRow("Rotation Angle:", self.rotation_angle_spin)
        self.property_layout.addRow("Rotation Speed:", self.rotation_speed_spin)
        self.property_layout.addRow("Show Label:", self.show_label_checkbox)
        self.property_layout.addRow("Use Micro Layers:", self.use_micro_layers_checkbox)
        self.property_layout.addRow("Number of Micro Layers:", self.num_micro_layers_spin)
        # If the body has a parent_layer, display it
        if 'parent_layer' in body:
            self.parent_layer_spin = QSpinBox()
            self.parent_layer_spin.setValue(body['parent_layer'])
            self.property_layout.addRow("Parent Layer:", self.parent_layer_spin)
        else:
            self.parent_layer_spin = None

    def choose_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.color_button.setStyleSheet(f"background-color: {color.name()}")
            self.current_body['color'] = color.name()

    def edit_layer_properties(self, layer, parent_body):
        self.current_layer = layer
        self.parent_body = parent_body
        # Create input fields for layer properties
        self.thickness_spin = QDoubleSpinBox()
        self.thickness_spin.setDecimals(4)
        self.thickness_spin.setSingleStep(0.01)
        self.thickness_spin.setRange(0.0, 1.0)
        self.thickness_spin.setValue(layer.get('thickness', 0.1))
        self.density_spin = QDoubleSpinBox()
        self.density_spin.setDecimals(4)
        self.density_spin.setSingleStep(0.01)
        self.density_spin.setRange(0.0, 1000.0)
        self.density_spin.setValue(layer.get('density', 1.0))
        self.num_micro_layers_spin = QSpinBox()
        self.num_micro_layers_spin.setValue(layer.get('num_micro_layers', 1))
        # Add widgets to property layout
        self.property_layout.addRow("Thickness:", self.thickness_spin)
        self.property_layout.addRow("Density:", self.density_spin)
        self.property_layout.addRow("Number of Micro Layers:", self.num_micro_layers_spin)

    def add_body(self):
        # Determine the parent body and parent layer from the selected item
        selected_item = self.tree_widget.currentItem()
        if not selected_item:
            QMessageBox.warning(self, "No Selection", "Please select a parent body or layer to add a child body.")
            return
        if selected_item.text(1) == 'Body':
            parent_body = selected_item.data(0, Qt.ItemDataRole.UserRole)
            parent_layer = None
        elif selected_item.text(1) == 'Layer':
            layer, parent_body = selected_item.data(0, Qt.ItemDataRole.UserRole)
            parent_layer = parent_body['layers'].index(layer)
        else:
            QMessageBox.warning(self, "Invalid Selection", "Please select a parent body or layer.")
            return
        # Create a new body with default properties
        new_body = {
            "name": "New Body",
            "color": "#0000dd",
            "placement_angle": 0.0,
            "angular_speed": 0.0,
            "rotation_angle": 0.0,
            "rotation_speed": 0.0,
            "show_label": True,
            "use_micro_layers": True,
            "num_micro_layers": 1,
            "layers": [],
            "child_bodies": []
        }
        new_body['parent_layer'] = parent_layer if parent_layer is not None else 0
        # Append the new body to the parent body's child_bodies list
        parent_body.setdefault('child_bodies', []).append(new_body)
        # Add the new body to the tree
        parent_item = selected_item if selected_item.text(1) == 'Body' else selected_item.parent()
        new_body_item = QTreeWidgetItem()
        new_body_item.setText(0, new_body['name'])
        new_body_item.setText(1, 'Body')
        new_body_item.setData(0, Qt.ItemDataRole.UserRole, new_body)
        parent_item.addChild(new_body_item)
        self.tree_widget.expandItem(parent_item)

    def add_layer(self):
        # Determine the parent body from the selected item
        selected_item = self.tree_widget.currentItem()
        if not selected_item:
            QMessageBox.warning(self, "No Selection", "Please select a body to add a layer.")
            return
        if selected_item.text(1) == 'Body':
            parent_body = selected_item.data(0, Qt.ItemDataRole.UserRole)
        elif selected_item.text(1) == 'Layer':
            layer, parent_body = selected_item.data(0, Qt.ItemDataRole.UserRole)
        else:
            QMessageBox.warning(self, "Invalid Selection", "Please select a body to add a layer.")
            return
        # Create a new layer with default properties
        new_layer = {
            "thickness": 0.1,
            "density": 1.0,
            "num_micro_layers": 1
        }
        parent_body['layers'].append(new_layer)
        # Add the new layer to the tree
        body_item = selected_item if selected_item.text(1) == 'Body' else selected_item.parent()
        new_layer_item = QTreeWidgetItem()
        new_layer_item.setText(0, f"Layer {len(parent_body['layers']) - 1}")
        new_layer_item.setText(1, 'Layer')
        new_layer_item.setData(0, Qt.ItemDataRole.UserRole, (new_layer, parent_body))
        body_item.addChild(new_layer_item)
        self.tree_widget.expandItem(body_item)
        # Update layer names
        self.update_layer_names(body_item, parent_body)

    def delete_item(self):
        selected_item = self.tree_widget.currentItem()
        if not selected_item:
            QMessageBox.warning(self, "No Selection", "Please select an item to delete.")
            return
        reply = QMessageBox.question(self, 'Delete', 'Are you sure you want to delete this item?', QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.No:
            return
        if selected_item.text(1) == 'Body':
            body = selected_item.data(0, Qt.ItemDataRole.UserRole)
            parent_item = selected_item.parent()
            if parent_item:
                parent_body = parent_item.data(0, Qt.ItemDataRole.UserRole)
                parent_body['child_bodies'].remove(body)
            else:
                QMessageBox.warning(self, "Cannot Delete", "Cannot delete the root body.")
                return
        elif selected_item.text(1) == 'Layer':
            layer, parent_body = selected_item.data(0, Qt.ItemDataRole.UserRole)
            parent_body['layers'].remove(layer)
            # Update layer names
            body_item = selected_item.parent()
            self.update_layer_names(body_item, parent_body)
        else:
            QMessageBox.warning(self, "Invalid Selection", "Please select a body or layer to delete.")
            return
        # Remove the item from the tree
        parent = selected_item.parent()
        if parent:
            parent.removeChild(selected_item)
        else:
            self.tree_widget.takeTopLevelItem(self.tree_widget.indexOfTopLevelItem(selected_item))

    def update_layer_names(self, body_item, body):
        # Update the names of layers under the specified body in the tree
        layer_count = 0
        for i in range(body_item.childCount()):
            child_item = body_item.child(i)
            if child_item.text(1) == 'Layer':
                child_item.setText(0, f"Layer {layer_count}")
                layer_count += 1

    def accept(self):
        # Update the properties of the current item before closing
        if self.current_item:
            if self.current_item.text(1) == 'Body':
                body = self.current_item.data(0, Qt.ItemDataRole.UserRole)
                body['name'] = self.name_edit.text()
                body['placement_angle'] = self.placement_angle_spin.value()
                body['angular_speed'] = self.angular_speed_spin.value()
                body['rotation_angle'] = self.rotation_angle_spin.value()
                body['rotation_speed'] = self.rotation_speed_spin.value()
                body['show_label'] = self.show_label_checkbox.isChecked()
                body['use_micro_layers'] = self.use_micro_layers_checkbox.isChecked()
                body['num_micro_layers'] = self.num_micro_layers_spin.value()
                if self.parent_layer_spin is not None:
                    body['parent_layer'] = self.parent_layer_spin.value()
                # Update the tree item text if the name has changed
                self.current_item.setText(0, body['name'])
            elif self.current_item.text(1) == 'Layer':
                layer, parent_body = self.current_item.data(0, Qt.ItemDataRole.UserRole)
                layer['thickness'] = self.thickness_spin.value()
                layer['density'] = self.density_spin.value()
                layer['num_micro_layers'] = self.num_micro_layers_spin.value()
        # Close the dialog
        super().accept()



class ColorStop:
    def __init__(self, position, color):
        self.position = position
        self.color = color

class ColorBar:
    def __init__(self, name="Custom"):
        self.name = name
        self.is_reversed = False
        self.color_stops = [
            ColorStop(0.0, QColor(0, 0, 255)),    # Blue
            ColorStop(0.25, QColor(0, 255, 255)), # Cyan
            ColorStop(0.5, QColor(0, 255, 0)),    # Green
            ColorStop(0.75, QColor(255, 255, 0)), # Yellow
            ColorStop(1.0, QColor(255, 0, 0))     # Red
        ]
        self.is_nonlinear = False

    def get_matplotlib_colormap(self):
        """Convert to matplotlib colormap format"""
        # Get the interpolated colors
        colors = self.interpolate_colors()

        # Convert QColor objects to RGB tuples
        rgb_colors = [(c.red()/255, c.green()/255, c.blue()/255) for c in colors]

        # Create the colormap
        if len(rgb_colors) > 0:
            return ListedColormap(rgb_colors, name=self.name)
        return matplotlib.colormaps['viridis']  # fallback colormap

    def interpolate_colors(self):
        positions = np.array([stop.position for stop in self.color_stops])
        colors = [stop.color for stop in self.color_stops]

        if self.is_reversed:  # Reverse positions if needed
            positions = 1 - positions
            positions = positions[::-1]
            colors = colors[::-1]

        # Create a linear space of positions
        interpolated_positions = np.linspace(0, 1, 256)

        # Interpolate red, green, blue channels separately
        reds = np.interp(interpolated_positions, positions, [c.red() for c in colors])
        greens = np.interp(interpolated_positions, positions, [c.green() for c in colors])
        blues = np.interp(interpolated_positions, positions, [c.blue() for c in colors])

        interpolated_colors = [QColor(int(r), int(g), int(b)) for r, g, b in zip(reds, greens, blues)]

        return interpolated_colors

    def save(self, filename):
        data = {
            "name": self.name,
            "color_stops": [(stop.position, stop.color.name()) for stop in self.color_stops],
            "is_nonlinear": self.is_nonlinear,
            "is_reversed": self.is_reversed  # Save reversed state
        }
        with open(filename, 'w') as f:
            json.dump(data, f)

    @classmethod
    def load(cls, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        colorbar = cls(data["name"])
        colorbar.color_stops = [ColorStop(pos, QColor(color)) for pos, color in data["color_stops"]]
        colorbar.is_nonlinear = data["is_nonlinear"]
        colorbar.is_reversed = data.get("is_reversed", False)  # Load reversed state with default
        return colorbar

class ColorBarWidget(QWidget):
    colorChanged = pyqtSignal()

    def __init__(self, colorbar, editor=None):
        super().__init__()
        self.colorbar = colorbar
        self.editor = editor
        self.dragging_stop = None
        self.setMinimumHeight(50)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.hovered_stop = None

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()

        # Draw the gradient background (white-to-gray checkerboard for transparency)
        self.draw_transparency_background(painter, rect)

        # Draw the gradient
        interpolated_colors = self.colorbar.interpolate_colors()
        width = rect.width()
        height = rect.height()

        for i, color in enumerate(interpolated_colors):
            x = i * (width / len(interpolated_colors))
            w = width / len(interpolated_colors) + 1  # Add 1 to prevent gaps
            painter.fillRect(x, 0, w, height, color)

        # Draw color stops
        for stop in self.colorbar.color_stops:
            x = stop.position * width

            # Draw selection circle
            if stop == self.dragging_stop or stop == self.hovered_stop:
                painter.setPen(QPen(Qt.GlobalColor.white, 2))
                painter.setBrush(stop.color)
                painter.drawEllipse(QPointF(x, height/2), 6, 6)

            # Draw normal circle
            painter.setPen(QPen(Qt.GlobalColor.black, 1))
            painter.setBrush(stop.color)
            painter.drawEllipse(QPointF(x, height/2), 5, 5)

    def draw_transparency_background(self, painter, rect):
        """Draw a checkerboard pattern to indicate transparency"""
        cell_size = 10
        light = QColor(255, 255, 255)
        dark = QColor(204, 204, 204)

        for i in range(0, rect.width(), cell_size):
            for j in range(0, rect.height(), cell_size):
                color = light if ((i + j) // cell_size) % 2 == 0 else dark
                painter.fillRect(i, j, cell_size, cell_size, color)

    def get_editor(self):
        # Walk up the parent chain to find the ColorBarEditor
        parent = self.parent()
        while parent is not None:
            if isinstance(parent, ColorBarEditor):
                return parent
            parent = parent.parent()
        return self.editor  # Fall back to stored editor reference

    def mousePressEvent(self, event):
        pos = event.position() if hasattr(event, 'position') else event.pos()
        clicked_pos = pos.x() / self.width()

        if event.button() == Qt.MouseButton.LeftButton:
            # Existing left-click functionality
            for stop in self.colorbar.color_stops:
                if abs(clicked_pos - stop.position) < 0.02:
                    self.dragging_stop = stop
                    self.update()
                    return

            # Add new color stop
            color = QColorDialog.getColor()
            if color.isValid():
                new_stop = ColorStop(clicked_pos, color)
                self.colorbar.color_stops.append(new_stop)
                self.colorbar.color_stops.sort(key=lambda stop: stop.position)
                self.dragging_stop = new_stop
                self.update()
                self.colorChanged.emit()
                editor = self.get_editor()
                if editor:
                    editor.add_to_undo_stack()

        elif event.button() == Qt.MouseButton.RightButton:
            # Right-click functionality to delete or change color stop
            for stop in self.colorbar.color_stops:
                if abs(clicked_pos - stop.position) < 0.02:
                    menu = QMenu(self)
                    delete_action = menu.addAction("Delete Color Stop")
                    change_color_action = menu.addAction("Change Color")
                    action = menu.exec(event.globalPosition().toPoint() if hasattr(event, 'globalPosition') else event.globalPos())
                    if action == delete_action:
                        if len(self.colorbar.color_stops) > 2:
                            self.colorbar.color_stops.remove(stop)
                            self.update()
                            self.colorChanged.emit()
                            editor = self.get_editor()
                            if editor:
                                editor.add_to_undo_stack()
                        else:
                            QMessageBox.warning(self, "Cannot Delete", "At least two color stops are required.")
                    elif action == change_color_action:
                        color = QColorDialog.getColor(stop.color)
                        if color.isValid():
                            stop.color = color
                            self.update()
                            self.colorChanged.emit()
                            editor = self.get_editor()
                            if editor:
                                editor.add_to_undo_stack()
                    return

    def mouseMoveEvent(self, event):
        pos = event.position() if hasattr(event, 'position') else event.pos()
        moved_pos = pos.x() / self.width()

        if self.dragging_stop:
            new_position = moved_pos
            new_position = max(0.0, min(1.0, new_position))
            self.dragging_stop.position = new_position
            self.colorbar.color_stops.sort(key=lambda stop: stop.position)
            self.update()
            self.colorChanged.emit()
        else:
            # Update hovered stop
            hovered = None
            for stop in self.colorbar.color_stops:
                if abs(moved_pos - stop.position) < 0.02:
                    hovered = stop
                    break
            if hovered != self.hovered_stop:
                self.hovered_stop = hovered
                self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.dragging_stop:
            self.dragging_stop = None
            self.update()
            self.colorChanged.emit()
            editor = self.get_editor()
            if editor:
                editor.add_to_undo_stack()

    def leaveEvent(self, event):
        self.hovered_stop = None
        self.update()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Delete and self.dragging_stop:
            if len(self.colorbar.color_stops) > 2:
                self.colorbar.color_stops.remove(self.dragging_stop)
                self.dragging_stop = None
                self.update()
                self.colorChanged.emit()
                editor = self.get_editor()
                if editor:
                    editor.add_to_undo_stack()
            else:
                QMessageBox.warning(self, "Cannot Delete", "At least two color stops are required.")

    def sizeHint(self):
        return QSize(400, 50)

class ColorBarEditor(QMainWindow):
    colorbar_saved = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        # Set window flags to keep it on top and independent
        self.setWindowFlags(
            Qt.WindowType.Window |  # Make it an independent window
            Qt.WindowType.WindowStaysOnTopHint  # Keep it on top
        )

        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)  # Prevent destruction on close
        self.setWindowTitle("Color Bar Editor")
        self.setGeometry(100, 100, 600, 400)

        # Initialize data
        self.colorbar = ColorBar("Custom Gradient")
        self.colorbars = []
        self.colorbar_folder = "colorbars"
        os.makedirs(self.colorbar_folder, exist_ok=True)

        self.undo_stack = []
        self.redo_stack = []

        # Setup UI
        self.init_ui()
        self.load_saved_colorbars()

    def init_ui(self):
        # Create main widget and layout
        self.main_widget = QWidget()
        layout = QVBoxLayout(self.main_widget)

        # Add spacing and margins for better appearance
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        # Name label with better visibility
        self.colorbar_name_label = QLabel(self.colorbar.name)
        self.colorbar_name_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(self.colorbar_name_label)

        # ColorBar widget
        self.colorbar_widget = ColorBarWidget(self.colorbar, editor=self)
        self.colorbar_widget.colorChanged.connect(self.enable_save_buttons)
        self.colorbar_widget.setMinimumHeight(50)
        layout.addWidget(self.colorbar_widget)

        # Reverse checkbox
        self.reverse_checkbox = QCheckBox("Reverse Colors")
        self.reverse_checkbox.setChecked(self.colorbar.is_reversed)
        self.reverse_checkbox.stateChanged.connect(self.toggle_reverse)
        layout.addWidget(self.reverse_checkbox)

        # Button layout with proper spacing
        button_layout = QHBoxLayout()
        button_layout.setSpacing(5)

        # Create and configure buttons
        buttons_config = [
            ("new_button", "New", self.new_colorbar),
            ("load_button", "Load", self.load_colorbar),
            ("save_button", "Save", self.save_colorbar),
            ("save_as_button", "Save As", self.save_colorbar_as),
            ("reset_button", "Reset", self.reset_colorbar),
            ("undo_button", "Undo", self.undo),
            ("redo_button", "Redo", self.redo)
        ]

        for attr_name, text, callback in buttons_config:
            button = QPushButton(text)
            button.setMinimumWidth(60)
            button.clicked.connect(callback)
            setattr(self, attr_name, button)
            button_layout.addWidget(button)

        layout.addLayout(button_layout)

        # Colorbar list
        list_label = QLabel("Saved Color Bars:")
        list_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(list_label)

        self.colorbar_list = QListWidget()
        self.colorbar_list.setMinimumHeight(150)
        self.colorbar_list.itemClicked.connect(self.on_colorbar_selected)
        self.colorbar_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.colorbar_list.customContextMenuRequested.connect(self.show_list_context_menu)
        layout.addWidget(self.colorbar_list)

        # Set the main widget
        self.setCentralWidget(self.main_widget)

        # Initial button states
        self.disable_save_buttons()
        self.update_undo_redo_buttons()
        self.update_colorbar_list()

    def closeEvent(self, event):
        # Hide instead of close
        self.hide()
        event.ignore()

    def show(self):
        super().show()
        self.raise_()  # Bring window to front
        self.activateWindow()  # Activate the window

    def toggle_reverse(self, state):
        self.colorbar.is_reversed = bool(state)
        self.colorbar_widget.update()
        self.enable_save_buttons()
        self.add_to_undo_stack()

    def on_colorbar_selected(self, item):
        selected_name = item.text()
        for colorbar in self.colorbars:
            if colorbar.name == selected_name:
                self.colorbar = colorbar
                self.colorbar_widget.colorbar = self.colorbar
                # Update reverse checkbox state
                self.reverse_checkbox.setChecked(self.colorbar.is_reversed)
                self.colorbar_widget.update()
                self.colorbar_name_label.setText(colorbar.name)
                self.disable_save_buttons()
                self.clear_undo_redo_stacks()
                break

    def save_colorbar(self):
        filename = os.path.join(self.colorbar_folder, f"{self.colorbar.name}.json")
        self.colorbar.save(filename)
        self.update_colorbar_list()
        self.disable_save_buttons()
        self.colorbar_saved.emit()

    def save_colorbar_as(self):
        while True:
            name, ok = QInputDialog.getText(self, "Save Color Bar As", "Enter new name:")
            if not ok:  # User canceled
                return
                
            if not name:  # Empty name
                QMessageBox.warning(self, "Invalid Name", "Please enter a valid name.")
                continue
                
            # Check for invalid characters in filename
            invalid_chars = '<>:"/\\|?*'
            if any(char in name for char in invalid_chars):
                QMessageBox.warning(self, "Invalid Name", 
                                f"Name cannot contain any of these characters: {invalid_chars}")
                continue
                
            filename = os.path.join(self.colorbar_folder, f"{name}.json")
            
            if os.path.exists(filename):
                reply = QMessageBox.question(self, "File Exists",
                                        f"A color bar named '{name}' already exists. Overwrite?",
                                        QMessageBox.StandardButton.Yes | 
                                        QMessageBox.StandardButton.No)
                if reply == QMessageBox.StandardButton.No:
                    continue
                    
            try:
                # Create a copy of the current colorbar with the new name
                new_colorbar = deepcopy(self.colorbar)
                new_colorbar.name = name
                
                # Save the new colorbar
                new_colorbar.save(filename)
                
                # Update the list of colorbars
                self.colorbars = [cb for cb in self.colorbars if cb.name != name]
                self.colorbars.append(new_colorbar)
                
                # Update current colorbar
                self.colorbar = new_colorbar
                self.colorbar_widget.colorbar = new_colorbar
                self.colorbar_name_label.setText(name)
                
                self.update_colorbar_list()
                self.disable_save_buttons()
                self.colorbar_saved.emit()
                break
                
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Error saving colorbar: {str(e)}")
                return

    def new_colorbar(self):
        name, ok = QInputDialog.getText(self, "New Color Bar", "Enter name:")
        if ok and name:
            if any(cb.name == name for cb in self.colorbars):
                QMessageBox.warning(self, "Name Exists", "A color bar with this name already exists.")
                return
            self.colorbar = ColorBar(name)
            self.colorbars.append(self.colorbar)
            self.update_colorbar_list()
            self.colorbar_widget.colorbar = self.colorbar
            self.colorbar_widget.update()
            self.colorbar_name_label.setText(name)
            self.disable_save_buttons()
            self.clear_undo_redo_stacks()

    def load_colorbar(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Load Color Bar", self.colorbar_folder, "JSON Files (*.json)")
        if filename:
            colorbar = ColorBar.load(filename)
            if any(cb.name == colorbar.name for cb in self.colorbars):
                QMessageBox.warning(self, "Name Exists", "A color bar with this name already exists.")
                return
            self.colorbars.append(colorbar)
            self.update_colorbar_list()
            self.colorbar = colorbar
            self.colorbar_widget.colorbar = self.colorbar
            self.colorbar_widget.update()
            self.colorbar_name_label.setText(colorbar.name)
            self.disable_save_buttons()
            self.clear_undo_redo_stacks()

    def load_saved_colorbars(self):
        for filename in os.listdir(self.colorbar_folder):
            if filename.endswith('.json'):
                full_path = os.path.join(self.colorbar_folder, filename)
                self.colorbars.append(ColorBar.load(full_path))
        self.update_colorbar_list()

    def reset_colorbar(self):
        self.colorbar.color_stops = [
            ColorStop(0.0, QColor(0, 0, 255)),
            ColorStop(0.25, QColor(0, 255, 255)),
            ColorStop(0.5, QColor(0, 255, 0)),
            ColorStop(0.75, QColor(255, 255, 0)),
            ColorStop(1.0, QColor(255, 0, 0))
        ]
        self.colorbar_widget.update()
        self.add_to_undo_stack()

    def update_colorbar_list(self):
        self.colorbar_list.clear()
        for colorbar in self.colorbars:
            self.colorbar_list.addItem(colorbar.name)

    def enable_save_buttons(self):
        self.save_button.setEnabled(True)
        self.save_as_button.setEnabled(True)

    def disable_save_buttons(self):
        self.save_button.setEnabled(False)
        self.save_as_button.setEnabled(False)

    def show_list_context_menu(self, pos):
        global_pos = self.colorbar_list.mapToGlobal(pos)
        item = self.colorbar_list.itemAt(pos)
        if item is not None:
            menu = QMenu()
            rename_action = menu.addAction("Rename")
            delete_action = menu.addAction("Delete")
            action = menu.exec(global_pos)
            if action == rename_action:
                selected_colorbar = next((cb for cb in self.colorbars if cb.name == item.text()), None)
                if selected_colorbar:
                    self.rename_colorbar(selected_colorbar)
            elif action == delete_action:
                self.delete_colorbar(item)

    def rename_colorbar(self, colorbar):
        new_name, ok = QInputDialog.getText(self, "Rename Color Bar", "Enter new name:", text=colorbar.name)

        if ok and new_name:
            old_filename = os.path.join(self.colorbar_folder, f"{colorbar.name}.json")
            new_filename = os.path.join(self.colorbar_folder, f"{new_name}.json")

            # Ensure no conflict with existing filenames
            if os.path.exists(new_filename):
                QMessageBox.warning(self, "Rename Error", "A color bar with this name already exists.")
                return

            # Rename the file on disk
            if os.path.exists(old_filename):
                os.rename(old_filename, new_filename)

            # Update the colorbar's name inside the object and save it with the new name
            colorbar.name = new_name
            colorbar.save(new_filename)

            # Update the UI and title to reflect the new name
            self.update_colorbar_list()
            self.colorbar_name_label.setText(new_name)

            # Update the current colorbar if it's the one being renamed
            if self.colorbar == colorbar:
                self.colorbar_name_label.setText(new_name)

    def delete_colorbar(self, item):
        name = item.text()
        reply = QMessageBox.question(self, "Delete Color Bar", f"Are you sure you want to delete '{name}'?",
                                     QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if reply == QMessageBox.StandardButton.Yes:
            filename = os.path.join(self.colorbar_folder, f"{name}.json")
            if os.path.exists(filename):
                os.remove(filename)
            self.colorbars = [cb for cb in self.colorbars if cb.name != name]
            self.update_colorbar_list()
            if self.colorbar.name == name:
                if self.colorbars:
                    self.colorbar = self.colorbars[0]
                    self.colorbar_widget.colorbar = self.colorbar
                    self.colorbar_widget.update()
                    self.colorbar_name_label.setText(self.colorbar.name)
                else:
                    self.colorbar = ColorBar("Custom Gradient")
                    self.colorbar_widget.colorbar = self.colorbar
                    self.colorbar_widget.update()
                    self.colorbar_name_label.setText(self.colorbar.name)
                self.clear_undo_redo_stacks()

    def add_to_undo_stack(self):
        if len(self.undo_stack) >= 10:
            self.undo_stack.pop(0)
        self.undo_stack.append(deepcopy(self.colorbar))
        self.redo_stack.clear()
        self.update_undo_redo_buttons()
        self.enable_save_buttons()

    def undo(self):
        if self.undo_stack:
            self.redo_stack.append(deepcopy(self.colorbar))
            self.colorbar = self.undo_stack.pop()
            self.colorbar_widget.colorbar = self.colorbar
            self.colorbar_widget.update()
            self.update_undo_redo_buttons()
            self.enable_save_buttons()

    def redo(self):
        if self.redo_stack:
            self.undo_stack.append(deepcopy(self.colorbar))
            self.colorbar = self.redo_stack.pop()
            self.colorbar_widget.colorbar = self.colorbar
            self.colorbar_widget.update()
            self.update_undo_redo_buttons()
            self.enable_save_buttons()

    def clear_undo_redo_stacks(self):
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.update_undo_redo_buttons()

    def update_undo_redo_buttons(self):
        self.undo_button.setEnabled(bool(self.undo_stack))
        self.redo_button.setEnabled(bool(self.redo_stack))

class ColorBarItemDelegate(QStyledItemDelegate):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.bar_width = 100
        self.bar_height = 20

    def paint(self, painter, option, index):
        # Draw the standard background
        option.widget.style().drawControl(QStyle.ControlElement.CE_ItemViewItem, option, painter, option.widget)
        
        # Get the colormap name
        cmap_name = index.data()
        try:
            # Use new style colormap access
            cmap = matplotlib.colormaps[cmap_name]
        except KeyError:
            # Fallback to default colormap if not found
            cmap = matplotlib.colormaps['viridis']
        
        # Calculate rectangles for text and color bar
        text_rect = option.rect.adjusted(4, 0, -(self.bar_width + 8), 0)
        bar_rect = QRectF(  # Use QRectF instead of QRect
            option.rect.right() - self.bar_width - 4,
            option.rect.top() + (option.rect.height() - self.bar_height) // 2,
            self.bar_width,
            self.bar_height
        )
        
        # Draw text
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter, cmap_name)
        
        # Create gradient with QPointF
        start = QPointF(bar_rect.left(), bar_rect.top())
        end = QPointF(bar_rect.right(), bar_rect.top())
        gradient = QLinearGradient(start, end)
        
        # Add colors to gradient
        n_colors = 100
        for i in range(n_colors):
            ratio = i / (n_colors - 1)
            rgba = cmap(ratio)
            color = QColor.fromRgbF(rgba[0], rgba[1], rgba[2])
            gradient.setColorAt(ratio, color)
        
        painter.fillRect(bar_rect, gradient)

    def sizeHint(self, option, index):
        size = super().sizeHint(option, index)
        return QSize(size.width() + self.bar_width + 8, max(size.height(), self.bar_height + 8))
