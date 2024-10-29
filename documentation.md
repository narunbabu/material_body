# Material Body Simulator Documentation

## Table of Contents
- [Introduction](#introduction)
- [Installation and Dependencies](#installation-and-dependencies)
- [Code Structure Overview](#code-structure-overview)
- [Simulation Mechanics](#simulation-mechanics)
- [Usage Instructions](#usage-instructions)
- [Algorithm and Implementation Details](#algorithm-and-implementation-details)
- [Understanding the Simulation](#understanding-the-simulation)
- [Potential Improvements](#potential-improvements)
- [Conclusion](#conclusion)

## Introduction

The Material Body Simulator is a Python application designed to visualize and simulate the behavior of layered material bodies within hierarchical Layered Density Structures (LDS). It models the interactions of bodies with varying densities, rotational dynamics, and adaptation mechanisms like the shedding of higher-order bodies. Users can define complex hierarchical bodies with multiple layers, micro-layers, and child bodies, each possessing unique physical properties such as density, thickness, and angular velocities.

This documentation provides a comprehensive guide to understanding, reproducing, and enhancing the Material Body Simulator. It covers the code structure, key classes and functions, simulation mechanics, usage instructions, and suggestions for future improvements.

## Installation and Dependencies

To run the Material Body Simulator, ensure you have the following dependencies installed:

- Python 3.8 or later
- PyQt6
- Matplotlib
- NumPy

You can install the required packages using pip:

```bash
pip install PyQt6 matplotlib numpy
```

## Code Structure Overview

### Main Classes

#### MaterialBodySimulator
- The main application window orchestrating the simulation
- Manages the GUI components and user interactions

#### MaterialBodyCanvas
- Handles rendering of material bodies using Matplotlib within a PyQt6 widget
- Manages plotting, user interactions like zooming and panning, and updates during the simulation

#### PropertiesDialog
- A dialog window for changing properties of material bodies
- Allows users to modify attributes like color, angular speeds, and density tolerances

#### RotationResult
- A data class used to store rotation results when minimizing density differences

### Utility Functions

These functions handle calculations related to densities, rotations, and patch mappings:

**Density Initialization and Updates**
- `initialize_densities()`
- `update_child_body_density()`

**Position Calculations**
- `calculate_child_patch_positions()`
- `calculate_parent_patch_positions()`
- `calculate_all_patch_centers()`

**Density Mismatch and Rotation Optimization**
- `find_optimal_rotation()`
- `calculate_density_mismatch()`

**Patch Mappings and Nearest Patches**
- `patch_mappings()`
- `find_nearest_patches_vectorized()`

## Simulation Mechanics

### Layered Density Structures (LDS)
1. **Multi-layered Bodies**: Each material body consists of multiple layers, and each layer can contain multiple micro-layers
2. **Density Gradient**: Densities increase towards the center of the body; inner layers have higher densities than outer layers
3. **Density Variations Along Layers**: Densities vary along the layers themselves (tangential variation), represented as patches

### Rotational Adjustments
- **Optimal Rotation Calculation**: Before updating positions, the program calculates the optimal rotation for the child body to minimize cumulative density differences with the parent layer
- **Rotation Adjustment**: After applying this rotation, the cumulative density difference is minimized through rotation alone, without further density adjustments
- **Minimal Density Changes**: This rotational adjustment allows the child body to satisfy the density conditions with minimal changes to its own density distribution

### Shedding Mechanism
- **Adaptation to External Density Fields**: Bodies can shed higher-order bodies when moving into regions of higher external density
- **Density Threshold**: Each body has a `density_threshold` property that determines when shedding should occur
- **Shedding Process**: When the external density exceeds this threshold and the body can shed (`can_shed` property), it removes its higher-order bodies and adjusts its properties

## Usage Instructions

### Running the Simulator

1. **Ensure Dependencies Are Installed**
   - Install the required packages as mentioned in the Installation and Dependencies section

2. **Define Material Bodies**
   - At the end of the script, material bodies are defined in a nested dictionary structure:

```python
material_object = {
    "name": "A",
    "color": "#00dd00",
    "placement_angle": 90,
    "angular_speed": 0.0,
    "rotation_angle": 0.0,
    "rotation_speed": 0.0,
    "show_label": True,
    "can_shed": False,
    "layers": [
        # Define layers for body A
    ],
    "child_bodies": [
        {
            "parent_layer": 0,
            "name": "B",
            "can_shed": False,
            "layers": [
                # Define layers for body B
            ],
            "child_bodies": [
                # Define further child bodies (e.g., C, D, E)
            ]
        }
    ]
}
```

3. **Run the Script**
```bash
python material_body_simulator.py
```

### Interacting with the GUI

#### Toolbar Controls
- Body Selector: Choose which body to focus on
- Arrest Revolutions Checkbox: Toggle the arrest of revolution movements
- Play/Pause Button: Start or stop the animation
- Step Forward Button: Advance the simulation by one time step
- Step Back Button: Go back one step in the simulation
- Restart Button: Restart the simulation to its initial state
- Change Properties: Open the properties dialog to modify body properties
- Save/Load Project: Save the current simulation state or load a previous one

#### Canvas Interactions
- Zooming: Use the scroll wheel to zoom in and out
- Panning: Right-click and drag to pan around the canvas

## Algorithm and Implementation Details

### Stage 1: LDS and Rotational Adjustments

#### Key Implementations

**Layered Density Structures**
- Bodies consist of multiple layers with densities increasing towards the center
- Layers can be subdivided into micro-layers for gradual density transitions
- Density variations are represented through patches along the layers

**Child Body Outer Layer Density**
- The outer layer density of a child body must always be greater than the parent body's layer density at the point of contact
- Child bodies adjust their outer layer densities during initialization to satisfy this rule

**Increasing Density in Inner Layers**
- Inner layer patches have densities greater than adjacent outer layer patches
- A compulsory increase (tolerance) is applied during initialization to maintain this gradient

**Minimizing Cumulative Density Difference via Rotation**
- Optimal rotation angles are calculated to minimize density mismatches between child and parent layers
- Rotation adjustments are applied without altering the densities during the simulation

#### Simulation Dynamics

**Initialization**
- Densities for each layer and micro-layer are initialized, ensuring they increase towards the center
- Child bodies adjust their outer layer densities based on the parent layer densities

**Rotation and Movement**
- Child bodies have properties like `placement_angle` and `rotation_angle`
- As a child body moves within the parent body, it must rotate to maintain minimal cumulative density differences

**Visualization**
- Real-time rendering of bodies, layers, micro-layers, and patches
- Animation of rotations to show how child bodies adjust their orientation
- Display of density values and variations along layers
- User interaction through zooming, panning, and property adjustments

### Stage 2: Implementing the Shedding Mechanism

#### Objective
To extend the simulation by implementing adaptation mechanisms that allow bodies to shed their higher-order bodies when the external density field increases beyond a threshold.

#### Key Tasks

**Construct Hierarchical Bodies (C, D, E)**
- Extend the material object hierarchy by adding bodies C, D, and E as nested child bodies
- Assign properties like `density_threshold` and `can_shed` to control shedding behavior

**Initialize Densities for Additional Bodies**
- Use the existing density initialization functions to set up densities for the new bodies

**Implement Shedding Logic**
- Develop functions to determine when a body should shed higher-order bodies based on external density
- Remove higher-order bodies from the `child_bodies` list when shedding occurs
- Adjust body properties post-shedding, including density profiles and layer radii

**Update Body Movements**
- Ensure bodies C, D, and E update their positions and rotations based on the movements of their parent bodies
- Account for changes in external density due to parent body movements

**Adjust Visualization**
- Modify plotting functions to reflect changes due to shedding
- Shed bodies are no longer visualized
- Layer sizes and densities are adjusted to represent updated body properties

#### Algorithm Steps

1. **Extend Material Object Structure**
   - Add properties `density_threshold` and `can_shed` to bodies C, D, and E

2. **Implement Shedding Functions**
   - `check_and_shed_higher_order_bodies(body, external_density)`: Determines if shedding should occur
   - `update_body_properties_after_shedding(body)`: Adjusts the body's properties post-shedding

3. **Calculate External Density**
   - `calculate_external_density(body)`: Calculates the external density experienced by a body

4. **Modify Update Functions**
   - Update `update_body_and_children()` to include calls to the shedding logic

5. **Adjust Visualization**
   - Ensure that shed bodies are no longer plotted
   - Update layer thicknesses and densities in the visualization

## Understanding the Simulation

### Material Body Representation

#### Bodies and Layers
- Each material body consists of multiple layers, defined from the outermost to the innermost
- Layers have properties like thickness, density, and the number of micro-layers

#### Micro-Layers
- Layers can be subdivided into micro-layers to simulate gradual changes in density
- This allows for more precise modeling of material properties

#### Child Bodies
- Bodies can have child bodies embedded within specific layers
- Each child body can have its own layers and child bodies, creating a hierarchical structure

### Dynamics and Rotations

#### Placement Angle (`placement_angle`)
- Defines the angle at which a body is placed relative to its parent
- Determines the body's position within the parent layer

#### Angular Speed (`angular_speed`)
- The speed at which a body revolves around its parent layer

#### Rotation Angle (`rotation_angle`)
- The body's own rotation angle
- Adjusts to minimize density mismatches with the parent layer

#### Density Matching
- The simulation adjusts the rotation of child bodies to minimize density differences between adjacent layers
- Crucial for simulating realistic interactions between materials

### Shedding Events

#### Shedding Conditions
- Occurs when the external density exceeds the body's `density_threshold`
- The body must have `can_shed` set to `True`

#### Shedding Process
- The body sheds its higher-order bodies (child bodies)
- Adjusts its own properties, such as layer thicknesses and densities

#### Effects of Shedding
- Allows the body to adapt to higher-density regions without violating density rules
- Shed bodies are no longer visualized in the simulation

## Potential Improvements

### Performance Optimization
- **Vectorization**: Further optimize calculations using NumPy vectorization
- **Parallel Processing**: Implement parallel processing for concurrent computations

### User Interface Enhancements
- **Interactive Editing**: Allow users to add or modify bodies and layers directly through the GUI
- **Visualization Options**: Provide options to change color maps, display settings, or layer visibility

### Advanced Physical Modeling
- **Dynamic Density Updates**: Implement models for density changes over time or due to interactions
- **Collision Detection**: Add physics for detecting and responding to collisions between bodies

### Error Handling and Validation
- **Input Validation**: Ensure all user inputs are validated to prevent runtime errors
- **Exception Handling**: Implement comprehensive exception handling throughout the code

### Documentation and Testing
- **Unit Tests**: Develop unit tests for key functions to ensure correctness
- **Expanded Documentation**: Provide examples and tutorials

## Conclusion

The Material Body Simulator offers a robust platform for simulating and visualizing complex material bodies within hierarchical layered density structures. By incorporating rotational adjustments and shedding mechanisms, it models the dynamic interactions of bodies in varying density fields.

This documentation provides the necessary understanding to reproduce the program, extend its functionality, and tailor it to specific simulation needs. Whether for educational purposes, research, or engineering applications, the simulator serves as a valuable tool for exploring the interactions of layered materials in a dynamic environment.