"""
Spacetime Curvature Module for Jugend forscht 2026 Physics Visualization Project
Raumzeit-Kruemmungs-Modul fuer Jugend forscht 2026 Physik-Visualisierungsprojekt

This module visualizes the curvature of spacetime around massive objects:
- Gravitational potential wells (2D and 3D)
- Schwarzschild radius and event horizons
- Compactness parameter comparison across stellar objects
- How gravity bends spacetime according to General Relativity

The key insight is how mass curves spacetime, and how compact objects
(white dwarfs, neutron stars, black holes) represent extreme curvature.

Author: Jugend forscht 2026 Project
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import os
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass

from .constants import get_constants, PhysicalConstants
from .color_scheme import COLORS, get_stellar_colors, get_sequence


# Output directory for visualizations
VIS_DIR = os.path.join(os.path.dirname(__file__), '..', 'visualizations')


@dataclass
class CompactObject:
    """
    Container for compact object properties.
    Behaelter fuer kompakte Objekteigenschaften.
    """
    name: str                # Object name
    name_de: str             # German name
    mass: float              # Mass [kg]
    radius: float            # Radius [m]
    schwarzschild_radius: float  # Schwarzschild radius [m]
    compactness: float       # R_s / R (dimensionless)
    surface_gravity: float   # Surface gravity [m/s²]
    escape_velocity: float   # Escape velocity [m/s]
    is_black_hole: bool      # Whether R < R_s


def calculate_compact_object(
    name: str,
    name_de: str,
    mass: float,
    radius: float,
    constants: Optional[PhysicalConstants] = None
) -> CompactObject:
    """
    Calculate properties of a compact object.
    Berechnet die Eigenschaften eines kompakten Objekts.

    Args:
        name: Object name in English
        name_de: Object name in German
        mass: Mass [kg]
        radius: Radius [m]
        constants: Physical constants

    Returns:
        CompactObject with calculated properties
    """
    if constants is None:
        constants = get_constants()

    R_s = constants.schwarzschild_radius(mass)
    compactness = R_s / radius if radius > 0 else float('inf')

    # Surface gravity: g = G*M/R²
    g = constants.surface_gravity(mass, radius) if radius > 0 else float('inf')

    # Escape velocity: v_esc = sqrt(2*G*M/R)
    v_esc = constants.escape_velocity(mass, radius) if radius > 0 else constants.c

    is_black_hole = radius <= R_s

    return CompactObject(
        name=name,
        name_de=name_de,
        mass=mass,
        radius=radius,
        schwarzschild_radius=R_s,
        compactness=compactness,
        surface_gravity=g,
        escape_velocity=v_esc,
        is_black_hole=is_black_hole
    )


def get_stellar_objects(constants: Optional[PhysicalConstants] = None) -> List[CompactObject]:
    """
    Get a list of stellar objects for comparison.
    Gibt eine Liste von Sternobjekten zum Vergleich zurueck.

    Returns:
        List of CompactObject instances
    """
    if constants is None:
        constants = get_constants()

    objects = [
        # Earth
        calculate_compact_object(
            'Earth', 'Erde',
            constants.M_earth, constants.R_earth, constants
        ),
        # Sun
        calculate_compact_object(
            'Sun', 'Sonne',
            constants.M_sun, constants.R_sun, constants
        ),
        # White Dwarf (typical: 0.6 M_sun, ~8000 km radius)
        calculate_compact_object(
            'White Dwarf', 'Weisser Zwerg',
            0.6 * constants.M_sun, 8000e3, constants
        ),
        # Neutron Star (typical: 1.4 M_sun, ~10 km radius)
        calculate_compact_object(
            'Neutron Star', 'Neutronenstern',
            1.4 * constants.M_sun, 10e3, constants
        ),
        # Black Hole (stellar mass: 10 M_sun)
        calculate_compact_object(
            'Black Hole\n(10 M☉)', 'Schwarzes Loch\n(10 M☉)',
            10 * constants.M_sun, 0, constants  # radius = 0 for simplicity
        ),
    ]

    # For black hole, set radius to Schwarzschild radius
    objects[-1] = CompactObject(
        name=objects[-1].name,
        name_de=objects[-1].name_de,
        mass=objects[-1].mass,
        radius=objects[-1].schwarzschild_radius,
        schwarzschild_radius=objects[-1].schwarzschild_radius,
        compactness=1.0,  # By definition for black hole
        surface_gravity=float('inf'),
        escape_velocity=constants.c,
        is_black_hole=True
    )

    return objects


def gravitational_potential(r: np.ndarray, M: float, constants: PhysicalConstants) -> np.ndarray:
    """
    Calculate Newtonian gravitational potential.
    Berechnet das Newtonsche Gravitationspotential.

    Formula: Φ = -G*M/r

    Args:
        r: Distance from center [m]
        M: Mass [kg]
        constants: Physical constants

    Returns:
        Gravitational potential [J/kg]
    """
    # Avoid division by zero
    r_safe = np.where(r > 0, r, 1e-10)
    return -constants.G * M / r_safe


def spacetime_embedding(x: np.ndarray, y: np.ndarray, M: float,
                        constants: PhysicalConstants, scale: float = 1.0) -> np.ndarray:
    """
    Calculate the embedding of curved spacetime for visualization.
    Berechnet die Einbettung der gekruemmten Raumzeit zur Visualisierung.

    This creates the "rubber sheet" visualization of spacetime curvature.
    The depth represents how much spacetime is curved by mass.

    Args:
        x, y: Coordinate grids [m]
        M: Mass [kg]
        constants: Physical constants
        scale: Scaling factor for visualization

    Returns:
        Z values representing spacetime curvature (depth)
    """
    r = np.sqrt(x**2 + y**2)
    R_s = constants.schwarzschild_radius(M)

    # Use a logarithmic-like function that shows the potential well
    # Avoid singularity at r=0 by setting minimum radius
    r_min = R_s * 0.5  # Don't go inside event horizon
    r_safe = np.maximum(r, r_min)

    # Embedding function: z = -sqrt(R_s * r) gives parabolic well shape
    # This is a simplified visualization, not exact GR embedding
    z = -scale * np.sqrt(R_s * r_safe)

    return z


def plot_potential_well_2d(
    constants: Optional[PhysicalConstants] = None,
    language: str = 'en',
    save: bool = True,
    show: bool = True
) -> plt.Figure:
    """
    Plot 2D cross-section of gravitational potential wells.
    Zeigt 2D-Querschnitt von Gravitationspotentialmulden.

    Shows the surface gravitational potential depth for different objects,
    demonstrating how compact objects have much deeper potential wells.

    Args:
        constants: Physical constants
        language: 'en' for English, 'de' for German
        save: Whether to save the figure
        show: Whether to display the figure

    Returns:
        matplotlib Figure object
    """
    if constants is None:
        constants = get_constants()

    objects = get_stellar_objects(constants)

    # Create figure with two subplots stacked vertically with proper spacing
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14), gridspec_kw={'hspace': 0.4})

    # Colors for different objects (Earth, Sun, White Dwarf, Neutron Star, Black Hole)
    colors = get_stellar_colors()

    # Top plot: Surface potential depth (shows compactness effect)
    names = [obj.name if language == 'en' else obj.name_de for obj in objects[:-1]]

    # Calculate |Φ_surface| / c² = GM/(Rc²) = R_s/(2R) = C/2
    # This is the dimensionless potential depth
    potential_depths = [obj.compactness / 2 for obj in objects[:-1]]

    bars = ax1.bar(names, potential_depths, color=colors[:-1], edgecolor='black', linewidth=1.5)
    ax1.set_yscale('log')

    # Add value labels
    for bar, depth in zip(bars, potential_depths):
        ax1.text(bar.get_x() + bar.get_width()/2, depth * 1.5, f'{depth:.1e}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Add reference line for black hole
    ax1.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
               label='Black Hole limit' if language == 'en' else 'Schwarzes-Loch-Grenze')

    if language == 'de':
        ax1.set_ylabel('|Φ_surface| / c²', fontsize=12)
        ax1.set_title('1. Potentialtiefe an der Oberfläche (dimensionslos)', fontsize=14, fontweight='bold', pad=15)
    else:
        ax1.set_ylabel('|Φ_surface| / c²', fontsize=12)
        ax1.set_title('1. Potential Depth at Surface (dimensionless)', fontsize=14, fontweight='bold', pad=15)

    ax1.legend(fontsize=9, loc='upper right', bbox_to_anchor=(1.0, -0.08), framealpha=0.7)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(1e-10, 1)

    # Right plot: Potential vs distance (showing shape of wells)
    # Use physical distance in km for comparison
    r_km = np.linspace(1, 1000, 500)  # 1 to 1000 km from center

    for obj, color in zip(objects[:-1], colors[:-1]):
        # Only plot if object radius is in this range
        r_surface_km = obj.radius / 1000

        if r_surface_km < 1000:
            # Calculate potential outside the object
            r_plot = r_km[r_km >= r_surface_km]
            r_m = r_plot * 1000  # Convert to meters

            # Potential in units of c²
            phi_over_c2 = -constants.G * obj.mass / (r_m * constants.c**2)

            label = obj.name if language == 'en' else obj.name_de
            ax2.plot(r_plot, phi_over_c2, color=color, linewidth=2.5, label=label)

            # Mark surface with a dot
            phi_surface = -constants.G * obj.mass / (obj.radius * constants.c**2)
            ax2.plot(r_surface_km, phi_surface, 'o', color=color, markersize=8)

    if language == 'de':
        ax2.set_xlabel('Abstand vom Zentrum (km)', fontsize=12)
        ax2.set_ylabel('Gravitationspotential Φ/c²', fontsize=12)
        ax2.set_title('2. Potentialmulden-Form (Punkte = Oberfläche)', fontsize=14, fontweight='bold', pad=15)
    else:
        ax2.set_xlabel('Distance from Center (km)', fontsize=12)
        ax2.set_ylabel('Gravitational Potential Φ/c²', fontsize=12)
        ax2.set_title('2. Potential Well Shape (dots = surface)', fontsize=14, fontweight='bold', pad=15)

    ax2.legend(fontsize=9, loc='lower right', bbox_to_anchor=(1.0, -0.1), framealpha=0.7)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(1, 1000)
    ax2.set_xscale('log')

    if save:
        os.makedirs(VIS_DIR, exist_ok=True)
        filepath = os.path.join(VIS_DIR, 'potential_well_2d.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")

    if show:
        plt.show()

    return fig


def plot_potential_well_3d(
    mass_solar: float = 1.0,
    constants: Optional[PhysicalConstants] = None,
    language: str = 'en',
    save: bool = True,
    show: bool = True
) -> plt.Figure:
    """
    Plot 3D visualization of spacetime curvature (potential well).
    Zeigt 3D-Visualisierung der Raumzeitkruemmung (Potentialmulde).

    Creates the classic "rubber sheet" visualization showing how
    mass curves spacetime.

    Args:
        mass_solar: Mass in solar masses
        constants: Physical constants
        language: 'en' for English, 'de' for German
        save: Whether to save the figure
        show: Whether to display the figure

    Returns:
        matplotlib Figure object
    """
    if constants is None:
        constants = get_constants()

    mass = mass_solar * constants.M_sun
    R_s = constants.schwarzschild_radius(mass)

    # Create coordinate grid
    # Use units of Schwarzschild radii for scale
    grid_size = 50
    extent = 20 * R_s  # Show out to 20 Schwarzschild radii

    x = np.linspace(-extent, extent, grid_size)
    y = np.linspace(-extent, extent, grid_size)
    X, Y = np.meshgrid(x, y)

    # Calculate spacetime embedding
    Z = spacetime_embedding(X, Y, mass, constants, scale=1.0)

    # Create 3D figure
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot surface with colormap
    surf = ax.plot_surface(X / R_s, Y / R_s, Z / R_s,
                           cmap=cm.viridis, alpha=0.8,
                           linewidth=0, antialiased=True)

    # Add event horizon circle at z=min
    theta = np.linspace(0, 2*np.pi, 100)
    x_horizon = np.cos(theta)
    y_horizon = np.sin(theta)
    z_horizon = np.min(Z / R_s) * np.ones_like(theta)
    ax.plot(x_horizon, y_horizon, z_horizon, 'r-', linewidth=2,
            label='Event Horizon' if language == 'en' else 'Ereignishorizont')

    # Labels
    if language == 'de':
        ax.set_xlabel('x (Schwarzschild-Radien)', fontsize=11)
        ax.set_ylabel('y (Schwarzschild-Radien)', fontsize=11)
        ax.set_zlabel('Raumzeit-Kruemmung', fontsize=11)
        title = f'Raumzeitkruemmung um {mass_solar} Sonnenmasse(n)\n"Gummituch"-Visualisierung'
    else:
        ax.set_xlabel('x (Schwarzschild radii)', fontsize=11)
        ax.set_ylabel('y (Schwarzschild radii)', fontsize=11)
        ax.set_zlabel('Spacetime Curvature', fontsize=11)
        title = f'Spacetime Curvature around {mass_solar} Solar Mass(es)\n"Rubber Sheet" Visualization'

    ax.set_title(title, fontsize=14, pad=20)

    # Add colorbar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
    cbar.set_label('Curvature Depth' if language == 'en' else 'Kruemmungstiefe', fontsize=10)

    # Add info text
    if language == 'de':
        info = (f'Schwarzschild-Radius: {R_s:.2e} m\n'
                f'Masse: {mass:.2e} kg')
    else:
        info = (f'Schwarzschild radius: {R_s:.2e} m\n'
                f'Mass: {mass:.2e} kg')

    # Position text in 2D overlay
    fig.text(0.02, 0.02, info,
            fontsize=10, va='bottom',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'),
            zorder=10)

    ax.legend(fontsize=9, loc='upper left', framealpha=0.7)

    # Set viewing angle
    ax.view_init(elev=25, azim=45)

    plt.tight_layout()

    if save:
        os.makedirs(VIS_DIR, exist_ok=True)
        filepath = os.path.join(VIS_DIR, 'potential_well_3d.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")

    if show:
        plt.show()

    return fig


def plot_compactness_comparison(
    constants: Optional[PhysicalConstants] = None,
    language: str = 'en',
    save: bool = True,
    show: bool = True
) -> plt.Figure:
    """
    Compare compactness parameter across different stellar objects.
    Vergleicht den Kompaktheitsparameter verschiedener Sternobjekte.

    Compactness = R_schwarzschild / R_object
    - Earth: ~10^-9
    - Sun: ~10^-6
    - White Dwarf: ~10^-4
    - Neutron Star: ~0.2-0.4
    - Black Hole: 1 (by definition)

    Args:
        constants: Physical constants
        language: 'en' for English, 'de' for German
        save: Whether to save the figure
        show: Whether to display the figure

    Returns:
        matplotlib Figure object
    """
    if constants is None:
        constants = get_constants()

    objects = get_stellar_objects(constants)

    # Create figure with two subplots stacked vertically with proper spacing
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16), gridspec_kw={'hspace': 0.4})

    # Top plot: Compactness on log scale
    names = [obj.name if language == 'en' else obj.name_de for obj in objects]
    compactness_values = [obj.compactness for obj in objects]
    colors = get_stellar_colors()

    bars = ax1.bar(names, compactness_values, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_yscale('log')

    # Add value labels on bars
    for bar, comp in zip(bars, compactness_values):
        if comp < 0.1:
            # Small values: label above bar
            ax1.text(bar.get_x() + bar.get_width()/2, comp * 2, f'{comp:.1e}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        elif comp < 1:
            # Neutron Star (~0.4): label inside bar to avoid overlap with red line
            ax1.text(bar.get_x() + bar.get_width()/2, comp * 0.4, f'{comp:.1e}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        else:
            # Black Hole (1.0): label inside bar
            ax1.text(bar.get_x() + bar.get_width()/2, comp * 0.5, f'{comp:.1f}',
                    ha='center', va='top', fontsize=9, fontweight='bold', color='white')

    # Add horizontal lines for reference
    ax1.axhline(y=1, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
               label='Black Hole limit (C=1)' if language == 'en' else 'Schwarzes-Loch-Grenze (C=1)')
    ax1.axhline(y=0.5, color='orange', linestyle=':', linewidth=1.5, alpha=0.7,
               label='Neutron Star typical' if language == 'en' else 'Neutronenstern typisch')

    if language == 'de':
        ax1.set_ylabel('Kompaktheit C = R_s / R', fontsize=12)
        ax1.set_title('1. Kompaktheit verschiedener Objekte (logarithmische Skala)', fontsize=14, fontweight='bold', pad=15)
    else:
        ax1.set_ylabel('Compactness C = R_s / R', fontsize=12)
        ax1.set_title('1. Compactness of Different Objects (logarithmic scale)', fontsize=14, fontweight='bold', pad=15)

    ax1.legend(fontsize=9, loc='upper right', bbox_to_anchor=(1.0, -0.08), framealpha=0.7)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(1e-10, 2)

    # Bottom plot: Visual size comparison (radius vs Schwarzschild radius)
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-0.5, len(objects) - 0.5)

    for i, obj in enumerate(objects):
        # Normalize to show relative sizes
        if obj.compactness < 1:
            # Object radius (blue) and Schwarzschild radius (red)
            r_norm = 1.0
            rs_norm = obj.compactness

            # Draw object
            circle_obj = plt.Circle((0, i), r_norm * 0.4, color=colors[i], alpha=0.6)
            ax2.add_patch(circle_obj)

            # Draw Schwarzschild radius (as a smaller red circle inside)
            if rs_norm > 0.01:  # Only draw if visible
                circle_rs = plt.Circle((0, i), rs_norm * 0.4, color='red', alpha=0.8)
                ax2.add_patch(circle_rs)
        else:
            # Black hole: R = R_s
            circle_bh = plt.Circle((0, i), 0.4, color='black', alpha=1.0)
            ax2.add_patch(circle_bh)
            circle_rs = plt.Circle((0, i), 0.4, color='red', alpha=0.3, linestyle='--', fill=False, linewidth=2)
            ax2.add_patch(circle_rs)

        # Add label
        name = obj.name if language == 'en' else obj.name_de
        ax2.text(1.2, i, name, fontsize=10, va='center')

    ax2.set_aspect('equal')
    ax2.axis('off')

    if language == 'de':
        ax2.set_title('2. Größenvergleich: Objekt vs. Schwarzschild-Radius (zeigt relatives Verhältnis)', fontsize=14, fontweight='bold', pad=15)
    else:
        ax2.set_title('2. Size Comparison: Object vs. Schwarzschild Radius (shows relative ratio)', fontsize=14, fontweight='bold', pad=15)

    # Add legend patches for the visual comparison - position at upper right to avoid overlap with circles
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='gray', edgecolor='black', alpha=0.6,
              label='Object radius' if language == 'en' else 'Objektradius'),
        Patch(facecolor='red', edgecolor='red', alpha=0.8,
              label='Schwarzschild radius (R_s)' if language == 'en' else 'Schwarzschild-Radius (R_s)')
    ]
    ax2.legend(handles=legend_elements, fontsize=9, loc='lower right', bbox_to_anchor=(1.0, -0.1), framealpha=0.7)

    if save:
        os.makedirs(VIS_DIR, exist_ok=True)
        filepath = os.path.join(VIS_DIR, 'compactness_comparison.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")

    if show:
        plt.show()

    return fig


def plot_escape_velocity(
    constants: Optional[PhysicalConstants] = None,
    language: str = 'en',
    save: bool = True,
    show: bool = True
) -> plt.Figure:
    """
    Plot escape velocity comparison and its relation to compactness.
    Zeigt Fluchtgeschwindigkeitsvergleich und Beziehung zur Kompaktheit.

    When escape velocity = c, the object becomes a black hole.

    Args:
        constants: Physical constants
        language: 'en' for English, 'de' for German
        save: Whether to save the figure
        show: Whether to display the figure

    Returns:
        matplotlib Figure object
    """
    if constants is None:
        constants = get_constants()

    objects = get_stellar_objects(constants)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    names = [obj.name if language == 'en' else obj.name_de for obj in objects]
    escape_velocities = [min(obj.escape_velocity, constants.c) for obj in objects]
    colors = get_stellar_colors()

    # Create bars showing escape velocity as fraction of c (log scale to show all)
    v_over_c = [v / constants.c for v in escape_velocities]
    bars = ax.bar(names, v_over_c, color=colors, edgecolor='black', linewidth=1.5)

    # Use log scale to show the huge range
    ax.set_yscale('log')

    # Add value labels
    for bar, v, v_frac in zip(bars, escape_velocities, v_over_c):
        if v < constants.c:
            label = f'{v/1000:.0f} km/s\n({v_frac:.1e} c)'
        else:
            label = 'c\n(light speed)'

        # Position label above bar
        ax.text(bar.get_x() + bar.get_width()/2, v_frac * 1.5, label,
               ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Add speed of light line
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.8,
              label='Speed of light c' if language == 'en' else 'Lichtgeschwindigkeit c')

    # Add Earth escape velocity reference
    v_earth_esc = objects[0].escape_velocity
    ax.axhline(y=v_earth_esc/constants.c, color='green', linestyle=':', alpha=0.7,
              label=f'Earth escape velocity ({v_earth_esc/1000:.1f} km/s)' if language == 'en'
                    else f'Erd-Fluchtgeschwindigkeit ({v_earth_esc/1000:.1f} km/s)')

    if language == 'de':
        ax.set_ylabel('Fluchtgeschwindigkeit (Anteil von c) - log Skala', fontsize=12)
        ax.set_title('Fluchtgeschwindigkeit verschiedener Objekte', fontsize=14, fontweight='bold', pad=15)
    else:
        ax.set_ylabel('Escape Velocity (fraction of c) - log scale', fontsize=12)
        ax.set_title('Escape Velocity of Different Objects', fontsize=14, fontweight='bold', pad=15)

    ax.legend(fontsize=9, loc='upper right', bbox_to_anchor=(1.0, -0.08), framealpha=0.7)
    ax.grid(True, alpha=0.3, axis='y', which='both')
    ax.set_ylim(1e-5, 2)

    if save:
        os.makedirs(VIS_DIR, exist_ok=True)
        filepath = os.path.join(VIS_DIR, 'escape_velocity.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")

    if show:
        plt.show()

    return fig


def plot_spacetime_summary(
    constants: Optional[PhysicalConstants] = None,
    language: str = 'en',
    save: bool = True,
    show: bool = True
) -> plt.Figure:
    """
    Create a comprehensive summary of spacetime curvature concepts.
    Erstellt eine umfassende Zusammenfassung der Raumzeitkruemmungskonzepte.

    Args:
        constants: Physical constants
        language: 'en' for English, 'de' for German
        save: Whether to save the figure
        show: Whether to display the figure

    Returns:
        matplotlib Figure object
    """
    if constants is None:
        constants = get_constants()

    objects = get_stellar_objects(constants)

    # Create figure with 3 subplots stacked vertically
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18), gridspec_kw={'hspace': 0.7})

    # Get names and colors
    names = [obj.name.split('\n')[0] if language == 'en' else obj.name_de.split('\n')[0]
             for obj in objects]
    compactness = [obj.compactness for obj in objects]
    colors = get_stellar_colors()

    # 1. Compactness bar chart - individual bars with legends
    for name, comp, color in zip(names, compactness, colors):
        ax1.bar(name, comp, color=color, edgecolor='black', label=name)
    ax1.set_yscale('log')
    ax1.set_ylabel('Compactness C' if language == 'en' else 'Kompaktheit C', fontsize=11)
    ax1.set_title('1. Compactness C = R_s/R' if language == 'en' else '1. Kompaktheit C = R_s/R',
                 fontsize=14, fontweight='bold', pad=15)
    ax1.tick_params(axis='x', rotation=45)
    ax1.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='BH limit')
    ax1.legend(fontsize=9, loc='upper right', bbox_to_anchor=(1.0, -0.35), ncol=3, framealpha=0.7)
    ax1.grid(True, alpha=0.3, axis='y')

    # 2. Escape velocity comparison (log scale to show all values)
    v_esc = [min(obj.escape_velocity, constants.c) / constants.c for obj in objects]

    for name, v, color in zip(names, v_esc, colors):
        ax2.bar(name, v, color=color, edgecolor='black', label=name)
    ax2.set_yscale('log')
    ax2.set_ylabel('v_esc / c (log)', fontsize=11)
    ax2.set_title('2. Escape Velocity' if language == 'en' else '2. Fluchtgeschwindigkeit',
                 fontsize=14, fontweight='bold', pad=15)
    ax2.tick_params(axis='x', rotation=45)
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='c (light)')
    ax2.legend(fontsize=9, loc='upper right', bbox_to_anchor=(1.0, -0.35), ncol=3, framealpha=0.7)
    ax2.grid(True, alpha=0.3, axis='y', which='both')
    ax2.set_ylim(1e-5, 2)

    # 3. Surface gravity comparison
    g_values = [obj.surface_gravity for obj in objects[:-1]]  # Exclude black hole (infinite)
    g_names = names[:-1]
    g_colors = colors[:-1]

    for name, g, color in zip(g_names, g_values, g_colors):
        ax3.bar(name, g, color=color, edgecolor='black', label=name)
    ax3.set_yscale('log')
    ax3.set_ylabel('g (m/s²)', fontsize=11)
    ax3.set_title('3. Surface Gravity' if language == 'en' else '3. Oberflaechengravitation',
                 fontsize=14, fontweight='bold', pad=15)
    ax3.tick_params(axis='x', rotation=45)
    ax3.axhline(y=9.8, color='green', linestyle=':', alpha=0.7, label='Earth g')
    ax3.legend(fontsize=9, loc='upper right', bbox_to_anchor=(1.0, -0.35), ncol=3, framealpha=0.7)
    ax3.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save:
        os.makedirs(VIS_DIR, exist_ok=True)
        filepath = os.path.join(VIS_DIR, 'spacetime_summary.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")

    if show:
        plt.show()

    return fig


def plot_gravitational_vector_field(
    constants: Optional[PhysicalConstants] = None,
    language: str = 'en',
    save: bool = True,
    show: bool = False
) -> plt.Figure:
    """
    Create a vector field visualization of gravitational and tidal forces.
    Erstellt eine Vektorfeldvisualisierung von Gravitations- und Gezeitenkraeften.

    Shows:
    - Left: Gravitational force vectors pointing toward central mass
    - Right: Tidal force vectors (stretching radially, compressing tangentially)

    Args:
        constants: Physical constants
        language: 'en' for English, 'de' for German
        save: Whether to save the plot
        show: Whether to display the plot

    Returns:
        matplotlib Figure object
    """
    if constants is None:
        constants = get_constants()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Create grid for vector field
    x = np.linspace(-4, 4, 15)
    y = np.linspace(-4, 4, 15)
    X, Y = np.meshgrid(x, y)

    # Distance from center
    R = np.sqrt(X**2 + Y**2)
    R = np.maximum(R, 0.5)  # Avoid singularity at center

    # ===== LEFT PLOT: Gravitational Force Field =====
    # F = -GM/r² in radial direction (pointing inward)
    # Normalized for visualization
    F_mag = 1 / R**2
    F_mag = F_mag / F_mag.max()  # Normalize

    # Unit vectors pointing toward center (negative radial direction)
    Fx = -X / R * F_mag
    Fy = -Y / R * F_mag

    # Mask center region
    mask = R < 0.6
    Fx[mask] = 0
    Fy[mask] = 0

    # Plot vector field
    ax1.quiver(X, Y, Fx, Fy, F_mag, cmap='Reds', scale=15, width=0.008,
               headwidth=4, headlength=5, alpha=0.9)

    # Draw central mass
    circle = plt.Circle((0, 0), 0.4, color=COLORS['scaled'], zorder=10)
    ax1.add_patch(circle)
    ax1.text(0, 0, 'M', ha='center', va='center', fontsize=14, fontweight='bold',
             color='white', zorder=11)

    # Add concentric circles showing equipotential lines
    for r in [1, 2, 3]:
        circle = plt.Circle((0, 0), r, fill=False, color='gray', linestyle='--',
                           alpha=0.5, linewidth=1)
        ax1.add_patch(circle)

    ax1.set_xlim(-4.5, 4.5)
    ax1.set_ylim(-4.5, 4.5)
    ax1.set_aspect('equal')
    ax1.set_xlabel('x' if language == 'en' else 'x', fontsize=12)
    ax1.set_ylabel('y' if language == 'en' else 'y', fontsize=12)

    if language == 'de':
        ax1.set_title('Gravitationsfeldvektoren\n$\\vec{F} = -\\frac{GM}{r^2}\\hat{r}$',
                     fontsize=14, fontweight='bold', pad=15)
    else:
        ax1.set_title('Gravitational Field Vectors\n$\\vec{F} = -\\frac{GM}{r^2}\\hat{r}$',
                     fontsize=14, fontweight='bold', pad=15)

    ax1.grid(True, alpha=0.3)

    # ===== RIGHT PLOT: Tidal Force Field =====
    # Tidal forces stretch objects radially and compress them tangentially
    # This is the differential of gravity

    # Create a denser grid for tidal forces
    x2 = np.linspace(-4, 4, 12)
    y2 = np.linspace(-4, 4, 12)
    X2, Y2 = np.meshgrid(x2, y2)
    R2 = np.sqrt(X2**2 + Y2**2)
    R2 = np.maximum(R2, 0.8)

    # Tidal acceleration components (simplified model)
    # Radial stretching: +2GM/r³ in radial direction
    # Tangential compression: -GM/r³ in tangential direction
    tidal_mag = 1 / R2**3
    tidal_mag = tidal_mag / tidal_mag.max()

    # Radial unit vectors
    r_hat_x = X2 / R2
    r_hat_y = Y2 / R2

    # Tangential unit vectors (perpendicular to radial)
    t_hat_x = -Y2 / R2
    t_hat_y = X2 / R2

    # Tidal force: stretches radially (outward from center of object)
    # We show this as vectors pointing outward along radial direction
    Tx = r_hat_x * tidal_mag * 2  # Radial stretching (factor of 2)
    Ty = r_hat_y * tidal_mag * 2

    # Mask center
    mask2 = R2 < 1.0
    Tx[mask2] = 0
    Ty[mask2] = 0

    # Plot tidal vectors (radial stretching - red)
    ax2.quiver(X2, Y2, Tx, Ty, color=COLORS['scaled'], scale=20, width=0.006,
               headwidth=4, headlength=4, alpha=0.8, label='Radial stretch' if language == 'en' else 'Radiale Dehnung')

    # Also show tangential compression (blue, pointing inward tangentially)
    # This is perpendicular to radial, compressing the object
    Cx = -t_hat_x * tidal_mag * 0.3  # Small tangential compression arrows
    Cy = -t_hat_y * tidal_mag * 0.3
    Cx[mask2] = 0
    Cy[mask2] = 0

    # Draw central mass
    circle2 = plt.Circle((0, 0), 0.6, color=COLORS['primary_blue'], zorder=10)
    ax2.add_patch(circle2)
    ax2.text(0, 0, 'M', ha='center', va='center', fontsize=14, fontweight='bold',
             color='white', zorder=11)

    # Draw a test object being tidally stretched
    # Ellipse representing deformation
    from matplotlib.patches import Ellipse
    test_obj = Ellipse((2.5, 0), 0.8, 0.4, angle=0, fill=True,
                       color=COLORS['standard'], alpha=0.7, zorder=5)
    ax2.add_patch(test_obj)

    # Add arrows showing the stretching on test object
    ax2.annotate('', xy=(3.1, 0), xytext=(2.9, 0),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax2.annotate('', xy=(1.9, 0), xytext=(2.1, 0),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax2.annotate('', xy=(2.5, 0.1), xytext=(2.5, 0.3),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    ax2.annotate('', xy=(2.5, -0.1), xytext=(2.5, -0.3),
                arrowprops=dict(arrowstyle='->', color='blue', lw=2))

    ax2.set_xlim(-4.5, 4.5)
    ax2.set_ylim(-4.5, 4.5)
    ax2.set_aspect('equal')
    ax2.set_xlabel('x' if language == 'en' else 'x', fontsize=12)
    ax2.set_ylabel('y' if language == 'en' else 'y', fontsize=12)

    if language == 'de':
        ax2.set_title('Gezeitenkraefte (Tidal Forces)\nRadiale Dehnung & Tangentiale Kompression',
                     fontsize=14, fontweight='bold', pad=15)
    else:
        ax2.set_title('Tidal Forces\nRadial Stretching & Tangential Compression',
                     fontsize=14, fontweight='bold', pad=15)

    ax2.grid(True, alpha=0.3)

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', marker='>', linestyle='-', markersize=8,
               label='Radial stretch' if language == 'en' else 'Radiale Dehnung'),
        Line2D([0], [0], color='blue', marker='>', linestyle='-', markersize=8,
               label='Tangential compression' if language == 'en' else 'Tangentiale Kompression')
    ]
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=10)

    plt.tight_layout()

    if save:
        os.makedirs(VIS_DIR, exist_ok=True)
        filepath = os.path.join(VIS_DIR, 'gravitational_vector_field.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")

    if show:
        plt.show()

    return fig


def plot_orbital_precession(
    constants: Optional[PhysicalConstants] = None,
    language: str = 'en',
    save: bool = True,
    show: bool = False,
    animate: bool = True
) -> plt.Figure:
    """
    Create visualization of orbital precession due to relativistic effects.
    Erstellt Visualisierung der Bahnpraezession durch relativistische Effekte.

    Shows how orbits precess (rotate) due to:
    - General relativistic corrections near massive objects
    - Mercury's perihelion precession as a famous example

    Args:
        constants: Physical constants
        language: 'en' for English, 'de' for German
        save: Whether to save the plot (and animation)
        show: Whether to display the plot
        animate: Whether to create animated GIF

    Returns:
        matplotlib Figure object
    """
    if constants is None:
        constants = get_constants()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # ===== LEFT PLOT: Static view of precessing orbit =====
    # Orbital parameters
    a = 1.0  # Semi-major axis (normalized)
    e = 0.6  # Eccentricity (exaggerated for visualization)

    # Precession rate (exaggerated for visualization)
    # Real Mercury: ~43 arcsec/century, here we use much larger for visibility
    precession_per_orbit = np.pi / 6  # 30 degrees per orbit (exaggerated)

    # Draw multiple orbits showing precession
    n_orbits = 6
    colors = plt.cm.viridis(np.linspace(0, 1, n_orbits))

    theta = np.linspace(0, 2*np.pi, 500)

    for i in range(n_orbits):
        # Precession angle for this orbit
        omega = i * precession_per_orbit

        # Orbital radius (ellipse in polar coordinates)
        r = a * (1 - e**2) / (1 + e * np.cos(theta))

        # Convert to Cartesian with precession rotation
        x = r * np.cos(theta + omega)
        y = r * np.sin(theta + omega)

        alpha = 0.3 + 0.7 * (i / n_orbits)  # Fade in
        ax1.plot(x, y, color=colors[i], linewidth=1.5, alpha=alpha)

        # Mark perihelion (closest approach)
        perihelion_angle = omega
        r_peri = a * (1 - e)
        px = r_peri * np.cos(perihelion_angle)
        py = r_peri * np.sin(perihelion_angle)
        ax1.plot(px, py, 'o', color=colors[i], markersize=6, alpha=alpha)

    # Draw central mass (Sun)
    sun = plt.Circle((0, 0), 0.08, color=COLORS['primary_amber'], zorder=10)
    ax1.add_patch(sun)
    ax1.text(0, -0.25, 'M' if language == 'en' else 'M', ha='center', fontsize=12, fontweight='bold')

    # Draw arrow showing precession direction
    arc_theta = np.linspace(0, precession_per_orbit * (n_orbits - 1), 50)
    arc_r = a * (1 - e) * 1.3
    arc_x = arc_r * np.cos(arc_theta)
    arc_y = arc_r * np.sin(arc_theta)
    ax1.plot(arc_x, arc_y, 'r--', linewidth=2, alpha=0.7)
    ax1.annotate('', xy=(arc_x[-1], arc_y[-1]), xytext=(arc_x[-5], arc_y[-5]),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))

    precession_label = 'Precession' if language == 'en' else 'Praezession'
    ax1.text(arc_r * 0.7, arc_r * 0.9, precession_label, fontsize=11, color='red', fontweight='bold')

    ax1.set_xlim(-2, 2)
    ax1.set_ylim(-2, 2)
    ax1.set_aspect('equal')
    ax1.set_xlabel('x (AU)' if language == 'en' else 'x (AE)', fontsize=12)
    ax1.set_ylabel('y (AU)' if language == 'en' else 'y (AE)', fontsize=12)

    if language == 'de':
        ax1.set_title('Perihelpraezession\n(Relativistische Bahnkorrektur)',
                     fontsize=14, fontweight='bold', pad=15)
    else:
        ax1.set_title('Perihelion Precession\n(Relativistic Orbital Correction)',
                     fontsize=14, fontweight='bold', pad=15)

    ax1.grid(True, alpha=0.3)

    # Add colorbar for orbit number
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(1, n_orbits))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1, shrink=0.6, label='Orbit number' if language == 'en' else 'Umlaufnummer')

    # ===== RIGHT PLOT: Comparison of Newtonian vs GR orbits =====
    theta2 = np.linspace(0, 4*np.pi, 1000)  # Two full orbits

    # Newtonian orbit (no precession)
    r_newton = a * (1 - e**2) / (1 + e * np.cos(theta2))
    x_newton = r_newton * np.cos(theta2)
    y_newton = r_newton * np.sin(theta2)

    # GR orbit with precession
    precession_rate = 0.15  # radians per radian of orbital motion
    omega_gr = precession_rate * theta2
    r_gr = a * (1 - e**2) / (1 + e * np.cos(theta2))
    x_gr = r_gr * np.cos(theta2 + omega_gr)
    y_gr = r_gr * np.sin(theta2 + omega_gr)

    ax2.plot(x_newton, y_newton, 'b-', linewidth=2, alpha=0.7,
             label='Newtonian (closed)' if language == 'en' else 'Newtonsch (geschlossen)')
    ax2.plot(x_gr, y_gr, 'r-', linewidth=2, alpha=0.7,
             label='General Relativity (precessing)' if language == 'en' else 'Allg. Relativitaet (praezedierend)')

    # Draw central mass
    sun2 = plt.Circle((0, 0), 0.08, color=COLORS['primary_amber'], zorder=10)
    ax2.add_patch(sun2)

    ax2.set_xlim(-2.5, 2.5)
    ax2.set_ylim(-2.5, 2.5)
    ax2.set_aspect('equal')
    ax2.set_xlabel('x (AU)' if language == 'en' else 'x (AE)', fontsize=12)
    ax2.set_ylabel('y (AU)' if language == 'en' else 'y (AE)', fontsize=12)

    if language == 'de':
        ax2.set_title('Newton vs. Allgemeine Relativitaet\n(2 Umlaeufe)',
                     fontsize=14, fontweight='bold', pad=15)
    else:
        ax2.set_title('Newton vs. General Relativity\n(2 orbits)',
                     fontsize=14, fontweight='bold', pad=15)

    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Add physics note
    note_text = ("Mercury's perihelion precesses by 43\"/century\n"
                "due to spacetime curvature near the Sun" if language == 'en' else
                "Merkurs Perihel praezediert um 43\"/Jahrhundert\n"
                "durch Raumzeitkruemmung nahe der Sonne")
    ax2.text(0.02, 0.02, note_text, transform=ax2.transAxes, fontsize=9,
            verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    if save:
        os.makedirs(VIS_DIR, exist_ok=True)
        filepath = os.path.join(VIS_DIR, 'orbital_precession.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")

        # Create animated GIF if requested
        if animate:
            create_precession_animation(constants, language, VIS_DIR)

    if show:
        plt.show()

    return fig


def create_precession_animation(
    constants: PhysicalConstants,
    language: str = 'en',
    output_dir: str = None
):
    """
    Create an animated GIF of orbital precession.
    Erstellt ein animiertes GIF der Bahnpraezession.
    """
    from matplotlib.animation import FuncAnimation, PillowWriter

    if output_dir is None:
        output_dir = VIS_DIR

    fig, ax = plt.subplots(figsize=(8, 8))

    # Orbital parameters
    a = 1.0
    e = 0.5
    precession_per_orbit = np.pi / 8  # Exaggerated

    # Initialize
    theta_full = np.linspace(0, 2*np.pi, 200)

    # Static elements
    sun = plt.Circle((0, 0), 0.06, color=COLORS['primary_amber'], zorder=10)
    ax.add_patch(sun)

    # Lines for orbit trail and current position
    orbit_line, = ax.plot([], [], 'b-', linewidth=1, alpha=0.5)
    current_orbit, = ax.plot([], [], 'r-', linewidth=2)
    planet, = ax.plot([], [], 'o', color=COLORS['primary_blue'], markersize=10, zorder=5)

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    if language == 'de':
        ax.set_title('Bahnpraezession Animation', fontsize=14, fontweight='bold')
    else:
        ax.set_title('Orbital Precession Animation', fontsize=14, fontweight='bold')

    # Store all previous orbits
    all_x = []
    all_y = []

    n_frames = 120  # Total frames
    n_orbits = 4    # Number of complete orbits

    def init():
        orbit_line.set_data([], [])
        current_orbit.set_data([], [])
        planet.set_data([], [])
        return orbit_line, current_orbit, planet

    def animate(frame):
        # Current orbital phase
        progress = frame / n_frames
        current_theta = progress * n_orbits * 2 * np.pi

        # Current precession angle
        omega = progress * n_orbits * precession_per_orbit

        # Current position
        r = a * (1 - e**2) / (1 + e * np.cos(current_theta))
        x = r * np.cos(current_theta + omega)
        y = r * np.sin(current_theta + omega)

        # Current orbit (full ellipse at current precession)
        r_orbit = a * (1 - e**2) / (1 + e * np.cos(theta_full))
        x_orbit = r_orbit * np.cos(theta_full + omega)
        y_orbit = r_orbit * np.sin(theta_full + omega)

        # Trail (previous positions)
        all_x.append(x)
        all_y.append(y)

        orbit_line.set_data(all_x, all_y)
        current_orbit.set_data(x_orbit, y_orbit)
        planet.set_data([x], [y])

        return orbit_line, current_orbit, planet

    anim = FuncAnimation(fig, animate, init_func=init, frames=n_frames,
                        interval=50, blit=True)

    # Save as GIF
    filepath = os.path.join(output_dir, 'orbital_precession_animation.gif')
    writer = PillowWriter(fps=20)
    anim.save(filepath, writer=writer)
    print(f"Saved animation: {filepath}")

    plt.close(fig)


# =============================================================================
# TIME DILATION VISUALIZATIONS
# =============================================================================

def gravitational_time_dilation_factor(r: float, M: float, constants: PhysicalConstants) -> float:
    """
    Calculate gravitational time dilation factor at distance r from mass M.
    Berechnet den gravitativen Zeitdilatationsfaktor im Abstand r von Masse M.

    Formula: τ/t = √(1 - R_s/r) = √(1 - 2GM/(rc²))

    This is the ratio of proper time (τ, experienced by observer at r)
    to coordinate time (t, experienced by distant observer).

    Args:
        r: Distance from center of mass [m]
        M: Mass of the object [kg]
        constants: Physical constants

    Returns:
        Time dilation factor (0 to 1, where 1 = no dilation)
    """
    R_s = constants.schwarzschild_radius(M)
    if r <= R_s:
        return 0.0  # At or inside event horizon
    return np.sqrt(1 - R_s / r)


def plot_time_dilation_comparison(
    constants: Optional[PhysicalConstants] = None,
    language: str = 'en',
    save: bool = True,
    show: bool = True
) -> plt.Figure:
    """
    Compare gravitational time dilation on different stellar object surfaces.
    Vergleicht gravitationelle Zeitdilatation auf verschiedenen Sternoberflaechen.

    Essay reference: "30% slower" on neutron star surface.

    Args:
        constants: Physical constants
        language: 'en' for English, 'de' for German
        save: Whether to save the figure
        show: Whether to display the figure

    Returns:
        matplotlib Figure object
    """
    if constants is None:
        constants = get_constants()

    # Create figure with 4x1 vertical layout for better readability
    fig, axes = plt.subplots(4, 1, figsize=(12, 24))
    fig.subplots_adjust(hspace=0.45)  # Add more space between plots

    # Get stellar objects
    objects = get_stellar_objects(constants)

    # Calculate time dilation for each object at surface
    dilation_factors = []
    time_slowdown_percent = []
    for obj in objects:
        if obj.radius > obj.schwarzschild_radius:
            factor = gravitational_time_dilation_factor(obj.radius, obj.mass, constants)
            dilation_factors.append(factor)
            time_slowdown_percent.append((1 - factor) * 100)
        else:
            dilation_factors.append(0.0)
            time_slowdown_percent.append(100.0)

    # Plot 1: Time dilation factor (bar chart)
    ax1 = axes[0]
    names = [obj.name if language == 'en' else obj.name_de for obj in objects]
    colors = get_stellar_colors()

    bars = ax1.bar(names, dilation_factors, color=colors, edgecolor='black', linewidth=1.5)

    # Add value labels
    for bar, factor in zip(bars, dilation_factors):
        if factor > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, factor + 0.02,
                    f'{factor:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax1.axhline(y=1.0, color=COLORS['standard'], linestyle='--', linewidth=1.5, alpha=0.7,
               label='No dilation' if language == 'en' else 'Keine Dilatation')
    ax1.axhline(y=0.7, color=COLORS['quantum'], linestyle=':', linewidth=1.5, alpha=0.7,
               label='30% slower' if language == 'en' else '30% langsamer')

    ax1.set_ylim(0, 1.1)
    if language == 'de':
        ax1.set_ylabel('Zeitdilatationsfaktor τ/t', fontsize=11)
        ax1.set_title('1. Zeitdilatation auf Oberfläche\n(τ/t = √(1 - R_s/R))', fontsize=12, fontweight='bold')
    else:
        ax1.set_ylabel('Time dilation factor τ/t', fontsize=11)
        ax1.set_title('1. Time Dilation at Surface\n(τ/t = √(1 - R_s/R))', fontsize=12, fontweight='bold')

    ax1.legend(fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2)
    ax1.grid(True, alpha=0.3, axis='y')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Plot 2: Time slowdown percentage (bar chart)
    ax2 = axes[1]
    bars2 = ax2.bar(names, time_slowdown_percent, color=colors, edgecolor='black', linewidth=1.5)

    # Add value labels
    for bar, pct in zip(bars2, time_slowdown_percent):
        if pct < 100:
            ax2.text(bar.get_x() + bar.get_width()/2, pct + 1,
                    f'{pct:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        else:
            ax2.text(bar.get_x() + bar.get_width()/2, pct/2,
                    '∞', ha='center', va='center', fontsize=14, fontweight='bold', color='white')

    # Mark the ~30% mentioned in essay
    ax2.axhline(y=30, color=COLORS['quantum'], linestyle='--', linewidth=2, alpha=0.8,
               label='~30% (essay reference)' if language == 'en' else '~30% (Essay-Referenz)')

    ax2.set_ylim(0, 110)
    if language == 'de':
        ax2.set_ylabel('Zeitverlangsamung (%)', fontsize=11)
        ax2.set_title('2. Prozentuale Zeitverlangsamung\n(relativ zu fernem Beobachter)', fontsize=12, fontweight='bold')
    else:
        ax2.set_ylabel('Time slowdown (%)', fontsize=11)
        ax2.set_title('2. Percentage Time Slowdown\n(relative to distant observer)', fontsize=12, fontweight='bold')

    ax2.legend(fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=2)
    ax2.grid(True, alpha=0.3, axis='y')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Plot 3: Time dilation vs distance from neutron star
    ax3 = axes[2]

    # Neutron star parameters
    M_ns = 1.4 * constants.M_sun
    R_ns = 10e3  # 10 km
    R_s_ns = constants.schwarzschild_radius(M_ns)

    # Distance range (from surface to 10 R_ns)
    r_range = np.linspace(R_ns, 10 * R_ns, 100)
    dilation_vs_r = np.array([gravitational_time_dilation_factor(r, M_ns, constants) for r in r_range])

    ax3.plot(r_range / 1e3, dilation_vs_r, '-', color=COLORS['quantum'], linewidth=2.5,
             label='τ/t = √(1 - R_s/r)')

    # Mark surface
    surface_dilation = gravitational_time_dilation_factor(R_ns, M_ns, constants)
    ax3.plot(R_ns / 1e3, surface_dilation, 'o', color=COLORS['scaled'], markersize=12,
             label=f'Surface: {(1-surface_dilation)*100:.1f}% slower' if language == 'en'
                   else f'Oberfläche: {(1-surface_dilation)*100:.1f}% langsamer')

    # Mark Schwarzschild radius
    ax3.axvline(x=R_s_ns / 1e3, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
               label=f'R_s = {R_s_ns/1e3:.1f} km')

    ax3.axhline(y=1.0, color=COLORS['standard'], linestyle=':', linewidth=1, alpha=0.5)
    ax3.axhline(y=0.7, color=COLORS['primary_amber'], linestyle=':', linewidth=1.5, alpha=0.7,
               label='30% slower line')

    ax3.set_xlim(0, 10 * R_ns / 1e3)
    ax3.set_ylim(0, 1.1)
    if language == 'de':
        ax3.set_xlabel('Abstand vom Zentrum (km)', fontsize=11)
        ax3.set_ylabel('Zeitdilatationsfaktor τ/t', fontsize=11)
        ax3.set_title('3. Zeitdilatation vs. Abstand\n(Neutronenstern: 1.4 M☉, R = 10 km)', fontsize=12, fontweight='bold')
    else:
        ax3.set_xlabel('Distance from center (km)', fontsize=11)
        ax3.set_ylabel('Time dilation factor τ/t', fontsize=11)
        ax3.set_title('3. Time Dilation vs. Distance\n(Neutron star: 1.4 M☉, R = 10 km)', fontsize=12, fontweight='bold')

    ax3.legend(fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Accumulated time difference example
    ax4 = axes[3]

    # Calculate how much time differs after 1 year on different objects
    years = np.linspace(0, 10, 100)  # 10 years coordinate time
    seconds_per_year = 365.25 * 24 * 3600

    # For each object, calculate proper time elapsed
    for i, obj in enumerate(objects):
        if obj.radius > obj.schwarzschild_radius:
            factor = gravitational_time_dilation_factor(obj.radius, obj.mass, constants)
            proper_years = years * factor
            time_lost_hours = (years - proper_years) * seconds_per_year / 3600
            label = obj.name if language == 'en' else obj.name_de
            ax4.plot(years, time_lost_hours, '-', color=colors[i], linewidth=2, label=label)

    if language == 'de':
        ax4.set_xlabel('Koordinatenzeit (Jahre)', fontsize=11)
        ax4.set_ylabel('Zeitverlust (Stunden)', fontsize=11)
        ax4.set_title('4. Akkumulierter Zeitunterschied\n(Stunden verloren auf Objektoberfläche)', fontsize=12, fontweight='bold')
    else:
        ax4.set_xlabel('Coordinate time (years)', fontsize=11)
        ax4.set_ylabel('Time lost (hours)', fontsize=11)
        ax4.set_title('4. Accumulated Time Difference\n(hours lost on object surface)', fontsize=12, fontweight='bold')

    ax4.legend(fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3)
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')

    plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()

    if save:
        os.makedirs(VIS_DIR, exist_ok=True)
        suffix = '_de' if language == 'de' else ''
        filepath = os.path.join(VIS_DIR, f'time_dilation_comparison{suffix}.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")

    if show:
        plt.show()

    return fig


def plot_time_dilation_scaling(
    constants: Optional[PhysicalConstants] = None,
    language: str = 'en',
    save: bool = True,
    show: bool = True
) -> plt.Figure:
    """
    Show how time dilation changes with G and ℏ scaling.
    Zeigt wie sich Zeitdilatation mit G- und ℏ-Skalierung aendert.

    Key insight: R_s ∝ G, so stronger gravity → more time dilation.
    When G increases by 10^36, Schwarzschild radii grow enormously.

    Args:
        constants: Physical constants
        language: 'en' for English, 'de' for German
        save: Whether to save the figure
        show: Whether to display the figure

    Returns:
        matplotlib Figure object
    """
    if constants is None:
        constants = get_constants()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    fig.subplots_adjust(hspace=0.4)  # Add more space between plots

    # Plot 1: Time dilation vs G scaling for Earth
    G_scales = np.logspace(0, 10, 50)  # G scaling from 1 to 10^10

    M_earth = constants.M_earth
    R_earth = constants.R_earth

    dilation_earth = []
    for G_scale in G_scales:
        R_s_scaled = 2 * constants.G * G_scale * M_earth / constants.c**2
        if R_earth > R_s_scaled:
            factor = np.sqrt(1 - R_s_scaled / R_earth)
        else:
            factor = 0.0
        dilation_earth.append(factor)

    ax1.semilogx(G_scales, dilation_earth, '-', color=COLORS['primary_blue'], linewidth=2.5,
                 label='Earth surface' if language == 'en' else 'Erdoberfläche')

    # Mark where Earth becomes a black hole (R_s = R_earth)
    G_bh = R_earth * constants.c**2 / (2 * constants.G * M_earth)
    ax1.axvline(x=G_bh, color='red', linestyle='--', linewidth=2, alpha=0.7,
               label=f'Earth → BH (G × {G_bh:.1e})' if language == 'en'
                     else f'Erde → SL (G × {G_bh:.1e})')

    # Mark standard G
    ax1.axvline(x=1, color=COLORS['standard'], linestyle=':', linewidth=1.5, alpha=0.7,
               label='Standard G')

    # Mark 30% slowdown threshold
    ax1.axhline(y=0.7, color=COLORS['quantum'], linestyle=':', linewidth=1.5, alpha=0.7,
               label='30% slower')

    ax1.set_ylim(0, 1.1)
    if language == 'de':
        ax1.set_xlabel('G-Skalierungsfaktor', fontsize=11)
        ax1.set_ylabel('Zeitdilatationsfaktor τ/t', fontsize=11)
        ax1.set_title('Zeitdilatation auf Erde vs. G-Skalierung', fontsize=12, fontweight='bold')
    else:
        ax1.set_xlabel('G scaling factor', fontsize=11)
        ax1.set_ylabel('Time dilation factor τ/t', fontsize=11)
        ax1.set_title('Time Dilation on Earth vs. G Scaling', fontsize=12, fontweight='bold')

    ax1.legend(fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Compare objects at different G scales
    ax2_colors = [COLORS['primary_blue'], COLORS['scaled'], COLORS['quantum'], COLORS['primary_amber']]
    objects_subset = [
        ('Earth', 'Erde', constants.M_earth, constants.R_earth),
        ('Sun', 'Sonne', constants.M_sun, 6.96e8),
        ('White Dwarf', 'Weißer Zwerg', 0.6 * constants.M_sun, 8e6),
        ('Neutron Star', 'Neutronenstern', 1.4 * constants.M_sun, 10e3),
    ]

    G_scales_2 = np.logspace(0, 6, 50)

    for (name_en, name_de, M, R), color in zip(objects_subset, ax2_colors):
        dilation = []
        for G_scale in G_scales_2:
            R_s_scaled = 2 * constants.G * G_scale * M / constants.c**2
            if R > R_s_scaled:
                factor = np.sqrt(1 - R_s_scaled / R)
            else:
                factor = 0.0
            dilation.append(factor)

        label = name_en if language == 'en' else name_de
        ax2.semilogx(G_scales_2, dilation, '-', color=color, linewidth=2, label=label)

    ax2.axhline(y=0.7, color=COLORS['text_dark'], linestyle=':', linewidth=1.5, alpha=0.7,
               label='30% slower')

    ax2.set_ylim(0, 1.1)
    if language == 'de':
        ax2.set_xlabel('G-Skalierungsfaktor', fontsize=11)
        ax2.set_ylabel('Zeitdilatationsfaktor τ/t', fontsize=11)
        ax2.set_title('Zeitdilatation verschiedener Objekte vs. G', fontsize=12, fontweight='bold')
    else:
        ax2.set_xlabel('G scaling factor', fontsize=11)
        ax2.set_ylabel('Time dilation factor τ/t', fontsize=11)
        ax2.set_title('Time Dilation of Objects vs. G', fontsize=12, fontweight='bold')

    ax2.legend(fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3)
    ax2.grid(True, alpha=0.3)

    plt.subplots_adjust(bottom=0.15)
    plt.tight_layout()

    if save:
        os.makedirs(VIS_DIR, exist_ok=True)
        suffix = '_de' if language == 'de' else ''
        filepath = os.path.join(VIS_DIR, f'time_dilation_scaling{suffix}.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")

    if show:
        plt.show()

    return fig


def plot_time_dilation_summary(
    constants: Optional[PhysicalConstants] = None,
    language: str = 'en',
    save: bool = True,
    show: bool = True
) -> plt.Figure:
    """
    Create a comprehensive summary of gravitational time dilation.
    Erstellt eine umfassende Zusammenfassung der gravitativen Zeitdilatation.

    Args:
        constants: Physical constants
        language: 'en' for English, 'de' for German
        save: Whether to save the figure
        show: Whether to display the figure

    Returns:
        matplotlib Figure object
    """
    if constants is None:
        constants = get_constants()

    fig = plt.figure(figsize=(12, 28))
    gs = fig.add_gridspec(5, 1, height_ratios=[1, 1, 1, 1, 0.8], hspace=0.45)

    # Plot 1: Formula visualization
    ax1 = fig.add_subplot(gs[0])

    r_over_Rs = np.linspace(1.01, 10, 100)  # r/R_s from just above 1 to 10
    dilation = np.sqrt(1 - 1/r_over_Rs)

    ax1.plot(r_over_Rs, dilation, '-', color=COLORS['primary_blue'], linewidth=2.5,
             label='τ/t = √(1 - R_s/r)')
    ax1.fill_between(r_over_Rs, 0, dilation, alpha=0.2, color=COLORS['primary_blue'])

    # Mark key points
    ax1.axvline(x=1, color='red', linestyle='--', linewidth=2, label='Event horizon (r = R_s)')
    ax1.axhline(y=0.7, color=COLORS['quantum'], linestyle=':', linewidth=1.5, alpha=0.7,
               label='30% slower')

    # Mark neutron star typical (r/R_s ≈ 2.4 for 10km, 1.4 M_sun)
    ns_r_over_Rs = 10e3 / (2 * constants.G * 1.4 * constants.M_sun / constants.c**2)
    ns_dilation = np.sqrt(1 - 1/ns_r_over_Rs)
    ax1.plot(ns_r_over_Rs, ns_dilation, 'o', color=COLORS['quantum'], markersize=12,
             label=f'NS surface: {(1-ns_dilation)*100:.0f}% slower')

    ax1.set_xlim(1, 10)
    ax1.set_ylim(0, 1.1)
    if language == 'de':
        ax1.set_xlabel('r / R_s (Abstand / Schwarzschild-Radius)', fontsize=11)
        ax1.set_ylabel('Zeitdilatation τ/t', fontsize=11)
        ax1.set_title('1. Zeitdilatationsformel', fontsize=12, fontweight='bold')
    else:
        ax1.set_xlabel('r / R_s (distance / Schwarzschild radius)', fontsize=11)
        ax1.set_ylabel('Time dilation τ/t', fontsize=11)
        ax1.set_title('1. Time Dilation Formula', fontsize=12, fontweight='bold')

    ax1.legend(fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Object comparison
    ax2 = fig.add_subplot(gs[1])

    objects = get_stellar_objects(constants)
    names = [obj.name if language == 'en' else obj.name_de for obj in objects]
    colors = get_stellar_colors()

    slowdown_pct = []
    for obj in objects:
        if obj.radius > obj.schwarzschild_radius:
            factor = gravitational_time_dilation_factor(obj.radius, obj.mass, constants)
            slowdown_pct.append((1 - factor) * 100)
        else:
            slowdown_pct.append(100.0)

    bars = ax2.barh(names, slowdown_pct, color=colors, edgecolor='black', linewidth=1.5)

    # Add value labels
    for bar, pct in zip(bars, slowdown_pct):
        if pct < 100:
            ax2.text(pct + 1, bar.get_y() + bar.get_height()/2,
                    f'{pct:.2f}%', va='center', fontsize=9, fontweight='bold')
        else:
            ax2.text(50, bar.get_y() + bar.get_height()/2,
                    '∞', va='center', ha='center', fontsize=14, fontweight='bold', color='white')

    ax2.axvline(x=30, color=COLORS['quantum'], linestyle='--', linewidth=2, alpha=0.8,
               label='~30% (essay)' if language == 'en' else '~30% (Essay)')

    if language == 'de':
        ax2.set_xlabel('Zeitverlangsamung (%)', fontsize=11)
        ax2.set_title('2. Zeitverlangsamung auf Objektoberflächen', fontsize=12, fontweight='bold')
    else:
        ax2.set_xlabel('Time slowdown (%)', fontsize=11)
        ax2.set_title('2. Time Slowdown on Object Surfaces', fontsize=12, fontweight='bold')

    ax2.legend(fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=1)
    ax2.grid(True, alpha=0.3, axis='x')

    # Plot 3: Time dilation vs G scaling
    ax3 = fig.add_subplot(gs[2])

    G_scales = np.logspace(0, 8, 50)
    M_earth = constants.M_earth
    R_earth = constants.R_earth

    dilation_values = []
    for G_scale in G_scales:
        R_s_scaled = 2 * constants.G * G_scale * M_earth / constants.c**2
        if R_earth > R_s_scaled:
            factor = np.sqrt(1 - R_s_scaled / R_earth)
        else:
            factor = 0.0
        dilation_values.append(factor)

    ax3.semilogx(G_scales, dilation_values, '-', color=COLORS['primary_blue'], linewidth=2.5,
                 label='Earth' if language == 'en' else 'Erde')

    # Mark where significant dilation occurs
    G_bh = R_earth * constants.c**2 / (2 * constants.G * M_earth)
    ax3.axvline(x=G_bh, color='red', linestyle='--', linewidth=2, alpha=0.7,
               label=f'BH threshold' if language == 'en' else 'SL Schwelle')
    ax3.axhline(y=0.7, color=COLORS['quantum'], linestyle=':', linewidth=1.5, alpha=0.7,
               label='30% slower')

    ax3.set_ylim(0, 1.1)
    if language == 'de':
        ax3.set_xlabel('G-Skalierungsfaktor', fontsize=11)
        ax3.set_ylabel('Zeitdilatation τ/t', fontsize=11)
        ax3.set_title('3. Zeitdilatation vs. Gravitationsstärke', fontsize=12, fontweight='bold')
    else:
        ax3.set_xlabel('G scaling factor', fontsize=11)
        ax3.set_ylabel('Time dilation τ/t', fontsize=11)
        ax3.set_title('3. Time Dilation vs. Gravity Strength', fontsize=12, fontweight='bold')

    ax3.legend(fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Accumulated time difference over years
    ax4 = fig.add_subplot(gs[3])

    years = np.linspace(0, 10, 100)
    seconds_per_year = 365.25 * 24 * 3600

    # Calculate time lost for different objects
    objects_data = [
        ('Earth', 'Erde', constants.R_earth, constants.M_earth, COLORS['primary_blue']),
        ('White Dwarf', 'Weißer Zwerg', 8e6, 0.6 * constants.M_sun, COLORS['scaled']),
        ('Neutron Star', 'Neutronenstern', 10e3, 1.4 * constants.M_sun, COLORS['quantum']),
    ]

    for name_en, name_de, R, M, color in objects_data:
        factor = gravitational_time_dilation_factor(R, M, constants)
        proper_years = years * factor
        time_lost_hours = (years - proper_years) * seconds_per_year / 3600
        label = name_en if language == 'en' else name_de
        ax4.plot(years, time_lost_hours, '-', color=color, linewidth=2, label=label)

    if language == 'de':
        ax4.set_xlabel('Koordinatenzeit (Jahre)', fontsize=11)
        ax4.set_ylabel('Akkumulierter Zeitunterschied (Stunden)', fontsize=11)
        ax4.set_title('4. Akkumulierter Zeitverlust', fontsize=12, fontweight='bold')
    else:
        ax4.set_xlabel('Coordinate time (years)', fontsize=11)
        ax4.set_ylabel('Accumulated time difference (hours)', fontsize=11)
        ax4.set_title('4. Accumulated Time Lost', fontsize=12, fontweight='bold')

    ax4.legend(fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3)
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')

    # Plot 5: Summary text panel
    ax5 = fig.add_subplot(gs[4])
    ax5.axis('off')

    # Calculate specific values for the summary
    earth_factor = gravitational_time_dilation_factor(constants.R_earth, constants.M_earth, constants)
    wd_factor = gravitational_time_dilation_factor(8e6, 0.6 * constants.M_sun, constants)
    ns_factor = gravitational_time_dilation_factor(10e3, 1.4 * constants.M_sun, constants)

    if language == 'de':
        summary_text = f"""
                    GRAVITATIONELLE ZEITDILATATION - Zusammenfassung
        ─────────────────────────────────────────────────────────────────

        FORMEL:     τ/t = √(1 - Rₛ/r) = √(1 - 2GM/(rc²))

                    τ = Eigenzeit (auf Objektoberfläche)
                    t = Koordinatenzeit (für fernen Beobachter)

                                    BEISPIELE:
            • Erde:                 {(1-earth_factor)*100:.6f}% langsamer   (kaum messbar)
            • Weißer Zwerg:         {(1-wd_factor)*100:.4f}% langsamer
            • Neutronenstern:       {(1-ns_factor)*100:.1f}% langsamer   (~30% wie im Essay)
            • Schwarzes Loch:       Zeit steht still am Ereignishorizont

                        SCHLÜSSELAUSSAGE AUS DEM ESSAY:
          "Auf der Oberfläche eines Neutronensterns vergeht die Zeit
           etwa 30% langsamer als für einen entfernten Beobachter."

                            PHYSIKALISCHE BEDEUTUNG:
        Je stärker die Gravitation (größeres G oder kompakteres Objekt),
             desto langsamer vergeht die Zeit → Extremfall: Schwarzes Loch
        """
    else:
        summary_text = f"""
                    GRAVITATIONAL TIME DILATION - Summary
        ─────────────────────────────────────────────────────────────────

        FORMULA:    τ/t = √(1 - Rₛ/r) = √(1 - 2GM/(rc²))

                    τ = proper time (on object surface)
                    t = coordinate time (for distant observer)

                                    EXAMPLES:
            • Earth:                {(1-earth_factor)*100:.6f}% slower   (barely measurable)
            • White Dwarf:          {(1-wd_factor)*100:.4f}% slower
            • Neutron Star:         {(1-ns_factor)*100:.1f}% slower   (~30% as in essay)
            • Black Hole:           Time stops at event horizon

                            KEY STATEMENT FROM ESSAY:
           "On the surface of a neutron star, time passes
            about 30% slower than for a distant observer."

                            PHYSICAL MEANING:
         The stronger gravity (larger G or more compact object),
              the slower time passes → Extreme case: Black Hole
        """

    ax5.text(0.5, 0.5, summary_text, transform=ax5.transAxes, fontsize=11,
             verticalalignment='center', horizontalalignment='center',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.95,
                      edgecolor=COLORS['primary_blue'], linewidth=2))

    # Overall title
    if language == 'de':
        fig.suptitle('Gravitationelle Zeitdilatation: Allgemeine Relativitätstheorie',
                    fontsize=16, fontweight='bold', y=0.98)
    else:
        fig.suptitle('Gravitational Time Dilation: General Relativity',
                    fontsize=16, fontweight='bold', y=0.98)

    plt.subplots_adjust(bottom=0.05)
    plt.tight_layout()

    if save:
        os.makedirs(VIS_DIR, exist_ok=True)
        suffix = '_de' if language == 'de' else ''
        filepath = os.path.join(VIS_DIR, f'time_dilation_summary{suffix}.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")

    if show:
        plt.show()

    return fig


def generate_all_spacetime_plots(language: str = 'en', show: bool = False) -> List[plt.Figure]:
    """
    Generate all spacetime curvature visualizations.
    Erzeugt alle Raumzeitkruemmungs-Visualisierungen.

    Args:
        language: 'en' for English, 'de' for German
        show: Whether to display plots

    Returns:
        List of matplotlib Figure objects
    """
    figures = []

    print("Generating spacetime curvature visualizations...")
    print("=" * 50)

    # 1. 2D Potential well
    print("1. 2D Potential well comparison...")
    figures.append(plot_potential_well_2d(language=language, show=show))

    # 2. 3D Potential well
    print("2. 3D Spacetime curvature visualization...")
    figures.append(plot_potential_well_3d(mass_solar=1.0, language=language, show=show))

    # 3. Compactness comparison
    print("3. Compactness comparison...")
    figures.append(plot_compactness_comparison(language=language, show=show))

    # 4. Escape velocity
    print("4. Escape velocity comparison...")
    figures.append(plot_escape_velocity(language=language, show=show))

    # 5. Summary
    print("5. Comprehensive summary...")
    figures.append(plot_spacetime_summary(language=language, show=show))

    # 6. Gravitational vector field
    print("6. Gravitational vector field...")
    figures.append(plot_gravitational_vector_field(language=language, show=show))

    # 7. Orbital precession
    print("7. Orbital precession (with animation)...")
    figures.append(plot_orbital_precession(language=language, show=show, animate=True))

    # 8. Light bending (gravitational lensing)
    print("8. Light bending / gravitational lensing (with animation)...")
    figures.append(plot_light_bending(language=language, show=show, animate=True))

    # 9. Penrose-Carter diagram (Schwarzschild)
    print("9. Penrose-Carter diagram (Schwarzschild)...")
    figures.append(plot_penrose_carter_diagram(language=language, show=show, diagram_type='schwarzschild'))

    # 10. Penrose diagram comparison
    print("10. Penrose diagram comparison...")
    figures.append(plot_penrose_comparison(language=language, show=show))

    # 11. Time dilation comparison
    print("11. Time dilation comparison...")
    figures.append(plot_time_dilation_comparison(language=language, show=show))

    # 12. Time dilation scaling
    print("12. Time dilation vs G scaling...")
    figures.append(plot_time_dilation_scaling(language=language, show=show))

    # 13. Time dilation summary
    print("13. Time dilation summary...")
    figures.append(plot_time_dilation_summary(language=language, show=show))

    print("=" * 50)
    print(f"Generated {len(figures)} visualizations in {VIS_DIR}")

    return figures


# =============================================================================
# LIGHT BENDING (GRAVITATIONAL LENSING)
# =============================================================================

def light_deflection_angle(b: float, M: float, constants: PhysicalConstants) -> float:
    """
    Calculate the deflection angle of light passing near a massive object.
    Berechnet den Ablenkungswinkel von Licht nahe eines massiven Objekts.

    Formula: alpha = 4GM / (c^2 * b)

    Args:
        b: Impact parameter (closest approach distance) [m]
        M: Mass of deflecting object [kg]
        constants: Physical constants

    Returns:
        Deflection angle [radians]
    """
    # Avoid division by zero
    if b <= 0:
        return float('inf')

    alpha = 4 * constants.G * M / (constants.c**2 * b)
    return alpha


def einstein_radius(M: float, D_ls: float, D_l: float, D_s: float,
                   constants: PhysicalConstants) -> float:
    """
    Calculate the Einstein radius for gravitational lensing.
    Berechnet den Einstein-Radius fuer Gravitationslinseneffekt.

    Formula: theta_E = sqrt(4GM/c^2 * D_ls / (D_l * D_s))

    Args:
        M: Mass of lens [kg]
        D_ls: Distance from lens to source [m]
        D_l: Distance from observer to lens [m]
        D_s: Distance from observer to source [m]
        constants: Physical constants

    Returns:
        Einstein radius [radians]
    """
    if D_l <= 0 or D_s <= 0:
        return 0.0

    R_s = constants.schwarzschild_radius(M)
    theta_E = np.sqrt(R_s * D_ls / (D_l * D_s))
    return theta_E


def calculate_light_path(x_start: float, y_start: float, direction: float,
                        M: float, constants: PhysicalConstants,
                        n_steps: int = 500, dt: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the path of a light ray bending around a massive object.
    Berechnet den Pfad eines Lichtstrahls, der um ein massives Objekt gebogen wird.

    Uses simple numerical integration of the geodesic equation in weak field limit.

    Args:
        x_start: Starting x position [Schwarzschild radii]
        y_start: Starting y position [Schwarzschild radii]
        direction: Initial direction angle [radians]
        M: Mass of central object [kg]
        constants: Physical constants
        n_steps: Number of integration steps
        dt: Time step

    Returns:
        Tuple of (x_array, y_array) positions
    """
    R_s = constants.schwarzschild_radius(M)

    # Initialize arrays
    x = np.zeros(n_steps)
    y = np.zeros(n_steps)

    # Initial position and velocity
    x[0] = x_start
    y[0] = y_start
    vx = np.cos(direction)
    vy = np.sin(direction)

    for i in range(1, n_steps):
        # Current position in Schwarzschild radii
        r = np.sqrt(x[i-1]**2 + y[i-1]**2)

        # Avoid singularity and stop if inside event horizon
        if r < 1.5:  # Stop at 1.5 R_s (photon sphere)
            x[i:] = x[i-1]
            y[i:] = y[i-1]
            break

        # Gravitational acceleration (Newtonian approximation for visualization)
        # a = -GM/r^2 in radial direction
        # In units of R_s, this simplifies
        a_mag = 0.5 / (r**2)  # Simplified for visualization

        # Radial unit vector
        rx = x[i-1] / r
        ry = y[i-1] / r

        # Acceleration components (toward center)
        ax = -a_mag * rx
        ay = -a_mag * ry

        # Update velocity
        vx += ax * dt
        vy += ay * dt

        # Normalize velocity (light always travels at c)
        v_mag = np.sqrt(vx**2 + vy**2)
        vx /= v_mag
        vy /= v_mag

        # Update position
        x[i] = x[i-1] + vx * dt
        y[i] = y[i-1] + vy * dt

        # Stop if ray escapes
        if abs(x[i]) > 30 or abs(y[i]) > 30:
            x[i+1:] = x[i]
            y[i+1:] = y[i]
            break

    return x, y


def plot_light_bending(
    constants: Optional[PhysicalConstants] = None,
    language: str = 'en',
    save: bool = True,
    show: bool = False,
    animate: bool = True
) -> plt.Figure:
    """
    Plot gravitational lensing / light bending visualization.
    Zeigt Gravitationslinsen- / Lichtbeugungs-Visualisierung.

    Shows:
    - Left: Multiple light rays bending around a central mass
    - Right: Einstein ring formation

    Args:
        constants: Physical constants
        language: 'en' for English, 'de' for German
        save: Whether to save the figure
        show: Whether to display the figure
        animate: Whether to create animation

    Returns:
        matplotlib Figure object
    """
    if constants is None:
        constants = get_constants()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # ===== LEFT PLOT: Light rays bending around mass =====

    # Central mass (black hole / star)
    central_mass = plt.Circle((0, 0), 1.0, color='black', zorder=10)
    ax1.add_patch(central_mass)

    # Photon sphere at 1.5 R_s
    photon_sphere = plt.Circle((0, 0), 1.5, facecolor='none',
                                edgecolor=COLORS['highlight'], linestyle='--',
                                linewidth=1.5, zorder=5)
    ax1.add_patch(photon_sphere)

    # Event horizon label
    ax1.text(0, 0, 'M', ha='center', va='center', fontsize=14,
             fontweight='bold', color='white', zorder=11)

    # Draw multiple light rays with different impact parameters
    impact_params = [2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]
    colors_rays = plt.cm.Blues(np.linspace(0.4, 0.9, len(impact_params)))

    M = 10 * constants.M_sun  # 10 solar mass black hole

    for b, color in zip(impact_params, colors_rays):
        # Ray coming from left, passing above center
        x_path, y_path = calculate_light_path(-25, b, 0, M, constants, n_steps=800, dt=0.08)

        # Only plot non-zero portion
        mask = (x_path != 0) | (y_path != 0)
        mask[0] = True
        ax1.plot(x_path[mask], y_path[mask], color=color, linewidth=1.5, alpha=0.8)

        # Mirror ray below
        x_path2, y_path2 = calculate_light_path(-25, -b, 0, M, constants, n_steps=800, dt=0.08)
        mask2 = (x_path2 != 0) | (y_path2 != 0)
        mask2[0] = True
        ax1.plot(x_path2[mask2], y_path2[mask2], color=color, linewidth=1.5, alpha=0.8)

    # Draw straight reference line (no bending)
    ax1.axhline(y=13, color='gray', linestyle=':', alpha=0.5, linewidth=1)

    ax1.set_xlim(-25, 25)
    ax1.set_ylim(-15, 15)
    ax1.set_aspect('equal')
    ax1.set_xlabel('x (Schwarzschild radii)' if language == 'en' else 'x (Schwarzschild-Radien)', fontsize=11)
    ax1.set_ylabel('y (Schwarzschild radii)' if language == 'en' else 'y (Schwarzschild-Radien)', fontsize=11)

    if language == 'de':
        ax1.set_title('Lichtablenkung durch Gravitation',
                     fontsize=14, fontweight='bold', pad=15)
    else:
        ax1.set_title('Light Bending by Gravity',
                     fontsize=14, fontweight='bold', pad=15)

    ax1.grid(True, alpha=0.3)

    # Add legend at bottom
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='black', edgecolor='black',
              label='Event Horizon (R_s)' if language == 'en' else 'Ereignishorizont (R_s)'),
        Line2D([0], [0], color=COLORS['highlight'], linestyle='--',
               label='Photon Sphere (1.5 R_s)' if language == 'en' else 'Photonensphäre (1,5 R_s)'),
        Line2D([0], [0], color=COLORS['primary_blue'], linewidth=2,
               label='Light rays' if language == 'en' else 'Lichtstrahlen'),
        Line2D([0], [0], color='gray', linestyle=':',
               label='No gravity (straight)' if language == 'en' else 'Ohne Gravitation (gerade)'),
    ]
    ax1.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, -0.08),
               fontsize=9, ncol=2, framealpha=0.7)

    # ===== RIGHT PLOT: Einstein Ring =====

    # Background source (star or galaxy)
    ax2.plot(0, 0, '*', color=COLORS['highlight'], markersize=20, zorder=15)

    # Lens (foreground mass)
    lens_circle = plt.Circle((0, 0), 0.3, color='black', zorder=10)
    ax2.add_patch(lens_circle)
    ax2.text(0, 0, 'L', ha='center', va='center', fontsize=10,
             fontweight='bold', color='white', zorder=11)

    # Einstein ring
    einstein_ring = plt.Circle((0, 0), 2.0, facecolor='none',
                               edgecolor=COLORS['primary_blue'], linewidth=3,
                               linestyle='-', zorder=5)
    ax2.add_patch(einstein_ring)

    # Add glow effect for Einstein ring
    for r in np.linspace(1.8, 2.2, 5):
        ring = plt.Circle((0, 0), r, facecolor='none',
                         edgecolor=COLORS['primary_blue'],
                         linewidth=1, alpha=0.2, zorder=4)
        ax2.add_patch(ring)

    # Arrow showing Einstein radius
    ax2.annotate('', xy=(2.0, 0), xytext=(0.5, 0),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    ax2.text(1.25, 0.3, r'$\theta_E$', fontsize=14, color='red', fontweight='bold')

    # Observer position indicator
    ax2.annotate('', xy=(0, -3.0), xytext=(0, -2.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    ax2.set_xlim(-4, 4)
    ax2.set_ylim(-4, 4)
    ax2.set_aspect('equal')
    ax2.set_xlabel('x (arcsec)' if language == 'en' else 'x (Bogensekunden)', fontsize=11)
    ax2.set_ylabel('y (arcsec)' if language == 'en' else 'y (Bogensekunden)', fontsize=11)

    if language == 'de':
        ax2.set_title('Einstein-Ring (Perfekte Ausrichtung)',
                     fontsize=14, fontweight='bold', pad=15)
    else:
        ax2.set_title('Einstein Ring (Perfect Alignment)',
                     fontsize=14, fontweight='bold', pad=15)

    ax2.grid(True, alpha=0.3)

    # Add legend at bottom
    legend_elements2 = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor=COLORS['highlight'],
               markersize=12, label='Source' if language == 'en' else 'Quelle'),
        Patch(facecolor='black', edgecolor='black',
              label='Lens (L)' if language == 'en' else 'Linse (L)'),
        Line2D([0], [0], color=COLORS['primary_blue'], linewidth=3,
               label='Einstein Ring' if language == 'en' else 'Einstein-Ring'),
        Line2D([0], [0], color='red', linewidth=2,
               label='Einstein Radius (θ_E)' if language == 'en' else 'Einstein-Radius (θ_E)'),
    ]
    ax2.legend(handles=legend_elements2, loc='upper right', bbox_to_anchor=(1.0, -0.08),
               fontsize=9, ncol=2, framealpha=0.7)

    plt.tight_layout()

    if save:
        os.makedirs(VIS_DIR, exist_ok=True)
        suffix = '_de' if language == 'de' else ''
        filepath = os.path.join(VIS_DIR, f'light_bending{suffix}.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")

        if animate:
            create_light_bending_animation(constants, language, VIS_DIR)

    if show:
        plt.show()

    return fig


def create_light_bending_animation(
    constants: PhysicalConstants,
    language: str = 'en',
    output_dir: str = None
):
    """
    Create an animated GIF of light bending around a massive object.
    Erstellt ein animiertes GIF der Lichtbeugung um ein massives Objekt.
    """
    from matplotlib.animation import FuncAnimation, PillowWriter

    if output_dir is None:
        output_dir = VIS_DIR

    fig, ax = plt.subplots(figsize=(10, 8))

    # Central mass
    central_mass = plt.Circle((0, 0), 1.0, color='black', zorder=10)
    ax.add_patch(central_mass)
    ax.text(0, 0, 'M', ha='center', va='center', fontsize=14,
            fontweight='bold', color='white', zorder=11)

    # Photon sphere
    photon_sphere = plt.Circle((0, 0), 1.5, facecolor='none',
                               edgecolor=COLORS['highlight'], linestyle='--',
                               linewidth=1.5, zorder=5)
    ax.add_patch(photon_sphere)

    ax.set_xlim(-20, 20)
    ax.set_ylim(-12, 12)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    if language == 'de':
        ax.set_title('Lichtablenkung Animation', fontsize=14, fontweight='bold')
    else:
        ax.set_title('Light Bending Animation', fontsize=14, fontweight='bold')

    # Pre-calculate all ray paths
    M = 10 * constants.M_sun
    impact_params = [2.5, 3.5, 5.0, 7.0]
    all_paths = []

    for b in impact_params:
        x_path, y_path = calculate_light_path(-20, b, 0, M, constants, n_steps=600, dt=0.07)
        all_paths.append((x_path, y_path))
        x_path2, y_path2 = calculate_light_path(-20, -b, 0, M, constants, n_steps=600, dt=0.07)
        all_paths.append((x_path2, y_path2))

    # Create line objects
    lines = []
    photons = []
    colors_rays = plt.cm.Blues(np.linspace(0.5, 0.9, len(all_paths)))

    for i, color in enumerate(colors_rays):
        line, = ax.plot([], [], color=color, linewidth=1.5, alpha=0.6)
        photon, = ax.plot([], [], 'o', color=color, markersize=6, zorder=6)
        lines.append(line)
        photons.append(photon)

    n_frames = 120

    def init():
        for line in lines:
            line.set_data([], [])
        for photon in photons:
            photon.set_data([], [])
        return lines + photons

    def animate(frame):
        progress = frame / n_frames

        for i, (x_path, y_path) in enumerate(all_paths):
            # Find valid points
            valid = (x_path != 0) | (y_path != 0)
            valid[0] = True
            n_valid = np.sum(valid)

            # Current position along path
            idx = int(progress * n_valid)
            idx = min(idx, n_valid - 1)

            # Trail
            x_valid = x_path[valid][:idx+1]
            y_valid = y_path[valid][:idx+1]
            lines[i].set_data(x_valid, y_valid)

            # Photon marker
            if idx < len(x_valid):
                photons[i].set_data([x_valid[-1]], [y_valid[-1]])

        return lines + photons

    anim = FuncAnimation(fig, animate, init_func=init, frames=n_frames,
                        interval=50, blit=True)

    suffix = '_de' if language == 'de' else ''
    filepath = os.path.join(output_dir, f'light_bending_animation{suffix}.gif')
    writer = PillowWriter(fps=20)
    anim.save(filepath, writer=writer)
    print(f"Saved animation: {filepath}")

    plt.close(fig)


# =============================================================================
# PENROSE-CARTER DIAGRAMS
# =============================================================================

def schwarzschild_to_kruskal(r: float, t: float, R_s: float) -> Tuple[float, float]:
    """
    Transform Schwarzschild coordinates to Kruskal-Szekeres coordinates.
    Transformiert Schwarzschild-Koordinaten zu Kruskal-Szekeres-Koordinaten.

    For r > R_s (outside event horizon):
        U = sqrt(r/R_s - 1) * exp(r/(2*R_s)) * cosh(t/(2*R_s))
        V = sqrt(r/R_s - 1) * exp(r/(2*R_s)) * sinh(t/(2*R_s))

    Args:
        r: Schwarzschild radial coordinate [R_s units]
        t: Schwarzschild time coordinate [R_s/c units]
        R_s: Schwarzschild radius

    Returns:
        Tuple of (U, V) Kruskal coordinates
    """
    if r <= 0:
        return (0, 0)

    r_norm = r / R_s  # Normalize to Schwarzschild radii
    t_norm = t / R_s

    if r_norm > 1:  # Outside horizon
        factor = np.sqrt(r_norm - 1) * np.exp(r_norm / 2)
        U = factor * np.cosh(t_norm / 2)
        V = factor * np.sinh(t_norm / 2)
    elif r_norm < 1:  # Inside horizon
        factor = np.sqrt(1 - r_norm) * np.exp(r_norm / 2)
        U = factor * np.sinh(t_norm / 2)
        V = factor * np.cosh(t_norm / 2)
    else:  # On horizon
        U = 0
        V = 0

    return (U, V)


def kruskal_to_penrose(U: float, V: float) -> Tuple[float, float]:
    """
    Transform Kruskal coordinates to Penrose diagram coordinates.
    Transformiert Kruskal-Koordinaten zu Penrose-Diagramm-Koordinaten.

    Compactification:
        u = U + V,  v = V - U
        T = arctan(u) + arctan(v)
        X = arctan(u) - arctan(v)

    Args:
        U, V: Kruskal-Szekeres coordinates

    Returns:
        Tuple of (X, T) Penrose coordinates
    """
    u = U + V
    v = V - U

    T = np.arctan(u) + np.arctan(v)
    X = np.arctan(u) - np.arctan(v)

    return (X, T)


def calculate_constant_r_curve(r: float, R_s: float,
                               t_range: Tuple[float, float] = (-10, 10),
                               n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate a constant-r curve in Penrose coordinates.
    Berechnet eine Kurve konstanter r in Penrose-Koordinaten.

    Args:
        r: Radial coordinate [m]
        R_s: Schwarzschild radius [m]
        t_range: Range of Schwarzschild time coordinate
        n_points: Number of points

    Returns:
        Tuple of (X, T) arrays in Penrose coordinates
    """
    t_values = np.linspace(t_range[0], t_range[1], n_points)
    X = np.zeros(n_points)
    T = np.zeros(n_points)

    for i, t in enumerate(t_values):
        U, V = schwarzschild_to_kruskal(r, t, R_s)
        X[i], T[i] = kruskal_to_penrose(U, V)

    return X, T


def calculate_infalling_worldline(R_s: float, r_start: float,
                                  n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate worldline of an infalling observer in Penrose coordinates.
    Berechnet die Weltlinie eines einfallenden Beobachters in Penrose-Koordinaten.

    Args:
        R_s: Schwarzschild radius
        r_start: Starting radius [R_s units]
        n_points: Number of points

    Returns:
        Tuple of (X, T) arrays in Penrose coordinates
    """
    # Simplified: use proper time parameterization for radial infall
    # For a radially infalling observer from rest at infinity:
    # dr/dtau = -sqrt(R_s/r) * c

    r_values = np.linspace(r_start * R_s, 0.01 * R_s, n_points)
    X = np.zeros(n_points)
    T = np.zeros(n_points)

    # Approximate the worldline using proper time
    tau = 0
    for i, r in enumerate(r_values):
        r_norm = r / R_s

        if r_norm > 1:
            # Outside horizon - use approximate Schwarzschild time
            t = tau * 2  # Simplified scaling
            U, V = schwarzschild_to_kruskal(r, t, R_s)
            X[i], T[i] = kruskal_to_penrose(U, V)
        else:
            # Inside horizon
            t = tau * 2
            U, V = schwarzschild_to_kruskal(r, t, R_s)
            X[i], T[i] = kruskal_to_penrose(U, V)

        tau += 0.1

    return X, T


def plot_penrose_carter_diagram(
    constants: Optional[PhysicalConstants] = None,
    language: str = 'en',
    save: bool = True,
    show: bool = False,
    diagram_type: str = 'schwarzschild'
) -> plt.Figure:
    """
    Plot a Penrose-Carter conformal diagram.
    Zeichnet ein Penrose-Carter Konformaldiagramm.

    Args:
        constants: Physical constants
        language: 'en' for English, 'de' for German
        save: Whether to save the figure
        show: Whether to display the figure
        diagram_type: 'schwarzschild', 'collapse', or 'flat'

    Returns:
        matplotlib Figure object
    """
    if constants is None:
        constants = get_constants()

    fig, ax = plt.subplots(figsize=(10, 12))

    # Define the diamond boundary
    # In Penrose coordinates, null infinities are at 45-degree lines

    if diagram_type == 'flat':
        # Minkowski spacetime: simple diamond
        # Future null infinity (J+) and past null infinity (J-)
        diamond_x = [0, np.pi/2, 0, -np.pi/2, 0]
        diamond_t = [-np.pi/2, 0, np.pi/2, 0, -np.pi/2]
        ax.plot(diamond_x, diamond_t, 'k-', linewidth=2)

        # Labels
        ax.text(0, np.pi/2 + 0.1, 'i+', fontsize=14, ha='center', fontweight='bold')
        ax.text(0, -np.pi/2 - 0.1, 'i-', fontsize=14, ha='center', va='top', fontweight='bold')
        ax.text(np.pi/2 + 0.1, 0, 'i0', fontsize=14, va='center', fontweight='bold')
        ax.text(-np.pi/2 - 0.1, 0, 'i0', fontsize=14, va='center', ha='right', fontweight='bold')

        # J+ and J- labels
        ax.text(np.pi/4 + 0.15, np.pi/4, 'J+', fontsize=12, rotation=-45)
        ax.text(-np.pi/4 - 0.15, np.pi/4, 'J+', fontsize=12, rotation=45, ha='right')
        ax.text(np.pi/4 + 0.15, -np.pi/4, 'J-', fontsize=12, rotation=45)
        ax.text(-np.pi/4 - 0.15, -np.pi/4, 'J-', fontsize=12, rotation=-45, ha='right')

        # Worldlines
        for x0 in np.linspace(-0.8, 0.8, 5):
            t_line = np.linspace(-np.pi/2 + abs(x0), np.pi/2 - abs(x0), 50)
            x_line = np.ones_like(t_line) * x0
            ax.plot(x_line, t_line, 'b-', alpha=0.3, linewidth=0.8)

        title = 'Minkowski Spacetime' if language == 'en' else 'Minkowski-Raumzeit'

    else:  # Schwarzschild
        # Schwarzschild Penrose diagram has 4 regions

        # Draw the diamond boundary (including singularity)
        # Region I (exterior) and Region II (interior/future)

        # Singularity (wavy line at top)
        x_sing = np.linspace(-np.pi/2, np.pi/2, 100)
        t_sing = np.pi/2 * np.ones_like(x_sing)
        # Add waviness
        t_sing_wavy = t_sing + 0.05 * np.sin(20 * x_sing)
        ax.plot(x_sing, t_sing_wavy, color=COLORS['relativistic'], linewidth=3,
               label='Singularity (r=0)' if language == 'en' else 'Singularität (r=0)')

        # Past singularity (for eternal black hole)
        t_sing_past = -np.pi/2 * np.ones_like(x_sing)
        t_sing_past_wavy = t_sing_past + 0.05 * np.sin(20 * x_sing)
        ax.plot(x_sing, t_sing_past_wavy, color=COLORS['relativistic'], linewidth=3, alpha=0.5)

        # Event horizons (45-degree lines)
        # Future horizon (from i- to singularity)
        ax.plot([0, np.pi/2], [-np.pi/2, 0], 'k--', linewidth=2,
               label='Event Horizon' if language == 'en' else 'Ereignishorizont')
        ax.plot([0, -np.pi/2], [-np.pi/2, 0], 'k--', linewidth=2)

        # Past horizon
        ax.plot([0, np.pi/2], [np.pi/2, 0], 'k--', linewidth=2, alpha=0.5)
        ax.plot([0, -np.pi/2], [np.pi/2, 0], 'k--', linewidth=2, alpha=0.5)

        # Outer boundaries (J+ and J-)
        ax.plot([np.pi/2, np.pi/2], [0, 0], 'k-', linewidth=2)  # i0
        ax.plot([-np.pi/2, -np.pi/2], [0, 0], 'k-', linewidth=2)  # i0

        # Right exterior (region I)
        ax.plot([np.pi/2, np.pi/2], [-np.pi/2, 0], 'k-', linewidth=2)  # J-
        ax.plot([np.pi/2, np.pi/2], [0, np.pi/2], 'k-', linewidth=2)   # J+

        # Left exterior (region III - "other universe")
        ax.plot([-np.pi/2, -np.pi/2], [-np.pi/2, 0], 'k-', linewidth=2, alpha=0.5)
        ax.plot([-np.pi/2, -np.pi/2], [0, np.pi/2], 'k-', linewidth=2, alpha=0.5)

        # Constant r curves
        R_s = 1.0  # Normalized
        for r_val in [1.5, 2.0, 3.0, 5.0]:
            X, T = calculate_constant_r_curve(r_val * R_s, R_s, (-5, 5), 100)
            # Clip to valid region
            valid = (abs(X) < np.pi/2) & (abs(T) < np.pi/2)
            if np.any(valid):
                ax.plot(X[valid], T[valid], color=COLORS['standard'],
                       linewidth=1, alpha=0.6)

        # Infalling observer worldline
        ax.plot([0.8, 0.3, 0], [-0.8, 0, 0.4], color=COLORS['highlight'],
               linewidth=2, label='Infalling observer' if language == 'en' else 'Einfallender Beobachter')
        ax.plot(0.8, -0.8, 'o', color=COLORS['highlight'], markersize=8)

        # Region labels
        ax.text(0.5, -0.3, 'I', fontsize=18, fontweight='bold', ha='center')
        ax.text(-0.5, -0.3, 'III', fontsize=18, fontweight='bold', ha='center', alpha=0.5)
        ax.text(0, 0.25, 'II', fontsize=18, fontweight='bold', ha='center')
        ax.text(0, -0.65, 'IV', fontsize=18, fontweight='bold', ha='center', alpha=0.5)

        # Infinity labels
        ax.text(0, np.pi/2 + 0.15, 'r = 0', fontsize=11, ha='center', color=COLORS['relativistic'])
        ax.text(np.pi/2 + 0.1, 0, 'i0', fontsize=14, va='center', fontweight='bold')
        ax.text(np.pi/2 + 0.1, np.pi/4, 'J+', fontsize=12)
        ax.text(np.pi/2 + 0.1, -np.pi/4, 'J-', fontsize=12)

        # Light cone example
        ax.annotate('', xy=(0.6, 0.2), xytext=(0.4, 0),
                   arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
        ax.annotate('', xy=(0.6, -0.2), xytext=(0.4, 0),
                   arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))

        title = 'Schwarzschild Black Hole' if language == 'en' else 'Schwarzschild Schwarzes Loch'

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')
    ax.set_xlabel('X (compactified space)' if language == 'en' else 'X (kompaktifizierter Raum)', fontsize=11)
    ax.set_ylabel('T (compactified time)' if language == 'en' else 'T (kompaktifizierte Zeit)', fontsize=11)

    ax.set_title(f'Penrose-Carter Diagram: {title}',
                fontsize=14, fontweight='bold', pad=15)

    # Legend at bottom
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=COLORS['relativistic'], linewidth=3,
               label='Singularity (r=0)' if language == 'en' else 'Singularität (r=0)'),
        Line2D([0], [0], color='black', linestyle='--', linewidth=2,
               label='Event Horizon' if language == 'en' else 'Ereignishorizont'),
        Line2D([0], [0], color=COLORS['highlight'], linewidth=2,
               label='Infalling observer' if language == 'en' else 'Einfallender Beobachter'),
        Line2D([0], [0], color=COLORS['standard'], linewidth=1,
               label='Constant r curves' if language == 'en' else 'Konstante r-Kurven'),
        Line2D([0], [0], color='blue', linewidth=1.5, marker='>',
               label='Light (45°)' if language == 'en' else 'Licht (45°)'),
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.08),
              fontsize=9, ncol=3, framealpha=0.7)

    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save:
        os.makedirs(VIS_DIR, exist_ok=True)
        suffix = '_de' if language == 'de' else ''
        filepath = os.path.join(VIS_DIR, f'penrose_{diagram_type}{suffix}.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")

    if show:
        plt.show()

    return fig


def plot_penrose_comparison(
    constants: Optional[PhysicalConstants] = None,
    language: str = 'en',
    save: bool = True,
    show: bool = False
) -> plt.Figure:
    """
    Plot comparison of Penrose diagrams for different spacetimes.
    Zeichnet Vergleich von Penrose-Diagrammen für verschiedene Raumzeiten.

    Shows: Minkowski, Schwarzschild, and Collapsing star

    Args:
        constants: Physical constants
        language: 'en' for English, 'de' for German
        save: Whether to save the figure
        show: Whether to display the figure

    Returns:
        matplotlib Figure object
    """
    if constants is None:
        constants = get_constants()

    fig, axes = plt.subplots(1, 3, figsize=(18, 10))

    # Common settings
    titles = {
        'en': ['Minkowski (Flat)', 'Schwarzschild (Eternal BH)', 'Stellar Collapse'],
        'de': ['Minkowski (Flach)', 'Schwarzschild (Ewiges SL)', 'Sternkollaps']
    }

    for idx, ax in enumerate(axes):
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X', fontsize=11)
        ax.set_ylabel('T', fontsize=11)
        ax.set_title(titles[language][idx], fontsize=12, fontweight='bold')

    # ===== MINKOWSKI =====
    ax = axes[0]
    # Diamond shape
    diamond_x = [0, np.pi/2, 0, -np.pi/2, 0]
    diamond_t = [-np.pi/2, 0, np.pi/2, 0, -np.pi/2]
    ax.plot(diamond_x, diamond_t, 'k-', linewidth=2)

    # Timelike worldlines
    for x0 in np.linspace(-0.6, 0.6, 4):
        t_line = np.linspace(-np.pi/2 + abs(x0) + 0.1, np.pi/2 - abs(x0) - 0.1, 30)
        ax.plot(np.ones_like(t_line) * x0, t_line, 'b-', alpha=0.4, linewidth=1)

    ax.text(0, np.pi/2 + 0.15, 'i+', fontsize=14, ha='center', fontweight='bold')
    ax.text(0, -np.pi/2 - 0.15, 'i-', fontsize=14, ha='center', va='top', fontweight='bold')
    ax.text(np.pi/2 + 0.1, 0, 'i0', fontsize=12, va='center')

    # ===== SCHWARZSCHILD =====
    ax = axes[1]

    # Singularities
    x_sing = np.linspace(-np.pi/2, np.pi/2, 50)
    t_top = np.pi/2 + 0.05 * np.sin(15 * x_sing)
    t_bot = -np.pi/2 + 0.05 * np.sin(15 * x_sing)
    ax.plot(x_sing, t_top, color=COLORS['relativistic'], linewidth=3)
    ax.plot(x_sing, t_bot, color=COLORS['relativistic'], linewidth=3, alpha=0.5)

    # Horizons
    ax.plot([0, np.pi/2], [-np.pi/2, 0], 'k--', linewidth=2)
    ax.plot([0, -np.pi/2], [-np.pi/2, 0], 'k--', linewidth=2, alpha=0.5)
    ax.plot([0, np.pi/2], [np.pi/2, 0], 'k--', linewidth=2)
    ax.plot([0, -np.pi/2], [np.pi/2, 0], 'k--', linewidth=2, alpha=0.5)

    # Region labels
    ax.text(0.5, -0.3, 'I', fontsize=16, fontweight='bold')
    ax.text(-0.5, -0.3, 'III', fontsize=16, fontweight='bold', alpha=0.5)
    ax.text(0, 0.2, 'II', fontsize=16, fontweight='bold')
    ax.text(0, -0.6, 'IV', fontsize=16, fontweight='bold', alpha=0.5)

    # ===== STELLAR COLLAPSE =====
    ax = axes[2]

    # Star surface worldline (before collapse)
    r_star = 0.4  # Initial radius in Penrose coords
    t_collapse = 0  # Time of horizon formation

    # Before collapse (matter region)
    ax.fill_betweenx(np.linspace(-np.pi/2, t_collapse, 20),
                     -r_star, r_star, color='gray', alpha=0.3)

    # Star surface falling in
    t_fall = np.linspace(t_collapse, np.pi/2 - 0.1, 30)
    x_fall = r_star * np.exp(-(t_fall - t_collapse))
    ax.plot(x_fall, t_fall, color=COLORS['highlight'], linewidth=2)
    ax.plot(-x_fall, t_fall, color=COLORS['highlight'], linewidth=2)

    # Singularity (only future)
    x_sing = np.linspace(-np.pi/2, np.pi/2, 50)
    t_sing = np.pi/2 + 0.05 * np.sin(15 * x_sing)
    ax.plot(x_sing, t_sing, color=COLORS['relativistic'], linewidth=3)

    # Horizon (only forms after collapse)
    ax.plot([0, np.pi/2], [t_collapse, np.pi/4 + t_collapse/2], 'k--', linewidth=2)

    # Outer boundary
    ax.plot([np.pi/2, np.pi/2], [-np.pi/2, np.pi/2], 'k-', linewidth=2)

    ax.text(0, np.pi/2 + 0.15, 'r = 0', fontsize=10, ha='center', color=COLORS['relativistic'])
    ax.text(0.5, -0.5, 'I', fontsize=16, fontweight='bold')
    ax.text(0, 0.3, 'II', fontsize=16, fontweight='bold')

    # Common legend at bottom for all three plots
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_elements = [
        Line2D([0], [0], color='k', linewidth=2,
               label='Boundary' if language == 'en' else 'Grenze'),
        Line2D([0], [0], color='k', linestyle='--', linewidth=2,
               label='Event Horizon' if language == 'en' else 'Ereignishorizont'),
        Line2D([0], [0], color=COLORS['relativistic'], linewidth=3,
               label='Singularity' if language == 'en' else 'Singularität'),
        Line2D([0], [0], color='blue', alpha=0.4, linewidth=1,
               label='Worldlines' if language == 'en' else 'Weltlinien'),
        Patch(facecolor='gray', alpha=0.3,
              label='Collapsing matter' if language == 'en' else 'Kollabierende Materie'),
        Line2D([0], [0], color=COLORS['highlight'], linewidth=2,
               label='Star surface' if language == 'en' else 'Sternoberfläche'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.02),
               fontsize=9, ncol=6, framealpha=0.7)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)  # Make room for legend

    if save:
        os.makedirs(VIS_DIR, exist_ok=True)
        suffix = '_de' if language == 'de' else ''
        filepath = os.path.join(VIS_DIR, f'penrose_comparison{suffix}.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")

    if show:
        plt.show()

    return fig


def verify_spacetime_physics():
    """
    Verify spacetime physics calculations.
    """
    print("=" * 70)
    print("SPACETIME CURVATURE PHYSICS VERIFICATION")
    print("=" * 70)

    c = get_constants()

    # 1. Earth Schwarzschild radius
    print("\n1. EARTH SCHWARZSCHILD RADIUS")
    print("-" * 50)
    R_s_earth = c.schwarzschild_radius(c.M_earth)
    print(f"   R_s(Earth) = {R_s_earth:.4f} m = {R_s_earth*1000:.2f} mm")
    print(f"   Expected: ~9 mm (CHECK: {'PASS' if 8 < R_s_earth*1000 < 10 else 'FAIL'})")

    # 2. Sun Schwarzschild radius
    print("\n2. SUN SCHWARZSCHILD RADIUS")
    print("-" * 50)
    R_s_sun = c.schwarzschild_radius(c.M_sun)
    print(f"   R_s(Sun) = {R_s_sun:.2f} m = {R_s_sun/1000:.2f} km")
    print(f"   Expected: ~3 km (CHECK: {'PASS' if 2500 < R_s_sun < 3500 else 'FAIL'})")

    # 3. Earth compactness
    print("\n3. EARTH COMPACTNESS")
    print("-" * 50)
    C_earth = c.compactness(c.M_earth, c.R_earth)
    print(f"   C(Earth) = R_s/R = {C_earth:.2e}")
    print(f"   Expected: ~10^-9 (CHECK: {'PASS' if 1e-10 < C_earth < 1e-8 else 'FAIL'})")

    # 4. Earth escape velocity
    print("\n4. EARTH ESCAPE VELOCITY")
    print("-" * 50)
    v_esc_earth = c.escape_velocity(c.M_earth, c.R_earth)
    print(f"   v_esc(Earth) = {v_esc_earth:.0f} m/s = {v_esc_earth/1000:.2f} km/s")
    print(f"   Expected: ~11.2 km/s (CHECK: {'PASS' if 11000 < v_esc_earth < 11500 else 'FAIL'})")

    # 5. Earth surface gravity
    print("\n5. EARTH SURFACE GRAVITY")
    print("-" * 50)
    g_earth = c.surface_gravity(c.M_earth, c.R_earth)
    print(f"   g(Earth) = {g_earth:.2f} m/s²")
    print(f"   Expected: ~9.8 m/s² (CHECK: {'PASS' if 9.7 < g_earth < 9.9 else 'FAIL'})")

    # 6. Neutron star compactness
    print("\n6. NEUTRON STAR COMPACTNESS")
    print("-" * 50)
    M_ns = 1.4 * c.M_sun
    R_ns = 10e3  # 10 km
    C_ns = c.compactness(M_ns, R_ns)
    print(f"   M = 1.4 M_sun, R = 10 km")
    print(f"   C(NS) = {C_ns:.3f}")
    print(f"   Expected: ~0.2-0.4 (CHECK: {'PASS' if 0.15 < C_ns < 0.5 else 'FAIL'})")

    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    print("=" * 60)
    print("Spacetime Curvature Module - Jugend forscht 2026")
    print("=" * 60)

    # Verify physics
    verify_spacetime_physics()

    # Generate visualizations
    print("\n")
    generate_all_spacetime_plots(language='en', show=False)

    print("\nDone! Check the 'visualizations' folder for output.")
