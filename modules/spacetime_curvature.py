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

    print("=" * 50)
    print(f"Generated {len(figures)} visualizations in {VIS_DIR}")

    return figures


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
