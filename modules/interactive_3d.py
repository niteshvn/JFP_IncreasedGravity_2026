"""
Interactive 3D Visualizations using Plotly for Jugend forscht 2026
Interaktive 3D-Visualisierungen mit Plotly fuer Jugend forscht 2026

This module provides interactive 3D visualizations that can be viewed in a web browser.
These complement the static matplotlib plots with interactive exploration capabilities.

Author: Jugend forscht 2026 Project
"""

import numpy as np
import os
from typing import Optional, List

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly not installed. Install with: pip install plotly")

from .constants import get_constants, PhysicalConstants
from .color_scheme import COLORS, get_stellar_colors


# Output directory for visualizations
VIS_DIR = os.path.join(os.path.dirname(__file__), '..', 'visualizations')


def check_plotly():
    """Check if plotly is available."""
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly is required for interactive 3D visualizations. "
                         "Install with: pip install plotly")


def plot_spacetime_curvature_3d_interactive(
    constants: Optional[PhysicalConstants] = None,
    language: str = 'en',
    save: bool = True
) -> 'go.Figure':
    """
    Create an interactive 3D visualization of spacetime curvature.
    Erstellt eine interaktive 3D-Visualisierung der Raumzeitkruemmung.

    This creates a "rubber sheet" analogy showing how mass curves spacetime.

    Args:
        constants: Physical constants
        language: 'en' for English, 'de' for German
        save: Whether to save as HTML file

    Returns:
        Plotly Figure object
    """
    check_plotly()

    if constants is None:
        constants = get_constants()

    # Create meshgrid for the surface
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)

    # Avoid division by zero
    R = np.maximum(R, 0.3)

    # Create potential well (gravitational "dent")
    # Z represents the curvature/depth
    Z = -1.5 / R

    # Clip extreme values for better visualization
    Z = np.clip(Z, -5, 0)

    # Create the figure
    fig = go.Figure()

    # Add the curved surface
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(
            title=dict(
                text='Depth' if language == 'en' else 'Tiefe',
                side='bottom',
                font=dict(size=13)
            ),
            orientation='h',
            x=0.5,
            y=0.02,
            xanchor='center',
            yanchor='top',
            len=0.5,
            thickness=18,
            tickformat='.1f'
        ),
        contours=dict(
            z=dict(show=True, usecolormap=True, highlightcolor="white", project_z=True)
        ),
        name='Spacetime curvature' if language == 'en' else 'Raumzeitkruemmung',
        showlegend=False  # Surface traces don't display well in legends
    ))

    # Add legend entry using Scatter3d (displays properly in legends)
    fig.add_trace(go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode='markers',
        marker=dict(size=10, color='#31688e', symbol='square'),  # Viridis mid color
        name='Spacetime curvature' if language == 'en' else 'Raumzeitkrümmung',
        showlegend=True
    ))

    # Add a marker for the central mass
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[-4.5],
        mode='markers',
        marker=dict(size=10, color=COLORS['scaled'], symbol='circle'),
        name='Mass' if language == 'en' else 'Masse',
        showlegend=True
    ))

    # Update layout
    if language == 'de':
        title = 'Interaktive Raumzeitkruemmung<br><sup>Drehen Sie die Ansicht mit der Maus</sup>'
        x_title = 'x-Position'
        y_title = 'y-Position'
        z_title = 'Kruemmung'
    else:
        title = 'Interactive Spacetime Curvature<br><sup>Rotate the view with your mouse</sup>'
        x_title = 'x position'
        y_title = 'y position'
        z_title = 'Curvature'

    fig.update_layout(
        title=dict(text=title, x=0.5),
        scene=dict(
            xaxis_title=x_title,
            yaxis_title=y_title,
            zaxis_title=z_title,
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            ),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.5),
            domain=dict(x=[0, 1], y=[0.18, 1])  # Full width, more vertical space for graph
        ),
        height=900,
        margin=dict(l=0, r=0, t=50, b=10),  # Minimal margins for max graph space
        template='plotly_white',
        showlegend=True,
        legend=dict(
            x=0.5,
            y=0.12,
            xanchor='center',
            yanchor='top',
            bgcolor='rgba(255,255,255,0.95)',
            bordercolor='rgba(180,180,180,0.8)',
            borderwidth=1,
            font=dict(size=14),
            orientation='v',
            entrywidth=0,  # Auto-size based on text width
            entrywidthmode='pixels'  # Use pixels mode for auto-sizing
        )
    )

    if save:
        os.makedirs(VIS_DIR, exist_ok=True)
        filepath = os.path.join(VIS_DIR, 'spacetime_3d_interactive.html')
        fig.write_html(filepath, config={'displaylogo': False, 'displayModeBar': True})
        print(f"Saved: {filepath}")

    return fig


def plot_multiple_masses_3d_interactive(
    constants: Optional[PhysicalConstants] = None,
    language: str = 'en',
    save: bool = True
) -> 'go.Figure':
    """
    Create an interactive 3D comparison of spacetime curvature for different masses.
    Erstellt einen interaktiven 3D-Vergleich der Raumzeitkruemmung fuer verschiedene Massen.

    Shows Earth, Sun, White Dwarf, and Neutron Star potential wells.

    Args:
        constants: Physical constants
        language: 'en' for English, 'de' for German
        save: Whether to save as HTML file

    Returns:
        Plotly Figure object
    """
    check_plotly()

    if constants is None:
        constants = get_constants()

    # Define objects with their relative "depths" (compactness)
    stellar_colors = get_stellar_colors()
    objects = [
        {'name': 'Earth' if language == 'en' else 'Erde',
         'depth': 0.5, 'color': stellar_colors[0], 'pos': (-3, -3)},
        {'name': 'Sun' if language == 'en' else 'Sonne',
         'depth': 1.5, 'color': stellar_colors[1], 'pos': (3, -3)},
        {'name': 'White Dwarf' if language == 'en' else 'Weisser Zwerg',
         'depth': 3.0, 'color': stellar_colors[2], 'pos': (-3, 3)},
        {'name': 'Neutron Star' if language == 'en' else 'Neutronenstern',
         'depth': 8.0, 'color': stellar_colors[3], 'pos': (3, 3)},
    ]

    # Create meshgrid
    x = np.linspace(-8, 8, 150)
    y = np.linspace(-8, 8, 150)
    X, Y = np.meshgrid(x, y)

    # Calculate combined potential
    Z = np.zeros_like(X)
    for obj in objects:
        px, py = obj['pos']
        R = np.sqrt((X - px)**2 + (Y - py)**2)
        R = np.maximum(R, 0.3)
        Z -= obj['depth'] / R

    # Clip for visualization
    Z = np.clip(Z, -15, 0)

    # Create figure
    fig = go.Figure()

    # Add surface (hidden from legend)
    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z,
        colorscale='Plasma',
        showscale=True,
        colorbar=dict(
            title=dict(
                text='Curvature' if language == 'en' else 'Kruemmung',
                side='bottom',
                font=dict(size=13)
            ),
            orientation='h',
            x=0.5,
            y=0.02,
            xanchor='center',
            yanchor='top',
            len=0.5,
            thickness=18
        ),
        opacity=0.9,
        showlegend=False
    ))

    # Add markers for each object
    for obj in objects:
        px, py = obj['pos']
        R = np.sqrt(px**2 + py**2)
        # Find z at this position
        z_val = -obj['depth'] / 0.3 - sum(
            o['depth'] / max(np.sqrt((px - o['pos'][0])**2 + (py - o['pos'][1])**2), 0.3)
            for o in objects if o != obj
        )
        z_val = max(z_val, -14)

        fig.add_trace(go.Scatter3d(
            x=[px], y=[py], z=[z_val],
            mode='markers+text',
            marker=dict(size=8, color=obj['color'],
                       line=dict(width=2, color='black')),
            text=[obj['name']],
            textposition='top center',
            name=obj['name']
        ))

    # Update layout
    if language == 'de':
        title = 'Vergleich der Raumzeitkruemmung verschiedener Objekte<br><sup>Interaktive 3D-Ansicht</sup>'
    else:
        title = 'Spacetime Curvature Comparison of Different Objects<br><sup>Interactive 3D View</sup>'

    # 4 legend items need more vertical space
    fig.update_layout(
        title=dict(text=title, x=0.5),
        scene=dict(
            xaxis_title='x',
            yaxis_title='y',
            zaxis_title='Curvature' if language == 'en' else 'Kruemmung',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.5),
            domain=dict(x=[0, 1], y=[0.25, 1])  # More bottom space for 4 legend items
        ),
        height=900,
        margin=dict(l=0, r=0, t=50, b=10),
        template='plotly_white',
        showlegend=True,
        legend=dict(
            x=0.5,
            y=0.18,  # Higher position to avoid colorbar
            xanchor='center',
            yanchor='top',
            bgcolor='rgba(255,255,255,0.95)',
            bordercolor='rgba(180,180,180,0.8)',
            borderwidth=1,
            font=dict(size=14),
            orientation='v',
            entrywidth=0,
            entrywidthmode='pixels'
        )
    )

    if save:
        os.makedirs(VIS_DIR, exist_ok=True)
        filepath = os.path.join(VIS_DIR, 'spacetime_comparison_3d_interactive.html')
        fig.write_html(filepath, config={'displaylogo': False, 'displayModeBar': True})
        print(f"Saved: {filepath}")

    return fig


def plot_atom_scaling_3d_interactive(
    hbar_scales: List[float] = [1.0, 0.5, 0.1],
    constants: Optional[PhysicalConstants] = None,
    language: str = 'en',
    save: bool = True
) -> 'go.Figure':
    """
    Create an interactive 3D visualization of atom sizes in different universes.
    Erstellt eine interaktive 3D-Visualisierung von Atomgroessen in verschiedenen Universen.

    Shows how atoms shrink when hbar is reduced.

    Args:
        hbar_scales: List of hbar scaling factors
        constants: Physical constants
        language: 'en' for English, 'de' for German
        save: Whether to save as HTML file

    Returns:
        Plotly Figure object
    """
    check_plotly()

    if constants is None:
        constants = get_constants()

    fig = go.Figure()

    # Standard Bohr radius
    a0_std = constants.a_0

    # Colors for different universes - distinct colors for orbitals
    orbital_colors = [COLORS['primary_blue'], COLORS['standard'], COLORS['scaled']]

    # Store legend info for later
    legend_entries = []

    for i, (scale, color) in enumerate(zip(hbar_scales, orbital_colors)):
        # Calculate scaled Bohr radius
        a0 = a0_std * scale**2

        # Create sphere for electron orbital
        u = np.linspace(0, 2 * np.pi, 50)
        v = np.linspace(0, np.pi, 50)

        # Normalize for display
        r = scale**2  # Relative radius

        x = r * np.outer(np.cos(u), np.sin(v))
        y = r * np.outer(np.sin(u), np.sin(v))
        z = r * np.outer(np.ones(np.size(u)), np.cos(v)) + i * 3  # Offset in z

        # Legend names
        if language == 'de':
            orbital_name = f'ℏ×{scale}: a₀={a0*1e12:.1f} pm'
        else:
            orbital_name = f'ℏ×{scale}: a₀={a0*1e12:.1f} pm'

        # Store for legend entry
        legend_entries.append((orbital_name, color, i * 3))

        # Add orbital surface (hidden from legend - Surface traces don't display well)
        fig.add_trace(go.Surface(
            x=x, y=y, z=z,
            colorscale=[[0, color], [1, color]],
            opacity=0.4,
            showscale=False,
            showlegend=False
        ))

        # Add nucleus (no legend - single entry for all nuclei below)
        fig.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[i * 3],
            mode='markers',
            marker=dict(size=6, color='darkred'),
            showlegend=False
        ))

    # Add legend entries using Scatter3d traces (these display properly in legends)
    for orbital_name, color, z_pos in legend_entries:
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode='markers',
            marker=dict(size=12, color=color, symbol='square'),
            name=orbital_name,
            showlegend=True
        ))

    # Add single nucleus legend entry
    nucleus_label = 'Nucleus' if language == 'en' else 'Kern'
    fig.add_trace(go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode='markers',
        marker=dict(size=8, color='darkred'),
        name=nucleus_label,
        showlegend=True
    ))

    # Update layout
    if language == 'de':
        title = 'Atomgrößen in verschiedenen Universen<br><sup>a₀ ∝ ℏ² — Interaktive 3D-Ansicht</sup>'
    else:
        title = 'Atom Sizes in Different Universes<br><sup>a₀ ∝ ℏ² — Interactive 3D View</sup>'

    # 4 legend items (3 orbitals + nucleus) need more vertical space
    fig.update_layout(
        title=dict(text=title, x=0.5),
        scene=dict(
            xaxis_title='x (relative)',
            yaxis_title='y (relative)',
            zaxis_title='Universe',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            aspectmode='data',
            domain=dict(x=[0, 1], y=[0.25, 1])  # More bottom space for 4 legend items
        ),
        height=900,
        margin=dict(l=0, r=0, t=50, b=10),
        template='plotly_white',
        showlegend=True,
        legend=dict(
            x=0.5,
            y=0.18,  # Higher position to avoid overlap
            xanchor='center',
            yanchor='top',
            bgcolor='rgba(255,255,255,0.95)',
            bordercolor='rgba(180,180,180,0.8)',
            borderwidth=1,
            font=dict(size=14),
            orientation='v',
            entrywidth=0,
            entrywidthmode='pixels'
        )
    )

    if save:
        os.makedirs(VIS_DIR, exist_ok=True)
        filepath = os.path.join(VIS_DIR, 'atom_scaling_3d_interactive.html')
        fig.write_html(filepath, config={'displaylogo': False, 'displayModeBar': True})
        print(f"Saved: {filepath}")

    return fig


def plot_force_ratio_3d_interactive(
    constants: Optional[PhysicalConstants] = None,
    language: str = 'en',
    save: bool = True
) -> 'go.Figure':
    """
    Create an interactive 3D surface showing force ratios across scales.
    Erstellt eine interaktive 3D-Oberflaeche der Kraftverhaeltnisse ueber verschiedene Skalen.

    Args:
        constants: Physical constants
        language: 'en' for English, 'de' for German
        save: Whether to save as HTML file

    Returns:
        Plotly Figure object
    """
    check_plotly()

    if constants is None:
        constants = get_constants()

    # Create ranges for mass and distance
    log_mass = np.linspace(-30, 35, 50)  # log10(mass in kg)
    log_dist = np.linspace(-15, 10, 50)  # log10(distance in m)

    LOG_MASS, LOG_DIST = np.meshgrid(log_mass, log_dist)

    # Calculate log of force ratio: F_em / F_grav
    # F_em / F_grav = (k_e * e^2) / (G * m^2) for same mass particles
    # This is independent of distance!
    # But we want to show different masses...

    # For charged particles of mass m:
    # log(F_em/F_grav) = log(k_e * e^2) - log(G) - 2*log(m)
    log_k_e = np.log10(constants.k_e)
    log_e2 = 2 * np.log10(constants.e)
    log_G = np.log10(constants.G)

    # Force ratio (independent of distance for same force law)
    LOG_RATIO = log_k_e + log_e2 - log_G - 2 * LOG_MASS

    # Clip for visualization
    LOG_RATIO = np.clip(LOG_RATIO, -10, 50)

    fig = go.Figure()

    # Add surface (hidden from legend - Surface traces don't display well in legends)
    fig.add_trace(go.Surface(
        x=LOG_MASS, y=LOG_DIST, z=LOG_RATIO,
        colorscale='RdBu',
        showscale=True,
        colorbar=dict(
            title=dict(text='log₁₀(F_em/F_grav)', side='bottom', font=dict(size=13)),
            orientation='h',
            x=0.5,
            y=0.02,
            xanchor='center',
            yanchor='top',
            len=0.5,
            thickness=18,
            tickfont=dict(size=11)
        ),
        showlegend=False
    ))

    # Add a plane at z=0 (where forces are equal) - hidden from legend
    fig.add_trace(go.Surface(
        x=LOG_MASS, y=LOG_DIST, z=np.zeros_like(LOG_RATIO),
        colorscale=[[0, 'gray'], [1, 'gray']],
        opacity=0.3,
        showscale=False,
        showlegend=False
    ))

    # Add legend entries using Scatter3d (these display properly in legends)
    fig.add_trace(go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode='markers',
        marker=dict(size=10, color='#4575b4', symbol='square'),  # Blue from RdBu colorscale
        name='Force Ratio Surface' if language == 'en' else 'Kraftverhältnis',
        showlegend=True
    ))

    fig.add_trace(go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode='markers',
        marker=dict(size=10, color='gray', symbol='square', opacity=0.5),
        name='F_em = F_grav (equality plane)' if language == 'en' else 'F_em = F_grav (Gleichheitsebene)',
        showlegend=True
    ))

    # Update layout
    if language == 'de':
        title = 'Verhältnis elektromagnetischer zu gravitativer Kraft<br><sup>Abhängigkeit von der Teilchenmasse</sup>'
        x_title = 'log₁₀(Masse / kg)'
        y_title = 'log₁₀(Abstand / m)'
        z_title = 'log₁₀(F_em / F_grav)'
    else:
        title = 'Electromagnetic to Gravitational Force Ratio<br><sup>Dependence on particle mass</sup>'
        x_title = 'log₁₀(Mass / kg)'
        y_title = 'log₁₀(Distance / m)'
        z_title = 'log₁₀(F_em / F_grav)'

    fig.update_layout(
        title=dict(text=title, x=0.5),
        scene=dict(
            xaxis_title=x_title,
            yaxis_title=y_title,
            zaxis_title=z_title,
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
            domain=dict(x=[0, 1], y=[0.18, 1])  # Full width, more vertical space for graph
        ),
        height=900,
        margin=dict(l=0, r=0, t=50, b=10),  # Minimal margins for max graph space
        template='plotly_white',
        showlegend=True,
        legend=dict(
            x=0.5,
            y=0.12,
            xanchor='center',
            yanchor='top',
            bgcolor='rgba(255,255,255,0.95)',
            bordercolor='rgba(180,180,180,0.8)',
            borderwidth=1,
            font=dict(size=14),
            orientation='v',  # Vertical - one item per row for readability
            entrywidth=0,  # Auto-size based on text width
            entrywidthmode='pixels'  # Use pixels mode for auto-sizing
        )
    )

    if save:
        os.makedirs(VIS_DIR, exist_ok=True)
        filepath = os.path.join(VIS_DIR, 'force_ratio_3d_interactive.html')
        fig.write_html(filepath, config={'displaylogo': False, 'displayModeBar': True})
        print(f"Saved: {filepath}")

    return fig


def plot_temperature_profile_3d_interactive(
    constants: Optional[PhysicalConstants] = None,
    language: str = 'en',
    save: bool = True
) -> 'go.Figure':
    """
    Create an interactive 3D visualization of temperature vs altitude and gravity.
    Erstellt eine interaktive 3D-Visualisierung von Temperatur vs. Hoehe und Gravitation.

    Shows how temperature varies with:
    - X-axis: Altitude (negative = depth into Earth, positive = atmosphere)
    - Y-axis: Gravity scale factor (1× to 100×)
    - Z-axis: Temperature (K)
    - Color: Temperature gradient (cold blue to hot red)

    Args:
        constants: Physical constants
        language: 'en' for English, 'de' for German
        save: Whether to save as HTML file

    Returns:
        Plotly Figure object
    """
    check_plotly()

    if constants is None:
        constants = get_constants()

    # Earth surface gravity
    g_earth = constants.G * constants.M_earth / constants.R_earth**2

    # Create meshgrid
    # Altitude: -50 km (depth) to +50 km (atmosphere)
    altitudes = np.linspace(-50, 50, 100)  # km
    g_scales = np.linspace(1, 100, 80)

    ALT, G_SCALE = np.meshgrid(altitudes, g_scales)

    # Calculate temperature at each point
    T = np.zeros_like(ALT)
    T_surface = constants.T_surface_earth  # 288 K
    T_core = constants.T_core_earth  # 5778 K
    c_p = constants.c_p_air  # J/(kg·K)

    for i, g_scale in enumerate(g_scales):
        g = g_earth * g_scale
        lapse_rate = -g / c_p  # K/m (negative = cooling with altitude)

        for j, alt_km in enumerate(altitudes):
            alt_m = alt_km * 1000  # Convert to meters

            if alt_m >= 0:
                # Atmosphere: temperature decreases with altitude (adiabatic lapse)
                T_val = T_surface + lapse_rate * alt_m
                T_val = max(T_val, 2.7)  # Cosmic background minimum
            else:
                # Interior: temperature increases with depth
                # Simplified model: linear increase to core temperature
                # Depth fraction (0 at surface, 1 at core)
                depth_fraction = min(abs(alt_m) / constants.R_earth, 1.0)
                # Core temperature scales with gravity (gravitational compression)
                T_core_scaled = T_core * g_scale**(1/3)
                T_val = T_surface + depth_fraction * (T_core_scaled - T_surface)

            T[i, j] = T_val

    # Clip temperature for visualization
    T = np.clip(T, 0, 25000)

    # Create figure
    fig = go.Figure()

    # Add temperature surface
    fig.add_trace(go.Surface(
        x=ALT,
        y=G_SCALE,
        z=T,
        colorscale='RdBu_r',  # Red-Blue reversed (blue=cold, red=hot)
        showscale=True,
        colorbar=dict(
            title=dict(text='Temperature [K]' if language == 'en' else 'Temperatur [K]', side='bottom', font=dict(size=13)),
            orientation='h',
            x=0.5,
            y=0.02,
            xanchor='center',
            yanchor='top',
            len=0.5,
            thickness=18,
            tickfont=dict(size=11)
        ),
        opacity=0.95,
        contours=dict(
            z=dict(show=True, usecolormap=True, highlightcolor="white", project_z=False)
        ),
        showlegend=False,
        name='Temperature Surface' if language == 'en' else 'Temperaturoberfläche'
    ))

    # Add reference lines at key gravity levels for the legend
    ref_g_scales = [1, 10, 50, 100]
    ref_colors = [COLORS['standard'], COLORS['primary_blue'], COLORS['primary_amber'], COLORS['scaled']]

    for g_scale, color in zip(ref_g_scales, ref_colors):
        # Temperature profile at this gravity
        g = g_earth * g_scale
        lapse_rate = -g / c_p

        T_profile = []
        for alt_km in altitudes:
            alt_m = alt_km * 1000
            if alt_m >= 0:
                T_val = T_surface + lapse_rate * alt_m
                T_val = max(T_val, 2.7)
            else:
                depth_fraction = min(abs(alt_m) / constants.R_earth, 1.0)
                T_core_scaled = T_core * g_scale**(1/3)
                T_val = T_surface + depth_fraction * (T_core_scaled - T_surface)
            T_profile.append(T_val)

        fig.add_trace(go.Scatter3d(
            x=altitudes,
            y=np.full(len(altitudes), g_scale),
            z=T_profile,
            mode='lines',
            line=dict(color=color, width=5),
            name=f'{g_scale}× g',
            showlegend=True
        ))

    # Add vertical line at altitude=0 (surface boundary)
    alt_zero = np.zeros(len(g_scales))
    T_at_surface = np.full(len(g_scales), T_surface)
    fig.add_trace(go.Scatter3d(
        x=alt_zero,
        y=g_scales,
        z=T_at_surface,
        mode='lines',
        line=dict(color='black', width=4, dash='dash'),
        name='Surface (z=0)' if language == 'en' else 'Oberfläche (z=0)',
        showlegend=True
    ))

    # Update layout
    if language == 'de':
        title = 'Temperatur vs. Höhe & Gravitation<br><sup>3D interaktiv - Atmosphäre (rechts) und Erdinneres (links)</sup>'
        x_title = 'Höhe [km] (negativ = Tiefe)'
        y_title = 'Gravitationsskala (× g)'
        z_title = 'Temperatur [K]'
    else:
        title = 'Temperature vs. Altitude & Gravity<br><sup>3D Interactive - Atmosphere (right) and Earth interior (left)</sup>'
        x_title = 'Altitude [km] (negative = depth)'
        y_title = 'Gravity Scale (× g)'
        z_title = 'Temperature [K]'

    # 6 legend items (5 lines + surface label) need more vertical space
    fig.update_layout(
        title=dict(text=title, x=0.5),
        scene=dict(
            xaxis_title=x_title,
            yaxis_title=y_title,
            zaxis_title=z_title,
            camera=dict(eye=dict(x=1.5, y=-1.8, z=1.0)),
            aspectmode='manual',
            aspectratio=dict(x=1.2, y=1, z=0.7),
            domain=dict(x=[0, 1], y=[0.30, 1])  # More bottom space for 6 legend items
        ),
        height=900,
        margin=dict(l=0, r=0, t=50, b=10),
        template='plotly_white',
        showlegend=True,
        legend=dict(
            x=0.5,
            y=0.23,  # Higher position to avoid colorbar
            xanchor='center',
            yanchor='top',
            bgcolor='rgba(255,255,255,0.95)',
            bordercolor='rgba(180,180,180,0.8)',
            borderwidth=1,
            font=dict(size=14),
            orientation='v',
            entrywidth=0,
            entrywidthmode='pixels'
        )
    )

    if save:
        os.makedirs(VIS_DIR, exist_ok=True)
        filepath = os.path.join(VIS_DIR, 'temperature_profile_3d_interactive.html')
        fig.write_html(filepath, config={'displaylogo': False, 'displayModeBar': True})
        print(f"Saved: {filepath}")

    return fig


def plot_light_bending_3d_interactive(
    constants: Optional[PhysicalConstants] = None,
    language: str = 'en',
    save: bool = True
) -> 'go.Figure':
    """
    Create an interactive 3D visualization of light bending around a massive object.
    Erstellt eine interaktive 3D-Visualisierung der Lichtbeugung um ein massives Objekt.

    Shows multiple light rays bending around a central mass in 3D space.

    Args:
        constants: Physical constants
        language: 'en' for English, 'de' for German
        save: Whether to save as HTML file

    Returns:
        Plotly Figure object
    """
    check_plotly()

    if constants is None:
        constants = get_constants()

    fig = go.Figure()

    # Central mass (sphere)
    # Create a sphere for the black hole / massive object
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    r_mass = 1.0  # Event horizon radius

    x_sphere = r_mass * np.outer(np.cos(u), np.sin(v))
    y_sphere = r_mass * np.outer(np.sin(u), np.sin(v))
    z_sphere = r_mass * np.outer(np.ones(np.size(u)), np.cos(v))

    fig.add_trace(go.Surface(
        x=x_sphere, y=y_sphere, z=z_sphere,
        colorscale=[[0, 'black'], [1, 'black']],
        showscale=False,
        opacity=1.0,
        name='Event Horizon' if language == 'en' else 'Ereignishorizont',
        showlegend=False
    ))

    # Add legend entry for mass
    fig.add_trace(go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode='markers',
        marker=dict(size=10, color='black', symbol='circle'),
        name='Event Horizon (R_s)' if language == 'en' else 'Ereignishorizont (R_s)',
        showlegend=True
    ))

    # Photon sphere (transparent sphere at 1.5 R_s)
    r_photon = 1.5
    x_photon = r_photon * np.outer(np.cos(u), np.sin(v))
    y_photon = r_photon * np.outer(np.sin(u), np.sin(v))
    z_photon = r_photon * np.outer(np.ones(np.size(u)), np.cos(v))

    fig.add_trace(go.Surface(
        x=x_photon, y=y_photon, z=z_photon,
        colorscale=[[0, COLORS['highlight']], [1, COLORS['highlight']]],
        showscale=False,
        opacity=0.15,
        name='Photon Sphere' if language == 'en' else 'Photonensphäre',
        showlegend=False
    ))

    # Calculate and plot light ray paths in 3D
    # Different impact parameters and entry angles (phi)
    impact_params = [2.0, 2.5, 3.0, 4.0, 5.0, 7.0]
    phi_angles = [0, np.pi/2, np.pi, 3*np.pi/2]  # Entry from different directions

    colors_rays = ['#1f77b4', '#2ca02c', '#9467bd', '#17becf']

    for phi_idx, phi in enumerate(phi_angles):
        for b in impact_params:
            # Starting position (far away, coming toward center)
            n_points = 150
            x_ray = np.zeros(n_points)
            y_ray = np.zeros(n_points)
            z_ray = np.zeros(n_points)

            # Initial conditions
            # Ray starts at x = -20, with y = b*cos(phi), z = b*sin(phi)
            x_ray[0] = -20
            y_ray[0] = b * np.cos(phi)
            z_ray[0] = b * np.sin(phi)

            # Initial velocity (toward +x direction)
            vx, vy, vz = 1.0, 0.0, 0.0
            dt = 0.15

            for i in range(1, n_points):
                # Current position
                r = np.sqrt(x_ray[i-1]**2 + y_ray[i-1]**2 + z_ray[i-1]**2)

                if r < 1.2:  # Inside photon sphere - stop
                    x_ray[i:] = x_ray[i-1]
                    y_ray[i:] = y_ray[i-1]
                    z_ray[i:] = z_ray[i-1]
                    break

                # Gravitational acceleration (simplified)
                a_mag = 0.5 / (r**2)

                # Radial unit vector
                rx = x_ray[i-1] / r
                ry = y_ray[i-1] / r
                rz = z_ray[i-1] / r

                # Acceleration toward center
                ax = -a_mag * rx
                ay = -a_mag * ry
                az = -a_mag * rz

                # Update velocity
                vx += ax * dt
                vy += ay * dt
                vz += az * dt

                # Normalize (light travels at c)
                v_mag = np.sqrt(vx**2 + vy**2 + vz**2)
                vx /= v_mag
                vy /= v_mag
                vz /= v_mag

                # Update position
                x_ray[i] = x_ray[i-1] + vx * dt
                y_ray[i] = y_ray[i-1] + vy * dt
                z_ray[i] = z_ray[i-1] + vz * dt

                # Stop if ray escapes
                if abs(x_ray[i]) > 20 or abs(y_ray[i]) > 20 or abs(z_ray[i]) > 20:
                    break

            # Plot the ray
            fig.add_trace(go.Scatter3d(
                x=x_ray[:i+1], y=y_ray[:i+1], z=z_ray[:i+1],
                mode='lines',
                line=dict(color=colors_rays[phi_idx], width=3),
                opacity=0.7,
                showlegend=False
            ))

    # Add legend entries for rays from different directions
    ray_labels = ['From -X', 'From -Y', 'From +X', 'From +Y'] if language == 'en' else ['Von -X', 'Von -Y', 'Von +X', 'Von +Y']
    for i, (color, label) in enumerate(zip(colors_rays, ray_labels)):
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode='lines',
            line=dict(color=color, width=4),
            name=f'Light rays ({label})' if language == 'en' else f'Lichtstrahlen ({label})',
            showlegend=True
        ))

    # Add straight reference ray (no bending)
    fig.add_trace(go.Scatter3d(
        x=np.linspace(-20, 20, 50),
        y=np.full(50, 12),
        z=np.zeros(50),
        mode='lines',
        line=dict(color='gray', width=2, dash='dash'),
        opacity=0.5,
        name='No gravity (straight)' if language == 'en' else 'Ohne Gravitation (gerade)',
        showlegend=True
    ))

    # Update layout
    if language == 'de':
        title = 'Lichtablenkung durch Gravitation (3D)<br><sup>Drehen Sie die Ansicht mit der Maus</sup>'
    else:
        title = 'Light Bending by Gravity (3D)<br><sup>Rotate the view with your mouse</sup>'

    fig.update_layout(
        title=dict(text=title, x=0.5),
        scene=dict(
            xaxis_title='X (Schwarzschild radii)' if language == 'en' else 'X (Schwarzschild-Radien)',
            yaxis_title='Y (Schwarzschild radii)' if language == 'en' else 'Y (Schwarzschild-Radien)',
            zaxis_title='Z (Schwarzschild radii)' if language == 'en' else 'Z (Schwarzschild-Radien)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
            aspectmode='data',
            domain=dict(x=[0, 1], y=[0.22, 1])
        ),
        height=900,
        margin=dict(l=0, r=0, t=50, b=10),
        template='plotly_white',
        showlegend=True,
        legend=dict(
            x=0.5,
            y=0.15,
            xanchor='center',
            yanchor='top',
            bgcolor='rgba(255,255,255,0.95)',
            bordercolor='rgba(180,180,180,0.8)',
            borderwidth=1,
            font=dict(size=12),
            orientation='h',
            entrywidth=0,
            entrywidthmode='pixels'
        )
    )

    if save:
        os.makedirs(VIS_DIR, exist_ok=True)
        suffix = '_de' if language == 'de' else ''
        filepath = os.path.join(VIS_DIR, f'light_bending_3d_interactive{suffix}.html')
        fig.write_html(filepath, config={'displaylogo': False, 'displayModeBar': True})
        print(f"Saved: {filepath}")

    return fig


def plot_neutron_star_3d_interactive(
    constants: Optional[PhysicalConstants] = None,
    language: str = 'en',
    save: bool = True
) -> 'go.Figure':
    """
    Create an interactive 3D visualization of neutron star mass-radius-density relation.
    Erstellt eine interaktive 3D-Visualisierung der Neutronenstern Masse-Radius-Dichte-Beziehung.

    Args:
        constants: Physical constants
        language: 'en' for English, 'de' for German
        save: Whether to save as HTML file

    Returns:
        Plotly Figure object
    """
    check_plotly()

    if constants is None:
        constants = get_constants()

    # Import neutron star functions
    from .neutron_star import tov_mass_limit, calculate_neutron_star

    # Create data for neutron stars at different masses
    M_tov = tov_mass_limit(constants) / constants.M_sun
    masses = np.linspace(0.5, min(M_tov - 0.1, 2.2), 50)

    radii = []
    densities = []
    pressures = []

    for m in masses:
        ns = calculate_neutron_star(m, constants)
        radii.append(ns.radius_km)
        densities.append(np.log10(ns.density))
        pressures.append(np.log10(ns.pressure))

    fig = go.Figure()

    # 3D scatter plot of mass-radius-density
    fig.add_trace(go.Scatter3d(
        x=masses,
        y=radii,
        z=densities,
        mode='markers+lines',
        marker=dict(
            size=6,
            color=pressures,
            colorscale='Plasma',
            colorbar=dict(
                title=dict(
                    text='log₁₀(P) [Pa]' if language == 'en' else 'log₁₀(P) [Pa]',
                    side='bottom',
                    font=dict(size=12)
                ),
                orientation='h',
                x=0.5,
                y=0.02,
                xanchor='center',
                yanchor='top',
                len=0.5,
                thickness=18
            ),
            symbol='circle'
        ),
        line=dict(color=COLORS['neutron_star'], width=3),
        name='Neutron Star' if language == 'en' else 'Neutronenstern'
    ))

    # Add TOV limit plane
    x_plane = np.array([M_tov, M_tov])
    y_plane = np.array([8, 14])
    X_plane, Y_plane = np.meshgrid(x_plane, y_plane)
    Z_plane = np.ones_like(X_plane) * np.mean(densities)

    fig.add_trace(go.Surface(
        x=X_plane, y=Y_plane, z=Z_plane,
        opacity=0.3,
        colorscale=[[0, 'red'], [1, 'red']],
        showscale=False,
        name=f'TOV Limit ({M_tov:.2f} M☉)' if language == 'en' else f'TOV-Grenze ({M_tov:.2f} M☉)'
    ))

    # Add legend entry for TOV limit
    fig.add_trace(go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode='markers',
        marker=dict(size=10, color='red', symbol='square'),
        name=f'TOV Limit ({M_tov:.2f} M☉)' if language == 'en' else f'TOV-Grenze ({M_tov:.2f} M☉)',
        showlegend=True
    ))

    if language == 'de':
        title = 'Neutronenstern: Masse-Radius-Dichte<br><sup>Farbskala zeigt Zentraldruck</sup>'
        x_title = 'Masse (M☉)'
        y_title = 'Radius (km)'
        z_title = 'log₁₀(Dichte) [kg/m³]'
    else:
        title = 'Neutron Star: Mass-Radius-Density<br><sup>Color shows central pressure</sup>'
        x_title = 'Mass (M☉)'
        y_title = 'Radius (km)'
        z_title = 'log₁₀(Density) [kg/m³]'

    fig.update_layout(
        title=dict(text=title, x=0.5),
        scene=dict(
            xaxis_title=x_title,
            yaxis_title=y_title,
            zaxis_title=z_title,
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.8),
            domain=dict(x=[0, 1], y=[0.18, 1])
        ),
        height=900,
        margin=dict(l=0, r=0, t=50, b=10),
        template='plotly_white',
        legend=dict(
            x=0.5,
            y=0.12,
            xanchor='center',
            yanchor='top',
            bgcolor='rgba(255,255,255,0.95)',
            bordercolor='rgba(180,180,180,0.8)',
            borderwidth=1,
            font=dict(size=12),
            orientation='h'
        )
    )

    if save:
        os.makedirs(VIS_DIR, exist_ok=True)
        suffix = '_de' if language == 'de' else ''
        filepath = os.path.join(VIS_DIR, f'neutron_star_3d_interactive{suffix}.html')
        fig.write_html(filepath, config={'displaylogo': False, 'displayModeBar': True})
        print(f"Saved: {filepath}")

    return fig


def plot_heisenberg_3d_interactive(
    constants: Optional[PhysicalConstants] = None,
    language: str = 'en',
    save: bool = True
) -> 'go.Figure':
    """
    Create an interactive 3D visualization of Heisenberg uncertainty and degeneracy pressure.
    Erstellt eine interaktive 3D-Visualisierung der Heisenberg-Unschaerfe und Entartungsdruck.

    Shows how confinement (Δx) affects momentum uncertainty (Δp) and resulting pressure.

    Args:
        constants: Physical constants
        language: 'en' for English, 'de' for German
        save: Whether to save as HTML file

    Returns:
        Plotly Figure object
    """
    check_plotly()

    if constants is None:
        constants = get_constants()

    # Create mesh for Δx vs ℏ scaling
    delta_x = np.logspace(-15, -10, 40)  # fm to pm scale
    hbar_scale = np.logspace(-1, 1, 40)  # 0.1 to 10
    Delta_X, Hbar_Scale = np.meshgrid(delta_x, hbar_scale)

    # Calculate momentum uncertainty: Δp = ℏ / (2 * Δx)
    hbar_scaled = constants.hbar * Hbar_Scale
    Delta_P = hbar_scaled / (2 * Delta_X)

    # Calculate pressure contribution (proportional to Δp² / m)
    # Using log scale for visualization
    Pressure = np.log10(Delta_P**2 / constants.m_e)

    fig = go.Figure()

    # 3D surface of pressure vs confinement vs ℏ
    fig.add_trace(go.Surface(
        x=np.log10(Delta_X * 1e15),  # Convert to fm, then log
        y=np.log10(Hbar_Scale),
        z=Pressure,
        colorscale='Viridis',
        colorbar=dict(
            title=dict(
                text='log₁₀(P) [arb.]' if language == 'en' else 'log₁₀(P) [willk.]',
                side='bottom',
                font=dict(size=12)
            ),
            orientation='h',
            x=0.5,
            y=0.02,
            xanchor='center',
            yanchor='top',
            len=0.5,
            thickness=18
        ),
        name='Degeneracy Pressure' if language == 'en' else 'Entartungsdruck',
        showlegend=False
    ))

    # Add legend entry
    fig.add_trace(go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode='markers',
        marker=dict(size=10, color='#31688e', symbol='square'),
        name='P ∝ ℏ²/Δx²',
        showlegend=True
    ))

    # Mark standard universe point
    std_x = np.log10(constants.a_0 * 1e15)  # Bohr radius in fm (log)
    std_hbar = 0  # log10(1)
    std_pressure = np.log10((constants.hbar / (2 * constants.a_0))**2 / constants.m_e)

    fig.add_trace(go.Scatter3d(
        x=[std_x], y=[std_hbar], z=[std_pressure],
        mode='markers',
        marker=dict(size=12, color=COLORS['standard'], symbol='diamond'),
        name='Standard (Bohr radius)' if language == 'en' else 'Standard (Bohr-Radius)'
    ))

    if language == 'de':
        title = 'Heisenberg-Unschärfe → Entartungsdruck<br><sup>P ∝ ℏ²/Δx² (kleiner Δx = höherer Druck)</sup>'
        x_title = 'log₁₀(Δx) [fm]'
        y_title = 'log₁₀(ℏ/ℏ₀)'
        z_title = 'log₁₀(Druck)'
    else:
        title = 'Heisenberg Uncertainty → Degeneracy Pressure<br><sup>P ∝ ℏ²/Δx² (smaller Δx = higher pressure)</sup>'
        x_title = 'log₁₀(Δx) [fm]'
        y_title = 'log₁₀(ℏ/ℏ₀)'
        z_title = 'log₁₀(Pressure)'

    fig.update_layout(
        title=dict(text=title, x=0.5),
        scene=dict(
            xaxis_title=x_title,
            yaxis_title=y_title,
            zaxis_title=z_title,
            camera=dict(eye=dict(x=1.8, y=1.8, z=1.0)),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.6),
            domain=dict(x=[0, 1], y=[0.18, 1])
        ),
        height=900,
        margin=dict(l=0, r=0, t=50, b=10),
        template='plotly_white',
        legend=dict(
            x=0.5,
            y=0.12,
            xanchor='center',
            yanchor='top',
            bgcolor='rgba(255,255,255,0.95)',
            bordercolor='rgba(180,180,180,0.8)',
            borderwidth=1,
            font=dict(size=12),
            orientation='h'
        )
    )

    if save:
        os.makedirs(VIS_DIR, exist_ok=True)
        suffix = '_de' if language == 'de' else ''
        filepath = os.path.join(VIS_DIR, f'heisenberg_3d_interactive{suffix}.html')
        fig.write_html(filepath, config={'displaylogo': False, 'displayModeBar': True})
        print(f"Saved: {filepath}")

    return fig


def plot_time_dilation_3d_interactive(
    constants: Optional[PhysicalConstants] = None,
    language: str = 'en',
    save: bool = True
) -> 'go.Figure':
    """
    Create an interactive 3D visualization of gravitational time dilation.
    Erstellt eine interaktive 3D-Visualisierung der gravitativen Zeitdilatation.

    Shows time dilation factor vs distance and mass.

    Args:
        constants: Physical constants
        language: 'en' for English, 'de' for German
        save: Whether to save as HTML file

    Returns:
        Plotly Figure object
    """
    check_plotly()

    if constants is None:
        constants = get_constants()

    # Create mesh for distance vs mass
    # Distance in units of Schwarzschild radius
    r_over_Rs = np.linspace(1.1, 10, 50)
    # Mass in solar masses
    mass_solar = np.linspace(0.5, 3, 50)
    R_over_Rs, Mass_solar = np.meshgrid(r_over_Rs, mass_solar)

    # Calculate time dilation: τ/t = √(1 - 1/(r/R_s))
    # Since r/R_s is already our variable, this simplifies
    Time_dilation = np.sqrt(1 - 1/R_over_Rs)

    fig = go.Figure()

    # 3D surface
    fig.add_trace(go.Surface(
        x=R_over_Rs,
        y=Mass_solar,
        z=Time_dilation,
        colorscale='RdYlBu',
        colorbar=dict(
            title=dict(
                text='τ/t' if language == 'en' else 'τ/t',
                side='bottom',
                font=dict(size=12)
            ),
            orientation='h',
            x=0.5,
            y=0.02,
            xanchor='center',
            yanchor='top',
            len=0.5,
            thickness=18
        ),
        name='Time dilation' if language == 'en' else 'Zeitdilatation',
        showlegend=False
    ))

    # Add legend entry
    fig.add_trace(go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode='markers',
        marker=dict(size=10, color='#4575b4', symbol='square'),
        name='τ/t = √(1 - R_s/r)',
        showlegend=True
    ))

    # Mark neutron star surface (r/R_s ≈ 2.4 for typical NS)
    ns_r_over_Rs = 2.4
    ns_dilation = np.sqrt(1 - 1/ns_r_over_Rs)
    fig.add_trace(go.Scatter3d(
        x=[ns_r_over_Rs], y=[1.4], z=[ns_dilation],
        mode='markers',
        marker=dict(size=12, color=COLORS['quantum'], symbol='diamond'),
        name=f'Neutron Star (~{(1-ns_dilation)*100:.0f}% slower)' if language == 'en'
             else f'Neutronenstern (~{(1-ns_dilation)*100:.0f}% langsamer)'
    ))

    # Mark event horizon
    fig.add_trace(go.Scatter3d(
        x=[1.1], y=[1.5], z=[0.3],
        mode='markers',
        marker=dict(size=10, color='red', symbol='x'),
        name='Near event horizon' if language == 'en' else 'Nahe Ereignishorizont'
    ))

    if language == 'de':
        title = 'Gravitationelle Zeitdilatation<br><sup>τ/t = √(1 - R_s/r) — Zeit verlangsamt sich nahe Masse</sup>'
        x_title = 'r / R_s (Abstand / Schwarzschild-Radius)'
        y_title = 'Masse (M☉)'
        z_title = 'Zeitdilatation τ/t'
    else:
        title = 'Gravitational Time Dilation<br><sup>τ/t = √(1 - R_s/r) — Time slows near mass</sup>'
        x_title = 'r / R_s (distance / Schwarzschild radius)'
        y_title = 'Mass (M☉)'
        z_title = 'Time dilation τ/t'

    fig.update_layout(
        title=dict(text=title, x=0.5),
        scene=dict(
            xaxis_title=x_title,
            yaxis_title=y_title,
            zaxis_title=z_title,
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.6),
            domain=dict(x=[0, 1], y=[0.18, 1])
        ),
        height=900,
        margin=dict(l=0, r=0, t=50, b=10),
        template='plotly_white',
        legend=dict(
            x=0.5,
            y=0.12,
            xanchor='center',
            yanchor='top',
            bgcolor='rgba(255,255,255,0.95)',
            bordercolor='rgba(180,180,180,0.8)',
            borderwidth=1,
            font=dict(size=12),
            orientation='h'
        )
    )

    if save:
        os.makedirs(VIS_DIR, exist_ok=True)
        suffix = '_de' if language == 'de' else ''
        filepath = os.path.join(VIS_DIR, f'time_dilation_3d_interactive{suffix}.html')
        fig.write_html(filepath, config={'displaylogo': False, 'displayModeBar': True})
        print(f"Saved: {filepath}")

    return fig


def plot_gravity_pauli_3d_interactive(
    constants: Optional[PhysicalConstants] = None,
    language: str = 'en',
    save: bool = True
) -> 'go.Figure':
    """
    Create an interactive 3D visualization of gravity vs Pauli pressure balance.
    Erstellt eine interaktive 3D-Visualisierung des Gravitation-vs-Pauli-Druck-Gleichgewichts.

    Shows how the pressure ratio P_grav/P_Pauli varies with G and ℏ scaling.

    Args:
        constants: Physical constants
        language: 'en' for English, 'de' for German
        save: Whether to save as HTML file

    Returns:
        Plotly Figure object
    """
    check_plotly()

    if constants is None:
        constants = get_constants()

    # Create mesh for G scaling vs ℏ scaling
    G_scale = np.logspace(0, 20, 50)
    hbar_scale = np.logspace(0, 12, 50)
    G_Scale, Hbar_Scale = np.meshgrid(G_scale, hbar_scale)

    # Pressure ratio: P_grav/P_Pauli ∝ G/ℏ²
    # Using log scale
    Pressure_ratio = np.log10(G_Scale / Hbar_Scale**2)

    fig = go.Figure()

    # 3D surface
    fig.add_trace(go.Surface(
        x=np.log10(G_Scale),
        y=np.log10(Hbar_Scale),
        z=Pressure_ratio,
        colorscale='RdBu_r',
        colorbar=dict(
            title=dict(
                text='log₁₀(P_g/P_p)' if language == 'en' else 'log₁₀(P_g/P_p)',
                side='bottom',
                font=dict(size=12)
            ),
            orientation='h',
            x=0.5,
            y=0.02,
            xanchor='center',
            yanchor='top',
            len=0.5,
            thickness=18
        ),
        name='Pressure ratio' if language == 'en' else 'Druckverhältnis',
        showlegend=False
    ))

    # Add equilibrium plane (log ratio = 0)
    x_plane = np.log10(G_scale)
    y_plane = np.log10(hbar_scale)
    X_eq, Y_eq = np.meshgrid(x_plane, y_plane)
    Z_eq = np.zeros_like(X_eq)

    fig.add_trace(go.Surface(
        x=X_eq, y=Y_eq, z=Z_eq,
        opacity=0.3,
        colorscale=[[0, 'green'], [1, 'green']],
        showscale=False,
        name='Equilibrium'
    ))

    # Legend entries
    fig.add_trace(go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode='markers',
        marker=dict(size=10, color='#b2182b', symbol='square'),
        name='P_grav/P_Pauli ∝ G/ℏ²',
        showlegend=True
    ))

    fig.add_trace(go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode='markers',
        marker=dict(size=10, color='green', symbol='square', opacity=0.5),
        name='Balance (ratio = 1)' if language == 'en' else 'Gleichgewicht (Verhältnis = 1)',
        showlegend=True
    ))

    # Mark standard universe
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        marker=dict(size=12, color=COLORS['standard'], symbol='diamond'),
        name='Standard universe' if language == 'en' else 'Standarduniversum'
    ))

    # Mark essay scenario (G × 10^36, ℏ × 10^18)
    fig.add_trace(go.Scatter3d(
        x=[18], y=[9], z=[0],  # log10(10^36)/2 and log10(10^18) to get ratio of 1
        mode='markers',
        marker=dict(size=12, color=COLORS['quantum'], symbol='cross'),
        name='Essay scenario (G×10³⁶, ℏ×10¹⁸)' if language == 'en'
             else 'Essay-Szenario (G×10³⁶, ℏ×10¹⁸)'
    ))

    if language == 'de':
        title = 'Gravitation vs. Pauli: Stabilitätslandschaft<br><sup>Rot = Kollaps, Blau = Stabil, Grün = Gleichgewicht</sup>'
        x_title = 'log₁₀(G/G₀)'
        y_title = 'log₁₀(ℏ/ℏ₀)'
        z_title = 'log₁₀(P_grav/P_Pauli)'
    else:
        title = 'Gravity vs. Pauli: Stability Landscape<br><sup>Red = Collapse, Blue = Stable, Green = Balance</sup>'
        x_title = 'log₁₀(G/G₀)'
        y_title = 'log₁₀(ℏ/ℏ₀)'
        z_title = 'log₁₀(P_grav/P_Pauli)'

    fig.update_layout(
        title=dict(text=title, x=0.5),
        scene=dict(
            xaxis_title=x_title,
            yaxis_title=y_title,
            zaxis_title=z_title,
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.6),
            domain=dict(x=[0, 1], y=[0.18, 1])
        ),
        height=900,
        margin=dict(l=0, r=0, t=50, b=10),
        template='plotly_white',
        legend=dict(
            x=0.5,
            y=0.12,
            xanchor='center',
            yanchor='top',
            bgcolor='rgba(255,255,255,0.95)',
            bordercolor='rgba(180,180,180,0.8)',
            borderwidth=1,
            font=dict(size=12),
            orientation='h'
        )
    )

    if save:
        os.makedirs(VIS_DIR, exist_ok=True)
        suffix = '_de' if language == 'de' else ''
        filepath = os.path.join(VIS_DIR, f'gravity_pauli_3d_interactive{suffix}.html')
        fig.write_html(filepath, config={'displaylogo': False, 'displayModeBar': True})
        print(f"Saved: {filepath}")

    return fig


def generate_all_interactive_plots(language: str = 'en') -> List['go.Figure']:
    """
    Generate all interactive 3D visualizations.
    Erzeugt alle interaktiven 3D-Visualisierungen.

    Args:
        language: 'en' for English, 'de' for German

    Returns:
        List of Plotly Figure objects
    """
    check_plotly()

    figures = []

    print("Generating interactive 3D visualizations...")
    print("=" * 50)

    print("1. Spacetime curvature 3D...")
    figures.append(plot_spacetime_curvature_3d_interactive(language=language))

    print("2. Multiple masses comparison 3D...")
    figures.append(plot_multiple_masses_3d_interactive(language=language))

    print("3. Atom scaling 3D...")
    figures.append(plot_atom_scaling_3d_interactive(language=language))

    print("4. Force ratio 3D surface...")
    figures.append(plot_force_ratio_3d_interactive(language=language))

    print("5. Temperature profile 3D...")
    figures.append(plot_temperature_profile_3d_interactive(language=language))

    print("6. Light bending 3D...")
    figures.append(plot_light_bending_3d_interactive(language=language))

    print("7. Neutron star 3D...")
    figures.append(plot_neutron_star_3d_interactive(language=language))

    print("8. Heisenberg uncertainty 3D...")
    figures.append(plot_heisenberg_3d_interactive(language=language))

    print("9. Time dilation 3D...")
    figures.append(plot_time_dilation_3d_interactive(language=language))

    print("10. Gravity vs Pauli 3D...")
    figures.append(plot_gravity_pauli_3d_interactive(language=language))

    print("=" * 50)
    print(f"Generated {len(figures)} interactive visualizations")
    print("Open the .html files in a web browser to interact with them!")

    return figures


if __name__ == "__main__":
    print("=" * 60)
    print("Interactive 3D Visualizations - Jugend forscht 2026")
    print("=" * 60)

    if PLOTLY_AVAILABLE:
        generate_all_interactive_plots(language='en')
    else:
        print("\nPlotly is not installed. Install with:")
        print("  pip install plotly")
