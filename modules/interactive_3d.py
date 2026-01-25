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
            y=0.01,  # Move down to avoid legend overlap
            xanchor='center',
            yanchor='top',
            len=0.5,
            thickness=15  # Slightly thinner
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

    fig.update_layout(
        title=dict(text=title, x=0.5),
        scene=dict(
            xaxis_title='x',
            yaxis_title='y',
            zaxis_title='Curvature' if language == 'en' else 'Kruemmung',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.5),
            domain=dict(x=[0, 1], y=[0.18, 1])  # Same as spacetime_3d
        ),
        height=900,
        margin=dict(l=0, r=0, t=50, b=10),
        template='plotly_white',
        showlegend=True,
        legend=dict(
            x=0.5,
            y=0.14,  # Moved up slightly
            xanchor='center',
            yanchor='top',
            bgcolor='rgba(255,255,255,0.95)',
            bordercolor='rgba(180,180,180,0.8)',
            borderwidth=1,
            font=dict(size=12),
            orientation='h',  # Horizontal layout to save vertical space
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

    fig.update_layout(
        title=dict(text=title, x=0.5),
        scene=dict(
            xaxis_title='x (relative)',
            yaxis_title='y (relative)',
            zaxis_title='Universe',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            aspectmode='data',
            domain=dict(x=[0, 1], y=[0.18, 1])  # Full width, same as force_ratio
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

    fig.update_layout(
        title=dict(text=title, x=0.5),
        scene=dict(
            xaxis_title=x_title,
            yaxis_title=y_title,
            zaxis_title=z_title,
            camera=dict(eye=dict(x=1.5, y=-1.8, z=1.0)),
            aspectmode='manual',
            aspectratio=dict(x=1.2, y=1, z=0.7),
            domain=dict(x=[0, 1], y=[0.18, 1])  # Full width, same as force_ratio
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
            orientation='v',  # Vertical - one item per row
            entrywidth=0,  # Auto-size based on text width
            entrywidthmode='pixels'  # Use pixels mode for auto-sizing
        )
    )

    if save:
        os.makedirs(VIS_DIR, exist_ok=True)
        filepath = os.path.join(VIS_DIR, 'temperature_profile_3d_interactive.html')
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
