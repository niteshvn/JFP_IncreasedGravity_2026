"""
Atomic Scale Effects Module for Jugend forscht 2026 Physics Visualization Project
Atomare-Skalen-Effekte-Modul fuer Jugend forscht 2026 Physik-Visualisierungsprojekt

This module visualizes atomic-scale physics including:
- Bohr radius and its dependence on fundamental constants
- How atoms would shrink/expand in alternative universes
- Electron energy levels and orbital structure
- The interplay between quantum mechanics (hbar) and electromagnetism

Key insight: The Bohr radius a_0 = 4πε₀ℏ²/(m_e × e²) determines atomic size.
When ℏ decreases, atoms shrink, and gravity becomes relatively more important.

Author: Jugend forscht 2026 Project
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.collections import PatchCollection
import os
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass

from .constants import get_constants, PhysicalConstants
from .color_scheme import COLORS, get_stellar_colors, get_sequence


# Output directory for visualizations
VIS_DIR = os.path.join(os.path.dirname(__file__), '..', 'visualizations')


@dataclass
class AtomicProperties:
    """
    Container for atomic properties.
    Behaelter fuer atomare Eigenschaften.
    """
    bohr_radius: float           # Bohr radius [m]
    ground_state_energy: float   # Ground state energy [J]
    ionization_energy: float     # Ionization energy [eV]
    orbital_velocity: float      # Electron orbital velocity [m/s]
    fine_structure_constant: float  # α ≈ 1/137


def calculate_atomic_properties(constants: PhysicalConstants) -> AtomicProperties:
    """
    Calculate atomic properties for hydrogen atom.
    Berechnet atomare Eigenschaften fuer Wasserstoffatom.

    Args:
        constants: Physical constants

    Returns:
        AtomicProperties object
    """
    # Bohr radius: a_0 = 4πε₀ℏ²/(m_e × e²) = ℏ/(m_e × c × α)
    a_0 = constants.a_0

    # Ground state energy: E_1 = -13.6 eV = -m_e × c² × α² / 2
    E_1_J = -constants.m_e * constants.c**2 * constants.alpha**2 / 2
    E_1_eV = abs(E_1_J) / constants.e  # Convert to eV (positive for ionization)

    # Ionization energy (same as |E_1|)
    ionization_eV = E_1_eV

    # Orbital velocity in ground state: v_1 = α × c
    v_orbital = constants.alpha * constants.c

    return AtomicProperties(
        bohr_radius=a_0,
        ground_state_energy=E_1_J,
        ionization_energy=ionization_eV,
        orbital_velocity=v_orbital,
        fine_structure_constant=constants.alpha
    )


def energy_level(n: int, constants: PhysicalConstants) -> float:
    """
    Calculate hydrogen energy level for principal quantum number n.
    Berechnet Wasserstoff-Energieniveau fuer Hauptquantenzahl n.

    Formula: E_n = -13.6 eV / n²

    Args:
        n: Principal quantum number (1, 2, 3, ...)
        constants: Physical constants

    Returns:
        Energy in eV (negative, as bound states)
    """
    # E_n = -m_e × c² × α² / (2n²)
    E_n_J = -constants.m_e * constants.c**2 * constants.alpha**2 / (2 * n**2)
    E_n_eV = E_n_J / constants.e
    return E_n_eV


def orbital_radius(n: int, constants: PhysicalConstants) -> float:
    """
    Calculate orbital radius for principal quantum number n.
    Berechnet Orbitalradius fuer Hauptquantenzahl n.

    Formula: r_n = n² × a_0

    Args:
        n: Principal quantum number
        constants: Physical constants

    Returns:
        Orbital radius [m]
    """
    return n**2 * constants.a_0


def plot_bohr_radius_scaling(
    constants: Optional[PhysicalConstants] = None,
    language: str = 'en',
    save: bool = True,
    show: bool = True
) -> plt.Figure:
    """
    Plot how Bohr radius changes with hbar scaling.
    Zeigt, wie sich der Bohr-Radius mit hbar-Skalierung aendert.

    Key insight: a_0 ∝ ℏ², so reducing ℏ dramatically shrinks atoms.

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

    # Range of hbar scaling factors
    hbar_scales = np.logspace(-2, 1, 100)  # 0.01 to 10

    # Calculate Bohr radii for each scaling
    # a_0 ∝ ℏ², so a_0(scaled) = a_0(standard) × hbar_scale²
    a_0_standard = constants.a_0
    a_0_scaled = a_0_standard * hbar_scales**2

    # Also calculate alpha_G for each scaling
    # alpha_G ∝ 1/ℏ, so alpha_G(scaled) = alpha_G(standard) / hbar_scale
    alpha_G_standard = constants.alpha_G
    alpha_G_scaled = alpha_G_standard / hbar_scales

    # Create figure with two subplots stacked vertically
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14), gridspec_kw={'hspace': 0.4})

    # Top plot: Bohr radius vs hbar scaling
    ax1.loglog(hbar_scales, a_0_scaled * 1e12, '-', color=COLORS['primary_blue'], linewidth=2.5,
               label='Bohr radius a_0' if language == 'en' else 'Bohr-Radius a_0')

    # Mark standard universe
    ax1.axvline(x=1, color=COLORS['standard'], linestyle='--', linewidth=1.5, alpha=0.7,
               label='Standard universe' if language == 'en' else 'Standarduniversum')
    ax1.axhline(y=a_0_standard * 1e12, color=COLORS['standard'], linestyle=':', linewidth=1, alpha=0.5)

    # Mark some interesting points
    ax1.plot(1, a_0_standard * 1e12, 'o', color=COLORS['standard'], markersize=10)
    ax1.plot(0.1, a_0_standard * 0.01 * 1e12, 'o', color=COLORS['scaled'], markersize=8)
    ax1.plot(0.01, a_0_standard * 0.0001 * 1e12, 'o', color=COLORS['quantum'], markersize=8)

    if language == 'de':
        ax1.set_xlabel('hbar-Skalierungsfaktor', fontsize=12)
        ax1.set_ylabel('Bohr-Radius (pm)', fontsize=12)
        ax1.set_title('1. Bohr-Radius vs. hbar-Skalierung (a_0 ∝ ℏ²)', fontsize=14, fontweight='bold', pad=15)
    else:
        ax1.set_xlabel('ℏ scaling factor', fontsize=12)
        ax1.set_ylabel('Bohr radius (pm)', fontsize=12)
        ax1.set_title('1. Bohr Radius vs. ℏ Scaling (a_0 ∝ ℏ²)', fontsize=14, fontweight='bold', pad=15)

    ax1.legend(fontsize=9, loc='upper right', bbox_to_anchor=(1.0, -0.08), framealpha=0.7)
    ax1.grid(True, alpha=0.3, which='both')

    # Bottom plot: Gravitational coupling constant vs hbar scaling
    ax2.loglog(hbar_scales, alpha_G_scaled, '-', color=COLORS['scaled'], linewidth=2.5,
               label='α_G (gravity strength)' if language == 'en' else 'α_G (Gravitationsstaerke)')

    ax2.axvline(x=1, color=COLORS['standard'], linestyle='--', linewidth=1.5, alpha=0.7,
               label='Standard universe' if language == 'en' else 'Standarduniversum')
    ax2.axhline(y=alpha_G_standard, color=COLORS['standard'], linestyle=':', linewidth=1, alpha=0.5)

    ax2.plot(1, alpha_G_standard, 'o', color=COLORS['standard'], markersize=10)
    ax2.plot(0.1, alpha_G_standard * 10, 'o', color=COLORS['scaled'], markersize=8)

    if language == 'de':
        ax2.set_xlabel('hbar-Skalierungsfaktor', fontsize=12)
        ax2.set_ylabel('Gravitationskopplungskonstante α_G', fontsize=12)
        ax2.set_title('2. Gravitations-Kopplungskonstante vs. hbar (α_G ∝ 1/ℏ)', fontsize=14, fontweight='bold', pad=15)
    else:
        ax2.set_xlabel('ℏ scaling factor', fontsize=12)
        ax2.set_ylabel('Gravitational coupling constant α_G', fontsize=12)
        ax2.set_title('2. Gravitational Coupling vs. ℏ (α_G ∝ 1/ℏ)', fontsize=14, fontweight='bold', pad=15)

    ax2.legend(fontsize=9, loc='upper right', bbox_to_anchor=(1.0, -0.08), framealpha=0.7)
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()

    if save:
        os.makedirs(VIS_DIR, exist_ok=True)
        filepath = os.path.join(VIS_DIR, 'bohr_radius_scaling.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")

    if show:
        plt.show()

    return fig


def plot_atom_size_comparison(
    hbar_scales: List[float] = [1.0, 0.5, 0.1],
    constants: Optional[PhysicalConstants] = None,
    language: str = 'en',
    save: bool = True,
    show: bool = True
) -> plt.Figure:
    """
    Visualize atom sizes in different universes.
    Visualisiert Atomgroessen in verschiedenen Universen.

    Shows how atoms shrink when hbar is reduced.

    Args:
        hbar_scales: List of hbar scaling factors to compare
        constants: Physical constants
        language: 'en' for English, 'de' for German
        save: Whether to save the figure
        show: Whether to display the figure

    Returns:
        matplotlib Figure object
    """
    if constants is None:
        constants = get_constants()

    # Create figure - use a tall aspect ratio
    fig, ax = plt.subplots(figsize=(12, 10))

    # Standard Bohr radius
    a_0_std = constants.a_0

    # Use consistent colors from color scheme
    universe_colors = [COLORS['standard'], COLORS['primary_blue'], COLORS['scaled']]

    # Fixed y-positions for each atom (evenly spaced, from top to bottom)
    y_positions = [0.75, 0.45, 0.15]
    x_center = 0.25

    # Maximum display radius (for the standard/largest atom)
    max_r_display = 0.18

    # Minimum orbital radius so small atoms are still visible
    min_r_display = 0.02

    for i, (scale, color, y_pos) in enumerate(zip(hbar_scales, universe_colors, y_positions)):
        a_0 = a_0_std * scale**2

        # Calculate relative radius (scale^2 because a_0 ∝ ℏ²)
        r_display = max(scale**2 * max_r_display, min_r_display)

        # Draw electron orbital (Bohr model visualization)
        orbital = Circle((x_center, y_pos), r_display, fill=False, color=color,
                         linewidth=3, linestyle='-', zorder=4)
        ax.add_patch(orbital)

        # Draw nucleus (fixed size)
        nucleus_size = 0.015
        nucleus = Circle((x_center, y_pos), nucleus_size, color=COLORS['scaled'], zorder=5)
        ax.add_patch(nucleus)

        # Draw electron on the orbital
        electron_x = x_center + r_display
        electron_size = 0.012
        electron = Circle((electron_x, y_pos), electron_size, color=COLORS['primary_blue'], zorder=6)
        ax.add_patch(electron)

        # Add label to the right of the atom
        if language == 'de':
            label = f'ℏ × {scale}: a₀ = {a_0*1e12:.2f} pm'
            if scale == 1.0:
                label += ' (Standard)'
        else:
            label = f'ℏ × {scale}: a₀ = {a_0*1e12:.2f} pm'
            if scale == 1.0:
                label += ' (Standard)'

        ax.text(0.55, y_pos, label, fontsize=12, va='center',
               bbox=dict(facecolor='white', alpha=0.9, edgecolor=color, linewidth=2))

        # Add shrink factor below label
        if scale != 1.0:
            shrink = 1 / scale**2
            shrink_text = f'{shrink:.0f}× smaller' if language == 'en' else f'{shrink:.0f}× kleiner'
            ax.text(0.55, y_pos - 0.05, shrink_text, fontsize=10, va='center',
                   color=COLORS['scaled'], style='italic')

    # Add legend for components
    ax.plot([], [], 'o', color=COLORS['scaled'], markersize=8,
            label='Nucleus' if language == 'en' else 'Atomkern')
    ax.plot([], [], 'o', color=COLORS['primary_blue'], markersize=8,
            label='Electron' if language == 'en' else 'Elektron')
    ax.plot([], [], '-', color=COLORS['standard'], linewidth=3,
            label='Bohr orbit' if language == 'en' else 'Bohr-Bahn')

    ax.legend(fontsize=10, loc='upper right', bbox_to_anchor=(1.0, -0.02), framealpha=0.9)

    # Title
    if language == 'de':
        ax.set_title('Atomgroesse in verschiedenen Universen\n'
                    'Bohr-Radius a₀ ∝ ℏ² (kleineres ℏ → kleinere Atome)', fontsize=14, fontweight='bold', pad=15)
    else:
        ax.set_title('Atom Size in Different Universes\n'
                    'Bohr radius a₀ ∝ ℏ² (smaller ℏ → smaller atoms)', fontsize=14, fontweight='bold', pad=15)

    # Set axis limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.95)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.tight_layout()

    if save:
        os.makedirs(VIS_DIR, exist_ok=True)
        filepath = os.path.join(VIS_DIR, 'atom_size_comparison.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")

    if show:
        plt.show()

    return fig


def plot_energy_levels(
    constants: Optional[PhysicalConstants] = None,
    language: str = 'en',
    save: bool = True,
    show: bool = True
) -> plt.Figure:
    """
    Plot hydrogen energy levels and transitions.
    Zeigt Wasserstoff-Energieniveaus und Uebergaenge.

    Shows the quantized energy levels E_n = -13.6 eV / n²

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

    # Create figure with vertical stacking
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14), gridspec_kw={'hspace': 0.4})

    # Top plot: Energy level diagram
    n_levels = 6
    energies = [energy_level(n, constants) for n in range(1, n_levels + 1)]

    # Draw energy levels
    for n in range(1, n_levels + 1):
        E = energies[n-1]
        ax1.hlines(E, 0.2, 0.8, colors=COLORS['primary_blue'], linewidth=2)
        # Label positioning - stagger horizontally for upper levels to avoid overlap
        if n <= 3:
            ax1.text(0.85, E, f'n={n}: {E:.2f} eV', fontsize=10, va='center')
        elif n == 4:
            ax1.text(0.85, E - 0.3, f'n=4', fontsize=9, va='center')
        elif n == 5:
            ax1.text(0.92, E, f'n=5', fontsize=9, va='center')
        elif n == 6:
            ax1.text(0.85, E + 0.5, f'n=6', fontsize=9, va='center')

    # Draw some transitions (Lyman and Balmer series)
    transitions = [(2, 1), (3, 1), (3, 2), (4, 2)]
    colors_trans = [COLORS['quantum'], COLORS['primary_blue'], COLORS['scaled'], COLORS['primary_amber']]
    series_names = ['Lyman α', 'Lyman β', 'Balmer α', 'Balmer β']

    for (n_high, n_low), color in zip(transitions, colors_trans):
        E_high = energies[n_high - 1]
        E_low = energies[n_low - 1]
        delta_E = E_high - E_low

        # Draw arrow
        ax1.annotate('', xy=(0.5, E_low), xytext=(0.5, E_high),
                    arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

        # Label transition
        ax1.text(0.52, (E_high + E_low) / 2, f'ΔE={delta_E:.2f} eV',
                fontsize=8, color=color, va='center')

    # Add legend for transitions
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=c, marker='>', linestyle='-',
                              markersize=5, label=name, linewidth=1.5)
                      for c, name in zip(colors_trans, series_names)]
    ax1.legend(handles=legend_elements, fontsize=9, loc='upper right', bbox_to_anchor=(1.0, -0.08),
              framealpha=0.7, ncol=2, title='Transitions' if language == 'en' else 'Übergänge')

    # Add ionization level
    ax1.axhline(y=0, color=COLORS['scaled'], linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.text(0.85, 0.5, 'Ionization\n(E=0)' if language == 'en' else 'Ionisation\n(E=0)',
            fontsize=10, va='bottom')

    if language == 'de':
        ax1.set_ylabel('Energie (eV)', fontsize=12)
        ax1.set_title('1. Wasserstoff-Energieniveaus (E_n = -13.6 eV / n²)', fontsize=14, fontweight='bold', pad=15)
    else:
        ax1.set_ylabel('Energy (eV)', fontsize=12)
        ax1.set_title('1. Hydrogen Energy Levels (E_n = -13.6 eV / n²)', fontsize=14, fontweight='bold', pad=15)

    ax1.set_xlim(0, 1)
    ax1.set_ylim(-15, 2)
    ax1.set_xticks([])
    ax1.grid(True, alpha=0.3, axis='y')

    # Bottom plot: Orbital radii
    radii = [orbital_radius(n, constants) for n in range(1, n_levels + 1)]

    # Single color bars - no legend needed since x-axis shows n and values are labeled
    ax2.bar(range(1, n_levels + 1), [r * 1e12 for r in radii],
           color=COLORS['primary_blue'], alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels
    for n, r in enumerate(radii, 1):
        ax2.text(n, r * 1e12 * 1.05, f'{r*1e12:.1f} pm',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    if language == 'de':
        ax2.set_xlabel('Hauptquantenzahl n', fontsize=12)
        ax2.set_ylabel('Orbitalradius (pm)', fontsize=12)
        ax2.set_title('2. Orbitalradien r_n = n² × a_0 (Bohr-Modell)', fontsize=14, fontweight='bold', pad=15)
    else:
        ax2.set_xlabel('Principal quantum number n', fontsize=12)
        ax2.set_ylabel('Orbital radius (pm)', fontsize=12)
        ax2.set_title('2. Orbital Radii r_n = n² × a_0 (Bohr model)', fontsize=14, fontweight='bold', pad=15)

    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save:
        os.makedirs(VIS_DIR, exist_ok=True)
        filepath = os.path.join(VIS_DIR, 'energy_levels.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")

    if show:
        plt.show()

    return fig


def plot_quantum_gravity_connection(
    constants: Optional[PhysicalConstants] = None,
    language: str = 'en',
    save: bool = True,
    show: bool = True
) -> plt.Figure:
    """
    Visualize the connection between quantum mechanics and gravity.
    Visualisiert die Verbindung zwischen Quantenmechanik und Gravitation.

    Shows how changing ℏ affects both atomic structure and gravitational coupling.

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

    # Create figure with 4 subplots stacked vertically
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 20), gridspec_kw={'hspace': 0.5})

    # Range of hbar scaling
    hbar_scales = np.logspace(-1.5, 0.5, 50)

    # Standard values
    a_0_std = constants.a_0
    alpha_G_std = constants.alpha_G
    E_1_std = abs(energy_level(1, constants))

    # Calculate scaled values
    a_0_scaled = a_0_std * hbar_scales**2
    alpha_G_scaled = alpha_G_std / hbar_scales
    # Energy scales as 1/ℏ² (since E ∝ m_e × c² × α² and α is kept constant)
    # Actually, E_1 = -m_e c² α² / 2, which doesn't depend on ℏ if α is constant
    # But if we consider E_1 = -ℏ²/(2 m_e a_0²), then E_1 ∝ 1/ℏ² when a_0 ∝ ℏ²
    # Let's use E_1 ∝ 1/a_0 ∝ 1/ℏ² for the bound state shrinking effect
    E_1_scaled = E_1_std / hbar_scales**2

    # 1. Bohr radius
    ax1.loglog(hbar_scales, a_0_scaled * 1e12, '-', color=COLORS['primary_blue'], linewidth=2.5, label='a_0(ℏ)')
    ax1.axvline(x=1, color=COLORS['standard'], linestyle='--', alpha=0.7)
    ax1.axhline(y=a_0_std * 1e12, color=COLORS['standard'], linestyle=':', alpha=0.5)
    ax1.plot(1, a_0_std * 1e12, 'o', color=COLORS['standard'], markersize=10,
             label='Standard' if language == 'en' else 'Standard')

    if language == 'de':
        ax1.set_xlabel('ℏ-Skalierung', fontsize=11)
        ax1.set_ylabel('Bohr-Radius (pm)', fontsize=11)
        ax1.set_title('1. Atomgroesse: a_0 ∝ ℏ²', fontsize=13, fontweight='bold', pad=15)
    else:
        ax1.set_xlabel('ℏ scaling', fontsize=11)
        ax1.set_ylabel('Bohr radius (pm)', fontsize=11)
        ax1.set_title('1. Atom Size: a_0 ∝ ℏ²', fontsize=13, fontweight='bold', pad=15)

    ax1.legend(fontsize=9, loc='upper right', bbox_to_anchor=(1.0, -0.08), framealpha=0.7)
    ax1.grid(True, alpha=0.3, which='both')

    # 2. Gravitational coupling
    ax2.loglog(hbar_scales, alpha_G_scaled, '-', color=COLORS['scaled'], linewidth=2.5, label='α_G(ℏ)')
    ax2.axvline(x=1, color=COLORS['standard'], linestyle='--', alpha=0.7)
    ax2.axhline(y=alpha_G_std, color=COLORS['standard'], linestyle=':', alpha=0.5)
    ax2.plot(1, alpha_G_std, 'o', color=COLORS['standard'], markersize=10, label='Standard')

    if language == 'de':
        ax2.set_xlabel('ℏ-Skalierung', fontsize=11)
        ax2.set_ylabel('α_G', fontsize=11)
        ax2.set_title('2. Gravitations-Kopplung: α_G ∝ 1/ℏ', fontsize=13, fontweight='bold', pad=15)
    else:
        ax2.set_xlabel('ℏ scaling', fontsize=11)
        ax2.set_ylabel('α_G', fontsize=11)
        ax2.set_title('2. Gravity Coupling: α_G ∝ 1/ℏ', fontsize=13, fontweight='bold', pad=15)

    ax2.legend(fontsize=9, loc='upper right', bbox_to_anchor=(1.0, -0.08), framealpha=0.7)
    ax2.grid(True, alpha=0.3, which='both')

    # 3. Matter density (∝ 1/a_0³ ∝ 1/ℏ⁶)
    density_ratio = 1 / hbar_scales**6
    ax3.loglog(hbar_scales, density_ratio, '-', color=COLORS['quantum'], linewidth=2.5, label='ρ(ℏ)/ρ₀')
    ax3.axvline(x=1, color=COLORS['standard'], linestyle='--', alpha=0.7)
    ax3.plot(1, 1, 'o', color=COLORS['standard'], markersize=10, label='Standard')

    if language == 'de':
        ax3.set_xlabel('ℏ-Skalierung', fontsize=11)
        ax3.set_ylabel('Dichte / Standard-Dichte', fontsize=11)
        ax3.set_title('3. Materiedichte: ρ ∝ 1/a_0³ ∝ 1/ℏ⁶', fontsize=13, fontweight='bold', pad=15)
    else:
        ax3.set_xlabel('ℏ scaling', fontsize=11)
        ax3.set_ylabel('Density / Standard density', fontsize=11)
        ax3.set_title('3. Matter Density: ρ ∝ 1/a_0³ ∝ 1/ℏ⁶', fontsize=13, fontweight='bold', pad=15)

    ax3.legend(fontsize=9, loc='upper right', bbox_to_anchor=(1.0, -0.08), framealpha=0.7)
    ax3.grid(True, alpha=0.3, which='both')

    # 4. Summary: alpha_G × density (gravity importance)
    # Gravitational effect ∝ α_G × ρ ∝ (1/ℏ) × (1/ℏ⁶) = 1/ℏ⁷
    gravity_importance = 1 / hbar_scales**7
    ax4.loglog(hbar_scales, gravity_importance, '-', color=COLORS['text_dark'], linewidth=2.5, label='α_G×ρ')
    ax4.axvline(x=1, color=COLORS['standard'], linestyle='--', alpha=0.7)
    ax4.plot(1, 1, 'o', color=COLORS['standard'], markersize=10, label='Standard')

    if language == 'de':
        ax4.set_xlabel('ℏ-Skalierung', fontsize=11)
        ax4.set_ylabel('Gravitations-Bedeutung (relativ)', fontsize=11)
        ax4.set_title('4. Gravitations-Relevanz: ∝ α_G × ρ ∝ 1/ℏ⁷', fontsize=13, fontweight='bold', pad=15)
    else:
        ax4.set_xlabel('ℏ scaling', fontsize=11)
        ax4.set_ylabel('Gravity importance (relative)', fontsize=11)
        ax4.set_title('4. Gravity Relevance: ∝ α_G × ρ ∝ 1/ℏ⁷', fontsize=13, fontweight='bold', pad=15)

    ax4.legend(fontsize=9, loc='upper right', bbox_to_anchor=(1.0, -0.08), framealpha=0.7)
    ax4.grid(True, alpha=0.3, which='both')

    plt.tight_layout()

    if save:
        os.makedirs(VIS_DIR, exist_ok=True)
        filepath = os.path.join(VIS_DIR, 'quantum_gravity_connection.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")

    if show:
        plt.show()

    return fig


def plot_atomic_summary(
    constants: Optional[PhysicalConstants] = None,
    language: str = 'en',
    save: bool = True,
    show: bool = True
) -> plt.Figure:
    """
    Create a comprehensive summary of atomic scale effects.
    Erstellt eine umfassende Zusammenfassung der atomaren Skaleneffekte.

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

    # Create figure with 3 subplots stacked vertically
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 16), gridspec_kw={'hspace': 0.5})

    # 1. Energy levels
    n_levels = 5
    energies = [energy_level(n, constants) for n in range(1, n_levels + 1)]

    for n in range(1, n_levels + 1):
        E = energies[n-1]
        ax1.hlines(E, 0.2, 0.8, colors=COLORS['primary_blue'], linewidth=2, label=f'n={n}' if n <= 3 else None)
        ax1.text(0.85, E, f'n={n}', fontsize=9, va='center')

    ax1.axhline(y=0, color=COLORS['scaled'], linestyle='--', alpha=0.7, label='Ionization' if language == 'en' else 'Ionisation')
    if language == 'de':
        ax1.set_ylabel('Energie (eV)', fontsize=11)
        ax1.set_title('1. H-Energieniveaus (E_n = -13.6/n² eV)', fontsize=14, fontweight='bold', pad=15)
    else:
        ax1.set_ylabel('Energy (eV)', fontsize=11)
        ax1.set_title('1. H Energy Levels (E_n = -13.6/n² eV)', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(-15, 2)
    ax1.set_xticks([])
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.legend(fontsize=9, loc='upper right', bbox_to_anchor=(1.0, -0.08), ncol=4, framealpha=0.7)

    # 2. Bohr radius scaling
    hbar_scales = np.logspace(-1, 0.5, 50)
    a_0_std = constants.a_0
    a_0_scaled = a_0_std * hbar_scales**2

    ax2.loglog(hbar_scales, a_0_scaled * 1e12, '-', color=COLORS['primary_blue'], linewidth=2.5, label='Bohr radius' if language == 'en' else 'Bohr-Radius')
    ax2.axvline(x=1, color=COLORS['standard'], linestyle='--', alpha=0.7)
    ax2.plot(1, a_0_std * 1e12, 'o', color=COLORS['standard'], markersize=8, label='Standard')

    if language == 'de':
        ax2.set_xlabel('ℏ-Skalierung', fontsize=11)
        ax2.set_ylabel('Bohr-Radius (pm)', fontsize=11)
        ax2.set_title('2. Atomgroesse (a_0 ∝ ℏ²)', fontsize=14, fontweight='bold', pad=15)
    else:
        ax2.set_xlabel('ℏ scaling', fontsize=11)
        ax2.set_ylabel('Bohr radius (pm)', fontsize=11)
        ax2.set_title('2. Atom Size (a_0 ∝ ℏ²)', fontsize=14, fontweight='bold', pad=15)

    ax2.legend(fontsize=9, loc='upper right', bbox_to_anchor=(1.0, -0.08), framealpha=0.7)
    ax2.grid(True, alpha=0.3, which='both')

    # 3. Key constants comparison
    props = calculate_atomic_properties(constants)

    constants_labels = ['Bohr radius', 'Ionization E', 'Orbital v', 'α (fine str.)'] if language == 'en' else ['Bohr-Radius', 'Ionisierungsenergie', 'Orbital-v', 'α (Feinstruktur)']
    constants_values = [
        props.bohr_radius*1e12,
        props.ionization_energy,
        props.orbital_velocity/1e6,
        1/props.fine_structure_constant
    ]
    constants_units = ['pm', 'eV', 'Mm/s', '']
    bar_colors = [COLORS['primary_blue'], COLORS['scaled'], COLORS['quantum'], COLORS['standard']]

    # Create bar chart with individual labels for legend
    for i, (lbl, val, unit, color) in enumerate(zip(constants_labels, constants_values, constants_units, bar_colors)):
        ax3.barh(lbl, 1, color=color, edgecolor='black', alpha=0.7, label=lbl)
        if unit:
            ax3.text(0.5, i, f'{val:.2f} {unit}', ha='center', va='center', fontsize=11, fontweight='bold')
        else:
            ax3.text(0.5, i, f'1/{val:.1f}', ha='center', va='center', fontsize=11, fontweight='bold')

    ax3.set_xlim(0, 1)
    ax3.set_xticks([])

    if language == 'de':
        ax3.set_title('3. Wichtige Atomkonstanten (Wasserstoff)', fontsize=14, fontweight='bold', pad=15)
    else:
        ax3.set_title('3. Key Atomic Constants (Hydrogen)', fontsize=14, fontweight='bold', pad=15)

    ax3.legend(fontsize=9, loc='upper right', bbox_to_anchor=(1.0, -0.08), ncol=2, framealpha=0.7)

    if save:
        os.makedirs(VIS_DIR, exist_ok=True)
        filepath = os.path.join(VIS_DIR, 'atomic_summary.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")

    if show:
        plt.show()

    return fig


def generate_all_atomic_plots(language: str = 'en', show: bool = False) -> List[plt.Figure]:
    """
    Generate all atomic scale visualizations.
    Erzeugt alle Visualisierungen fuer atomare Skalen.

    Args:
        language: 'en' for English, 'de' for German
        show: Whether to display plots

    Returns:
        List of matplotlib Figure objects
    """
    figures = []

    print("Generating atomic scale visualizations...")
    print("=" * 50)

    # 1. Bohr radius scaling
    print("1. Bohr radius scaling...")
    figures.append(plot_bohr_radius_scaling(language=language, show=show))

    # 2. Atom size comparison
    print("2. Atom size comparison...")
    figures.append(plot_atom_size_comparison(language=language, show=show))

    # 3. Energy levels
    print("3. Energy levels diagram...")
    figures.append(plot_energy_levels(language=language, show=show))

    # 4. Quantum-gravity connection
    print("4. Quantum-gravity connection...")
    figures.append(plot_quantum_gravity_connection(language=language, show=show))

    # 5. Summary
    print("5. Comprehensive summary...")
    figures.append(plot_atomic_summary(language=language, show=show))

    print("=" * 50)
    print(f"Generated {len(figures)} visualizations in {VIS_DIR}")

    return figures


def verify_atomic_physics():
    """
    Verify atomic physics calculations.
    """
    print("=" * 70)
    print("ATOMIC SCALE PHYSICS VERIFICATION")
    print("=" * 70)

    c = get_constants()
    props = calculate_atomic_properties(c)

    # 1. Bohr radius
    print("\n1. BOHR RADIUS")
    print("-" * 50)
    print(f"   a_0 = {props.bohr_radius:.4e} m = {props.bohr_radius*1e12:.2f} pm")
    print(f"   Expected: ~52.9 pm (CHECK: {'PASS' if 52 < props.bohr_radius*1e12 < 54 else 'FAIL'})")

    # 2. Ground state energy
    print("\n2. HYDROGEN GROUND STATE ENERGY")
    print("-" * 50)
    E_1 = energy_level(1, c)
    print(f"   E_1 = {E_1:.2f} eV")
    print(f"   Expected: ~-13.6 eV (CHECK: {'PASS' if -14 < E_1 < -13 else 'FAIL'})")

    # 3. Ionization energy
    print("\n3. IONIZATION ENERGY")
    print("-" * 50)
    print(f"   Ionization E = {props.ionization_energy:.2f} eV")
    print(f"   Expected: ~13.6 eV (CHECK: {'PASS' if 13 < props.ionization_energy < 14 else 'FAIL'})")

    # 4. Fine structure constant
    print("\n4. FINE STRUCTURE CONSTANT")
    print("-" * 50)
    print(f"   alpha = {props.fine_structure_constant:.6f}")
    print(f"   1/alpha = {1/props.fine_structure_constant:.2f}")
    print(f"   Expected: alpha = 1/137 (CHECK: {'PASS' if 136 < 1/props.fine_structure_constant < 138 else 'FAIL'})")

    # 5. Orbital velocity
    print("\n5. ELECTRON ORBITAL VELOCITY (n=1)")
    print("-" * 50)
    print(f"   v_1 = {props.orbital_velocity:.2e} m/s")
    print(f"   v_1/c = {props.orbital_velocity/c.c:.6f} = alpha")
    print(f"   Expected: v_1 = alpha*c = 2.19e6 m/s (CHECK: {'PASS' if 2.1e6 < props.orbital_velocity < 2.3e6 else 'FAIL'})")

    # 6. Orbital radii scaling
    print("\n6. ORBITAL RADII (r_n = n²×a_0)")
    print("-" * 50)
    for n in range(1, 4):
        r_n = orbital_radius(n, c)
        print(f"   r_{n} = {r_n*1e12:.2f} pm = {n**2} × a_0")

    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    print("=" * 60)
    print("Atomic Scale Effects Module - Jugend forscht 2026")
    print("=" * 60)

    # Verify physics
    verify_atomic_physics()

    # Generate visualizations
    print("\n")
    generate_all_atomic_plots(language='en', show=False)

    print("\nDone! Check the 'visualizations' folder for output.")
