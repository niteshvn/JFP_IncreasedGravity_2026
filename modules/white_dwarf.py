"""
White Dwarf Physics Module for Jugend forscht 2026 Physics Visualization Project
Weisse-Zwerg-Physik-Modul fuer Jugend forscht 2026 Physik-Visualisierungsprojekt

This module visualizes the physics of white dwarf stars, including:
- Electron degeneracy pressure (from Pauli exclusion principle)
- Non-relativistic vs relativistic pressure regimes
- Mass-radius relationship
- The Chandrasekhar limit (~1.4 solar masses)

The key insight is how increasing gravitational pressure eventually overcomes
electron degeneracy pressure, leading to stellar collapse.

Author: Jugend forscht 2026 Project
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import os
from typing import Optional, List, Tuple
from dataclasses import dataclass

from .constants import get_constants, PhysicalConstants
from .color_scheme import COLORS, get_stellar_colors, get_sequence


# Output directory for visualizations
VIS_DIR = os.path.join(os.path.dirname(__file__), '..', 'visualizations')


@dataclass
class WhiteDwarfProperties:
    """
    Container for white dwarf star properties.
    Behaelter fuer Weisse-Zwerg-Eigenschaften.
    """
    mass: float              # Mass [kg]
    mass_solar: float        # Mass in solar masses
    radius: float            # Radius [m]
    radius_solar: float      # Radius in solar radii
    density: float           # Central density [kg/m³]
    pressure: float          # Central pressure [Pa]
    is_stable: bool          # Whether below Chandrasekhar limit
    compactness: float       # R_schwarzschild / R


def electron_degeneracy_pressure_nr(density: float, constants: PhysicalConstants) -> float:
    """
    Calculate non-relativistic electron degeneracy pressure.
    Berechnet nicht-relativistischen Elektronen-Entartungsdruck.

    Formula: P = K_nr * rho^(5/3)
    where K_nr = (hbar^2 / m_e) * (3 / (8*pi))^(2/3) * (1 / (m_p * mu_e))^(5/3)

    For a fully ionized gas with equal numbers of protons and neutrons: mu_e ≈ 2

    Args:
        density: Mass density [kg/m³]
        constants: Physical constants

    Returns:
        Pressure [Pa]
    """
    # Mean molecular weight per electron (for typical white dwarf composition)
    mu_e = 2.0  # Carbon/Oxygen white dwarf

    # Non-relativistic constant
    # K_nr = (hbar^2 / m_e) * (3*pi^2)^(2/3) * (1 / (mu_e * m_p))^(5/3) / 5
    # Simplified form using n_e = rho / (mu_e * m_p)
    K_nr = (constants.hbar**2 / (5 * constants.m_e)) * \
           (3 * np.pi**2)**(2/3) * \
           (1 / (mu_e * constants.m_p))**(5/3)

    return K_nr * density**(5/3)


def electron_degeneracy_pressure_r(density: float, constants: PhysicalConstants) -> float:
    """
    Calculate relativistic electron degeneracy pressure.
    Berechnet relativistischen Elektronen-Entartungsdruck.

    Formula: P = K_r * rho^(4/3)

    In the ultra-relativistic limit, electrons move at nearly the speed of light.

    Args:
        density: Mass density [kg/m³]
        constants: Physical constants

    Returns:
        Pressure [Pa]
    """
    mu_e = 2.0

    # Relativistic constant
    # K_r = (hbar * c / 4) * (3*pi^2)^(1/3) * (1 / (mu_e * m_p))^(4/3)
    K_r = (constants.hbar * constants.c / 4) * \
          (3 * np.pi**2)**(1/3) * \
          (1 / (mu_e * constants.m_p))**(4/3)

    return K_r * density**(4/3)


def gravitational_pressure(mass: float, radius: float, constants: PhysicalConstants) -> float:
    """
    Estimate central gravitational pressure in a star.
    Schaetzt den zentralen Gravitationsdruck in einem Stern.

    Using virial theorem approximation: P_c ~ G * M^2 / R^4

    Args:
        mass: Stellar mass [kg]
        radius: Stellar radius [m]
        constants: Physical constants

    Returns:
        Approximate central pressure [Pa]
    """
    # Approximate central pressure from hydrostatic equilibrium
    # P_c ~ (3 / 8*pi) * G * M^2 / R^4 for uniform density
    return (3 / (8 * np.pi)) * constants.G * mass**2 / radius**4


def white_dwarf_radius(mass: float, constants: PhysicalConstants) -> float:
    """
    Calculate white dwarf radius using mass-radius relation.
    Berechnet den Weissen-Zwerg-Radius mit der Masse-Radius-Beziehung.

    For non-relativistic degeneracy: R ∝ M^(-1/3)

    This uses the Chandrasekhar mass-radius relation with corrections.

    Args:
        mass: White dwarf mass [kg]
        constants: Physical constants

    Returns:
        Radius [m]
    """
    mu_e = 2.0
    M_ch = chandrasekhar_mass(constants)

    # Mass-radius relation: R = R_0 * (M/M_ch)^(-1/3) * (1 - (M/M_ch)^(4/3))^(1/2)
    # Where R_0 is a characteristic radius

    # Characteristic radius (approximate)
    # R_0 ~ (hbar / m_e*c) * (M_ch / m_p)^(1/3) / mu_e^(5/3)
    R_0 = 0.0126 * constants.R_sun  # ~8800 km, calibrated to observations

    mass_ratio = mass / M_ch

    if mass_ratio >= 1.0:
        return 0.0  # Collapse - no stable radius

    # Mass-radius relation with relativistic correction
    radius = R_0 * mass_ratio**(-1/3) * np.sqrt(1 - mass_ratio**(4/3))

    return radius


def chandrasekhar_mass(constants: PhysicalConstants) -> float:
    """
    Calculate the Chandrasekhar mass limit.
    Berechnet die Chandrasekhar-Massengrenze.

    Formula: M_Ch = (hbar*c/G)^(3/2) * (1/m_p)^2 * (omega_3 / (mu_e)^2)

    where omega_3 ≈ 2.018 is a numerical constant from the Lane-Emden equation.

    Args:
        constants: Physical constants

    Returns:
        Chandrasekhar mass [kg]
    """
    mu_e = 2.0
    omega_3 = 2.018  # Lane-Emden numerical constant for n=3 polytrope

    # Chandrasekhar mass formula
    M_ch = (omega_3 * np.sqrt(3 * np.pi) / 2) * \
           (constants.hbar * constants.c / constants.G)**(3/2) * \
           (1 / (mu_e * constants.m_p))**2

    return M_ch


def calculate_white_dwarf(
    mass_solar: float,
    constants: Optional[PhysicalConstants] = None
) -> WhiteDwarfProperties:
    """
    Calculate properties of a white dwarf given its mass.
    Berechnet die Eigenschaften eines Weissen Zwergs bei gegebener Masse.

    Args:
        mass_solar: Mass in solar masses
        constants: Physical constants (uses standard if None)

    Returns:
        WhiteDwarfProperties object
    """
    if constants is None:
        constants = get_constants()

    mass = mass_solar * constants.M_sun
    M_ch = chandrasekhar_mass(constants)

    is_stable = mass < M_ch

    if is_stable:
        radius = white_dwarf_radius(mass, constants)
        # Average density (assuming uniform, which is approximate)
        density = mass / ((4/3) * np.pi * radius**3) if radius > 0 else float('inf')
        # Central pressure (using degeneracy pressure)
        pressure = electron_degeneracy_pressure_nr(density, constants)
    else:
        radius = 0
        density = float('inf')
        pressure = float('inf')

    # Compactness = R_s / R
    R_s = constants.schwarzschild_radius(mass)
    compactness = R_s / radius if radius > 0 else float('inf')

    return WhiteDwarfProperties(
        mass=mass,
        mass_solar=mass_solar,
        radius=radius,
        radius_solar=radius / constants.R_sun if radius > 0 else 0,
        density=density,
        pressure=pressure,
        is_stable=is_stable,
        compactness=compactness
    )


def plot_pressure_vs_density(
    constants: Optional[PhysicalConstants] = None,
    language: str = 'en',
    save: bool = True,
    show: bool = True
) -> plt.Figure:
    """
    Plot electron degeneracy pressure vs density showing both regimes.
    Zeigt Elektronen-Entartungsdruck vs. Dichte in beiden Regimen.

    Shows:
    - Non-relativistic: P ∝ ρ^(5/3) (steeper slope)
    - Relativistic: P ∝ ρ^(4/3) (shallower slope)
    - Transition region where relativistic effects become important

    Args:
        constants: Physical constants (uses standard if None)
        language: 'en' for English, 'de' for German
        save: Whether to save the figure
        show: Whether to display the figure

    Returns:
        matplotlib Figure object
    """
    if constants is None:
        constants = get_constants()

    # Density range: 10^6 to 10^12 kg/m³ (typical white dwarf range)
    rho = np.logspace(6, 12, 200)

    # Calculate pressures
    P_nr = np.array([electron_degeneracy_pressure_nr(r, constants) for r in rho])
    P_r = np.array([electron_degeneracy_pressure_r(r, constants) for r in rho])

    # Transition density (where P_nr ≈ P_r)
    transition_idx = np.argmin(np.abs(P_nr - P_r))
    rho_transition = rho[transition_idx]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot both pressure curves
    ax.loglog(rho, P_nr, '-', color=COLORS['non_relativistic'], linewidth=2.5,
              label='Non-relativistic: P ∝ ρ^(5/3)' if language == 'en' else 'Nicht-relativistisch: P ∝ ρ^(5/3)')
    ax.loglog(rho, P_r, '-', color=COLORS['relativistic'], linewidth=2.5,
              label='Relativistic: P ∝ ρ^(4/3)' if language == 'en' else 'Relativistisch: P ∝ ρ^(4/3)')

    # Mark transition region
    ax.axvline(x=rho_transition, color=COLORS['standard'], linestyle='--', linewidth=1.5, alpha=0.7,
              label=f'Transition: ρ ≈ {rho_transition:.1e} kg/m³' if language == 'en' else f'Übergang: ρ ≈ {rho_transition:.1e} kg/m³')
    ax.axvspan(rho_transition/10, rho_transition*10, alpha=0.1, color=COLORS['standard'])

    # Annotations
    if language == 'de':
        ax.set_xlabel('Dichte ρ (kg/m³)', fontsize=12)
        ax.set_ylabel('Elektronen-Entartungsdruck P (Pa)', fontsize=12)
        ax.set_title('Elektronen-Entartungsdruck: Nicht-relativistisch vs. Relativistisch', fontsize=14, fontweight='bold', pad=15)
    else:
        ax.set_xlabel('Density ρ (kg/m³)', fontsize=12)
        ax.set_ylabel('Electron Degeneracy Pressure P (Pa)', fontsize=12)
        ax.set_title('Electron Degeneracy Pressure: Non-relativistic vs. Relativistic', fontsize=14, fontweight='bold', pad=15)

    # Legend at bottom right, outside graph
    ax.legend(fontsize=10, loc='upper right', bbox_to_anchor=(1.0, -0.08), framealpha=0.7)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim(1e6, 1e12)

    if save:
        os.makedirs(VIS_DIR, exist_ok=True)
        filepath = os.path.join(VIS_DIR, 'pressure_vs_density.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")

    if show:
        plt.show()

    return fig


def plot_mass_radius_relation(
    constants: Optional[PhysicalConstants] = None,
    language: str = 'en',
    save: bool = True,
    show: bool = True
) -> plt.Figure:
    """
    Plot the mass-radius relationship for white dwarfs.
    Zeigt die Masse-Radius-Beziehung fuer Weisse Zwerge.

    Key insight: R ∝ M^(-1/3) - more massive white dwarfs are SMALLER!
    This is opposite to normal stars where more mass means larger size.

    Args:
        constants: Physical constants (uses standard if None)
        language: 'en' for English, 'de' for German
        save: Whether to save the figure
        show: Whether to display the figure

    Returns:
        matplotlib Figure object
    """
    if constants is None:
        constants = get_constants()

    # Mass range: 0.1 to 1.5 solar masses
    mass_range = np.linspace(0.1, 1.45, 200)
    M_ch = chandrasekhar_mass(constants) / constants.M_sun

    # Calculate radii
    radii = []
    for m in mass_range:
        wd = calculate_white_dwarf(m, constants)
        radii.append(wd.radius_solar)
    radii = np.array(radii)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot mass-radius curve with label for legend
    curve_label = 'White Dwarf: R ∝ M^(-1/3)' if language == 'en' else 'Weisser Zwerg: R ∝ M^(-1/3)'
    ax.plot(mass_range, radii, '-', color=COLORS['white_dwarf'], linewidth=2.5, label=curve_label)

    # Mark Chandrasekhar limit
    limit_label = f'Chandrasekhar limit (M_Ch ≈ {M_ch:.2f} M☉)' if language == 'en' else f'Chandrasekhar-Grenze (M_Ch ≈ {M_ch:.2f} M☉)'
    ax.axvline(x=M_ch, color=COLORS['scaled'], linestyle='--', linewidth=2, alpha=0.8, label=limit_label)
    ax.axvspan(M_ch, 1.5, alpha=0.2, color=COLORS['scaled'], label='Unstable region' if language == 'en' else 'Instabiler Bereich')

    # Mark some typical white dwarfs
    typical_wds = [
        (0.6, 'Typical WD' if language == 'en' else 'Typischer WZ'),
        (1.0, 'Massive WD' if language == 'en' else 'Massiver WZ'),
        (1.3, 'Near limit' if language == 'en' else 'Nahe Grenze'),
    ]

    # Plot first point with label for legend
    first = True
    for m, label in typical_wds:
        wd = calculate_white_dwarf(m, constants)
        if wd.is_stable:
            if first:
                ax.plot(m, wd.radius_solar, 'ko', markersize=10,
                       label='Example white dwarfs' if language == 'en' else 'Beispiel-Weisse-Zwerge')
                first = False
            else:
                ax.plot(m, wd.radius_solar, 'ko', markersize=10)

            # Offset annotations to avoid overlap
            if m == 0.6:
                offset = (0.08, 0.002)
            elif m == 1.0:
                offset = (0.08, 0.001)
            else:
                offset = (-0.25, 0.001)

            ax.annotate(f'{label}\n({m} M☉, {wd.radius_solar:.3f} R☉)',
                       xy=(m, wd.radius_solar),
                       xytext=(m + offset[0], wd.radius_solar + offset[1]),
                       fontsize=9,
                       bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'),
                       zorder=10)

    # Labels
    if language == 'de':
        ax.set_xlabel('Masse (Sonnenmassen M☉)', fontsize=12)
        ax.set_ylabel('Radius (Sonnenradien R☉)', fontsize=12)
        ax.set_title('Masse-Radius-Beziehung für Weiße Zwerge', fontsize=14, fontweight='bold', pad=15)
    else:
        ax.set_xlabel('Mass (Solar masses M☉)', fontsize=12)
        ax.set_ylabel('Radius (Solar radii R☉)', fontsize=12)
        ax.set_title('Mass-Radius Relationship for White Dwarfs', fontsize=14, fontweight='bold', pad=15)

    ax.set_xlim(0, 1.5)
    ax.set_ylim(0, 0.025)
    ax.grid(True, alpha=0.3)

    # Legend at bottom right, outside graph
    ax.legend(fontsize=9, loc='upper right', bbox_to_anchor=(1.0, -0.08), framealpha=0.7)

    if save:
        os.makedirs(VIS_DIR, exist_ok=True)
        filepath = os.path.join(VIS_DIR, 'mass_radius_relation.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")

    if show:
        plt.show()

    return fig


def plot_chandrasekhar_limit(
    constants: Optional[PhysicalConstants] = None,
    language: str = 'en',
    save: bool = True,
    show: bool = True
) -> plt.Figure:
    """
    Visualize why the Chandrasekhar limit exists.
    Visualisiert, warum die Chandrasekhar-Grenze existiert.

    Shows the competition between degeneracy pressure and gravitational pressure,
    and why masses > 1.4 M_sun have no stable equilibrium.

    Args:
        constants: Physical constants (uses standard if None)
        language: 'en' for English, 'de' for German
        save: Whether to save the figure
        show: Whether to display the figure

    Returns:
        matplotlib Figure object
    """
    if constants is None:
        constants = get_constants()

    M_ch = chandrasekhar_mass(constants) / constants.M_sun

    # Create figure with two subplots stacked vertically with proper spacing
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14), gridspec_kw={'hspace': 0.4})

    # Top plot: Pressure balance at different masses
    masses = [0.5, 0.8, 1.0, 1.2, 1.35]
    colors = [COLORS['primary_blue'], COLORS['primary_teal'], COLORS['standard'], COLORS['primary_amber'], COLORS['scaled']]

    rho = np.logspace(8, 11, 100)

    for mass, color in zip(masses, colors):
        wd = calculate_white_dwarf(mass, constants)
        if wd.is_stable:
            # Degeneracy pressure at various densities
            P_deg = np.array([electron_degeneracy_pressure_nr(r, constants) for r in rho])
            # Gravitational pressure needed (scales with M^2)
            P_grav = gravitational_pressure(mass * constants.M_sun, wd.radius, constants) * (rho / wd.density)

            label = f'M = {mass} M☉'
            ax1.loglog(rho, P_deg, '-', color=color, linewidth=2, label=label + ' (deg)')
            ax1.loglog(rho, P_grav, '--', color=color, linewidth=1.5, alpha=0.7)

    if language == 'de':
        ax1.set_xlabel('Dichte ρ (kg/m³)', fontsize=12)
        ax1.set_ylabel('Druck P (Pa)', fontsize=12)
        ax1.set_title('1. Gleichgewicht: Entartungsdruck vs. Gravitationsdruck', fontsize=14, fontweight='bold', pad=15)
    else:
        ax1.set_xlabel('Density ρ (kg/m³)', fontsize=12)
        ax1.set_ylabel('Pressure P (Pa)', fontsize=12)
        ax1.set_title('1. Equilibrium: Degeneracy Pressure vs. Gravitational Pressure', fontsize=14, fontweight='bold', pad=15)

    ax1.legend(fontsize=9, loc='upper right', bbox_to_anchor=(1.0, -0.08), framealpha=0.7)
    ax1.grid(True, alpha=0.3, which='both')

    # Right plot: Stability diagram
    mass_range = np.linspace(0.1, 1.6, 100)

    # Calculate "stability margin" - how much degeneracy pressure exceeds gravity
    stability = []
    for m in mass_range:
        if m < M_ch:
            wd = calculate_white_dwarf(m, constants)
            # Ratio of degeneracy to gravitational pressure at center
            P_deg = electron_degeneracy_pressure_nr(wd.density, constants)
            P_grav = gravitational_pressure(m * constants.M_sun, wd.radius, constants)
            margin = (P_deg - P_grav) / P_grav if P_grav > 0 else 0
            stability.append(margin)
        else:
            stability.append(-1)  # Unstable

    stability = np.array(stability)

    # Plot stability
    stable_mask = mass_range < M_ch
    ax2.fill_between(mass_range[stable_mask], 0, stability[stable_mask],
                     alpha=0.3, color=COLORS['standard'], label='Stable region' if language == 'en' else 'Stabiler Bereich')
    ax2.fill_between(mass_range[~stable_mask], 0, -0.5 * np.ones(sum(~stable_mask)),
                     alpha=0.3, color=COLORS['scaled'], label='Unstable (collapse)' if language == 'en' else 'Instabil (Kollaps)')

    limit_label = f'M_Ch = {M_ch:.2f} M☉'
    ax2.axvline(x=M_ch, color=COLORS['scaled'], linestyle='--', linewidth=2, label=limit_label)
    ax2.axhline(y=0, color='black', linewidth=1)

    if language == 'de':
        ax2.set_xlabel('Masse (Sonnenmassen M☉)', fontsize=12)
        ax2.set_ylabel('Stabilitäts-Marge', fontsize=12)
        ax2.set_title('2. Stabilität vs. Masse', fontsize=14, fontweight='bold', pad=15)
    else:
        ax2.set_xlabel('Mass (Solar masses M☉)', fontsize=12)
        ax2.set_ylabel('Stability Margin', fontsize=12)
        ax2.set_title('2. Stability vs. Mass', fontsize=14, fontweight='bold', pad=15)

    ax2.set_xlim(0, 1.6)
    ax2.set_ylim(-0.5, 1.5)
    ax2.legend(fontsize=9, loc='upper right', bbox_to_anchor=(1.0, -0.08), framealpha=0.7)
    ax2.grid(True, alpha=0.3)

    if save:
        os.makedirs(VIS_DIR, exist_ok=True)
        filepath = os.path.join(VIS_DIR, 'chandrasekhar_limit.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")

    if show:
        plt.show()

    return fig


def plot_white_dwarf_summary(
    constants: Optional[PhysicalConstants] = None,
    language: str = 'en',
    save: bool = True,
    show: bool = True
) -> plt.Figure:
    """
    Create a comprehensive summary visualization of white dwarf physics.
    Erstellt eine umfassende Zusammenfassung der Weisse-Zwerg-Physik.

    Args:
        constants: Physical constants (uses standard if None)
        language: 'en' for English, 'de' for German
        save: Whether to save the figure
        show: Whether to display the figure

    Returns:
        matplotlib Figure object
    """
    if constants is None:
        constants = get_constants()

    M_ch = chandrasekhar_mass(constants) / constants.M_sun

    # Create figure with 3 subplots stacked vertically
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18), gridspec_kw={'hspace': 0.4})

    # Main title
    if language == 'de':
        fig.suptitle('Physik der Weißen Zwerge: Pauli-Prinzip vs. Gravitation',
                    fontsize=16, fontweight='bold', y=0.98)
    else:
        fig.suptitle('White Dwarf Physics: Pauli Principle vs. Gravity',
                    fontsize=16, fontweight='bold', y=0.98)

    # 1. Pressure vs Density (top left)
    rho = np.logspace(7, 11, 100)
    P_nr = np.array([electron_degeneracy_pressure_nr(r, constants) for r in rho])
    P_r = np.array([electron_degeneracy_pressure_r(r, constants) for r in rho])

    ax1.loglog(rho, P_nr, '-', color=COLORS['non_relativistic'], linewidth=2, label='Non-rel: P ∝ ρ^(5/3)')
    ax1.loglog(rho, P_r, '-', color=COLORS['relativistic'], linewidth=2, label='Rel: P ∝ ρ^(4/3)')
    ax1.set_xlabel('Density (kg/m³)' if language == 'en' else 'Dichte (kg/m³)', fontsize=11)
    ax1.set_ylabel('Pressure (Pa)' if language == 'en' else 'Druck (Pa)', fontsize=11)
    ax1.set_title('1. Degeneracy Pressure' if language == 'en' else '1. Entartungsdruck', fontsize=14, fontweight='bold', pad=15)
    ax1.legend(fontsize=9, loc='upper right', bbox_to_anchor=(1.0, -0.08), framealpha=0.7)
    ax1.grid(True, alpha=0.3)

    # 2. Mass-Radius (top middle)
    masses = np.linspace(0.2, 1.38, 50)
    radii = [calculate_white_dwarf(m, constants).radius_solar for m in masses]

    ax2.plot(masses, radii, '-', color=COLORS['white_dwarf'], linewidth=2, label='R ∝ M^(-1/3)')
    ax2.axvline(x=M_ch, color=COLORS['scaled'], linestyle='--', linewidth=1.5, label=f'M_Ch ≈ {M_ch:.1f} M☉')
    ax2.set_xlabel('Mass (M☉)' if language == 'en' else 'Masse (M☉)', fontsize=11)
    ax2.set_ylabel('Radius (R☉)' if language == 'en' else 'Radius (R☉)', fontsize=11)
    ax2.set_title('2. Mass-Radius Relation' if language == 'en' else '2. Masse-Radius', fontsize=14, fontweight='bold', pad=15)
    ax2.legend(fontsize=9, loc='upper right', bbox_to_anchor=(1.0, -0.08), framealpha=0.7)
    ax2.grid(True, alpha=0.3)

    # 3. Comparison chart (top right)
    objects = ['Earth', 'White\nDwarf', 'Neutron\nStar']
    if language == 'de':
        objects = ['Erde', 'Weisser\nZwerg', 'Neutronen-\nstern']
    densities = [5500, 1e9, 1e17]  # kg/m³ (approximate)
    colors_bar = [COLORS['earth'], COLORS['white_dwarf'], COLORS['neutron_star']]

    # Create bars individually for separate legend entries
    bar_labels = ['Earth', 'White Dwarf', 'Neutron Star'] if language == 'en' else ['Erde', 'Weißer Zwerg', 'Neutronenstern']
    for i, (obj, density, color, lbl) in enumerate(zip(objects, densities, colors_bar, bar_labels)):
        ax3.bar(obj, density, color=color, edgecolor='black', label=lbl)

    ax3.set_yscale('log')
    ax3.set_ylabel('Density (kg/m³)' if language == 'en' else 'Dichte (kg/m³)', fontsize=11)
    ax3.set_title('3. Density Comparison' if language == 'en' else '3. Dichtevergleich', fontsize=14, fontweight='bold', pad=15)

    # Add value labels on bars
    for i, (obj, density) in enumerate(zip(objects, densities)):
        ax3.text(i, density * 2, f'{density:.0e}', ha='center', va='bottom', fontsize=9)

    # Set y-axis limit to give space for the label above the tallest bar
    ax3.set_ylim(1e3, 1e19)
    ax3.legend(fontsize=9, loc='upper right', bbox_to_anchor=(1.0, -0.08), ncol=3, framealpha=0.7)
    ax3.grid(True, alpha=0.3, axis='y')

    if save:
        os.makedirs(VIS_DIR, exist_ok=True)
        filepath = os.path.join(VIS_DIR, 'white_dwarf_summary.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")

    if show:
        plt.show()

    return fig


def generate_all_white_dwarf_plots(language: str = 'en', show: bool = False) -> List[plt.Figure]:
    """
    Generate all white dwarf visualizations.
    Erzeugt alle Weisse-Zwerg-Visualisierungen.

    Args:
        language: 'en' for English, 'de' for German
        show: Whether to display plots

    Returns:
        List of matplotlib Figure objects
    """
    figures = []

    print("Generating white dwarf visualizations...")
    print("=" * 50)

    # 1. Pressure vs density
    print("1. Pressure vs density curves...")
    figures.append(plot_pressure_vs_density(language=language, show=show))

    # 2. Mass-radius relation
    print("2. Mass-radius relationship...")
    figures.append(plot_mass_radius_relation(language=language, show=show))

    # 3. Chandrasekhar limit
    print("3. Chandrasekhar limit visualization...")
    figures.append(plot_chandrasekhar_limit(language=language, show=show))

    # 4. Summary
    print("4. Comprehensive summary...")
    figures.append(plot_white_dwarf_summary(language=language, show=show))

    print("=" * 50)
    print(f"Generated {len(figures)} visualizations in {VIS_DIR}")

    return figures


def verify_white_dwarf_physics():
    """
    Verify white dwarf physics calculations.
    """
    print("=" * 70)
    print("WHITE DWARF PHYSICS VERIFICATION")
    print("=" * 70)

    c = get_constants()

    # 1. Chandrasekhar mass
    print("\n1. CHANDRASEKHAR MASS")
    print("-" * 50)
    M_ch = chandrasekhar_mass(c)
    M_ch_solar = M_ch / c.M_sun
    print(f"   Calculated M_Ch = {M_ch:.4e} kg")
    print(f"                   = {M_ch_solar:.3f} M_sun")
    print(f"   Expected: ~1.4 M_sun (CHECK: {'PASS' if 1.35 < M_ch_solar < 1.45 else 'FAIL'})")

    # 2. Typical white dwarf properties
    print("\n2. TYPICAL WHITE DWARF (0.6 M_sun)")
    print("-" * 50)
    wd = calculate_white_dwarf(0.6, c)
    print(f"   Mass:     {wd.mass_solar:.2f} M_sun")
    print(f"   Radius:   {wd.radius_solar:.4f} R_sun = {wd.radius/1000:.0f} km")
    print(f"   Density:  {wd.density:.2e} kg/m³")
    print(f"   Stable:   {wd.is_stable}")
    expected_radius_km = 8000  # ~8000 km for typical WD
    actual_radius_km = wd.radius / 1000
    print(f"   Expected radius: ~{expected_radius_km} km (CHECK: {'PASS' if 5000 < actual_radius_km < 12000 else 'FAIL'})")

    # 3. Near limit
    print("\n3. NEAR CHANDRASEKHAR LIMIT (1.3 M_sun)")
    print("-" * 50)
    wd2 = calculate_white_dwarf(1.3, c)
    print(f"   Mass:     {wd2.mass_solar:.2f} M_sun")
    print(f"   Radius:   {wd2.radius_solar:.4f} R_sun = {wd2.radius/1000:.0f} km")
    print(f"   Density:  {wd2.density:.2e} kg/m³")
    print(f"   Stable:   {wd2.is_stable}")
    print(f"   Compactness: {wd2.compactness:.2e}")

    # 4. Above limit
    print("\n4. ABOVE LIMIT (1.5 M_sun)")
    print("-" * 50)
    wd3 = calculate_white_dwarf(1.5, c)
    print(f"   Mass:     {wd3.mass_solar:.2f} M_sun")
    print(f"   Stable:   {wd3.is_stable}")
    print(f"   Result:   {'Collapse (as expected)' if not wd3.is_stable else 'ERROR - should collapse!'}")

    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    print("=" * 60)
    print("White Dwarf Physics Module - Jugend forscht 2026")
    print("=" * 60)

    # Verify physics
    verify_white_dwarf_physics()

    # Generate visualizations
    print("\n")
    generate_all_white_dwarf_plots(language='en', show=False)

    print("\nDone! Check the 'visualizations' folder for output.")
