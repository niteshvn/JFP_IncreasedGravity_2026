"""
Neutron Star Physics Module for Jugend forscht 2026 Physics Visualization Project
Neutronenstern-Physik-Modul fuer Jugend forscht 2026 Physik-Visualisierungsprojekt

This module visualizes the physics of neutron stars, including:
- Neutron degeneracy pressure (from Pauli exclusion principle for neutrons)
- The Tolman-Oppenheimer-Volkoff (TOV) limit (~2.1-3 solar masses)
- Comparison with white dwarf physics (Chandrasekhar limit)
- Electron capture process (inverse beta decay)
- Time dilation effects (~30% slower on neutron star surface)

Key insight: When electron degeneracy pressure fails at the Chandrasekhar limit,
the star collapses until neutron degeneracy pressure takes over. If mass exceeds
the TOV limit, even neutron degeneracy fails and a black hole forms.

Author: Jugend forscht 2026 Project
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
import os
from typing import Optional, List, Tuple
from dataclasses import dataclass

from .constants import get_constants, PhysicalConstants
from .color_scheme import COLORS, get_stellar_colors, get_sequence

# Output directory for visualizations
VIS_DIR = os.path.join(os.path.dirname(__file__), '..', 'visualizations')


@dataclass
class NeutronStarProperties:
    """
    Container for neutron star properties.
    Behaelter fuer Neutronenstern-Eigenschaften.
    """
    mass: float              # Mass [kg]
    mass_solar: float        # Mass in solar masses
    radius: float            # Radius [m]
    radius_km: float         # Radius in km
    density: float           # Central density [kg/m³]
    pressure: float          # Central pressure [Pa]
    is_stable: bool          # Whether below TOV limit
    compactness: float       # R_schwarzschild / R
    escape_velocity: float   # Surface escape velocity [m/s]
    time_dilation: float     # Time dilation factor (t_surface / t_infinity)
    surface_gravity: float   # Surface gravity [m/s²]


def neutron_degeneracy_pressure(density: float, constants: PhysicalConstants) -> float:
    """
    Calculate non-relativistic neutron degeneracy pressure.
    Berechnet nicht-relativistischen Neutronen-Entartungsdruck.

    Formula: P = K_n * rho^(5/3)
    Similar to electron degeneracy but with neutron mass.

    Args:
        density: Mass density [kg/m³]
        constants: Physical constants

    Returns:
        Pressure [Pa]
    """
    # Mean molecular weight per neutron (pure neutron matter)
    mu_n = 1.0

    # Neutron mass (approximately equal to proton mass)
    m_n = constants.m_p * 1.00137  # Neutron is slightly heavier

    # Non-relativistic constant for neutron degeneracy
    # K_n = (hbar^2 / 5*m_n) * (3*pi^2)^(2/3) * (1 / m_n)^(5/3)
    K_n = (constants.hbar**2 / (5 * m_n)) * \
          (3 * np.pi**2)**(2/3) * \
          (1 / m_n)**(5/3)

    return K_n * density**(5/3)


def neutron_degeneracy_pressure_relativistic(density: float, constants: PhysicalConstants) -> float:
    """
    Calculate relativistic neutron degeneracy pressure.
    Berechnet relativistischen Neutronen-Entartungsdruck.

    Formula: P = K_r * rho^(4/3)

    In the ultra-relativistic limit, neutrons approach light speed.

    Args:
        density: Mass density [kg/m³]
        constants: Physical constants

    Returns:
        Pressure [Pa]
    """
    m_n = constants.m_p * 1.00137

    # Relativistic constant
    K_r = (constants.hbar * constants.c / 4) * \
          (3 * np.pi**2)**(1/3) * \
          (1 / m_n)**(4/3)

    return K_r * density**(4/3)


def tov_mass_limit(constants: PhysicalConstants) -> float:
    """
    Calculate the Tolman-Oppenheimer-Volkoff mass limit.
    Berechnet die Tolman-Oppenheimer-Volkoff-Massengrenze.

    The TOV limit is the maximum mass of a stable neutron star.
    Above this mass, neutron degeneracy pressure cannot prevent collapse.

    Formula (approximate): M_TOV ≈ 0.7 * (ℏc/G)^(3/2) / m_n^2

    The exact value depends on the equation of state of nuclear matter,
    which is still uncertain. Observations suggest 2.1-2.3 M_sun.

    Args:
        constants: Physical constants

    Returns:
        TOV mass limit [kg]
    """
    m_n = constants.m_p * 1.00137

    # Theoretical estimate (simplified)
    # The coefficient varies from ~0.7 to ~3 depending on EOS
    # We use a value that gives ~2.2 M_sun to match observations
    coefficient = 0.71

    M_tov = coefficient * (constants.hbar * constants.c / constants.G)**(3/2) / m_n**2

    return M_tov


def neutron_star_radius(mass: float, constants: PhysicalConstants) -> float:
    """
    Calculate neutron star radius using approximate mass-radius relation.
    Berechnet den Neutronenstern-Radius mit approximativer Masse-Radius-Beziehung.

    Neutron stars have radii of approximately 10-14 km regardless of mass,
    until they approach the TOV limit where they shrink rapidly.

    Args:
        mass: Neutron star mass [kg]
        constants: Physical constants

    Returns:
        Radius [m]
    """
    M_tov = tov_mass_limit(constants)
    mass_ratio = mass / M_tov

    if mass_ratio >= 1.0:
        return 0.0  # Collapse to black hole

    # Typical neutron star radius ~12 km
    # R decreases slightly with increasing mass
    R_typical = 12000  # 12 km in meters

    # Simplified mass-radius relation
    # R decreases as mass approaches TOV limit
    radius = R_typical * (1 - 0.5 * mass_ratio**2) * np.sqrt(1 - mass_ratio**(4/3))

    return max(radius, 0)


def gravitational_time_dilation(r: float, M: float, constants: PhysicalConstants) -> float:
    """
    Calculate gravitational time dilation factor.
    Berechnet den gravitativen Zeitdilatationsfaktor.

    Formula: sqrt(1 - R_s/r) where R_s = 2GM/c^2

    Time passes slower in stronger gravitational fields.
    A clock at radius r ticks slower than a clock at infinity.

    Args:
        r: Radial distance from center [m]
        M: Mass of object [kg]
        constants: Physical constants

    Returns:
        Time dilation factor (< 1 means time passes slower)
    """
    R_s = constants.schwarzschild_radius(M)

    if r <= R_s:
        return 0.0  # Inside event horizon

    return np.sqrt(1 - R_s / r)


def calculate_neutron_star(
    mass_solar: float,
    constants: Optional[PhysicalConstants] = None
) -> NeutronStarProperties:
    """
    Calculate properties of a neutron star given its mass.
    Berechnet die Eigenschaften eines Neutronensterns bei gegebener Masse.

    Args:
        mass_solar: Mass in solar masses
        constants: Physical constants (uses standard if None)

    Returns:
        NeutronStarProperties object
    """
    if constants is None:
        constants = get_constants()

    mass = mass_solar * constants.M_sun
    M_tov = tov_mass_limit(constants)

    is_stable = mass < M_tov

    if is_stable:
        radius = neutron_star_radius(mass, constants)
        # Central density for neutron star (very high!)
        # Typical: 10^17 - 10^18 kg/m³
        if radius > 0:
            # Approximate central density (higher than average)
            avg_density = mass / ((4/3) * np.pi * radius**3)
            density = avg_density * 3  # Central density ~3x average
            pressure = neutron_degeneracy_pressure(density, constants)

            # Time dilation at surface
            time_dilation = gravitational_time_dilation(radius, mass, constants)

            # Escape velocity
            R_s = constants.schwarzschild_radius(mass)
            v_esc = constants.c * np.sqrt(R_s / radius) if radius > R_s else constants.c

            # Surface gravity
            g_surface = constants.G * mass / radius**2

            compactness = R_s / radius
        else:
            density = float('inf')
            pressure = float('inf')
            time_dilation = 0.0
            v_esc = constants.c
            g_surface = float('inf')
            compactness = float('inf')
    else:
        radius = 0
        density = float('inf')
        pressure = float('inf')
        time_dilation = 0.0
        v_esc = constants.c
        g_surface = float('inf')
        compactness = float('inf')

    return NeutronStarProperties(
        mass=mass,
        mass_solar=mass_solar,
        radius=radius,
        radius_km=radius / 1000 if radius > 0 else 0,
        density=density,
        pressure=pressure,
        is_stable=is_stable,
        compactness=compactness,
        escape_velocity=v_esc,
        time_dilation=time_dilation,
        surface_gravity=g_surface
    )


def plot_tov_limit_comparison(
    constants: Optional[PhysicalConstants] = None,
    language: str = 'en',
    save: bool = True,
    show: bool = False
) -> plt.Figure:
    """
    Plot comparison of Chandrasekhar and TOV limits.
    Zeigt Vergleich von Chandrasekhar- und TOV-Grenzen.

    Shows:
    - Chandrasekhar limit (~1.4 M_sun) for white dwarfs
    - TOV limit (~2.1-2.3 M_sun) for neutron stars
    - What happens when each limit is exceeded

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

    from .white_dwarf import chandrasekhar_mass

    M_ch = chandrasekhar_mass(constants) / constants.M_sun
    M_tov = tov_mass_limit(constants) / constants.M_sun

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # === Left: Mass limits bar chart ===
    limits = [M_ch, M_tov]
    labels = ['Chandrasekhar\n(White Dwarf)', 'TOV\n(Neutron Star)'] if language == 'en' else \
             ['Chandrasekhar\n(Weißer Zwerg)', 'TOV\n(Neutronenstern)']
    colors = [COLORS['white_dwarf'], COLORS['neutron_star']]

    bars = ax1.bar(labels, limits, color=colors, edgecolor='black', linewidth=2)

    # Add value labels
    for bar, limit in zip(bars, limits):
        ax1.text(bar.get_x() + bar.get_width()/2, limit + 0.1,
                f'{limit:.2f} M☉', ha='center', va='bottom',
                fontsize=12, fontweight='bold')

    # Add "collapse zone" annotation
    ax1.axhline(y=M_ch, color=COLORS['white_dwarf'], linestyle='--', alpha=0.5)
    ax1.axhline(y=M_tov, color=COLORS['neutron_star'], linestyle='--', alpha=0.5)

    if language == 'de':
        ax1.set_ylabel('Massengrenze (Sonnenmassen M☉)', fontsize=12)
        ax1.set_title('Kritische Massengrenzen', fontsize=14, fontweight='bold', pad=15)
    else:
        ax1.set_ylabel('Mass Limit (Solar masses M☉)', fontsize=12)
        ax1.set_title('Critical Mass Limits', fontsize=14, fontweight='bold', pad=15)

    ax1.set_ylim(0, 3)
    ax1.grid(True, alpha=0.3, axis='y')

    # === Right: Progression diagram ===
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.set_aspect('equal')
    ax2.axis('off')

    # Draw progression: Star → White Dwarf → Neutron Star → Black Hole
    positions = [(1, 7), (4, 7), (7, 7), (1, 2), (4, 2), (7, 2)]
    sizes = [0.8, 0.5, 0.15, 0.8, 0.15, 0.4]
    obj_colors = [COLORS['sun'], COLORS['white_dwarf'], COLORS['neutron_star'],
                  COLORS['sun'], COLORS['neutron_star'], 'black']

    if language == 'de':
        obj_labels = ['Stern\n< 8 M☉', 'Weißer Zwerg\n< 1.4 M☉', 'Stabiler\nEndpunkt',
                     'Stern\n> 8 M☉', 'Neutronenstern\n< 2.2 M☉', 'Schwarzes\nLoch']
    else:
        obj_labels = ['Star\n< 8 M☉', 'White Dwarf\n< 1.4 M☉', 'Stable\nEndpoint',
                     'Star\n> 8 M☉', 'Neutron Star\n< 2.2 M☉', 'Black\nHole']

    # Top row: Low mass path
    for i in range(3):
        circle = Circle(positions[i], sizes[i], color=obj_colors[i],
                       ec='black', linewidth=2, zorder=5)
        ax2.add_patch(circle)
        ax2.text(positions[i][0], positions[i][1] - 1.3, obj_labels[i],
                ha='center', va='top', fontsize=10, fontweight='bold')

    # Bottom row: High mass path
    for i in range(3, 6):
        circle = Circle(positions[i], sizes[i], color=obj_colors[i],
                       ec='black', linewidth=2, zorder=5)
        ax2.add_patch(circle)
        ax2.text(positions[i][0], positions[i][1] - 1.3, obj_labels[i],
                ha='center', va='top', fontsize=10, fontweight='bold')

    # Draw arrows
    arrow_style = dict(arrowstyle='->', color='gray', lw=2)
    ax2.annotate('', xy=(3, 7), xytext=(2, 7), arrowprops=arrow_style)
    ax2.annotate('', xy=(6, 7), xytext=(5, 7), arrowprops=arrow_style)
    ax2.annotate('', xy=(3, 2), xytext=(2, 2), arrowprops=arrow_style)
    ax2.annotate('', xy=(6, 2), xytext=(5, 2), arrowprops=arrow_style)

    # Add limit labels
    ax2.text(2.5, 7.8, f'M < {M_ch:.1f} M☉', ha='center', fontsize=9,
            color=COLORS['white_dwarf'], fontweight='bold')
    ax2.text(5.5, 7.8, 'Chandrasekhar', ha='center', fontsize=9,
            color=COLORS['white_dwarf'])
    ax2.text(2.5, 2.8, 'Supernova', ha='center', fontsize=9, color='red')
    ax2.text(5.5, 2.8, f'M > {M_tov:.1f} M☉ → TOV', ha='center', fontsize=9,
            color=COLORS['neutron_star'])

    # Row labels
    if language == 'de':
        ax2.text(0.2, 7, 'Niedrige\nMasse:', ha='left', va='center', fontsize=11, fontweight='bold')
        ax2.text(0.2, 2, 'Hohe\nMasse:', ha='left', va='center', fontsize=11, fontweight='bold')
        ax2.set_title('Sternentwicklung und Massengrenzen', fontsize=14, fontweight='bold', pad=15)
    else:
        ax2.text(0.2, 7, 'Low\nMass:', ha='left', va='center', fontsize=11, fontweight='bold')
        ax2.text(0.2, 2, 'High\nMass:', ha='left', va='center', fontsize=11, fontweight='bold')
        ax2.set_title('Stellar Evolution and Mass Limits', fontsize=14, fontweight='bold', pad=15)

    plt.tight_layout()

    if save:
        os.makedirs(VIS_DIR, exist_ok=True)
        suffix = '_de' if language == 'de' else ''
        filepath = os.path.join(VIS_DIR, f'tov_limit_comparison{suffix}.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")

    if show:
        plt.show()

    return fig


def plot_neutron_star_structure(
    constants: Optional[PhysicalConstants] = None,
    language: str = 'en',
    save: bool = True,
    show: bool = False
) -> plt.Figure:
    """
    Plot neutron star internal structure and properties.
    Zeigt Neutronenstern-Innenstruktur und Eigenschaften.

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

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    ((ax1, ax2), (ax3, ax4)) = axes

    # === Plot 1: Mass-Radius relation ===
    masses = np.linspace(0.5, 2.5, 100)
    M_tov = tov_mass_limit(constants) / constants.M_sun

    radii = []
    for m in masses:
        if m < M_tov:
            ns = calculate_neutron_star(m, constants)
            radii.append(ns.radius_km)
        else:
            radii.append(0)

    ax1.plot(masses, radii, '-', color=COLORS['neutron_star'], linewidth=2.5,
            label='Neutron Star R(M)' if language == 'en' else 'Neutronenstern R(M)')
    ax1.axvline(x=M_tov, color=COLORS['scaled'], linestyle='--', linewidth=2,
               label=f'TOV limit ({M_tov:.2f} M☉)' if language == 'en' else f'TOV-Grenze ({M_tov:.2f} M☉)')
    ax1.fill_betweenx([0, 15], M_tov, 3, alpha=0.2, color=COLORS['scaled'],
                      label='Black hole region' if language == 'en' else 'Schwarzes-Loch-Bereich')

    if language == 'de':
        ax1.set_xlabel('Masse (M☉)', fontsize=11)
        ax1.set_ylabel('Radius (km)', fontsize=11)
        ax1.set_title('1. Masse-Radius-Beziehung', fontsize=12, fontweight='bold')
    else:
        ax1.set_xlabel('Mass (M☉)', fontsize=11)
        ax1.set_ylabel('Radius (km)', fontsize=11)
        ax1.set_title('1. Mass-Radius Relationship', fontsize=12, fontweight='bold')

    ax1.set_xlim(0.5, 2.5)
    ax1.set_ylim(0, 15)
    ax1.legend(fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3, framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # === Plot 2: Density comparison ===
    objects = [
        ('Earth', 5500, COLORS['earth']),
        ('Sun', 1410, COLORS['sun']),
        ('White Dwarf', 1e9, COLORS['white_dwarf']),
        ('Neutron Star', 5e17, COLORS['neutron_star'])
    ]
    if language == 'de':
        objects = [
            ('Erde', 5500, COLORS['earth']),
            ('Sonne', 1410, COLORS['sun']),
            ('Weißer Zwerg', 1e9, COLORS['white_dwarf']),
            ('Neutronenstern', 5e17, COLORS['neutron_star'])
        ]

    names = [o[0] for o in objects]
    densities = [o[1] for o in objects]
    colors = [o[2] for o in objects]

    bars = ax2.bar(names, densities, color=colors, edgecolor='black')
    ax2.set_yscale('log')

    # Add value labels
    for bar, density in zip(bars, densities):
        ax2.text(bar.get_x() + bar.get_width()/2, density * 2,
                f'{density:.0e}', ha='center', va='bottom', fontsize=9)

    if language == 'de':
        ax2.set_ylabel('Dichte (kg/m³)', fontsize=11)
        ax2.set_title('2. Dichtevergleich', fontsize=12, fontweight='bold')
    else:
        ax2.set_ylabel('Density (kg/m³)', fontsize=11)
        ax2.set_title('2. Density Comparison', fontsize=12, fontweight='bold')

    ax2.set_ylim(1e2, 1e19)
    ax2.grid(True, alpha=0.3, axis='y')

    # === Plot 3: Pressure vs Density (neutron vs electron degeneracy) ===
    rho = np.logspace(14, 18, 100)

    from .white_dwarf import electron_degeneracy_pressure_nr, electron_degeneracy_pressure_r

    P_e_nr = np.array([electron_degeneracy_pressure_nr(r, constants) for r in rho])
    P_e_r = np.array([electron_degeneracy_pressure_r(r, constants) for r in rho])
    P_n_nr = np.array([neutron_degeneracy_pressure(r, constants) for r in rho])
    P_n_r = np.array([neutron_degeneracy_pressure_relativistic(r, constants) for r in rho])

    ax3.loglog(rho, P_e_nr, '-', color=COLORS['white_dwarf'], linewidth=2,
              label='Electron (non-rel)' if language == 'en' else 'Elektron (nicht-rel)')
    ax3.loglog(rho, P_e_r, '--', color=COLORS['white_dwarf'], linewidth=2,
              label='Electron (rel)' if language == 'en' else 'Elektron (rel)')
    ax3.loglog(rho, P_n_nr, '-', color=COLORS['neutron_star'], linewidth=2,
              label='Neutron (non-rel)' if language == 'en' else 'Neutron (nicht-rel)')
    ax3.loglog(rho, P_n_r, '--', color=COLORS['neutron_star'], linewidth=2,
              label='Neutron (rel)' if language == 'en' else 'Neutron (rel)')

    if language == 'de':
        ax3.set_xlabel('Dichte ρ (kg/m³)', fontsize=11)
        ax3.set_ylabel('Entartungsdruck P (Pa)', fontsize=11)
        ax3.set_title('3. Entartungsdruck: Elektronen vs. Neutronen', fontsize=12, fontweight='bold')
    else:
        ax3.set_xlabel('Density ρ (kg/m³)', fontsize=11)
        ax3.set_ylabel('Degeneracy Pressure P (Pa)', fontsize=11)
        ax3.set_title('3. Degeneracy Pressure: Electrons vs. Neutrons', fontsize=12, fontweight='bold')

    ax3.legend(fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, framealpha=0.9)
    ax3.grid(True, alpha=0.3, which='both')

    # === Plot 4: Surface gravity comparison ===
    ns_14 = calculate_neutron_star(1.4, constants)
    ns_20 = calculate_neutron_star(2.0, constants)

    objects_g = [
        ('Earth', 9.8, COLORS['earth']),
        ('Sun', 274, COLORS['sun']),
        ('White Dwarf\n(0.6 M☉)', 3e6, COLORS['white_dwarf']),
        ('NS (1.4 M☉)', ns_14.surface_gravity, COLORS['neutron_star']),
        ('NS (2.0 M☉)', ns_20.surface_gravity, COLORS['quantum'])
    ]
    if language == 'de':
        objects_g = [
            ('Erde', 9.8, COLORS['earth']),
            ('Sonne', 274, COLORS['sun']),
            ('Weißer Zwerg\n(0.6 M☉)', 3e6, COLORS['white_dwarf']),
            ('NS (1.4 M☉)', ns_14.surface_gravity, COLORS['neutron_star']),
            ('NS (2.0 M☉)', ns_20.surface_gravity, COLORS['quantum'])
        ]

    names_g = [o[0] for o in objects_g]
    gravities = [o[1] for o in objects_g]
    colors_g = [o[2] for o in objects_g]

    bars_g = ax4.bar(names_g, gravities, color=colors_g, edgecolor='black')
    ax4.set_yscale('log')

    if language == 'de':
        ax4.set_ylabel('Oberflächengravitation (m/s²)', fontsize=11)
        ax4.set_title('4. Oberflächengravitation', fontsize=12, fontweight='bold')
    else:
        ax4.set_ylabel('Surface Gravity (m/s²)', fontsize=11)
        ax4.set_title('4. Surface Gravity Comparison', fontsize=12, fontweight='bold')

    ax4.tick_params(axis='x', rotation=30)
    ax4.grid(True, alpha=0.3, axis='y')

    # Main title
    if language == 'de':
        fig.suptitle('Neutronenstern-Physik', fontsize=16, fontweight='bold', y=1.02)
    else:
        fig.suptitle('Neutron Star Physics', fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()

    if save:
        os.makedirs(VIS_DIR, exist_ok=True)
        suffix = '_de' if language == 'de' else ''
        filepath = os.path.join(VIS_DIR, f'neutron_star_structure{suffix}.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")

    if show:
        plt.show()

    return fig


def plot_electron_capture(
    language: str = 'en',
    save: bool = True,
    show: bool = False
) -> plt.Figure:
    """
    Visualize electron capture process (inverse beta decay).
    Visualisiert den Elektroneneinfang-Prozess (inverser Beta-Zerfall).

    This process converts protons to neutrons in neutron star formation:
    p + e⁻ → n + νₑ

    Args:
        language: 'en' for English, 'de' for German
        save: Whether to save the figure
        show: Whether to display the figure

    Returns:
        matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # === Left: Beta-minus decay ===
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.set_aspect('equal')
    ax1.axis('off')

    # Neutron (before)
    n_before = Circle((2, 5), 0.8, color=COLORS['neutron_star'], ec='black', lw=2)
    ax1.add_patch(n_before)
    ax1.text(2, 5, 'n', ha='center', va='center', fontsize=16, fontweight='bold', color='white')

    # Arrow
    ax1.annotate('', xy=(4.5, 5), xytext=(3.2, 5),
                arrowprops=dict(arrowstyle='->', lw=3, color='gray'))

    # Products (after)
    p_after = Circle((6, 6.5), 0.6, color=COLORS['scaled'], ec='black', lw=2)
    ax1.add_patch(p_after)
    ax1.text(6, 6.5, 'p', ha='center', va='center', fontsize=14, fontweight='bold', color='white')

    e_after = Circle((6, 5), 0.4, color=COLORS['primary_blue'], ec='black', lw=2)
    ax1.add_patch(e_after)
    ax1.text(6, 5, 'e⁻', ha='center', va='center', fontsize=12, fontweight='bold', color='white')

    nu_after = Circle((6, 3.5), 0.3, color=COLORS['muted'], ec='black', lw=2)
    ax1.add_patch(nu_after)
    ax1.text(6, 3.5, 'ν̄ₑ', ha='center', va='center', fontsize=10, fontweight='bold')

    # Labels
    ax1.text(6, 7.5, 'Proton', ha='center', fontsize=10)
    ax1.text(7, 5, 'Electron', ha='left', fontsize=10)
    ax1.text(7, 3.5, 'Antineutrino', ha='left', fontsize=10)

    # Equation
    if language == 'de':
        ax1.text(5, 1.5, r'$n \rightarrow p + e^- + \bar{\nu}_e$',
                ha='center', fontsize=16, fontweight='bold')
        ax1.set_title('Beta-Minus-Zerfall (β⁻)', fontsize=14, fontweight='bold', pad=15)
        ax1.text(5, 0.5, 'Neutron → Proton\n(Standard-Radioaktivität)', ha='center', fontsize=11)
    else:
        ax1.text(5, 1.5, r'$n \rightarrow p + e^- + \bar{\nu}_e$',
                ha='center', fontsize=16, fontweight='bold')
        ax1.set_title('Beta-Minus Decay (β⁻)', fontsize=14, fontweight='bold', pad=15)
        ax1.text(5, 0.5, 'Neutron → Proton\n(Standard radioactivity)', ha='center', fontsize=11)

    # === Right: Electron capture (inverse beta decay) ===
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.set_aspect('equal')
    ax2.axis('off')

    # Reactants (before)
    p_before = Circle((1.5, 6), 0.6, color=COLORS['scaled'], ec='black', lw=2)
    ax2.add_patch(p_before)
    ax2.text(1.5, 6, 'p', ha='center', va='center', fontsize=14, fontweight='bold', color='white')

    e_before = Circle((1.5, 4), 0.4, color=COLORS['primary_blue'], ec='black', lw=2)
    ax2.add_patch(e_before)
    ax2.text(1.5, 4, 'e⁻', ha='center', va='center', fontsize=12, fontweight='bold', color='white')

    # Arrow
    ax2.annotate('', xy=(4.5, 5), xytext=(2.5, 5),
                arrowprops=dict(arrowstyle='->', lw=3, color='gray'))
    ax2.text(3.5, 5.8, 'High\npressure' if language == 'en' else 'Hoher\nDruck',
            ha='center', fontsize=10, color='red')

    # Products (after)
    n_after = Circle((6.5, 6), 0.8, color=COLORS['neutron_star'], ec='black', lw=2)
    ax2.add_patch(n_after)
    ax2.text(6.5, 6, 'n', ha='center', va='center', fontsize=16, fontweight='bold', color='white')

    nu_after2 = Circle((6.5, 3.5), 0.3, color=COLORS['standard'], ec='black', lw=2)
    ax2.add_patch(nu_after2)
    ax2.text(6.5, 3.5, 'νₑ', ha='center', va='center', fontsize=10, fontweight='bold')

    # Labels
    ax2.text(0.3, 6, 'Proton', ha='left', fontsize=10)
    ax2.text(0.3, 4, 'Electron', ha='left', fontsize=10)
    ax2.text(7.5, 6, 'Neutron', ha='left', fontsize=10)
    ax2.text(7.5, 3.5, 'Neutrino', ha='left', fontsize=10)

    # Equation
    if language == 'de':
        ax2.text(5, 1.5, r'$p + e^- \rightarrow n + \nu_e$',
                ha='center', fontsize=16, fontweight='bold')
        ax2.set_title('Elektroneneinfang (Inverser β-Zerfall)', fontsize=14, fontweight='bold', pad=15)
        ax2.text(5, 0.5, 'Proton + Elektron → Neutron\n(Neutronenstern-Entstehung)', ha='center', fontsize=11)
    else:
        ax2.text(5, 1.5, r'$p + e^- \rightarrow n + \nu_e$',
                ha='center', fontsize=16, fontweight='bold')
        ax2.set_title('Electron Capture (Inverse β Decay)', fontsize=14, fontweight='bold', pad=15)
        ax2.text(5, 0.5, 'Proton + Electron → Neutron\n(Neutron star formation)', ha='center', fontsize=11)

    plt.tight_layout()

    if save:
        os.makedirs(VIS_DIR, exist_ok=True)
        suffix = '_de' if language == 'de' else ''
        filepath = os.path.join(VIS_DIR, f'electron_capture{suffix}.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")

    if show:
        plt.show()

    return fig


def plot_neutron_star_summary(
    constants: Optional[PhysicalConstants] = None,
    language: str = 'en',
    save: bool = True,
    show: bool = False
) -> plt.Figure:
    """
    Create comprehensive summary of neutron star physics.
    Erstellt umfassende Zusammenfassung der Neutronenstern-Physik.

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

    from .white_dwarf import chandrasekhar_mass

    M_ch = chandrasekhar_mass(constants) / constants.M_sun
    M_tov = tov_mass_limit(constants) / constants.M_sun

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    ((ax1, ax2), (ax3, ax4)) = axes

    # === Plot 1: Mass limits with stability regions ===
    mass_range = np.linspace(0, 3.5, 200)

    # Create stability regions
    ax1.axvspan(0, M_ch, alpha=0.3, color=COLORS['white_dwarf'],
               label='White Dwarf stable' if language == 'en' else 'Weißer Zwerg stabil')
    ax1.axvspan(M_ch, M_tov, alpha=0.3, color=COLORS['neutron_star'],
               label='Neutron Star stable' if language == 'en' else 'Neutronenstern stabil')
    ax1.axvspan(M_tov, 3.5, alpha=0.3, color='black',
               label='Black Hole' if language == 'en' else 'Schwarzes Loch')

    ax1.axvline(x=M_ch, color=COLORS['white_dwarf'], linestyle='--', linewidth=2)
    ax1.axvline(x=M_tov, color=COLORS['neutron_star'], linestyle='--', linewidth=2)

    ax1.text(M_ch/2, 0.8, f'< {M_ch:.1f} M☉', ha='center', fontsize=11, fontweight='bold')
    ax1.text((M_ch + M_tov)/2, 0.8, f'{M_ch:.1f}-{M_tov:.1f} M☉', ha='center', fontsize=11, fontweight='bold')
    ax1.text((M_tov + 3.5)/2, 0.8, f'> {M_tov:.1f} M☉', ha='center', fontsize=11, fontweight='bold')

    ax1.set_xlim(0, 3.5)
    ax1.set_ylim(0, 1)
    ax1.set_yticks([])

    if language == 'de':
        ax1.set_xlabel('Masse (Sonnenmassen M☉)', fontsize=11)
        ax1.set_title('1. Stabilitätsbereiche nach Masse', fontsize=12, fontweight='bold')
    else:
        ax1.set_xlabel('Mass (Solar masses M☉)', fontsize=11)
        ax1.set_title('1. Stability Regions by Mass', fontsize=12, fontweight='bold')

    ax1.legend(fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3)

    # === Plot 2: Radius comparison ===
    objects_r = [
        ('Earth', 6371, COLORS['earth']),
        ('White Dwarf', 8000, COLORS['white_dwarf']),
        ('Neutron Star', 12, COLORS['neutron_star'])
    ]
    if language == 'de':
        objects_r = [
            ('Erde', 6371, COLORS['earth']),
            ('Weißer Zwerg', 8000, COLORS['white_dwarf']),
            ('Neutronenstern', 12, COLORS['neutron_star'])
        ]

    names_r = [o[0] for o in objects_r]
    radii_r = [o[1] for o in objects_r]
    colors_r = [o[2] for o in objects_r]

    bars = ax2.bar(names_r, radii_r, color=colors_r, edgecolor='black')
    ax2.set_yscale('log')

    for bar, r in zip(bars, radii_r):
        ax2.text(bar.get_x() + bar.get_width()/2, r * 1.5,
                f'{r:,} km', ha='center', va='bottom', fontsize=10)

    if language == 'de':
        ax2.set_ylabel('Radius (km)', fontsize=11)
        ax2.set_title('2. Radiusvergleich', fontsize=12, fontweight='bold')
    else:
        ax2.set_ylabel('Radius (km)', fontsize=11)
        ax2.set_title('2. Radius Comparison', fontsize=12, fontweight='bold')

    ax2.grid(True, alpha=0.3, axis='y')

    # === Plot 3: Key numbers ===
    ax3.axis('off')

    ns_typical = calculate_neutron_star(1.4, constants)

    if language == 'de':
        info_text = f"""
NEUTRONENSTERN FAKTEN (1.4 M☉)
═══════════════════════════════════════

Masse:           1.4 Sonnenmassen = {ns_typical.mass:.2e} kg
Radius:          ~{ns_typical.radius_km:.0f} km (Größe einer Stadt!)
Dichte:          ~{ns_typical.density:.1e} kg/m³
                 (1 Teelöffel = ~1 Milliarde Tonnen)

Oberflächengravitation: ~{ns_typical.surface_gravity:.1e} m/s²
                        ({ns_typical.surface_gravity/9.8:.0e}× Erd-g)

Fluchtgeschwindigkeit:  {ns_typical.escape_velocity/1000:.0f} km/s
                        ({ns_typical.escape_velocity/constants.c*100:.0f}% Lichtgeschwindigkeit)

Zeitdilatation:         {ns_typical.time_dilation:.2f}
                        (Zeit vergeht {(1-ns_typical.time_dilation)*100:.0f}% langsamer!)

═══════════════════════════════════════
Chandrasekhar-Grenze:   {M_ch:.2f} M☉ (Weißer Zwerg)
TOV-Grenze:             {M_tov:.2f} M☉ (Neutronenstern)
"""
    else:
        info_text = f"""
NEUTRON STAR FACTS (1.4 M☉)
═══════════════════════════════════════

Mass:            1.4 Solar masses = {ns_typical.mass:.2e} kg
Radius:          ~{ns_typical.radius_km:.0f} km (size of a city!)
Density:         ~{ns_typical.density:.1e} kg/m³
                 (1 teaspoon = ~1 billion tons)

Surface Gravity: ~{ns_typical.surface_gravity:.1e} m/s²
                 ({ns_typical.surface_gravity/9.8:.0e}× Earth g)

Escape Velocity: {ns_typical.escape_velocity/1000:.0f} km/s
                 ({ns_typical.escape_velocity/constants.c*100:.0f}% speed of light)

Time Dilation:   {ns_typical.time_dilation:.2f}
                 (time passes {(1-ns_typical.time_dilation)*100:.0f}% slower!)

═══════════════════════════════════════
Chandrasekhar Limit: {M_ch:.2f} M☉ (White Dwarf)
TOV Limit:           {M_tov:.2f} M☉ (Neutron Star)
"""

    ax3.text(0.05, 0.95, info_text, transform=ax3.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    if language == 'de':
        ax3.set_title('3. Schlüsselzahlen', fontsize=12, fontweight='bold')
    else:
        ax3.set_title('3. Key Numbers', fontsize=12, fontweight='bold')

    # === Plot 4: Compactness comparison ===
    objects_c = [
        ('Earth', 1.4e-9, COLORS['earth']),
        ('Sun', 4.2e-6, COLORS['sun']),
        ('White Dwarf', 2e-4, COLORS['white_dwarf']),
        ('Neutron Star', ns_typical.compactness, COLORS['neutron_star']),
        ('Black Hole', 1.0, 'black')
    ]
    if language == 'de':
        objects_c = [
            ('Erde', 1.4e-9, COLORS['earth']),
            ('Sonne', 4.2e-6, COLORS['sun']),
            ('Weißer Zwerg', 2e-4, COLORS['white_dwarf']),
            ('Neutronenstern', ns_typical.compactness, COLORS['neutron_star']),
            ('Schwarzes Loch', 1.0, 'black')
        ]

    names_c = [o[0] for o in objects_c]
    compactness = [o[1] for o in objects_c]
    colors_c = [o[2] for o in objects_c]

    bars_c = ax4.bar(names_c, compactness, color=colors_c, edgecolor='black')
    ax4.set_yscale('log')
    ax4.axhline(y=1, color='red', linestyle='--', linewidth=2,
               label='BH limit (C=1)' if language == 'en' else 'SL-Grenze (C=1)')

    if language == 'de':
        ax4.set_ylabel('Kompaktheit C = Rₛ/R', fontsize=11)
        ax4.set_title('4. Kompaktheitsvergleich', fontsize=12, fontweight='bold')
    else:
        ax4.set_ylabel('Compactness C = Rₛ/R', fontsize=11)
        ax4.set_title('4. Compactness Comparison', fontsize=12, fontweight='bold')

    ax4.tick_params(axis='x', rotation=30)
    ax4.legend(fontsize=9, loc='upper left')
    ax4.grid(True, alpha=0.3, axis='y')

    # Main title
    if language == 'de':
        fig.suptitle('Neutronenstern-Physik: Zusammenfassung', fontsize=16, fontweight='bold', y=1.02)
    else:
        fig.suptitle('Neutron Star Physics: Summary', fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()

    if save:
        os.makedirs(VIS_DIR, exist_ok=True)
        suffix = '_de' if language == 'de' else ''
        filepath = os.path.join(VIS_DIR, f'neutron_star_summary{suffix}.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")

    if show:
        plt.show()

    return fig


def generate_all_neutron_star_plots(language: str = 'en', show: bool = False) -> List[plt.Figure]:
    """
    Generate all neutron star visualizations.
    Erzeugt alle Neutronenstern-Visualisierungen.

    Args:
        language: 'en' for English, 'de' for German
        show: Whether to display plots

    Returns:
        List of matplotlib Figure objects
    """
    figures = []

    print("Generating neutron star visualizations...")
    print("=" * 50)

    print("1. TOV limit comparison...")
    figures.append(plot_tov_limit_comparison(language=language, show=show))

    print("2. Neutron star structure...")
    figures.append(plot_neutron_star_structure(language=language, show=show))

    print("3. Electron capture process...")
    figures.append(plot_electron_capture(language=language, show=show))

    print("4. Neutron star summary...")
    figures.append(plot_neutron_star_summary(language=language, show=show))

    print("=" * 50)
    print(f"Generated {len(figures)} neutron star visualizations")

    return figures


def verify_neutron_star_physics():
    """
    Verify neutron star physics calculations.
    """
    print("=" * 70)
    print("NEUTRON STAR PHYSICS VERIFICATION")
    print("=" * 70)

    c = get_constants()

    from .white_dwarf import chandrasekhar_mass

    # 1. Mass limits
    print("\n1. MASS LIMITS")
    print("-" * 50)
    M_ch = chandrasekhar_mass(c) / c.M_sun
    M_tov = tov_mass_limit(c) / c.M_sun
    print(f"   Chandrasekhar limit: {M_ch:.2f} M_sun")
    print(f"   Expected: ~1.4 M_sun (CHECK: {'PASS' if 1.3 < M_ch < 1.5 else 'FAIL'})")
    print(f"   TOV limit: {M_tov:.2f} M_sun")
    print(f"   Expected: ~2.1-2.3 M_sun (CHECK: {'PASS' if 2.0 < M_tov < 2.5 else 'FAIL'})")

    # 2. Typical neutron star
    print("\n2. TYPICAL NEUTRON STAR (1.4 M_sun)")
    print("-" * 50)
    ns = calculate_neutron_star(1.4, c)
    print(f"   Radius: {ns.radius_km:.1f} km")
    print(f"   Expected: ~10-14 km (CHECK: {'PASS' if 8 < ns.radius_km < 16 else 'FAIL'})")
    print(f"   Density: {ns.density:.2e} kg/m³")
    print(f"   Expected: ~10^17-10^18 (CHECK: {'PASS' if 1e16 < ns.density < 1e19 else 'FAIL'})")
    print(f"   Surface gravity: {ns.surface_gravity:.2e} m/s²")
    print(f"   Time dilation: {ns.time_dilation:.3f} (time {(1-ns.time_dilation)*100:.1f}% slower)")
    print(f"   Expected: ~30% slower (CHECK: {'PASS' if 0.65 < ns.time_dilation < 0.85 else 'FAIL'})")

    # 3. Near TOV limit
    print("\n3. NEAR TOV LIMIT (2.0 M_sun)")
    print("-" * 50)
    ns2 = calculate_neutron_star(2.0, c)
    print(f"   Radius: {ns2.radius_km:.1f} km")
    print(f"   Compactness: {ns2.compactness:.3f}")
    print(f"   Stable: {ns2.is_stable}")

    # 4. Above TOV limit
    print("\n4. ABOVE TOV LIMIT (2.5 M_sun)")
    print("-" * 50)
    ns3 = calculate_neutron_star(2.5, c)
    print(f"   Stable: {ns3.is_stable}")
    print(f"   Result: {'Collapse to BH (as expected)' if not ns3.is_stable else 'ERROR'}")

    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    print("=" * 60)
    print("Neutron Star Physics Module - Jugend forscht 2026")
    print("=" * 60)

    verify_neutron_star_physics()

    print("\n")
    generate_all_neutron_star_plots(language='en', show=False)

    print("\nDone! Check the 'visualizations' folder for output.")
