"""
Gravity vs Pauli Balance Module for Jugend forscht 2026 Physics Visualization Project
Gravitation-vs-Pauli-Gleichgewicht-Modul fuer Jugend forscht 2026 Physik-Visualisierungsprojekt

This module addresses the core hypothesis of the essay:
"How does increasing gravitational pressure modify the balance between
 electron degeneracy pressure (Pauli) and Coulomb interactions?"

Key visualizations:
1. Earth radius/structural effects under modified constants
2. Explicit comparison of gravity vs Pauli (degeneracy) pressure
3. Critical mass thresholds and collapse conditions
4. Validation/negation of the essay hypothesis

The hypothesis: When G increases, gravitational pressure increases, potentially
overcoming electron degeneracy pressure in objects that are normally stable.

Author: Jugend forscht 2026 Project
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
import os
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass

from .constants import get_constants, PhysicalConstants
from .color_scheme import COLORS, get_sequence


# Output directory for visualizations
VIS_DIR = os.path.join(os.path.dirname(__file__), '..', 'visualizations')


@dataclass
class PlanetaryEquilibrium:
    """
    Container for planetary equilibrium properties.
    Behaelter fuer planetares Gleichgewichts-Eigenschaften.
    """
    mass: float                    # Mass [kg]
    radius: float                  # Equilibrium radius [m]
    central_pressure: float        # Central pressure [Pa]
    gravitational_pressure: float  # Gravitational pressure [Pa]
    degeneracy_pressure: float     # Electron degeneracy pressure [Pa]
    coulomb_pressure: float        # Coulomb/thermal pressure [Pa]
    pressure_ratio: float          # P_gravity / P_support
    is_stable: bool                # Whether object can support itself
    compression_factor: float      # Radius / standard_radius


def gravitational_central_pressure(mass: float, radius: float, constants: PhysicalConstants) -> float:
    """
    Estimate central gravitational pressure in a uniform density sphere.
    Schaetzt den zentralen Gravitationsdruck in einer Kugel mit gleichmaessiger Dichte.

    Formula: P_c ≈ (3/8π) × G × M² / R⁴

    Args:
        mass: Object mass [kg]
        radius: Object radius [m]
        constants: Physical constants

    Returns:
        Central pressure [Pa]
    """
    return (3 / (8 * np.pi)) * constants.G * mass**2 / radius**4


def electron_degeneracy_pressure_simple(density: float, constants: PhysicalConstants) -> float:
    """
    Calculate electron degeneracy pressure (non-relativistic).
    Berechnet Elektronen-Entartungsdruck (nicht-relativistisch).

    Formula: P = K × (ρ/μ_e m_p)^(5/3)
    where K = (ℏ²/m_e) × (3π²)^(2/3) / 5

    Args:
        density: Mass density [kg/m³]
        constants: Physical constants

    Returns:
        Degeneracy pressure [Pa]
    """
    # Electron fraction for typical matter (Z/A ≈ 0.5 for elements like C, O, Fe)
    mu_e = 2.0  # Mean molecular weight per electron

    # Number density of electrons
    n_e = density / (mu_e * constants.m_p)

    # Non-relativistic degeneracy pressure coefficient
    K = (constants.hbar**2 / (5 * constants.m_e)) * (3 * np.pi**2)**(2/3)

    return K * n_e**(5/3)


def coulomb_thermal_pressure(density: float, temperature: float, constants: PhysicalConstants) -> float:
    """
    Calculate Coulomb/thermal pressure from ideal gas law.
    Berechnet Coulomb/thermischen Druck aus idealem Gasgesetz.

    Formula: P = n × k_B × T

    Args:
        density: Mass density [kg/m³]
        temperature: Temperature [K]
        constants: Physical constants

    Returns:
        Thermal pressure [Pa]
    """
    # Average particle mass (assume ~2 m_p for typical rock/metal)
    mu = 2.0 * constants.m_p
    n = density / mu
    return n * constants.k_B * temperature


def calculate_planetary_equilibrium(
    mass: float,
    standard_radius: float,
    temperature: float,
    constants: PhysicalConstants,
    G_scale: float = 1.0,
    hbar_scale: float = 1.0
) -> PlanetaryEquilibrium:
    """
    Calculate equilibrium properties for a planet under modified constants.
    Berechnet Gleichgewichtseigenschaften fuer einen Planeten bei veraenderten Konstanten.

    Args:
        mass: Object mass [kg]
        standard_radius: Standard radius (no scaling) [m]
        temperature: Core temperature [K]
        constants: Physical constants
        G_scale: Gravitational constant scaling factor
        hbar_scale: Planck constant scaling factor

    Returns:
        PlanetaryEquilibrium object
    """
    # Scaled constants
    G_scaled = constants.G * G_scale
    hbar_scaled = constants.hbar * hbar_scale

    # Simple model: radius scales with balance point
    # Degeneracy pressure P_deg ∝ ℏ² × ρ^(5/3)
    # Gravitational pressure P_grav ∝ G × M² / R⁴
    # For equilibrium: P_deg = P_grav
    # This gives R ∝ (ℏ²/G)^(1/2) × M^(-1/3) for degenerate matter

    # For normal matter, thermal pressure dominates:
    # P_thermal = n × k_B × T, independent of ℏ
    # But density ρ ∝ M/R³
    # P_grav ∝ G × ρ × M / R

    # For Earth-like objects, thermal pressure dominates over degeneracy
    # Equilibrium: P_thermal ≈ P_grav
    # This gives roughly R ∝ (k_B × T / G)^(1/2) × M^(-1/2) × (some function of composition)

    # Simplified scaling: assume radius scales inversely with G^(1/4) for rocky planets
    # (based on balance between gravity and material strength/thermal pressure)
    compression_factor = G_scale**(-0.25)
    radius = standard_radius * compression_factor

    # Ensure minimum radius (can't compress below nuclear density)
    min_radius = (3 * mass / (4 * np.pi * 1e18))**(1/3)  # Nuclear density limit
    radius = max(radius, min_radius)
    compression_factor = radius / standard_radius

    # Calculate average density
    density = mass / ((4/3) * np.pi * radius**3)

    # Calculate pressures
    P_grav = (3 / (8 * np.pi)) * G_scaled * mass**2 / radius**4
    P_thermal = coulomb_thermal_pressure(density, temperature, constants)

    # For degeneracy pressure, need to account for ℏ scaling
    # P_deg ∝ ℏ²
    density_for_deg = density
    # Base degeneracy pressure (with scaled ℏ)
    n_e = density_for_deg / (2.0 * constants.m_p)
    K_scaled = (hbar_scaled**2 / (5 * constants.m_e)) * (3 * np.pi**2)**(2/3)
    P_deg = K_scaled * n_e**(5/3)

    # Total support pressure
    P_support = P_thermal + P_deg

    # Pressure ratio (gravity / support)
    pressure_ratio = P_grav / P_support if P_support > 0 else float('inf')

    # Stability: object is stable if support pressure can balance gravity
    is_stable = pressure_ratio <= 1.5  # Allow some margin

    return PlanetaryEquilibrium(
        mass=mass,
        radius=radius,
        central_pressure=P_grav,
        gravitational_pressure=P_grav,
        degeneracy_pressure=P_deg,
        coulomb_pressure=P_thermal,
        pressure_ratio=pressure_ratio,
        is_stable=is_stable,
        compression_factor=compression_factor
    )


def plot_earth_structural_effects(
    constants: Optional[PhysicalConstants] = None,
    language: str = 'en',
    save: bool = True,
    show: bool = True
) -> plt.Figure:
    """
    Visualize how Earth's structure changes with modified G and ℏ.
    Visualisiert wie sich die Erdstruktur mit veraendertem G und ℏ aendert.

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

    # Earth parameters
    M_earth = constants.M_earth
    R_earth = constants.R_earth
    T_core = 5500  # Core temperature in K

    # Plot 1: Earth radius vs G scaling
    ax1 = axes[0, 0]
    G_scales = np.logspace(0, 10, 50)

    radii = []
    is_stable_list = []
    for G_scale in G_scales:
        eq = calculate_planetary_equilibrium(M_earth, R_earth, T_core, constants, G_scale=G_scale)
        radii.append(eq.radius / R_earth)
        is_stable_list.append(eq.is_stable)

    ax1.semilogx(G_scales, radii, '-', color=COLORS['primary_blue'], linewidth=2.5)

    # Mark stability threshold
    stable_mask = np.array(is_stable_list)
    if not all(stable_mask):
        collapse_idx = np.where(~stable_mask)[0][0]
        G_collapse = G_scales[collapse_idx]
        ax1.axvline(x=G_collapse, color='red', linestyle='--', linewidth=2,
                   label=f'Collapse (G × {G_collapse:.1e})' if language == 'en'
                         else f'Kollaps (G × {G_collapse:.1e})')

    ax1.axhline(y=1, color=COLORS['standard'], linestyle=':', linewidth=1.5, alpha=0.7,
               label='Standard radius' if language == 'en' else 'Standard-Radius')
    ax1.axvline(x=1, color=COLORS['standard'], linestyle=':', linewidth=1.5, alpha=0.7)
    ax1.plot(1, 1, 'o', color=COLORS['standard'], markersize=12, label='Standard Earth')

    ax1.set_ylim(0, 1.2)
    if language == 'de':
        ax1.set_xlabel('G-Skalierungsfaktor', fontsize=11)
        ax1.set_ylabel('Erdradius / Standard-Radius', fontsize=11)
        ax1.set_title('1. Erdradius vs. Gravitationsstärke', fontsize=12, fontweight='bold')
    else:
        ax1.set_xlabel('G scaling factor', fontsize=11)
        ax1.set_ylabel('Earth radius / standard radius', fontsize=11)
        ax1.set_title('1. Earth Radius vs. Gravity Strength', fontsize=12, fontweight='bold')

    ax1.legend(fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Pressure components vs G
    ax2 = axes[0, 1]

    P_grav_list = []
    P_thermal_list = []
    P_deg_list = []

    for G_scale in G_scales:
        eq = calculate_planetary_equilibrium(M_earth, R_earth, T_core, constants, G_scale=G_scale)
        P_grav_list.append(eq.gravitational_pressure)
        P_thermal_list.append(eq.coulomb_pressure)
        P_deg_list.append(eq.degeneracy_pressure)

    ax2.loglog(G_scales, P_grav_list, '-', color=COLORS['scaled'], linewidth=2.5,
               label='P_gravity' if language == 'en' else 'P_Gravitation')
    ax2.loglog(G_scales, P_thermal_list, '--', color=COLORS['primary_amber'], linewidth=2,
               label='P_thermal (Coulomb)' if language == 'en' else 'P_thermisch (Coulomb)')
    ax2.loglog(G_scales, P_deg_list, ':', color=COLORS['quantum'], linewidth=2,
               label='P_degeneracy (Pauli)' if language == 'en' else 'P_Entartung (Pauli)')

    ax2.axvline(x=1, color=COLORS['standard'], linestyle=':', linewidth=1, alpha=0.5)

    if language == 'de':
        ax2.set_xlabel('G-Skalierungsfaktor', fontsize=11)
        ax2.set_ylabel('Druck (Pa)', fontsize=11)
        ax2.set_title('2. Druckkomponenten im Erdkern', fontsize=12, fontweight='bold')
    else:
        ax2.set_xlabel('G scaling factor', fontsize=11)
        ax2.set_ylabel('Pressure (Pa)', fontsize=11)
        ax2.set_title('2. Pressure Components in Earth\'s Core', fontsize=12, fontweight='bold')

    ax2.legend(fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3)
    ax2.grid(True, alpha=0.3, which='both')

    # Plot 3: Effect of ℏ scaling on degeneracy pressure
    ax3 = axes[1, 0]

    hbar_scales = np.logspace(-1, 1, 50)
    G_fixed = 1e6  # Fix G at high value to see ℏ effect

    P_deg_hbar = []
    P_grav_hbar = []
    for hbar_scale in hbar_scales:
        eq = calculate_planetary_equilibrium(M_earth, R_earth, T_core, constants,
                                           G_scale=G_fixed, hbar_scale=hbar_scale)
        P_deg_hbar.append(eq.degeneracy_pressure)
        P_grav_hbar.append(eq.gravitational_pressure)

    ax3.loglog(hbar_scales, P_deg_hbar, '-', color=COLORS['quantum'], linewidth=2.5,
               label='P_degeneracy (∝ ℏ²)' if language == 'en' else 'P_Entartung (∝ ℏ²)')
    ax3.loglog(hbar_scales, P_grav_hbar, '--', color=COLORS['scaled'], linewidth=2,
               label=f'P_gravity (G × {G_fixed:.0e})' if language == 'en'
                     else f'P_Gravitation (G × {G_fixed:.0e})')

    ax3.axvline(x=1, color=COLORS['standard'], linestyle=':', linewidth=1, alpha=0.5,
               label='Standard ℏ')

    # Find crossover point
    P_deg_arr = np.array(P_deg_hbar)
    P_grav_arr = np.array(P_grav_hbar)
    if np.any(P_deg_arr > P_grav_arr[0]):
        crossover_idx = np.where(P_deg_arr >= P_grav_arr)[0]
        if len(crossover_idx) > 0:
            hbar_crossover = hbar_scales[crossover_idx[0]]
            ax3.axvline(x=hbar_crossover, color='green', linestyle='--', linewidth=2, alpha=0.7,
                       label=f'Stability restored (ℏ × {hbar_crossover:.1f})')

    if language == 'de':
        ax3.set_xlabel('ℏ-Skalierungsfaktor', fontsize=11)
        ax3.set_ylabel('Druck (Pa)', fontsize=11)
        ax3.set_title(f'3. Entartungsdruck vs. ℏ (G × {G_fixed:.0e})', fontsize=12, fontweight='bold')
    else:
        ax3.set_xlabel('ℏ scaling factor', fontsize=11)
        ax3.set_ylabel('Pressure (Pa)', fontsize=11)
        ax3.set_title(f'3. Degeneracy Pressure vs. ℏ (G × {G_fixed:.0e})', fontsize=12, fontweight='bold')

    ax3.legend(fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3)
    ax3.grid(True, alpha=0.3, which='both')

    # Plot 4: Stability map in G-ℏ space
    ax4 = axes[1, 1]

    G_range = np.logspace(0, 12, 30)
    hbar_range = np.logspace(-1, 2, 30)
    G_grid, hbar_grid = np.meshgrid(G_range, hbar_range)

    stability = np.zeros_like(G_grid)
    for i in range(len(hbar_range)):
        for j in range(len(G_range)):
            eq = calculate_planetary_equilibrium(M_earth, R_earth, T_core, constants,
                                               G_scale=G_range[j], hbar_scale=hbar_range[i])
            stability[i, j] = 1 if eq.is_stable else 0

    im = ax4.contourf(np.log10(G_range), np.log10(hbar_range), stability,
                      levels=[0, 0.5, 1], colors=[COLORS['scaled'], COLORS['primary_blue']], alpha=0.5)
    ax4.contour(np.log10(G_range), np.log10(hbar_range), stability,
               levels=[0.5], colors=['red'], linewidths=2)

    ax4.plot(0, 0, 'o', color=COLORS['standard'], markersize=15, label='Standard universe')

    if language == 'de':
        ax4.set_xlabel('log₁₀(G/G₀)', fontsize=11)
        ax4.set_ylabel('log₁₀(ℏ/ℏ₀)', fontsize=11)
        ax4.set_title('4. Stabilitätskarte der Erde im G-ℏ-Raum\n(Blau = stabil, Rot = Kollaps)', fontsize=12, fontweight='bold')
    else:
        ax4.set_xlabel('log₁₀(G/G₀)', fontsize=11)
        ax4.set_ylabel('log₁₀(ℏ/ℏ₀)', fontsize=11)
        ax4.set_title('4. Earth Stability Map in G-ℏ Space\n(Blue = stable, Red = collapse)', fontsize=12, fontweight='bold')

    ax4.legend(fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)

    if save:
        os.makedirs(VIS_DIR, exist_ok=True)
        suffix = '_de' if language == 'de' else ''
        filepath = os.path.join(VIS_DIR, f'earth_structural_effects{suffix}.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")

    if show:
        plt.show()

    return fig


def plot_gravity_vs_pauli(
    constants: Optional[PhysicalConstants] = None,
    language: str = 'en',
    save: bool = True,
    show: bool = True
) -> plt.Figure:
    """
    Explicit comparison of gravitational pressure vs Pauli exclusion (degeneracy) pressure.
    Expliziter Vergleich von Gravitationsdruck vs. Pauli-Ausschlussdruck (Entartung).

    This is the core visualization addressing the essay hypothesis.

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

    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    # Plot 1: Pressure vs density comparison
    ax1 = fig.add_subplot(gs[0, 0])

    densities = np.logspace(3, 18, 100)  # From rock to nuclear density

    # Gravitational pressure for Earth-mass sphere
    M = constants.M_earth
    R_from_rho = (3 * M / (4 * np.pi * densities))**(1/3)
    P_grav = (3 / (8 * np.pi)) * constants.G * M**2 / R_from_rho**4

    # Degeneracy pressure (non-relativistic electrons)
    P_deg_e = np.array([electron_degeneracy_pressure_simple(rho, constants) for rho in densities])

    # Degeneracy pressure (relativistic electrons, for high density)
    # P_rel ∝ ℏc × n^(4/3)
    n_e = densities / (2.0 * constants.m_p)
    P_deg_r = (constants.hbar * constants.c / 4) * (3 / np.pi)**(1/3) * n_e**(4/3)

    ax1.loglog(densities, P_grav, '-', color=COLORS['scaled'], linewidth=2.5,
               label='P_gravity (Earth mass)' if language == 'en' else 'P_Gravitation (Erdmasse)')
    ax1.loglog(densities, P_deg_e, '--', color=COLORS['quantum'], linewidth=2.5,
               label='P_Pauli (non-rel, ∝ ρ^{5/3})' if language == 'en'
                     else 'P_Pauli (nicht-rel, ∝ ρ^{5/3})')
    ax1.loglog(densities, P_deg_r, ':', color=COLORS['primary_blue'], linewidth=2,
               label='P_Pauli (ultra-rel, ∝ ρ^{4/3})' if language == 'en'
                     else 'P_Pauli (ultra-rel, ∝ ρ^{4/3})')

    # Mark Earth, WD, NS densities
    rho_earth = 5500  # kg/m³
    rho_wd = 1e9
    rho_ns = 4e17
    ax1.axvline(x=rho_earth, color=COLORS['primary_amber'], linestyle=':', alpha=0.7, label='Earth')
    ax1.axvline(x=rho_wd, color=COLORS['standard'], linestyle=':', alpha=0.7, label='White dwarf')
    ax1.axvline(x=rho_ns, color='red', linestyle=':', alpha=0.7, label='Neutron star')

    if language == 'de':
        ax1.set_xlabel('Dichte ρ (kg/m³)', fontsize=11)
        ax1.set_ylabel('Druck P (Pa)', fontsize=11)
        ax1.set_title('1. Gravitationsdruck vs. Pauli-Druck', fontsize=12, fontweight='bold')
    else:
        ax1.set_xlabel('Density ρ (kg/m³)', fontsize=11)
        ax1.set_ylabel('Pressure P (Pa)', fontsize=11)
        ax1.set_title('1. Gravitational Pressure vs. Pauli Pressure', fontsize=12, fontweight='bold')

    ax1.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3)
    ax1.grid(True, alpha=0.3, which='both')

    # Plot 2: Pressure ratio vs mass
    ax2 = fig.add_subplot(gs[0, 1])

    masses = np.logspace(23, 32, 50)  # From asteroids to stellar masses
    M_sun = constants.M_sun

    ratios_normal = []  # P_gravity / P_Pauli at typical density
    for M in masses:
        # Assume object at WD density
        rho = 1e9
        R = (3 * M / (4 * np.pi * rho))**(1/3)
        P_g = (3 / (8 * np.pi)) * constants.G * M**2 / R**4
        P_p = electron_degeneracy_pressure_simple(rho, constants)
        ratios_normal.append(P_g / P_p if P_p > 0 else float('inf'))

    ax2.loglog(masses / M_sun, ratios_normal, '-', color=COLORS['text_dark'], linewidth=2.5)

    ax2.axhline(y=1, color='red', linestyle='--', linewidth=2, alpha=0.8,
               label='Balance (P_grav = P_Pauli)' if language == 'en'
                     else 'Gleichgewicht (P_grav = P_Pauli)')

    # Mark Chandrasekhar limit
    M_ch = 1.44 * M_sun
    ax2.axvline(x=1.44, color=COLORS['quantum'], linestyle=':', linewidth=2, alpha=0.8,
               label='Chandrasekhar limit (1.44 M☉)')

    # Mark key masses
    ax2.axvline(x=constants.M_earth / M_sun, color=COLORS['primary_amber'], linestyle=':', alpha=0.7,
               label='Earth')

    if language == 'de':
        ax2.set_xlabel('Masse (M☉)', fontsize=11)
        ax2.set_ylabel('P_grav / P_Pauli', fontsize=11)
        ax2.set_title('2. Druckverhältnis vs. Masse (bei WD-Dichte)', fontsize=12, fontweight='bold')
    else:
        ax2.set_xlabel('Mass (M☉)', fontsize=11)
        ax2.set_ylabel('P_gravity / P_Pauli', fontsize=11)
        ax2.set_title('2. Pressure Ratio vs. Mass (at WD density)', fontsize=12, fontweight='bold')

    ax2.legend(fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax2.grid(True, alpha=0.3, which='both')

    # Plot 3: Effect of G scaling on balance
    ax3 = fig.add_subplot(gs[1, 0])

    G_scales = np.logspace(0, 40, 100)

    # For different objects
    objects = [
        ('Earth', constants.M_earth, 5500, COLORS['primary_amber']),
        ('Jupiter', 1.9e27, 1300, COLORS['primary_blue']),
        ('White Dwarf (0.6 M☉)', 0.6 * M_sun, 1e9, COLORS['standard']),
    ]

    for name, M, rho, color in objects:
        R = (3 * M / (4 * np.pi * rho))**(1/3)
        ratios = []
        for G_scale in G_scales:
            G_scaled = constants.G * G_scale
            P_g = (3 / (8 * np.pi)) * G_scaled * M**2 / R**4
            P_p = electron_degeneracy_pressure_simple(rho, constants)
            ratios.append(P_g / P_p if P_p > 0 else float('inf'))

        ax3.loglog(G_scales, ratios, '-', color=color, linewidth=2, label=name)

    ax3.axhline(y=1, color='red', linestyle='--', linewidth=2, alpha=0.8,
               label='Collapse threshold')
    ax3.axvline(x=1, color=COLORS['standard'], linestyle=':', linewidth=1, alpha=0.5)

    if language == 'de':
        ax3.set_xlabel('G-Skalierungsfaktor', fontsize=11)
        ax3.set_ylabel('P_grav / P_Pauli', fontsize=11)
        ax3.set_title('3. Druckverhältnis vs. G-Skalierung', fontsize=12, fontweight='bold')
    else:
        ax3.set_xlabel('G scaling factor', fontsize=11)
        ax3.set_ylabel('P_gravity / P_Pauli', fontsize=11)
        ax3.set_title('3. Pressure Ratio vs. G Scaling', fontsize=12, fontweight='bold')

    ax3.legend(fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3)
    ax3.grid(True, alpha=0.3, which='both')

    # Plot 4: ℏ scaling to restore balance
    ax4 = fig.add_subplot(gs[1, 1])

    # For G × 10^36 (essay value), what ℏ scaling restores balance?
    G_essay = 1e36
    hbar_scales = np.logspace(0, 20, 50)

    for name, M, rho, color in objects:
        R = (3 * M / (4 * np.pi * rho))**(1/3)
        ratios = []
        for hbar_scale in hbar_scales:
            G_scaled = constants.G * G_essay
            hbar_scaled = constants.hbar * hbar_scale
            P_g = (3 / (8 * np.pi)) * G_scaled * M**2 / R**4
            # Degeneracy pressure scales as ℏ²
            n_e = rho / (2.0 * constants.m_p)
            K_scaled = (hbar_scaled**2 / (5 * constants.m_e)) * (3 * np.pi**2)**(2/3)
            P_p = K_scaled * n_e**(5/3)
            ratios.append(P_g / P_p if P_p > 0 else float('inf'))

        ax4.loglog(hbar_scales, ratios, '-', color=color, linewidth=2, label=name)

    ax4.axhline(y=1, color='red', linestyle='--', linewidth=2, alpha=0.8,
               label='Balance threshold')

    # Mark essay scaling: ℏ should scale to keep Bohr radius same
    # a_0 ∝ ℏ², so for G × 10^36, need ℏ × 10^18 to maintain atoms
    ax4.axvline(x=1e18, color='green', linestyle='--', linewidth=2, alpha=0.8,
               label='ℏ × 10¹⁸ (essay)')

    if language == 'de':
        ax4.set_xlabel('ℏ-Skalierungsfaktor', fontsize=11)
        ax4.set_ylabel('P_grav / P_Pauli', fontsize=11)
        ax4.set_title(f'4. Gleichgewicht bei G × 10³⁶', fontsize=12, fontweight='bold')
    else:
        ax4.set_xlabel('ℏ scaling factor', fontsize=11)
        ax4.set_ylabel('P_gravity / P_Pauli', fontsize=11)
        ax4.set_title(f'4. Balance at G × 10³⁶', fontsize=12, fontweight='bold')

    ax4.legend(fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3)
    ax4.grid(True, alpha=0.3, which='both')

    # Overall title
    if language == 'de':
        fig.suptitle('Gravitation vs. Pauli: Die Kernfrage des Essays',
                    fontsize=16, fontweight='bold', y=0.98)
    else:
        fig.suptitle('Gravity vs. Pauli: The Core Question of the Essay',
                    fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12, top=0.93)

    if save:
        os.makedirs(VIS_DIR, exist_ok=True)
        suffix = '_de' if language == 'de' else ''
        filepath = os.path.join(VIS_DIR, f'gravity_vs_pauli{suffix}.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")

    if show:
        plt.show()

    return fig


def plot_hypothesis_summary(
    constants: Optional[PhysicalConstants] = None,
    language: str = 'en',
    save: bool = True,
    show: bool = True
) -> plt.Figure:
    """
    Create a comprehensive summary visualization for the essay hypothesis.
    Erstellt eine umfassende Zusammenfassung fuer die Essay-Hypothese.

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

    M_sun = constants.M_sun

    # Plot 1: Object stability under standard physics
    ax1 = axes[0, 0]

    objects_data = [
        ('Earth', constants.M_earth, 5500, 'Stable (thermal)'),
        ('Jupiter', 1.9e27, 1300, 'Stable (thermal)'),
        ('White Dwarf', 0.6 * M_sun, 1e9, 'Stable (Pauli)'),
        ('Chandrasekhar', 1.44 * M_sun, 1e10, 'Limit'),
        ('Neutron Star', 1.4 * M_sun, 4e17, 'Stable (Pauli-n)'),
    ]

    masses = [obj[1] / M_sun for obj in objects_data]
    densities = [obj[2] for obj in objects_data]
    colors_obj = [COLORS['primary_amber'], COLORS['primary_blue'], COLORS['standard'],
                  'red', COLORS['quantum']]

    scatter = ax1.scatter(masses, densities, c=colors_obj, s=200, edgecolors='black', linewidth=2)

    for i, (name, M, rho, status) in enumerate(objects_data):
        ax1.annotate(f'{name}\n({status})', xy=(M/M_sun, rho),
                    xytext=(10, 10), textcoords='offset points', fontsize=8)

    ax1.set_xscale('log')
    ax1.set_yscale('log')

    if language == 'de':
        ax1.set_xlabel('Masse (M☉)', fontsize=11)
        ax1.set_ylabel('Dichte (kg/m³)', fontsize=11)
        ax1.set_title('1. Objektstabilität (Standardphysik)', fontsize=12, fontweight='bold')
    else:
        ax1.set_xlabel('Mass (M☉)', fontsize=11)
        ax1.set_ylabel('Density (kg/m³)', fontsize=11)
        ax1.set_title('1. Object Stability (Standard Physics)', fontsize=12, fontweight='bold')

    ax1.grid(True, alpha=0.3, which='both')

    # Plot 2: Pressure scaling with G
    ax2 = axes[0, 1]

    G_scales = np.logspace(0, 40, 100)

    # Gravitational pressure ∝ G
    P_grav_scaling = G_scales

    # Degeneracy pressure ∝ ℏ² (constant ℏ)
    P_deg_scaling = np.ones_like(G_scales)

    ax2.loglog(G_scales, P_grav_scaling, '-', color=COLORS['scaled'], linewidth=2.5,
               label='P_gravity (∝ G)' if language == 'en' else 'P_Gravitation (∝ G)')
    ax2.loglog(G_scales, P_deg_scaling, '--', color=COLORS['quantum'], linewidth=2.5,
               label='P_Pauli (constant ℏ)' if language == 'en' else 'P_Pauli (konstant ℏ)')

    # Mark where gravity exceeds Pauli
    ax2.axvline(x=1, color=COLORS['standard'], linestyle=':', alpha=0.7, label='Standard G')
    ax2.axvline(x=1e36, color='red', linestyle='--', alpha=0.7, label='G × 10³⁶')
    ax2.axhline(y=1e36, color=COLORS['primary_amber'], linestyle=':', alpha=0.5)

    if language == 'de':
        ax2.set_xlabel('G-Skalierung', fontsize=11)
        ax2.set_ylabel('Relative Druckskalierung', fontsize=11)
        ax2.set_title('2. Druckskalierung mit G (ℏ konstant)', fontsize=12, fontweight='bold')
    else:
        ax2.set_xlabel('G scaling', fontsize=11)
        ax2.set_ylabel('Relative pressure scaling', fontsize=11)
        ax2.set_title('2. Pressure Scaling with G (ℏ constant)', fontsize=12, fontweight='bold')

    ax2.legend(fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax2.grid(True, alpha=0.3, which='both')

    # Plot 3: Stability ratio vs G for different ℏ scalings
    ax3 = axes[1, 0]

    hbar_scalings = [1, 1e9, 1e18]
    labels_en = ['Standard ℏ', 'ℏ × 10⁹', 'ℏ × 10¹⁸']
    labels_de = ['Standard ℏ', 'ℏ × 10⁹', 'ℏ × 10¹⁸']
    colors_hbar = [COLORS['primary_blue'], COLORS['primary_amber'], COLORS['quantum']]

    for hbar_s, label_en, label_de, color in zip(hbar_scalings, labels_en, labels_de, colors_hbar):
        # P_grav / P_Pauli ∝ G / ℏ²
        ratio = G_scales / (hbar_s ** 2)
        label = label_de if language == 'de' else label_en
        ax3.loglog(G_scales, ratio, '-', color=color, linewidth=2, label=label)

    ax3.axhline(y=1, color='red', linestyle='--', linewidth=2, alpha=0.8,
               label='Collapse threshold' if language == 'en' else 'Kollapsgrenze')
    ax3.axvline(x=1e36, color='gray', linestyle=':', alpha=0.5, label='G × 10³⁶')

    if language == 'de':
        ax3.set_xlabel('G-Skalierung', fontsize=11)
        ax3.set_ylabel('P_grav / P_Pauli', fontsize=11)
        ax3.set_title('3. Stabilitätsverhältnis vs G', fontsize=12, fontweight='bold')
    else:
        ax3.set_xlabel('G scaling', fontsize=11)
        ax3.set_ylabel('P_gravity / P_Pauli', fontsize=11)
        ax3.set_title('3. Stability Ratio vs G', fontsize=12, fontweight='bold')

    ax3.legend(fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3)
    ax3.grid(True, alpha=0.3, which='both')

    # Plot 4: Required ℏ scaling for stability at different G
    ax4 = axes[1, 1]

    G_values = np.logspace(0, 40, 50)
    # For stability: P_grav = P_Pauli → G_scaled = ℏ_scaled²
    # So ℏ_needed = sqrt(G_scaled)
    hbar_needed = np.sqrt(G_values)

    ax4.loglog(G_values, hbar_needed, '-', color=COLORS['quantum'], linewidth=2.5,
               label='Required ℏ for stability' if language == 'en' else 'Erforderliches ℏ für Stabilität')

    # Mark essay scenario
    ax4.plot(1e36, 1e18, 'o', color=COLORS['primary_amber'], markersize=15,
             label='Essay (G×10³⁶, ℏ×10¹⁸)' if language == 'en' else 'Essay (G×10³⁶, ℏ×10¹⁸)')
    ax4.plot(1, 1, 's', color=COLORS['standard'], markersize=12,
             label='Standard universe' if language == 'en' else 'Standarduniversum')

    # Shade stable and unstable regions
    ax4.fill_between(G_values, hbar_needed, 1e25, alpha=0.2, color=COLORS['primary_blue'])
    ax4.fill_between(G_values, 1e-5, hbar_needed, alpha=0.2, color=COLORS['scaled'])

    if language == 'de':
        ax4.set_xlabel('G-Skalierung', fontsize=11)
        ax4.set_ylabel('Erforderliche ℏ-Skalierung', fontsize=11)
        ax4.set_title('4. ℏ erforderlich für Gleichgewicht: ℏ ∝ √G', fontsize=12, fontweight='bold')
    else:
        ax4.set_xlabel('G scaling', fontsize=11)
        ax4.set_ylabel('Required ℏ scaling', fontsize=11)
        ax4.set_title('4. ℏ Required for Balance: ℏ ∝ √G', fontsize=12, fontweight='bold')

    ax4.legend(fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax4.grid(True, alpha=0.3, which='both')
    ax4.set_xlim(1, 1e42)
    ax4.set_ylim(1e-2, 1e25)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)

    if save:
        os.makedirs(VIS_DIR, exist_ok=True)
        suffix = '_de' if language == 'de' else ''
        filepath = os.path.join(VIS_DIR, f'hypothesis_summary{suffix}.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")

    if show:
        plt.show()

    return fig


def generate_all_gravity_pauli_plots(language: str = 'en', show: bool = False) -> List[plt.Figure]:
    """
    Generate all gravity vs Pauli balance visualizations.
    Erzeugt alle Gravitation-vs-Pauli-Gleichgewichts-Visualisierungen.

    Args:
        language: 'en' for English, 'de' for German
        show: Whether to display plots

    Returns:
        List of matplotlib Figure objects
    """
    figures = []

    print("Generating gravity vs Pauli balance visualizations...")
    print("=" * 50)

    # 1. Earth structural effects
    print("1. Earth structural effects...")
    figures.append(plot_earth_structural_effects(language=language, show=show))

    # 2. Gravity vs Pauli comparison
    print("2. Gravity vs Pauli comparison...")
    figures.append(plot_gravity_vs_pauli(language=language, show=show))

    # 3. Hypothesis summary
    print("3. Hypothesis summary...")
    figures.append(plot_hypothesis_summary(language=language, show=show))

    print("=" * 50)
    print(f"Generated {len(figures)} visualizations in {VIS_DIR}")

    return figures


def verify_gravity_pauli_physics():
    """
    Verify gravity vs Pauli calculations.
    Verifiziert Gravitation-vs-Pauli-Berechnungen.
    """
    print("=" * 70)
    print("GRAVITY VS PAULI PHYSICS VERIFICATION")
    print("=" * 70)

    c = get_constants()

    # 1. Earth core pressure
    print("\n1. EARTH CORE PRESSURE")
    print("-" * 50)
    P_earth = gravitational_central_pressure(c.M_earth, c.R_earth, c)
    print(f"   Central pressure: {P_earth:.2e} Pa")
    print(f"   Expected: ~3.6×10¹¹ Pa (CHECK: {'PASS' if 1e11 < P_earth < 1e12 else 'FAIL'})")

    # 2. White dwarf degeneracy pressure
    print("\n2. WHITE DWARF DEGENERACY PRESSURE")
    print("-" * 50)
    rho_wd = 1e9  # kg/m³
    P_deg_wd = electron_degeneracy_pressure_simple(rho_wd, c)
    print(f"   At density {rho_wd:.0e} kg/m³:")
    print(f"   P_degeneracy = {P_deg_wd:.2e} Pa")
    print(f"   Expected: ~10²² Pa (CHECK: {'PASS' if 1e20 < P_deg_wd < 1e24 else 'FAIL'})")

    # 3. Pressure ratio for WD
    print("\n3. PRESSURE RATIO FOR WHITE DWARF")
    print("-" * 50)
    M_wd = 0.6 * c.M_sun
    R_wd = 8e6  # 8000 km
    P_grav_wd = gravitational_central_pressure(M_wd, R_wd, c)
    ratio = P_grav_wd / P_deg_wd
    print(f"   P_gravity / P_Pauli = {ratio:.2f}")
    print(f"   (Ratio ~1 indicates equilibrium)")
    print(f"   CHECK: {'PASS - Near equilibrium' if 0.1 < ratio < 10 else 'CHECK VALUES'}")

    # 4. Scaling verification
    print("\n4. SCALING WITH G AND ℏ")
    print("-" * 50)
    print(f"   P_gravity ∝ G")
    print(f"   P_Pauli ∝ ℏ²")
    print(f"   For G × 10³⁶: P_gravity increases by 10³⁶")
    print(f"   For ℏ × 10¹⁸: P_Pauli increases by (10¹⁸)² = 10³⁶")
    print(f"   → Balance preserved when G/ℏ² = constant")

    # 5. Earth stability check
    print("\n5. EARTH STABILITY AT SCALED G")
    print("-" * 50)
    eq_std = calculate_planetary_equilibrium(c.M_earth, c.R_earth, 5500, c, G_scale=1.0)
    eq_scaled = calculate_planetary_equilibrium(c.M_earth, c.R_earth, 5500, c, G_scale=1e6)
    print(f"   Standard G: Stable = {eq_std.is_stable}, Ratio = {eq_std.pressure_ratio:.2f}")
    print(f"   G × 10⁶: Stable = {eq_scaled.is_stable}, Ratio = {eq_scaled.pressure_ratio:.2f}")

    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    print("=" * 60)
    print("Gravity vs Pauli Balance Module - Jugend forscht 2026")
    print("=" * 60)

    # Verify physics
    verify_gravity_pauli_physics()

    # Generate visualizations
    print("\n")
    generate_all_gravity_pauli_plots(language='en', show=False)

    print("\nDone! Check the 'visualizations' folder for output.")
