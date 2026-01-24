"""
Thermal Physics Module for Jugend forscht 2026
Thermophysik-Modul fuer Jugend forscht 2026

This module calculates and visualizes how increased gravity affects temperature
in Earth's interior, surface, and atmosphere. Connects degeneracy pressure
physics to observable temperature effects.

Key concepts:
- Atmospheric scale height: H = kT/(μg) - higher g → compressed atmosphere
- Adiabatic lapse rate: dT/dz = -g/c_p - higher g → steeper temperature gradient
- Fermi temperature: T_F = E_F/k_B - when T << T_F, degeneracy pressure dominates
- Virial temperature: T ~ GMm_p/(3k_B R) - gravitational compression heating

Author: Jugend forscht 2026 Project
"""

import numpy as np
import os
from dataclasses import dataclass
from typing import Optional, Tuple, List
import math

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from .constants import get_constants, PhysicalConstants
from .color_scheme import COLORS, get_sequence

# Output directory for visualizations
VIS_DIR = os.path.join(os.path.dirname(__file__), '..', 'visualizations')


@dataclass
class ThermalProperties:
    """
    Container for thermal properties at a given gravity scale.
    Behaelter fuer thermische Eigenschaften bei gegebener Gravitationsskalierung.
    """
    scale_height: float           # Atmospheric scale height [m]
    lapse_rate: float             # Temperature gradient [K/m]
    fermi_temperature: float      # Fermi temperature [K]
    surface_temp: float           # Surface temperature [K]
    core_temp_estimate: float     # Core temperature estimate [K]
    g_scale: float                # Gravity scale factor used


def atmospheric_scale_height(
    g: float,
    T: float,
    mu: float,
    k_B: float
) -> float:
    """
    Calculate the atmospheric scale height.
    Berechnet die atmosphaerische Skalenhoehe.

    Formula: H = kT / (μg)

    The scale height is the altitude over which pressure drops by a factor of e.
    Higher g → more compressed atmosphere → smaller scale height.

    Args:
        g: Gravitational acceleration [m/s²]
        T: Temperature [K]
        mu: Mean molecular mass [kg]
        k_B: Boltzmann constant [J/K]

    Returns:
        Scale height [m]
    """
    return (k_B * T) / (mu * g)


def adiabatic_lapse_rate(g: float, c_p: float) -> float:
    """
    Calculate the dry adiabatic lapse rate.
    Berechnet die trockene adiabatische Abkuehlungsrate.

    Formula: dT/dz = -g / c_p

    This is how temperature decreases with altitude in a convective atmosphere.
    Higher g → steeper temperature gradient.

    Args:
        g: Gravitational acceleration [m/s²]
        c_p: Specific heat at constant pressure [J/(kg·K)]

    Returns:
        Lapse rate [K/m] (negative means temperature decreases with altitude)
    """
    return -g / c_p


def fermi_energy(n_e: float, m_e: float, hbar: float) -> float:
    """
    Calculate the Fermi energy for a degenerate electron gas.
    Berechnet die Fermi-Energie fuer ein entartetes Elektronengas.

    Formula: E_F = (ℏ²/2m_e)(3π²n_e)^(2/3)

    Args:
        n_e: Electron number density [m⁻³]
        m_e: Electron mass [kg]
        hbar: Reduced Planck constant [J·s]

    Returns:
        Fermi energy [J]
    """
    return (hbar**2 / (2 * m_e)) * (3 * np.pi**2 * n_e)**(2/3)


def fermi_temperature(n_e: float, constants: PhysicalConstants) -> float:
    """
    Calculate the Fermi temperature.
    Berechnet die Fermi-Temperatur.

    Formula: T_F = E_F / k_B

    When T << T_F, quantum degeneracy pressure dominates over thermal pressure.
    This is key to white dwarf stability.

    Args:
        n_e: Electron number density [m⁻³]
        constants: Physical constants

    Returns:
        Fermi temperature [K]
    """
    E_F = fermi_energy(n_e, constants.m_e, constants.hbar)
    return E_F / constants.k_B


def thermal_pressure(n: float, T: float, k_B: float) -> float:
    """
    Calculate the thermal (ideal gas) pressure.
    Berechnet den thermischen (idealen Gas) Druck.

    Formula: P = n k_B T

    Args:
        n: Number density [m⁻³]
        T: Temperature [K]
        k_B: Boltzmann constant [J/K]

    Returns:
        Thermal pressure [Pa]
    """
    return n * k_B * T


def degeneracy_pressure_nr(n_e: float, constants: PhysicalConstants) -> float:
    """
    Calculate non-relativistic electron degeneracy pressure.
    Berechnet den nicht-relativistischen Elektronenentartungsdruck.

    Formula: P = (ℏ²/5m_e)(3π²)^(2/3) n_e^(5/3)

    Args:
        n_e: Electron number density [m⁻³]
        constants: Physical constants

    Returns:
        Degeneracy pressure [Pa]
    """
    prefactor = (constants.hbar**2 / (5 * constants.m_e)) * (3 * np.pi**2)**(2/3)
    return prefactor * n_e**(5/3)


def virial_temperature(M: float, R: float, constants: PhysicalConstants) -> float:
    """
    Calculate the virial temperature from gravitational compression.
    Berechnet die Virialtemperatur aus gravitativer Kompression.

    Formula: T_virial ≈ GM m_p / (3 k_B R)

    This gives the temperature that gravitational compression would produce.
    For stars, this is the temperature needed to support against gravity.

    Args:
        M: Mass [kg]
        R: Radius [m]
        constants: Physical constants

    Returns:
        Virial temperature [K]
    """
    return (constants.G * M * constants.m_p) / (3 * constants.k_B * R)


def calculate_thermal_properties(
    g_scale: float = 1.0,
    constants: Optional[PhysicalConstants] = None
) -> ThermalProperties:
    """
    Calculate thermal properties for a given gravity scaling.
    Berechnet thermische Eigenschaften fuer eine gegebene Gravitationsskalierung.

    Args:
        g_scale: Factor to multiply standard Earth gravity
        constants: Physical constants (uses default if None)

    Returns:
        ThermalProperties dataclass
    """
    if constants is None:
        constants = get_constants()

    # Standard Earth surface gravity
    g_earth = constants.G * constants.M_earth / constants.R_earth**2
    g = g_earth * g_scale

    # Scale height at surface temperature
    H = atmospheric_scale_height(g, constants.T_surface_earth, constants.mu_air, constants.k_B)

    # Lapse rate
    gamma = adiabatic_lapse_rate(g, constants.c_p_air)

    # Fermi temperature at white dwarf density (for reference)
    # White dwarf density ~ 10^9 kg/m³, electron density ~ 3×10^35 m⁻³
    n_e_wd = 3e35  # Typical white dwarf electron density
    T_F = fermi_temperature(n_e_wd, constants)

    # Core temperature estimate: scales with compression
    # Higher g → more compression → higher core temperature
    T_core = constants.T_core_earth * g_scale**(1/3)  # Rough scaling

    return ThermalProperties(
        scale_height=H,
        lapse_rate=gamma,
        fermi_temperature=T_F,
        surface_temp=constants.T_surface_earth,
        core_temp_estimate=T_core,
        g_scale=g_scale
    )


def temperature_vs_altitude(
    altitudes: np.ndarray,
    g: float,
    T_surface: float,
    c_p: float
) -> np.ndarray:
    """
    Calculate temperature profile vs altitude (adiabatic).
    Berechnet das Temperaturprofil vs. Hoehe (adiabatisch).

    Args:
        altitudes: Array of altitudes [m]
        g: Gravitational acceleration [m/s²]
        T_surface: Surface temperature [K]
        c_p: Specific heat [J/(kg·K)]

    Returns:
        Array of temperatures [K]
    """
    lapse = adiabatic_lapse_rate(g, c_p)
    T = T_surface + lapse * altitudes
    # Temperature cannot go below ~2.7 K (cosmic microwave background)
    return np.maximum(T, 2.7)


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_temperature_atmosphere(
    language: str = 'en',
    save: bool = True,
    show: bool = False
) -> plt.Figure:
    """
    Plot atmospheric temperature profiles for different gravity values.
    Zeichnet atmosphaerische Temperaturprofile fuer verschiedene Gravitationswerte.

    Creates a figure with two vertically stacked plots:
    - Top: Temperature vs altitude for different g values
    - Bottom: Scale height H vs g (log-log plot)

    Args:
        language: 'en' for English, 'de' for German
        save: Whether to save the figure
        show: Whether to display the figure

    Returns:
        Matplotlib Figure object
    """
    constants = get_constants()
    g_earth = constants.G * constants.M_earth / constants.R_earth**2

    # Create figure with two subplots stacked vertically with proper spacing
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14), gridspec_kw={'hspace': 0.4})

    # --- Top plot: Temperature vs altitude ---
    g_scales = [1, 2, 5, 10, 50, 100]
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(g_scales)))
    altitudes = np.linspace(0, 50000, 500)  # 0 to 50 km

    for g_scale, color in zip(g_scales, colors):
        g = g_earth * g_scale
        T = temperature_vs_altitude(altitudes, g, constants.T_surface_earth, constants.c_p_air)
        label = f'{g_scale}× g'
        ax1.plot(altitudes / 1000, T, color=color, linewidth=2, label=label)

    ax1.axhline(y=273, color='gray', linestyle='--', alpha=0.5, label='0°C')

    if language == 'de':
        ax1.set_xlabel('Höhe [km]', fontsize=12)
        ax1.set_ylabel('Temperatur [K]', fontsize=12)
        ax1.set_title('1. Atmosphärische Temperatur vs. Höhe', fontsize=14, fontweight='bold', pad=15)
    else:
        ax1.set_xlabel('Altitude [km]', fontsize=12)
        ax1.set_ylabel('Temperature [K]', fontsize=12)
        ax1.set_title('1. Atmospheric Temperature vs. Altitude', fontsize=14, fontweight='bold', pad=15)

    ax1.legend(fontsize=9, loc='upper right', bbox_to_anchor=(1.0, -0.08), ncol=4, framealpha=0.7)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 50)
    ax1.set_ylim(0, 350)

    # --- Bottom plot: Scale height vs g ---
    g_range = np.logspace(0, 2, 50)  # 1× to 100× g
    scale_heights = []

    for gs in g_range:
        g = g_earth * gs
        H = atmospheric_scale_height(g, constants.T_surface_earth, constants.mu_air, constants.k_B)
        scale_heights.append(H)

    ax2.loglog(g_range, np.array(scale_heights) / 1000, color=COLORS['primary_blue'], linewidth=2.5,
              label='Scale Height H' if language == 'en' else 'Skalenhöhe H')
    ax2.axhline(y=8.5, color=COLORS['standard'], linestyle='--', alpha=0.7,
               label='Earth (8.5 km)' if language == 'en' else 'Erde (8,5 km)')

    if language == 'de':
        ax2.set_xlabel('Gravitationsskala (× Erd-g)', fontsize=12)
        ax2.set_ylabel('Skalenhöhe H [km]', fontsize=12)
        ax2.set_title('2. Atmosphärische Skalenhöhe vs. Gravitation', fontsize=14, fontweight='bold', pad=15)
    else:
        ax2.set_xlabel('Gravity Scale (× Earth g)', fontsize=12)
        ax2.set_ylabel('Scale Height H [km]', fontsize=12)
        ax2.set_title('2. Atmospheric Scale Height vs. Gravity', fontsize=14, fontweight='bold', pad=15)

    ax2.legend(fontsize=9, loc='upper right', bbox_to_anchor=(1.0, -0.08), framealpha=0.7)
    ax2.grid(True, alpha=0.3, which='both')

    if save:
        os.makedirs(VIS_DIR, exist_ok=True)
        filepath = os.path.join(VIS_DIR, 'temperature_atmosphere.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {filepath}")

    if show:
        plt.show()

    return fig


def plot_temperature_degeneracy(
    language: str = 'en',
    save: bool = True,
    show: bool = False
) -> plt.Figure:
    """
    Plot Fermi temperature and thermal vs degeneracy pressure crossover.
    Zeichnet Fermi-Temperatur und thermischen vs. Entartungsdruck-Uebergang.

    Creates a figure with two vertically stacked plots:
    - Top: Fermi temperature vs electron density
    - Bottom: Thermal vs degeneracy pressure crossover

    Args:
        language: 'en' for English, 'de' for German
        save: Whether to save the figure
        show: Whether to display the figure

    Returns:
        Matplotlib Figure object
    """
    constants = get_constants()

    # Create figure with two subplots stacked vertically with proper spacing
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14), gridspec_kw={'hspace': 0.4})

    # --- Top plot: Fermi temperature vs density ---
    n_e_range = np.logspace(28, 38, 100)  # m⁻³
    T_F_values = [fermi_temperature(n, constants) for n in n_e_range]

    ax1.loglog(n_e_range, T_F_values, color=COLORS['quantum'], linewidth=2.5,
              label='Fermi Temperature T_F' if language == 'en' else 'Fermi-Temperatur T_F')

    # Mark key densities with legend entries
    densities = [
        ('Metal (Cu)', 8.5e28, COLORS['earth']),
        ('White Dwarf', 3e35, COLORS['white_dwarf']),
        ('Neutron Star', 1e38, COLORS['neutron_star'])
    ]
    densities_de = [
        ('Metall (Cu)', 8.5e28, COLORS['earth']),
        ('Weißer Zwerg', 3e35, COLORS['white_dwarf']),
        ('Neutronenstern', 1e38, COLORS['neutron_star'])
    ]

    items = densities_de if language == 'de' else densities
    for name, n, color in items:
        T_F = fermi_temperature(n, constants)
        ax1.scatter([n], [T_F], color=color, s=100, zorder=5, label=name)

    # Reference temperature lines
    ax1.axhline(y=1e7, color='red', linestyle='--', alpha=0.5,
               label='T = 10⁷ K (star core)' if language == 'en' else 'T = 10⁷ K (Sternkern)')
    ax1.axhline(y=300, color='green', linestyle='--', alpha=0.5,
               label='T = 300 K (room temp)' if language == 'en' else 'T = 300 K (Raumtemp.)')

    if language == 'de':
        ax1.set_xlabel('Elektronendichte n_e [m⁻³]', fontsize=12)
        ax1.set_ylabel('Fermi-Temperatur T_F [K]', fontsize=12)
        ax1.set_title('1. Fermi-Temperatur vs. Elektronendichte', fontsize=14, fontweight='bold', pad=15)
    else:
        ax1.set_xlabel('Electron Density n_e [m⁻³]', fontsize=12)
        ax1.set_ylabel('Fermi Temperature T_F [K]', fontsize=12)
        ax1.set_title('1. Fermi Temperature vs. Electron Density', fontsize=14, fontweight='bold', pad=15)

    ax1.legend(fontsize=9, loc='upper right', bbox_to_anchor=(1.0, -0.08), ncol=3, framealpha=0.7)
    ax1.grid(True, alpha=0.3, which='both')

    # --- Bottom plot: Pressure crossover ---
    n_e_range2 = np.logspace(30, 36, 100)
    temperatures = [1e4, 1e6, 1e8]  # K
    temp_colors = [COLORS['standard'], COLORS['primary_amber'], COLORS['scaled']]

    for T, color in zip(temperatures, temp_colors):
        P_thermal = [thermal_pressure(n, T, constants.k_B) for n in n_e_range2]
        label = f'P_thermal (T={T:.0e} K)'
        ax2.loglog(n_e_range2, P_thermal, color=color, linewidth=2, linestyle='--', label=label)

    # Degeneracy pressure (independent of T)
    P_deg = [degeneracy_pressure_nr(n, constants) for n in n_e_range2]
    ax2.loglog(n_e_range2, P_deg, color=COLORS['quantum'], linewidth=3,
              label='P_degeneracy (Pauli)' if language == 'en' else 'P_Entartung (Pauli)')

    if language == 'de':
        ax2.set_xlabel('Elektronendichte n_e [m⁻³]', fontsize=12)
        ax2.set_ylabel('Druck [Pa]', fontsize=12)
        ax2.set_title('2. Thermischer vs. Entartungsdruck', fontsize=14, fontweight='bold', pad=15)
    else:
        ax2.set_xlabel('Electron Density n_e [m⁻³]', fontsize=12)
        ax2.set_ylabel('Pressure [Pa]', fontsize=12)
        ax2.set_title('2. Thermal vs. Degeneracy Pressure', fontsize=14, fontweight='bold', pad=15)

    ax2.legend(fontsize=9, loc='upper right', bbox_to_anchor=(1.0, -0.08), ncol=2, framealpha=0.7)
    ax2.grid(True, alpha=0.3, which='both')
    ax2.set_ylim(1e10, 1e35)

    if save:
        os.makedirs(VIS_DIR, exist_ok=True)
        filepath = os.path.join(VIS_DIR, 'temperature_degeneracy.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {filepath}")

    if show:
        plt.show()

    return fig


def plot_temperature_summary(
    language: str = 'en',
    save: bool = True,
    show: bool = False
) -> plt.Figure:
    """
    Create a comprehensive summary of temperature physics.
    Erstellt eine umfassende Zusammenfassung der Temperaturphysik.

    Three vertically stacked subplots:
    1. Scale height vs gravity
    2. Fermi temperature threshold
    3. Core temperature estimation

    Args:
        language: 'en' for English, 'de' for German
        save: Whether to save the figure
        show: Whether to display the figure

    Returns:
        Matplotlib Figure object
    """
    constants = get_constants()
    g_earth = constants.G * constants.M_earth / constants.R_earth**2

    # Create figure with 3 subplots stacked vertically
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 18), gridspec_kw={'hspace': 0.4})

    # --- Plot 1: Scale height effect ---
    g_scales = np.array([1, 2, 5, 10, 20, 50, 100])
    scale_heights = []
    for gs in g_scales:
        g = g_earth * gs
        H = atmospheric_scale_height(g, constants.T_surface_earth, constants.mu_air, constants.k_B)
        scale_heights.append(H / 1000)  # Convert to km

    colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(g_scales)))

    # Create bars with individual labels for legend
    for i, (gs, sh, color) in enumerate(zip(g_scales, scale_heights, colors)):
        ax1.bar(i, sh, color=color, edgecolor='black', label=f'{gs}× g')

    ax1.set_xticks(range(len(g_scales)))
    ax1.set_xticklabels([f'{gs}×' for gs in g_scales])
    ax1.set_yscale('log')

    if language == 'de':
        ax1.set_xlabel('Gravitationsskala', fontsize=12)
        ax1.set_ylabel('Skalenhöhe H [km]', fontsize=12)
        ax1.set_title('1. Atmosphärenkompression mit steigender Gravitation', fontsize=14, fontweight='bold', pad=15)
    else:
        ax1.set_xlabel('Gravity Scale', fontsize=12)
        ax1.set_ylabel('Scale Height H [km]', fontsize=12)
        ax1.set_title('1. Atmospheric Compression with Increasing Gravity', fontsize=14, fontweight='bold', pad=15)

    ax1.legend(fontsize=9, loc='upper right', bbox_to_anchor=(1.0, -0.08), ncol=4, framealpha=0.7)
    ax1.grid(True, alpha=0.3, axis='y')

    # --- Plot 2: T vs T_F threshold ---
    objects = [
        ('Metal (Cu)', 8.5e28, 300, COLORS['earth']),
        ('White Dwarf', 3e35, 1e7, COLORS['white_dwarf']),
        ('Neutron Star', 1e38, 1e8, COLORS['neutron_star'])
    ]
    objects_de = [
        ('Metall (Cu)', 8.5e28, 300, COLORS['earth']),
        ('Weißer Zwerg', 3e35, 1e7, COLORS['white_dwarf']),
        ('Neutronenstern', 1e38, 1e8, COLORS['neutron_star'])
    ]

    objs = objects_de if language == 'de' else objects
    x_pos = np.arange(len(objs))

    T_actuals = []
    T_fermis = []
    for name, n_e, T_actual, color in objs:
        T_F = fermi_temperature(n_e, constants)
        T_actuals.append(T_actual)
        T_fermis.append(T_F)

    width = 0.35
    bars_T = ax2.bar(x_pos - width/2, T_actuals, width, label='T (actual)' if language == 'en' else 'T (tatsächlich)',
                    color=COLORS['scaled'], edgecolor='black')
    bars_TF = ax2.bar(x_pos + width/2, T_fermis, width, label='T_F (Fermi)' if language == 'en' else 'T_F (Fermi)',
                     color=COLORS['quantum'], edgecolor='black')

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([o[0] for o in objs])
    ax2.set_yscale('log')

    if language == 'de':
        ax2.set_ylabel('Temperatur [K]', fontsize=12)
        ax2.set_title('2. Tatsächliche Temperatur vs. Fermi-Temperatur', fontsize=14, fontweight='bold', pad=15)
    else:
        ax2.set_ylabel('Temperature [K]', fontsize=12)
        ax2.set_title('2. Actual Temperature vs. Fermi Temperature', fontsize=14, fontweight='bold', pad=15)

    ax2.legend(fontsize=9, loc='upper right', bbox_to_anchor=(1.0, -0.08), ncol=2, framealpha=0.7)
    ax2.grid(True, alpha=0.3, axis='y')

    # --- Plot 3: Core temperature with gravity ---
    g_scales_core = np.linspace(1, 100, 50)
    T_core_vals = [constants.T_core_earth * gs**(1/3) for gs in g_scales_core]

    ax3.plot(g_scales_core, np.array(T_core_vals), color=COLORS['scaled'], linewidth=2.5,
            label='Core Temp (T ∝ g^(1/3))' if language == 'en' else 'Kerntemp. (T ∝ g^(1/3))')
    ax3.axhline(y=constants.T_core_earth, color=COLORS['standard'], linestyle='--',
               label='Earth core (5778 K)' if language == 'en' else 'Erdkern (5778 K)')

    # Mark key points
    for gs in [10, 50, 100]:
        T = constants.T_core_earth * gs**(1/3)
        ax3.scatter([gs], [T], color=COLORS['highlight'], s=80, zorder=5)

    if language == 'de':
        ax3.set_xlabel('Gravitationsskala', fontsize=12)
        ax3.set_ylabel('Geschätzte Kerntemperatur [K]', fontsize=12)
        ax3.set_title('3. Kerntemperatur durch gravitative Kompression', fontsize=14, fontweight='bold', pad=15)
    else:
        ax3.set_xlabel('Gravity Scale', fontsize=12)
        ax3.set_ylabel('Estimated Core Temperature [K]', fontsize=12)
        ax3.set_title('3. Core Temperature from Gravitational Compression', fontsize=14, fontweight='bold', pad=15)

    ax3.legend(fontsize=9, loc='upper right', bbox_to_anchor=(1.0, -0.08), framealpha=0.7)
    ax3.grid(True, alpha=0.3)

    if save:
        os.makedirs(VIS_DIR, exist_ok=True)
        filepath = os.path.join(VIS_DIR, 'temperature_summary.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {filepath}")

    if show:
        plt.show()

    return fig


def generate_all_thermal_plots(language: str = 'en', show: bool = False) -> List[plt.Figure]:
    """
    Generate all thermal physics visualizations.
    Erzeugt alle Temperaturphysik-Visualisierungen.

    Args:
        language: 'en' for English, 'de' for German
        show: Whether to display plots

    Returns:
        List of Matplotlib Figure objects
    """
    figures = []

    print("Generating thermal physics visualizations...")
    print("=" * 50)

    print("1. Temperature atmosphere profile...")
    figures.append(plot_temperature_atmosphere(language=language, show=show))

    print("2. Temperature degeneracy comparison...")
    figures.append(plot_temperature_degeneracy(language=language, show=show))

    print("3. Temperature summary...")
    figures.append(plot_temperature_summary(language=language, show=show))

    print("=" * 50)
    print(f"Generated {len(figures)} thermal physics visualizations")

    return figures


def verify_thermal_physics():
    """
    Verify thermal physics calculations with known values.
    Ueberprueft Temperaturphysik-Berechnungen mit bekannten Werten.
    """
    constants = get_constants()
    g_earth = constants.G * constants.M_earth / constants.R_earth**2

    print("=" * 60)
    print("Thermal Physics Verification")
    print("=" * 60)

    # Test scale height
    H = atmospheric_scale_height(g_earth, 288, constants.mu_air, constants.k_B)
    print(f"\nScale height at Earth surface:")
    print(f"  Calculated: {H/1000:.2f} km")
    print(f"  Expected:   ~8.5 km")

    # Test lapse rate
    gamma = adiabatic_lapse_rate(g_earth, constants.c_p_air)
    print(f"\nAdiabatic lapse rate:")
    print(f"  Calculated: {gamma*1000:.2f} K/km")
    print(f"  Expected:   ~-9.8 K/km")

    # Test Fermi temperature for metal
    n_e_metal = 8.5e28  # Copper electron density
    T_F = fermi_temperature(n_e_metal, constants)
    print(f"\nFermi temperature for copper:")
    print(f"  Calculated: {T_F:.0f} K")
    print(f"  Expected:   ~80,000 K")

    # Test thermal properties
    props = calculate_thermal_properties(g_scale=1.0, constants=constants)
    print(f"\nThermal properties at 1× g:")
    print(f"  Scale height: {props.scale_height/1000:.2f} km")
    print(f"  Lapse rate: {props.lapse_rate*1000:.2f} K/km")
    print(f"  Surface temp: {props.surface_temp} K")

    props_10x = calculate_thermal_properties(g_scale=10.0, constants=constants)
    print(f"\nThermal properties at 10× g:")
    print(f"  Scale height: {props_10x.scale_height/1000:.2f} km (10× compressed)")
    print(f"  Lapse rate: {props_10x.lapse_rate*1000:.2f} K/km (10× steeper)")

    print("\n" + "=" * 60)
    print("Verification complete!")


if __name__ == "__main__":
    verify_thermal_physics()
    print("\nGenerating visualizations...")
    generate_all_thermal_plots(language='en')
