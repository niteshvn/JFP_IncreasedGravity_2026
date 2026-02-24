"""
Solar Physics Module for Jugend forscht 2026 Physics Visualization Project
Sonnenphysik-Modul fuer Jugend forscht 2026 Physik-Visualisierungsprojekt

This module visualizes how the Sun changes when gravity is modified, including:
- Solar luminosity scaling with gravitational constant (L proportional to G^4)
- Stellar radius under modified gravity (R proportional to 1/G)
- Core and surface temperature response to gravity scaling
- Fusion rate sensitivity via the Gamow peak formalism
- Main-sequence lifetime and Eddington luminosity limits
- HR diagram shifts and stellar fate classification

The key insight is that even modest increases in G dramatically increase
luminosity (G^4 dependence), shorten stellar lifetimes, and can push
stars beyond the Eddington limit into instability.

Author: Jugend forscht 2026 Project
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Optional, List
from dataclasses import dataclass

from .constants import get_constants, PhysicalConstants
from .color_scheme import COLORS, get_sequence


# Output directory for visualizations
VIS_DIR = os.path.join(os.path.dirname(__file__), '..', 'visualizations')


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class SolarProperties:
    """
    Container for solar / stellar properties under modified gravity.
    Behaelter fuer Sonnen-/Stern-Eigenschaften bei modifizierter Gravitation.
    """
    luminosity: float           # [W]
    radius: float               # [m]
    surface_temperature: float  # Effective T [K]
    central_temperature: float  # Core T [K]
    central_pressure: float     # Core P [Pa]
    fusion_rate_factor: float   # Relative to standard
    lifetime: float             # Main-sequence lifetime [s]
    eddington_luminosity: float # Eddington limit [W]
    is_stable: bool             # Can sustain fusion without exceeding Eddington?


# ---------------------------------------------------------------------------
# Calculation functions
# ---------------------------------------------------------------------------

def solar_luminosity(G_scale: float, constants: PhysicalConstants) -> float:
    """
    Calculate solar luminosity under modified gravity.
    Berechnet die Sonnenleuchtkraft bei modifizierter Gravitation.

    From stellar structure theory the luminosity of a radiative star of
    fixed mass scales as L proportional to G^4:

        L = (64 pi^4 / 5) * (G^4 M^3 mu^4 m_p^4) / (kappa hbar^3)

    Simplifying for fixed composition and opacity:  L proportional to G^4.

    Args:
        G_scale: Scaling factor for the gravitational constant.
        constants: Physical constants (used for L_sun reference).

    Returns:
        Luminosity [W].
    """
    return constants.L_sun * G_scale**4


def solar_radius(G_scale: float, constants: PhysicalConstants) -> float:
    """
    Calculate solar radius under modified gravity.
    Berechnet den Sonnenradius bei modifizierter Gravitation.

    For a star in radiative equilibrium the radius scales as R proportional
    to 1/G (from combining hydrostatic equilibrium with radiative transport).
    The radius is capped at the Schwarzschild radius as a physical minimum.

    Args:
        G_scale: Scaling factor for the gravitational constant.
        constants: Physical constants.

    Returns:
        Radius [m].
    """
    R = constants.R_sun / G_scale

    # Physical minimum: Schwarzschild radius of the Sun at this G
    R_s = 2.0 * constants.G * G_scale * constants.M_sun / constants.c**2

    return max(R, R_s)


def solar_core_temperature(G_scale: float, constants: PhysicalConstants) -> float:
    """
    Calculate solar core temperature under modified gravity.
    Berechnet die Kerntemperatur der Sonne bei modifizierter Gravitation.

    From the virial theorem:
        T_c = G M mu m_p / (3 k_B R)

    With R proportional to 1/G this gives T_c proportional to G^2.

    Args:
        G_scale: Scaling factor for the gravitational constant.
        constants: Physical constants.

    Returns:
        Core temperature [K].
    """
    return constants.T_core_sun * G_scale**2


def solar_surface_temperature(luminosity: float, radius: float,
                              constants: PhysicalConstants) -> float:
    """
    Calculate effective surface temperature via Stefan-Boltzmann law.
    Berechnet die effektive Oberflaechentemperatur ueber Stefan-Boltzmann.

        T_eff = ( L / (4 pi R^2 sigma) )^(1/4)

    Args:
        luminosity: Stellar luminosity [W].
        radius: Stellar radius [m].
        constants: Physical constants (for sigma).

    Returns:
        Effective surface temperature [K].
    """
    return (luminosity / (4.0 * np.pi * radius**2 * constants.sigma))**0.25


def gamow_fusion_factor(T_core: float, alpha: float,
                        constants: PhysicalConstants) -> float:
    """
    Calculate the relative nuclear fusion rate via the Gamow peak formalism.
    Berechnet die relative Kernfusionsrate ueber den Gamow-Peak-Formalismus.

    The Gamow peak energy for the pp chain (Z1 = Z2 = 1, reduced mass = m_p/2):

        E_G = (pi alpha Z1 Z2)^2 * 2 mu_r c^2

    The rate scales as:

        rate proportional to exp( -3 (E_G / (4 k_B T))^(1/3) )

    The returned factor is normalised to unity at the standard solar core
    temperature so that it represents the relative change.

    Args:
        T_core: Core temperature [K].
        alpha: Fine-structure constant (dimensionless).
        constants: Physical constants.

    Returns:
        Fusion rate factor relative to the standard Sun (dimensionless).
    """
    Z1, Z2 = 1, 1
    mu_r = constants.m_p / 2.0  # reduced mass for pp

    E_G = (np.pi * alpha * Z1 * Z2)**2 * 2.0 * mu_r * constants.c**2

    # Exponent at the given temperature
    exponent = -3.0 * (E_G / (4.0 * constants.k_B * T_core))**(1.0 / 3.0)

    # Exponent at the standard solar core temperature (for normalisation)
    exponent_std = -3.0 * (E_G / (4.0 * constants.k_B * constants.T_core_sun))**(1.0 / 3.0)

    # Relative rate (avoid overflow by subtracting reference exponent first)
    log_ratio = exponent - exponent_std

    # Clamp to avoid numerical overflow / underflow
    log_ratio = np.clip(log_ratio, -500.0, 500.0)

    return np.exp(log_ratio)


def eddington_luminosity(G_scale: float, constants: PhysicalConstants) -> float:
    """
    Calculate the Eddington luminosity.
    Berechnet die Eddington-Leuchtkraft.

        L_Edd = 4 pi G M c / kappa

    where kappa ~ 0.02 m^2/kg (electron-scattering opacity).
    Scales linearly with G_scale for fixed mass.

    Args:
        G_scale: Scaling factor for the gravitational constant.
        constants: Physical constants.

    Returns:
        Eddington luminosity [W].
    """
    kappa = 0.02  # m^2/kg  (electron scattering, ~0.2 cm^2/g)
    G_eff = constants.G * G_scale
    return 4.0 * np.pi * G_eff * constants.M_sun * constants.c / kappa


def solar_lifetime(luminosity: float, constants: PhysicalConstants) -> float:
    """
    Calculate the main-sequence lifetime of the Sun.
    Berechnet die Hauptreihenlebensdauer der Sonne.

        tau = epsilon * M_sun * c^2 / L

    where epsilon ~ 0.007 is the hydrogen-burning mass-to-energy efficiency.

    Args:
        luminosity: Stellar luminosity [W].
        constants: Physical constants.

    Returns:
        Main-sequence lifetime [s].
    """
    epsilon = 0.007  # hydrogen burning efficiency
    return epsilon * constants.M_sun * constants.c**2 / luminosity


def solar_central_pressure(G_scale: float, constants: PhysicalConstants) -> float:
    """
    Estimate central pressure of the Sun under modified gravity.
    Schaetzt den zentralen Druck der Sonne bei modifizierter Gravitation.

    From hydrostatic equilibrium:  P_c ~ G M^2 / R^4.
    With R proportional to 1/G:  P_c proportional to G^5.

    Standard solar core pressure ~ 2.5e16 Pa.

    Args:
        G_scale: Scaling factor for the gravitational constant.
        constants: Physical constants.

    Returns:
        Central pressure [Pa].
    """
    P_core_sun = 2.5e16  # Pa, standard solar core pressure
    return P_core_sun * G_scale**5


def calculate_solar_properties(G_scale: float,
                               constants: Optional[PhysicalConstants] = None
                               ) -> SolarProperties:
    """
    Calculate all solar properties for a given gravity scaling.
    Berechnet alle Sonneneigenschaften fuer eine gegebene Gravitationsskalierung.

    Args:
        G_scale: Scaling factor for the gravitational constant.
        constants: Physical constants (uses standard if None).

    Returns:
        SolarProperties dataclass.
    """
    if constants is None:
        constants = get_constants()

    L = solar_luminosity(G_scale, constants)
    R = solar_radius(G_scale, constants)
    T_c = solar_core_temperature(G_scale, constants)
    T_eff = solar_surface_temperature(L, R, constants)
    P_c = solar_central_pressure(G_scale, constants)
    fusion = gamow_fusion_factor(T_c, constants.alpha, constants)
    L_edd = eddington_luminosity(G_scale, constants)
    tau = solar_lifetime(L, constants)
    stable = L < L_edd

    return SolarProperties(
        luminosity=L,
        radius=R,
        surface_temperature=T_eff,
        central_temperature=T_c,
        central_pressure=P_c,
        fusion_rate_factor=fusion,
        lifetime=tau,
        eddington_luminosity=L_edd,
        is_stable=stable,
    )


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

_SECONDS_PER_GYR = 1.0e9 * 365.25 * 24 * 3600  # seconds in 1 Gyr


def _g_range(n: int = 300) -> np.ndarray:
    """Return a log-spaced array of G_scale values from 0.1 to 10^40 (includes 10^36 threshold)."""
    return np.logspace(np.log10(0.1), 40, n)


# ---------------------------------------------------------------------------
# Plot 1: Solar Structure
# ---------------------------------------------------------------------------

def plot_solar_structure(
    constants: Optional[PhysicalConstants] = None,
    language: str = 'en',
    save: bool = True,
    show: bool = True,
) -> plt.Figure:
    """
    Plot how the Sun's internal structure changes with gravity scaling.
    Zeigt, wie sich die innere Struktur der Sonne mit der Gravitationsskalierung aendert.

    Four vertical subplots:
      1. Luminosity vs G_scale  (log-log)
      2. Radius vs G_scale
      3. Core temperature vs G_scale
      4. Fusion rate factor vs G_scale

    Args:
        constants: Physical constants (uses standard if None).
        language: 'en' for English, 'de' for German.
        save: Whether to save the figure.
        show: Whether to display the figure.

    Returns:
        matplotlib Figure object.
    """
    if constants is None:
        constants = get_constants()

    G_arr = _g_range()

    # Pre-compute arrays
    L_arr = np.array([solar_luminosity(g, constants) for g in G_arr])
    R_arr = np.array([solar_radius(g, constants) for g in G_arr])
    Tc_arr = np.array([solar_core_temperature(g, constants) for g in G_arr])
    fusion_arr = np.array([
        gamow_fusion_factor(solar_core_temperature(g, constants), constants.alpha, constants)
        for g in G_arr
    ])

    # Reference scaling lines
    L_ref = constants.L_sun * G_arr**4
    R_ref = constants.R_sun / G_arr
    Tc_ref = constants.T_core_sun * G_arr**2

    # Schwarzschild radii for reference
    R_schwarz = np.array([2.0 * constants.G * g * constants.M_sun / constants.c**2
                          for g in G_arr])

    # ----- figure -----
    fig, axes = plt.subplots(4, 1, figsize=(12, 32))
    fig.subplots_adjust(hspace=0.5, top=0.95, bottom=0.04)

    if language == 'de':
        fig.suptitle('Sonnenstruktur bei modifizierter Gravitation',
                     fontsize=16, fontweight='bold', y=0.98)
    else:
        fig.suptitle('Solar Structure Under Modified Gravity',
                     fontsize=16, fontweight='bold', y=0.98)

    ax1, ax2, ax3, ax4 = axes

    # --- Subplot 1: Luminosity ---
    ax1.loglog(G_arr, L_arr / constants.L_sun, '-', color=COLORS['sun'],
               linewidth=2.5, label=r'$L/L_\odot$')
    ax1.loglog(G_arr, L_ref / constants.L_sun, '--', color=COLORS['muted'],
               linewidth=1.5, alpha=0.7,
               label=r'$L \propto G^4$' + (' Referenz' if language == 'de' else ' reference'))
    ax1.axvline(x=1.0, color=COLORS['standard'], linestyle=':', linewidth=1.5,
                label=r'G = G$_{\rm standard}$' if language == 'en' else r'G = G$_{\rm Standard}$')
    ax1.axhline(y=1.0, color=COLORS['muted'], linestyle=':', linewidth=1.0, alpha=0.5)

    ax1.set_xlabel('G / G_standard', fontsize=12)
    if language == 'de':
        ax1.set_ylabel('Leuchtkraft (Sonnenleuchtkraefte)', fontsize=12)
        ax1.set_title('Leuchtkraft vs. Gravitationsskalierung', fontsize=14, fontweight='bold', pad=15)
    else:
        ax1.set_ylabel('Luminosity (Solar luminosities)', fontsize=12)
        ax1.set_title('Luminosity vs. Gravity Scaling', fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3)
    # Add G×10^36 threshold marker (HIGHLY VISIBLE - altered universe)
    ax1.axvline(x=1e36, color='#FF0000', linestyle='-', linewidth=4, alpha=1.0, zorder=10)
    ax1.axvspan(1e35, 1e37, alpha=0.15, color='red', zorder=1)  # Highlight band
    lbl = r'$\mathbf{G \times 10^{36}}$' + ('\n(Veraendertes\nUniversum)' if language == 'de' else '\n(Altered\nUniverse)')
    ax1.text(1e36, (L_arr / constants.L_sun).max()*0.3, lbl, color='white', fontsize=12, fontweight='bold',
             rotation=90, va='center', ha='center', bbox=dict(boxstyle='round,pad=0.3', facecolor='#FF0000', edgecolor='darkred', linewidth=2))
    ax1.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

    # Add SUN EXPLOSION WARNING box (Peer Review 2 feedback)
    if language == 'de':
        explosion_text = (
            'SONNE EXPLODIERT BEI G×10³⁶!\n\n'
            'Formeln:\n'
            '• L ∝ G⁴ → L = 10¹⁴⁴ × L☉\n'
            '• L_Edd ∝ G → L/L_Edd ∝ G³\n'
            '• Bei G×10³⁶: L/L_Edd = 10¹⁰⁸\n\n'
            'Die Sonne überschreitet die\n'
            'Eddington-Grenze um Faktor 10¹⁰⁸\n'
            '→ SOFORTIGE EXPLOSION!\n\n'
            'Planeten-Überleben: UNMÖGLICH'
        )
    else:
        explosion_text = (
            'SUN EXPLODES AT G×10³⁶!\n\n'
            'Formulas:\n'
            '• L ∝ G⁴ → L = 10¹⁴⁴ × L☉\n'
            '• L_Edd ∝ G → L/L_Edd ∝ G³\n'
            '• At G×10³⁶: L/L_Edd = 10¹⁰⁸\n\n'
            'The Sun exceeds the Eddington\n'
            'limit by factor 10¹⁰⁸\n'
            '→ IMMEDIATE EXPLOSION!\n\n'
            'Planetary survival: IMPOSSIBLE'
        )
    ax1.text(0.02, 0.98, explosion_text, fontsize=9, va='top', ha='left',
             transform=ax1.transAxes,
             bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['box_error'],
                      edgecolor=COLORS['collapse'], linewidth=3, alpha=0.95),
             color=COLORS['collapse'], fontweight='bold', family='monospace')

    # --- Subplot 2: Radius ---
    ax2.loglog(G_arr, R_arr / constants.R_sun, '-', color=COLORS['primary_blue'],
               linewidth=2.5, label=r'$R/R_\odot$')
    ax2.loglog(G_arr, R_ref / constants.R_sun, '--', color=COLORS['muted'],
               linewidth=1.5, alpha=0.7,
               label=r'$R \propto G^{-1}$' + (' Referenz' if language == 'de' else ' reference'))
    ax2.loglog(G_arr, R_schwarz / constants.R_sun, '-.', color=COLORS['black_hole'],
               linewidth=1.5, alpha=0.7, label='Schwarzschild')
    ax2.axvline(x=1.0, color=COLORS['standard'], linestyle=':', linewidth=1.5)
    ax2.axhline(y=1.0, color=COLORS['muted'], linestyle=':', linewidth=1.0, alpha=0.5)
    ax2.set_xlabel('G / G_standard', fontsize=12)
    if language == 'de':
        ax2.set_ylabel('Radius (Sonnenradien)', fontsize=12)
        ax2.set_title('Sternradius vs. Gravitationsskalierung', fontsize=14, fontweight='bold', pad=15)
    else:
        ax2.set_ylabel('Radius (Solar radii)', fontsize=12)
        ax2.set_title('Stellar Radius vs. Gravity Scaling', fontsize=14, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3)
    # Add G×10^36 threshold marker (HIGHLY VISIBLE - altered universe)
    ax2.axvline(x=1e36, color='#FF0000', linestyle='-', linewidth=4, alpha=1.0, zorder=10)
    ax2.axvspan(1e35, 1e37, alpha=0.15, color='red', zorder=1)
    lbl = r'$\mathbf{G \times 10^{36}}$' + ('\n(Veraendertes\nUniversum)' if language == 'de' else '\n(Altered\nUniverse)')
    ax2.text(1e36, 1e-30, lbl, color='white', fontsize=12, fontweight='bold',
             rotation=90, va='center', ha='center', bbox=dict(boxstyle='round,pad=0.3', facecolor='#FF0000', edgecolor='darkred', linewidth=2))
    ax2.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

    # --- Subplot 3: Core Temperature ---
    ax3.loglog(G_arr, Tc_arr / constants.T_core_sun, '-', color=COLORS['temp_hot'],
               linewidth=2.5, label=r'$T_c / T_{c,\odot}$')
    ax3.loglog(G_arr, Tc_ref / constants.T_core_sun, '--', color=COLORS['muted'],
               linewidth=1.5, alpha=0.7,
               label=r'$T_c \propto G^2$' + (' Referenz' if language == 'de' else ' reference'))
    ax3.axvline(x=1.0, color=COLORS['standard'], linestyle=':', linewidth=1.5)
    ax3.axhline(y=1.0, color=COLORS['muted'], linestyle=':', linewidth=1.0, alpha=0.5)
    ax3.set_xlabel('G / G_standard', fontsize=12)
    if language == 'de':
        ax3.set_ylabel('Kerntemperatur (Vielfaches des Standardwerts)', fontsize=12)
        ax3.set_title('Kerntemperatur vs. Gravitationsskalierung', fontsize=14, fontweight='bold', pad=15)
    else:
        ax3.set_ylabel('Core Temperature (multiples of standard)', fontsize=12)
        ax3.set_title('Core Temperature vs. Gravity Scaling', fontsize=14, fontweight='bold', pad=15)
    ax3.grid(True, alpha=0.3)
    # Add G×10^36 threshold marker (HIGHLY VISIBLE - altered universe)
    ax3.axvline(x=1e36, color='#FF0000', linestyle='-', linewidth=4, alpha=1.0, zorder=10)
    ax3.axvspan(1e35, 1e37, alpha=0.15, color='red', zorder=1)
    lbl = r'$\mathbf{G \times 10^{36}}$' + ('\n(Veraendertes\nUniversum)' if language == 'de' else '\n(Altered\nUniverse)')
    ax3.text(1e36, (Tc_arr / constants.T_core_sun).max()*0.3, lbl, color='white', fontsize=12, fontweight='bold',
             rotation=90, va='center', ha='center', bbox=dict(boxstyle='round,pad=0.3', facecolor='#FF0000', edgecolor='darkred', linewidth=2))
    ax3.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

    # --- Subplot 4: Fusion Rate ---
    ax4.loglog(G_arr, fusion_arr, '-', color=COLORS['quantum'],
               linewidth=2.5, label='Fusion rate factor')
    ax4.axvline(x=1.0, color=COLORS['standard'], linestyle=':', linewidth=1.5)
    ax4.axhline(y=1.0, color=COLORS['muted'], linestyle=':', linewidth=1.0, alpha=0.5)
    ax4.set_xlabel('G / G_standard', fontsize=12)
    if language == 'de':
        ax4.set_ylabel('Fusionsrate (relativ)', fontsize=12)
        ax4.set_title('Kernfusionsrate vs. Gravitationsskalierung', fontsize=14, fontweight='bold', pad=15)
    else:
        ax4.set_ylabel('Fusion Rate (relative)', fontsize=12)
        ax4.set_title('Nuclear Fusion Rate vs. Gravity Scaling', fontsize=14, fontweight='bold', pad=15)
    ax4.grid(True, alpha=0.3)
    # Add G×10^36 threshold marker (HIGHLY VISIBLE - altered universe)
    ax4.axvline(x=1e36, color='#FF0000', linestyle='-', linewidth=4, alpha=1.0, zorder=10)
    ax4.axvspan(1e35, 1e37, alpha=0.15, color='red', zorder=1)
    lbl = r'$\mathbf{G \times 10^{36}}$' + ('\n(Veraendertes\nUniversum)' if language == 'de' else '\n(Altered\nUniverse)')
    ax4.text(1e36, fusion_arr.max()*0.3, lbl, color='white', fontsize=12, fontweight='bold',
             rotation=90, va='center', ha='center', bbox=dict(boxstyle='round,pad=0.3', facecolor='#FF0000', edgecolor='darkred', linewidth=2))
    ax4.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

    # ----- save / show -----
    if save:
        suffix = '_de' if language == 'de' else ''
        fname = os.path.join(VIS_DIR, f'solar_structure{suffix}.png')
        os.makedirs(VIS_DIR, exist_ok=True)
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        print(f'Saved: {fname}')
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Plot 2: Solar Lifetime and Stability
# ---------------------------------------------------------------------------

def plot_solar_lifetime(
    constants: Optional[PhysicalConstants] = None,
    language: str = 'en',
    save: bool = True,
    show: bool = True,
) -> plt.Figure:
    """
    Plot stellar lifetime and stability under modified gravity.
    Zeigt Sternlebensdauer und Stabilitaet bei modifizierter Gravitation.

    Four vertical subplots:
      1. Main-sequence lifetime vs G_scale
      2. Luminosity vs Eddington limit
      3. Surface temperature vs G_scale
      4. HR diagram shift

    Args:
        constants: Physical constants (uses standard if None).
        language: 'en' for English, 'de' for German.
        save: Whether to save the figure.
        show: Whether to display the figure.

    Returns:
        matplotlib Figure object.
    """
    if constants is None:
        constants = get_constants()

    G_arr = _g_range()

    # Pre-compute arrays
    L_arr = np.array([solar_luminosity(g, constants) for g in G_arr])
    R_arr = np.array([solar_radius(g, constants) for g in G_arr])
    Tc_arr = np.array([solar_core_temperature(g, constants) for g in G_arr])
    Teff_arr = np.array([
        solar_surface_temperature(solar_luminosity(g, constants),
                                  solar_radius(g, constants), constants)
        for g in G_arr
    ])
    tau_arr = np.array([solar_lifetime(solar_luminosity(g, constants), constants)
                        for g in G_arr])
    L_edd_arr = np.array([eddington_luminosity(g, constants) for g in G_arr])

    # ----- figure -----
    fig, axes = plt.subplots(4, 1, figsize=(12, 32))
    fig.subplots_adjust(hspace=0.5, top=0.95, bottom=0.04)

    if language == 'de':
        fig.suptitle('Sternlebensdauer und Stabilitaet bei modifizierter Gravitation',
                     fontsize=16, fontweight='bold', y=0.98)
    else:
        fig.suptitle('Stellar Lifetime and Stability Under Modified Gravity',
                     fontsize=16, fontweight='bold', y=0.98)

    ax1, ax2, ax3, ax4 = axes

    # --- Subplot 1: Lifetime ---
    ax1.loglog(G_arr, tau_arr / _SECONDS_PER_GYR, '-', color=COLORS['primary_blue'],
               linewidth=2.5, label='Main-sequence lifetime')
    ax1.axvline(x=1.0, color=COLORS['standard'], linestyle=':', linewidth=1.5)
    ax1.axhline(y=tau_arr[np.argmin(np.abs(G_arr - 1.0))] / _SECONDS_PER_GYR,
                color=COLORS['muted'], linestyle=':', linewidth=1.0, alpha=0.5)
    ax1.set_xlabel('G / G_standard', fontsize=12)
    if language == 'de':
        ax1.set_ylabel('Lebensdauer (Milliarden Jahre)', fontsize=12)
        ax1.set_title('Hauptreihen-Lebensdauer vs. Gravitationsskalierung', fontsize=14, fontweight='bold', pad=15)
    else:
        ax1.set_ylabel('Lifetime (Gyr)', fontsize=12)
        ax1.set_title('Main-Sequence Lifetime vs. Gravity Scaling', fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3)
    # Add G×10^36 threshold marker (HIGHLY VISIBLE - altered universe)
    ax1.axvline(x=1e36, color='#FF0000', linestyle='-', linewidth=4, alpha=1.0, zorder=10)
    ax1.axvspan(1e35, 1e37, alpha=0.15, color='red', zorder=1)
    lbl1 = r'$\mathbf{G \times 10^{36}}$' + ('\n(Veraendertes\nUniversum)' if language == 'de' else '\n(Altered\nUniverse)')
    ax1.text(1e36, (tau_arr / _SECONDS_PER_GYR).max()*0.01, lbl1, color='white', fontsize=12, fontweight='bold',
             rotation=90, va='center', ha='center', bbox=dict(boxstyle='round,pad=0.3', facecolor='#FF0000', edgecolor='darkred', linewidth=2))
    ax1.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

    # --- Subplot 2: Luminosity vs Eddington ---
    ax2.loglog(G_arr, L_arr / constants.L_sun, '-', color=COLORS['sun'],
               linewidth=2.5, label=r'$L / L_\odot$')
    ax2.loglog(G_arr, L_edd_arr / constants.L_sun, '--', color=COLORS['relativistic'],
               linewidth=2.0, label='Eddington')
    ax2.axvline(x=1.0, color=COLORS['standard'], linestyle=':', linewidth=1.5)
    # Shade unstable region
    ax2.fill_between(G_arr, L_arr / constants.L_sun, L_edd_arr / constants.L_sun,
                     where=L_arr > L_edd_arr, color=COLORS['relativistic'],
                     alpha=0.2, label='Unstable' if language == 'en' else 'Instabil')
    ax2.set_xlabel('G / G_standard', fontsize=12)
    if language == 'de':
        ax2.set_ylabel('Leuchtkraft (Sonnenleuchtkraefte)', fontsize=12)
        ax2.set_title('Leuchtkraft vs. Eddington-Grenze', fontsize=14, fontweight='bold', pad=15)
    else:
        ax2.set_ylabel('Luminosity (Solar luminosities)', fontsize=12)
        ax2.set_title('Luminosity vs. Eddington Limit', fontsize=14, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3)
    # Add G×10^36 threshold marker (HIGHLY VISIBLE - altered universe)
    ax2.axvline(x=1e36, color='#FF0000', linestyle='-', linewidth=4, alpha=1.0, zorder=10)
    ax2.axvspan(1e35, 1e37, alpha=0.15, color='red', zorder=1)
    lbl2 = r'$\mathbf{G \times 10^{36}}$' + ('\n(Veraendertes\nUniversum)' if language == 'de' else '\n(Altered\nUniverse)')
    ax2.text(1e36, (L_arr / constants.L_sun).max()*0.3, lbl2, color='white', fontsize=12, fontweight='bold',
             rotation=90, va='center', ha='center', bbox=dict(boxstyle='round,pad=0.3', facecolor='#FF0000', edgecolor='darkred', linewidth=2))
    ax2.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

    # Add EDDINGTON INSTABILITY explanation (Peer Review 2 feedback)
    if language == 'de':
        eddington_text = (
            'EDDINGTON-INSTABILITÄT:\n\n'
            'Wenn L > L_Edd:\n'
            '• Strahlungsdruck > Gravitation\n'
            '• Stern wird auseinandergerissen\n'
            '• Runaway-Kernfusion\n\n'
            'Bei G×10³⁶:\n'
            'L/L_Edd = G³ = 10¹⁰⁸\n'
            '→ HYPERNOVA-EXPLOSION!'
        )
    else:
        eddington_text = (
            'EDDINGTON INSTABILITY:\n\n'
            'When L > L_Edd:\n'
            '• Radiation pressure > Gravity\n'
            '• Star is torn apart\n'
            '• Runaway nuclear fusion\n\n'
            'At G×10³⁶:\n'
            'L/L_Edd = G³ = 10¹⁰⁸\n'
            '→ HYPERNOVA EXPLOSION!'
        )
    ax2.text(0.02, 0.98, eddington_text, fontsize=9, va='top', ha='left',
             transform=ax2.transAxes,
             bbox=dict(boxstyle='round,pad=0.4', facecolor=COLORS['box_error'],
                      edgecolor=COLORS['collapse'], linewidth=2, alpha=0.95),
             color=COLORS['collapse'], fontweight='bold', family='monospace')

    # --- Subplot 3: Surface Temperature ---
    ax3.semilogx(G_arr, Teff_arr, '-', color=COLORS['primary_amber'],
                 linewidth=2.5, label=r'$T_{\rm eff}$')
    ax3.axvline(x=1.0, color=COLORS['standard'], linestyle=':', linewidth=1.5)
    ax3.set_xlabel('G / G_standard', fontsize=12)
    if language == 'de':
        ax3.set_ylabel('Oberflaechentemperatur (K)', fontsize=12)
        ax3.set_title('Effektive Oberflaechentemperatur vs. Gravitationsskalierung', fontsize=14, fontweight='bold', pad=15)
    else:
        ax3.set_ylabel('Surface Temperature (K)', fontsize=12)
        ax3.set_title('Effective Surface Temperature vs. Gravity Scaling', fontsize=14, fontweight='bold', pad=15)
    ax3.grid(True, alpha=0.3)
    # Add G×10^36 threshold marker (HIGHLY VISIBLE - altered universe)
    ax3.axvline(x=1e36, color='#FF0000', linestyle='-', linewidth=4, alpha=1.0, zorder=10)
    ax3.axvspan(1e35, 1e37, alpha=0.15, color='red', zorder=1)
    lbl3 = r'$\mathbf{G \times 10^{36}}$' + ('\n(Veraendertes\nUniversum)' if language == 'de' else '\n(Altered\nUniverse)')
    ax3.text(1e36, Teff_arr.max()*0.5, lbl3, color='white', fontsize=12, fontweight='bold',
             rotation=90, va='center', ha='center', bbox=dict(boxstyle='round,pad=0.3', facecolor='#FF0000', edgecolor='darkred', linewidth=2))
    ax3.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

    # --- Subplot 4: HR Diagram ---
    colors_seq = get_sequence()
    sample_g = [0.5, 1.0, 2.0, 5.0, 10.0]
    for i, g in enumerate(sample_g):
        props = calculate_solar_properties(g, constants)
        lum_solar = props.luminosity / constants.L_sun
        marker = '*' if not props.is_stable else 'o'
        ax4.semilogy(props.surface_temperature, lum_solar,
                     marker=marker, markersize=12, color=colors_seq[i % len(colors_seq)],
                     label=f'G = {g}x')
    ax4.invert_xaxis()
    ax4.set_xlabel('Effective Temperature (K)' if language == 'en' else 'Effektive Temperatur (K)', fontsize=12)
    if language == 'de':
        ax4.set_ylabel('Leuchtkraft (Sonnenleuchtkraefte)', fontsize=12)
        ax4.set_title('Hertzsprung-Russell-Diagramm', fontsize=14, fontweight='bold', pad=15)
    else:
        ax4.set_ylabel('Luminosity (Solar luminosities)', fontsize=12)
        ax4.set_title('Hertzsprung-Russell Diagram', fontsize=14, fontweight='bold', pad=15)
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5)

    # ----- save / show -----
    if save:
        suffix = '_de' if language == 'de' else ''
        fname = os.path.join(VIS_DIR, f'solar_lifetime{suffix}.png')
        os.makedirs(VIS_DIR, exist_ok=True)
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        print(f'Saved: {fname}')
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Plot 3: Solar Summary
# ---------------------------------------------------------------------------

def plot_solar_summary(
    constants: Optional[PhysicalConstants] = None,
    language: str = 'en',
    save: bool = True,
    show: bool = True,
) -> plt.Figure:
    """
    Summary plot comparing solar properties at selected G values.
    Zusammenfassungsdiagramm der Sonneneigenschaften bei ausgewaehlten G-Werten.

    Four vertical subplots:
      1. Bar chart: luminosity at selected G values
      2. Multi-line: radius, temperature, pressure vs G
      3. Fusion rate at selected G values
      4. Stellar fate classification diagram

    Args:
        constants: Physical constants (uses standard if None).
        language: 'en' for English, 'de' for German.
        save: Whether to save the figure.
        show: Whether to display the figure.

    Returns:
        matplotlib Figure object.
    """
    if constants is None:
        constants = get_constants()

    sample_g = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    props_list = [calculate_solar_properties(g, constants) for g in sample_g]
    colors_seq = get_sequence()

    # ----- figure -----
    fig, axes = plt.subplots(4, 1, figsize=(12, 32))
    fig.subplots_adjust(hspace=0.5, top=0.95, bottom=0.04)

    if language == 'de':
        fig.suptitle('Zusammenfassung: Sonne bei modifizierter Gravitation',
                     fontsize=16, fontweight='bold', y=0.98)
    else:
        fig.suptitle('Summary: Sun Under Modified Gravity',
                     fontsize=16, fontweight='bold', y=0.98)

    ax1, ax2, ax3, ax4 = axes

    # --- Subplot 1: Luminosity bar chart ---
    lum_vals = [p.luminosity / constants.L_sun for p in props_list]
    bar_labels = [f'G={g}x' for g in sample_g]
    bar_colors = [colors_seq[i % len(colors_seq)] for i in range(len(sample_g))]
    bars = ax1.bar(bar_labels, lum_vals, color=bar_colors, edgecolor='black', linewidth=1.5)
    ax1.set_yscale('log')
    for bar, val in zip(bars, lum_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, val * 1.3,
                 f'{val:.1e}', ha='center', va='bottom',
                 fontsize=9, fontweight='bold')
    if language == 'de':
        ax1.set_ylabel('Leuchtkraft (Sonnenleuchtkraefte)', fontsize=12)
        ax1.set_title('Leuchtkraft bei verschiedenen Gravitationswerten', fontsize=14, fontweight='bold', pad=15)
    else:
        ax1.set_ylabel('Luminosity (Solar luminosities)', fontsize=12)
        ax1.set_title('Luminosity at Different Gravity Values', fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, axis='y')

    # --- Subplot 2: Multi-line comparison ---
    G_arr = _g_range()
    R_norm = np.array([solar_radius(g, constants) / constants.R_sun for g in G_arr])
    Tc_norm = np.array([solar_core_temperature(g, constants) / constants.T_core_sun for g in G_arr])
    Pc_norm = np.array([solar_central_pressure(g, constants) / 2.5e16 for g in G_arr])
    L_norm = np.array([solar_luminosity(g, constants) / constants.L_sun for g in G_arr])

    ax2.loglog(G_arr, L_norm, '-', color=COLORS['sun'], linewidth=2.0,
               label=r'$L/L_\odot \propto G^4$')
    ax2.loglog(G_arr, R_norm, '-', color=COLORS['primary_blue'], linewidth=2.0,
               label=r'$R/R_\odot \propto G^{-1}$')
    ax2.loglog(G_arr, Tc_norm, '-', color=COLORS['temp_hot'], linewidth=2.0,
               label=r'$T_c/T_{c,\odot} \propto G^2$')
    ax2.loglog(G_arr, Pc_norm, '-', color=COLORS['quantum'], linewidth=2.0,
               label=r'$P_c/P_{c,\odot} \propto G^5$')
    ax2.axvline(x=1.0, color=COLORS['standard'], linestyle=':', linewidth=1.5)
    ax2.axhline(y=1.0, color=COLORS['muted'], linestyle=':', linewidth=1.0, alpha=0.5)
    ax2.set_xlabel('G / G_standard', fontsize=12)
    if language == 'de':
        ax2.set_ylabel('Vielfaches des Standardwerts', fontsize=12)
        ax2.set_title('Alle Groessen vs. Gravitationsskalierung', fontsize=14, fontweight='bold', pad=15)
    else:
        ax2.set_ylabel('Multiple of standard value', fontsize=12)
        ax2.set_title('All Quantities vs. Gravity Scaling', fontsize=14, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)

    # --- Subplot 3: Fusion rate bars ---
    fusion_vals = [p.fusion_rate_factor for p in props_list]
    bars3 = ax3.bar(bar_labels, fusion_vals, color=bar_colors, edgecolor='black', linewidth=1.5)
    ax3.set_yscale('log')
    for bar, val in zip(bars3, fusion_vals):
        if val > 0 and np.isfinite(val):
            ax3.text(bar.get_x() + bar.get_width() / 2, max(val * 1.3, 1e-10),
                     f'{val:.1e}', ha='center', va='bottom',
                     fontsize=9, fontweight='bold')
    if language == 'de':
        ax3.set_ylabel('Fusionsrate (relativ)', fontsize=12)
        ax3.set_title('Kernfusionsrate bei verschiedenen Gravitationswerten', fontsize=14, fontweight='bold', pad=15)
    else:
        ax3.set_ylabel('Fusion Rate (relative)', fontsize=12)
        ax3.set_title('Nuclear Fusion Rate at Different Gravity Values', fontsize=14, fontweight='bold', pad=15)
    ax3.grid(True, alpha=0.3, axis='y')

    # --- Subplot 4: Stellar fate diagram ---
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 6)
    ax4.axis('off')

    fate_data = []
    for g, props in zip(sample_g, props_list):
        if props.is_stable:
            if g <= 1.0:
                fate = 'Stable MS' if language == 'en' else 'Stabile HR'
                fc = COLORS['standard']
            else:
                fate = 'Hot, fast' if language == 'en' else 'Heiss, schnell'
                fc = COLORS['sun']
        else:
            fate = 'Unstable' if language == 'en' else 'Instabil'
            fc = COLORS['relativistic']
        fate_data.append((g, fate, fc))

    x_positions = np.linspace(1, 9, len(fate_data))
    for i, (g, fate, fc) in enumerate(fate_data):
        circle = plt.Circle((x_positions[i], 3), 0.6, color=fc,
                            ec='black', linewidth=2, zorder=5)
        ax4.add_patch(circle)
        ax4.text(x_positions[i], 3, f'G={g}x',
                 ha='center', va='center', fontsize=9, fontweight='bold',
                 color='white')
        ax4.text(x_positions[i], 1.8, fate,
                 ha='center', va='top', fontsize=10, fontweight='bold')

    if language == 'de':
        ax4.set_title('Schicksal des Sterns bei verschiedenen Gravitationswerten',
                     fontsize=14, fontweight='bold', pad=15)
    else:
        ax4.set_title('Stellar Fate at Different Gravity Values',
                     fontsize=14, fontweight='bold', pad=15)

    # ----- save / show -----
    if save:
        suffix = '_de' if language == 'de' else ''
        fname = os.path.join(VIS_DIR, f'solar_summary{suffix}.png')
        os.makedirs(VIS_DIR, exist_ok=True)
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        print(f'Saved: {fname}')
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Generate all plots
# ---------------------------------------------------------------------------

def generate_all_solar_plots(
    language: str = 'en',
    save: bool = True,
    show: bool = False,
) -> List[plt.Figure]:
    """
    Generate all solar physics plots.
    Erzeugt alle Sonnenphysik-Diagramme.

    Args:
        language: 'en' or 'de'.
        save: Whether to save figures.
        show: Whether to display figures.

    Returns:
        List of matplotlib Figure objects.
    """
    figures = []
    figures.append(plot_solar_structure(language=language, save=save, show=show))
    figures.append(plot_solar_lifetime(language=language, save=save, show=show))
    figures.append(plot_solar_summary(language=language, save=save, show=show))
    return figures


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_solar_physics() -> bool:
    """
    Run basic sanity checks on the solar physics calculations.
    Fuehrt grundlegende Plausibilitaetspruefungen der Sonnenphysik-Berechnungen durch.

    Returns:
        True if all checks pass.
    """
    constants = get_constants()
    ok = True

    # Standard Sun: G_scale = 1
    props = calculate_solar_properties(1.0, constants)
    assert abs(props.luminosity / constants.L_sun - 1.0) < 1e-10, 'Luminosity at G=1 should be L_sun'
    assert abs(props.radius / constants.R_sun - 1.0) < 1e-10, 'Radius at G=1 should be R_sun'
    assert abs(props.central_temperature / constants.T_core_sun - 1.0) < 1e-10, 'T_core at G=1 should be T_core_sun'
    assert props.is_stable, 'Standard Sun should be stable'
    print('[OK] Standard Sun properties correct.')

    # Doubled gravity: L should be ~16x, R ~0.5x, T_c ~4x
    props2 = calculate_solar_properties(2.0, constants)
    assert abs(props2.luminosity / constants.L_sun - 16.0) < 0.01, 'L at G=2 should be 16 L_sun'
    assert abs(props2.radius / constants.R_sun - 0.5) < 0.01, 'R at G=2 should be 0.5 R_sun'
    assert abs(props2.central_temperature / constants.T_core_sun - 4.0) < 0.01, 'T_c at G=2 should be 4 T_core_sun'
    print('[OK] G=2 scaling relations correct.')

    # Large G: should become unstable (exceed Eddington)
    props100 = calculate_solar_properties(100.0, constants)
    assert not props100.is_stable, 'At G=100 the Sun should exceed Eddington limit'
    print('[OK] G=100 instability check passed.')

    # Lifetime scaling: tau proportional to 1/G^4
    tau1 = calculate_solar_properties(1.0, constants).lifetime
    tau2 = calculate_solar_properties(2.0, constants).lifetime
    ratio = tau1 / tau2
    assert abs(ratio - 16.0) < 0.01, f'Lifetime ratio should be 16, got {ratio}'
    print('[OK] Lifetime scaling correct.')

    print('All solar physics checks passed!')
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print('Solar Physics Module - Jugend forscht 2026')
    print('==================================================')
    verify_solar_physics()
    print()
    print('Generating plots...')
    generate_all_solar_plots(language='en', show=False)
    generate_all_solar_plots(language='de', show=False)
    print('Done! Check the visualizations folder for output.')

