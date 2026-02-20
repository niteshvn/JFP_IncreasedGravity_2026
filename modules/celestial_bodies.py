"""
Celestial Bodies Module for Jugend forscht 2026 Physics Visualization Project
Himmelskoerper-Modul fuer Jugend forscht 2026 Physik-Visualisierungsprojekt

This module analyzes how all solar system celestial bodies are affected by
modified gravity (scaling the gravitational constant G). It produces:
- Planetary property comparisons (radius, surface gravity, escape velocity)
- Solar system positioning and orbital compression under stronger gravity
- Mass-radius diagrams, compactness, Jeans mass, and system-scale summaries

Key insight: Increasing G compresses orbits (a proportional to 1/G for fixed
orbital energy), shrinks rocky planets via hydrostatic balance (R proportional
to G^(-1/4)), and eventually drives bodies past stability thresholds into
gravitational collapse.

Author: Jugend forscht 2026 Project
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import math
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass

from .constants import get_constants, PhysicalConstants
from .color_scheme import COLORS, get_sequence, get_planet_colors, PLANET_COLORS


# Output directory for visualizations
VIS_DIR = os.path.join(os.path.dirname(__file__), '..', 'visualizations')


# ============================================================================
# DATA CLASS
# ============================================================================

@dataclass
class CelestialBodyProperties:
    """
    Container for celestial body properties under modified gravity.
    Behaelter fuer Himmelskoerper-Eigenschaften bei veraenderter Gravitation.
    """
    name: str                    # English name
    name_de: str                 # German name
    mass: float                  # [kg]
    radius: float                # [m]
    orbital_radius: float        # [m]
    density: float               # [kg/m^3]
    surface_gravity: float       # [m/s^2]
    escape_velocity: float       # [m/s]
    compactness: float           # R_s / R  (dimensionless)
    is_stable: bool              # Survives modified gravity?
    collapse_G_threshold: float  # G_scale at which body becomes unstable


# ============================================================================
# HELPER: SOLAR SYSTEM BODY LIST
# ============================================================================

def get_solar_system_bodies(
    constants: PhysicalConstants
) -> List[Tuple[str, str, float, float, float, str]]:
    """
    Return a list of all 8 planets with their properties and plot colors.
    Gibt eine Liste aller 8 Planeten mit Eigenschaften und Plotfarben zurueck.

    Returns:
        List of (name, name_de, mass_kg, radius_m, orbital_radius_m, hex_color)
    """
    colors = get_planet_colors()
    bodies = [
        ('Mercury', 'Merkur',   constants.M_mercury, constants.R_mercury, constants.a_mercury, colors[0]),
        ('Venus',   'Venus',    constants.M_venus,   constants.R_venus,   constants.a_venus,   colors[1]),
        ('Earth',   'Erde',     constants.M_earth,   constants.R_earth,   constants.a_earth,   colors[2]),
        ('Mars',    'Mars',     constants.M_mars,    constants.R_mars,    constants.a_mars,     colors[3]),
        ('Jupiter', 'Jupiter',  constants.M_jupiter, constants.R_jupiter, constants.a_jupiter,  colors[4]),
        ('Saturn',  'Saturn',   constants.M_saturn,  constants.R_saturn,  constants.a_saturn,   colors[5]),
        ('Uranus',  'Uranus',   constants.M_uranus,  constants.R_uranus,  constants.a_uranus,   colors[6]),
        ('Neptune', 'Neptun',   constants.M_neptune, constants.R_neptune, constants.a_neptune,  colors[7]),
    ]
    return bodies


# ============================================================================
# CALCULATION FUNCTIONS
# ============================================================================

def body_radius_scaled(R_standard: float, G_scale: float) -> float:
    """
    Scale a rocky-planet radius under modified gravity via hydrostatic balance.
    Skaliert einen Gesteinsplanet-Radius bei veraenderter Gravitation.

    For a self-gravitating body in hydrostatic equilibrium the radius scales as
    R proportional to G^(-1/4) (balancing gravitational pressure against bulk
    modulus / material strength).  We also impose a hard floor at the nuclear
    density minimum radius to avoid unphysical results.

    Args:
        R_standard: Radius at G_scale = 1  [m]
        G_scale:    Multiplicative factor on G

    Returns:
        Scaled radius [m]
    """
    R_scaled = R_standard * G_scale ** (-0.25)
    # Nuclear-density floor: rho_nuc ~ 2.3e17 kg/m^3 => R_min for a given mass
    # Approximate minimum radius ~ 10 km (neutron-star scale)
    R_min = 1.0e4  # 10 km
    return max(R_scaled, R_min)


def body_surface_gravity(M: float, R: float, G: float) -> float:
    """
    Surface gravitational acceleration g = GM / R^2.
    Gravitationsbeschleunigung an der Oberflaeche.

    Args:
        M: Mass [kg]
        R: Radius [m]
        G: Gravitational constant [m^3 kg^-1 s^-2]

    Returns:
        Surface gravity [m/s^2]
    """
    return G * M / R ** 2


def body_escape_velocity(M: float, R: float, G: float) -> float:
    """
    Escape velocity v_esc = sqrt(2GM / R).
    Fluchtgeschwindigkeit.

    Args:
        M: Mass [kg]
        R: Radius [m]
        G: Gravitational constant [m^3 kg^-1 s^-2]

    Returns:
        Escape velocity [m/s]
    """
    return math.sqrt(2.0 * G * M / R)


def body_compactness(M: float, R: float, G: float, c: float) -> float:
    """
    Compactness parameter C = 2GM / (R c^2)  =  R_s / R.
    Kompaktheitsparameter.

    Args:
        M: Mass [kg]
        R: Radius [m]
        G: Gravitational constant [m^3 kg^-1 s^-2]
        c: Speed of light [m/s]

    Returns:
        Compactness (dimensionless)
    """
    return 2.0 * G * M / (R * c ** 2)


def collapse_threshold(
    M: float,
    R_standard: float,
    T_core: float,
    constants: PhysicalConstants
) -> float:
    """
    Find the G_scale at which gravitational pressure exceeds support pressure.
    Findet den G-Skalierungsfaktor, bei dem der Gravitationsdruck den
    Stuetzdruck uebersteigt.

    Uses a binary search over G_scale in [1, 1e12].  Gravitational central
    pressure scales as P_grav ~ G M^2 / R^4 while thermal + Coulomb support
    pressure is roughly constant (depends on material, not G).

    Args:
        M:           Body mass [kg]
        R_standard:  Radius at G_scale = 1 [m]
        T_core:      Core temperature [K]
        constants:   PhysicalConstants instance

    Returns:
        G_scale at collapse threshold (float)
    """
    # Approximate central support pressure (thermal + Coulomb lattice)
    # P_support ~ n k_B T  with n ~ rho / m_p
    rho_0 = M / ((4.0 / 3.0) * math.pi * R_standard ** 3)
    n_0 = rho_0 / constants.m_p
    P_support = n_0 * constants.k_B * T_core

    # Binary search
    lo, hi = 1.0, 1.0e12
    for _ in range(100):
        mid = math.sqrt(lo * hi)  # geometric midpoint for log-scale search
        R_sc = body_radius_scaled(R_standard, mid)
        G_eff = constants.G * mid
        # Gravitational central pressure estimate (virial-like)
        P_grav = (3.0 * G_eff * M ** 2) / (8.0 * math.pi * R_sc ** 4)
        if P_grav > P_support:
            hi = mid
        else:
            lo = mid
        if hi / lo < 1.001:
            break
    return math.sqrt(lo * hi)


def jeans_mass_at_scale(
    G_scale: float,
    T: float,
    rho: float,
    constants: PhysicalConstants
) -> float:
    """
    Jeans mass: minimum cloud mass for gravitational collapse.
    Jeans-Masse: Minimale Wolkenmasse fuer gravitativen Kollaps.

    M_J = (5 k_B T / (G mu m_p))^(3/2) * (3 / (4 pi rho))^(1/2)

    As G increases the Jeans mass drops, so smaller clouds can collapse.

    Args:
        G_scale:   Multiplicative factor on G
        T:         Temperature [K]
        rho:       Density [kg/m^3]
        constants: PhysicalConstants instance

    Returns:
        Jeans mass [kg]
    """
    G_eff = constants.G * G_scale
    mu = 2.0  # mean molecular weight for molecular hydrogen
    term1 = (5.0 * constants.k_B * T / (G_eff * mu * constants.m_p)) ** 1.5
    term2 = math.sqrt(3.0 / (4.0 * math.pi * rho))
    return term1 * term2


def calculate_body_properties(
    name: str,
    name_de: str,
    M: float,
    R: float,
    a_orbit: float,
    G_scale: float,
    constants: PhysicalConstants
) -> CelestialBodyProperties:
    """
    Compute all derived properties for a single celestial body.
    Berechnet alle abgeleiteten Eigenschaften fuer einen Himmelskoerper.

    Args:
        name:      English name
        name_de:   German name
        M:         Mass [kg]
        R:         Standard radius [m]
        a_orbit:   Orbital semi-major axis [m]
        G_scale:   Gravity scaling factor
        constants: PhysicalConstants instance

    Returns:
        CelestialBodyProperties
    """
    G_eff = constants.G * G_scale
    R_sc = body_radius_scaled(R, G_scale)
    volume = (4.0 / 3.0) * math.pi * R_sc ** 3
    density = M / volume
    g = body_surface_gravity(M, R_sc, G_eff)
    v_esc = body_escape_velocity(M, R_sc, G_eff)
    C = body_compactness(M, R_sc, G_eff, constants.c)

    # Core-temperature estimate: scale with planet mass
    T_core_est = constants.T_core_earth * (M / constants.M_earth) ** 0.25
    T_core_est = max(T_core_est, 300.0)  # at least 300 K

    G_thresh = collapse_threshold(M, R, T_core_est, constants)
    is_stable = G_scale < G_thresh

    # Orbital radius also shrinks: a proportional to 1/G for fixed energy
    a_scaled = a_orbit / G_scale

    return CelestialBodyProperties(
        name=name,
        name_de=name_de,
        mass=M,
        radius=R_sc,
        orbital_radius=a_scaled,
        density=density,
        surface_gravity=g,
        escape_velocity=v_esc,
        compactness=C,
        is_stable=is_stable,
        collapse_G_threshold=G_thresh,
    )


# ============================================================================
# PLOT 1: PLANETARY COMPARISON
# ============================================================================

def plot_planetary_comparison(
    constants: Optional[PhysicalConstants] = None,
    language: str = 'en',
    save: bool = True, show: bool = True,
) -> plt.Figure:
    """Four-panel comparison of planetary properties vs G_scale."""
    if constants is None:
        constants = get_constants()
    bodies = get_solar_system_bodies(constants)
    G_scales = np.logspace(0, 6, 200)
    planet_names, planet_names_de, planet_colors = [], [], []
    radii_arrays, grav_arrays, vesc_arrays, thresholds = [], [], [], []
    for name, name_de, M, R, a_orb, color in bodies:
        planet_names.append(name)
        planet_names_de.append(name_de)
        planet_colors.append(color)
        radii = np.array([body_radius_scaled(R, gs) for gs in G_scales])
        radii_arrays.append(radii / constants.R_earth)
        g_arr = np.array([body_surface_gravity(M, body_radius_scaled(R, gs), constants.G * gs) for gs in G_scales])
        grav_arrays.append(g_arr)
        v_arr = np.array([body_escape_velocity(M, body_radius_scaled(R, gs), constants.G * gs) for gs in G_scales])
        vesc_arrays.append(v_arr / 1e3)
        T_c = max(constants.T_core_earth * (M / constants.M_earth) ** 0.25, 300.0)
        thresholds.append(collapse_threshold(M, R, T_c, constants))
    de = language == 'de'
    nd = planet_names_de if de else planet_names
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 38))
    fig.subplots_adjust(hspace=0.55, top=0.96, bottom=0.04)
    st = ('Planeteneigenschaften bei ver\u00e4nderter Gravitation'
         if de else 'Planetary Properties Under Modified Gravity')
    fig.suptitle(st, fontsize=18, fontweight='bold', y=0.99)
    for i, (lbl, arr, col) in enumerate(zip(nd, radii_arrays, planet_colors)):
        ax1.loglog(G_scales, arr, color=col, linewidth=2, label=lbl)
    ax1.axvline(1, color=COLORS['muted'], ls='--', alpha=0.5)
    ax1.set_xlabel('G / G$', fontsize=12)
    ax1.set_ylabel('Radius (Rextglob\\oplus$)', fontsize=12)
    t1 = '1. ' + ('Planetradien vs. Gravitationsskala' if de else 'Planet Radii vs. Gravity Scale')
    ax1.set_title(t1, fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, which='both')
    ax1.tick_params(axis='both', labelsize=11)
    ax1.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, framealpha=0.7)
    for i, (lbl, arr, col) in enumerate(zip(nd, grav_arrays, planet_colors)):
        ax2.loglog(G_scales, arr, color=col, linewidth=2, label=lbl)
    ax2.axvline(1, color=COLORS['muted'], ls='--', alpha=0.5)
    ax2.set_xlabel('G / G$', fontsize=12)
    ax2.set_ylabel('g (m/s$^2$)', fontsize=12)
    t2 = '2. ' + ('Oberfl\u00e4chengravitation' if de else 'Surface Gravity')
    ax2.set_title(t2, fontsize=14, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, which='both')
    ax2.tick_params(axis='both', labelsize=11)
    ax2.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, framealpha=0.7)
    for i, (lbl, arr, col) in enumerate(zip(nd, vesc_arrays, planet_colors)):
        ax3.loglog(G_scales, arr, color=col, linewidth=2, label=lbl)
    ax3.axvline(1, color=COLORS['muted'], ls='--', alpha=0.5)
    ax3.set_xlabel('G / G$', fontsize=12)
    ax3.set_ylabel('vextglob{esc}$ (km/s)', fontsize=12)
    t3 = '3. ' + ('Fluchtgeschwindigkeit' if de else 'Escape Velocity')
    ax3.set_title(t3, fontsize=14, fontweight='bold', pad=15)
    ax3.grid(True, alpha=0.3, which='both')
    ax3.tick_params(axis='both', labelsize=11)
    ax3.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, framealpha=0.7)
    y_pos = np.arange(len(nd))
    ax4.barh(y_pos, thresholds, color=planet_colors, edgecolor='black', alpha=0.85)
    ax4.set_yticks(y_pos)
    ax4.set_yticklabels(nd, fontsize=11)
    ax4.set_xscale('log')
    ax4.set_xlabel('Gextglob{collapse}$ / G$', fontsize=12)
    t4 = '4. ' + ('Kollaps-Schwellenwert je Planet' if de else 'Collapse Threshold per Planet')
    ax4.set_title(t4, fontsize=14, fontweight='bold', pad=15)
    ax4.grid(True, alpha=0.3, axis='x', which='both')
    ax4.tick_params(axis='both', labelsize=11)
    if save:
        os.makedirs(VIS_DIR, exist_ok=True)
        suffix = '_de' if de else ''
        filepath = os.path.join(VIS_DIR, 'planetary_comparison' + suffix + '.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print('Saved: ' + filepath)
    if show:
        plt.show()
    return fig


# ============================================================================
# PLOT 2: SOLAR SYSTEM POSITIONING
# ============================================================================

def plot_solar_system_positioning(
    constants: Optional[PhysicalConstants] = None,
    language: str = 'en',
    save: bool = True, show: bool = True,
) -> plt.Figure:
    """Four-panel solar system positioning under modified G."""
    if constants is None:
        constants = get_constants()
    bodies = get_solar_system_bodies(constants)
    de = language == 'de'
    planet_names = [b[0] for b in bodies]
    planet_names_de = [b[1] for b in bodies]
    masses = np.array([b[2] for b in bodies])
    radii_std = np.array([b[3] for b in bodies])
    orbits_std = np.array([b[4] for b in bodies])
    colors = [b[5] for b in bodies]
    nd = planet_names_de if de else planet_names
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 38))
    fig.subplots_adjust(hspace=0.55, top=0.96, bottom=0.04)
    st = 'Sonnensystem-Positionierung' if de else 'Solar System Positioning'
    fig.suptitle(st, fontsize=18, fontweight='bold', y=0.99)
    angles = np.linspace(0, 2 * np.pi, len(bodies), endpoint=False)
    def _scatter_ss(ax, G_scale, title_str):
        orbits_au = orbits_std / (G_scale * constants.AU)
        lmin, lmax = np.log10(radii_std.min()), np.log10(radii_std.max())
        lr = lmax - lmin if lmax > lmin else 1.0
        sizes = 30 + 120 * (np.log10(radii_std) - lmin) / lr
        xs = orbits_au * np.cos(angles)
        ys = orbits_au * np.sin(angles)
        ax.scatter([0], [0], s=200, color=COLORS['sun'], marker='*', edgecolors='black', zorder=5, label='Sonne' if de else 'Sun')
        for i in range(len(bodies)):
            ax.scatter(xs[i], ys[i], s=sizes[i], color=colors[i], edgecolors='black', zorder=4, label=nd[i])
            theta = np.linspace(0, 2 * np.pi, 200)
            ax.plot(orbits_au[i] * np.cos(theta), orbits_au[i] * np.sin(theta), color=colors[i], alpha=0.25, linewidth=0.8)
        ax.set_xlabel('x (AU)', fontsize=12)
        ax.set_ylabel('y (AU)', fontsize=12)
        ax.set_title(title_str, fontsize=14, fontweight='bold', pad=15)
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=11)
    _scatter_ss(ax1, 1.0, '1. ' + ('Sonnensystem bei G = G$' if de else 'Solar System at G = G$'))
    ax1.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5, framealpha=0.7)
    _scatter_ss(ax2, 10.0, '2. ' + ('Sonnensystem bei G = 10 G$' if de else 'Solar System at G = 10 G$'))
    ax2.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5, framealpha=0.7)
    G_sc = np.logspace(0, 4, 200)
    for i in range(len(bodies)):
        ax3.loglog(G_sc, orbits_std[i] / G_sc / constants.AU, color=colors[i], linewidth=2, label=nd[i])
    ax3.axvline(1, color=COLORS['muted'], ls='--', alpha=0.5)
    ax3.set_xlabel('G / G$', fontsize=12)
    ax3.set_ylabel('a (AU)', fontsize=12)
    t3 = '3. ' + ('Orbitalradius-Skalierung (a $\\propto$ 1/G)' if de else 'Orbital Radius Scaling (a $\\propto$ 1/G)')
    ax3.set_title(t3, fontsize=14, fontweight='bold', pad=15)
    ax3.grid(True, alpha=0.3, which='both')
    ax3.tick_params(axis='both', labelsize=11)
    ax3.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, framealpha=0.7)
    def hill_r(a, Mp, Ms):
        return a * (Mp / (3.0 * Ms)) ** (1.0 / 3.0)
    h1 = np.array([hill_r(orbits_std[i], masses[i], constants.M_sun) for i in range(len(bodies))]) / constants.AU
    h100 = np.array([hill_r(orbits_std[i] / 100.0, masses[i], constants.M_sun) for i in range(len(bodies))]) / constants.AU
    yp = np.arange(len(bodies))
    bh = 0.35
    ax4.barh(yp - bh/2, h1, height=bh, color=COLORS['standard'], edgecolor='black', alpha=0.8, label='G = G$')
    ax4.barh(yp + bh/2, h100, height=bh, color=COLORS['scaled'], edgecolor='black', alpha=0.8, label='G = 100 G$')
    ax4.set_yticks(yp)
    ax4.set_yticklabels(nd, fontsize=11)
    ax4.set_xscale('log')
    xl4 = 'Hill-Sph\u00e4re Radius (AU)' if de else 'Hill Sphere Radius (AU)'
    ax4.set_xlabel(xl4, fontsize=12)
    t4 = '4. ' + ('Hill-Sph\u00e4ren: G$ vs. 100 G$' if de else 'Hill Spheres: G$ vs. 100 G$')
    ax4.set_title(t4, fontsize=14, fontweight='bold', pad=15)
    ax4.grid(True, alpha=0.3, axis='x', which='both')
    ax4.tick_params(axis='both', labelsize=11)
    ax4.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, framealpha=0.7)
    if save:
        os.makedirs(VIS_DIR, exist_ok=True)
        suffix = '_de' if de else ''
        fp = os.path.join(VIS_DIR, 'solar_system_positioning' + suffix + '.png')
        fig.savefig(fp, dpi=150, bbox_inches='tight')
        print('Saved: ' + fp)
    if show:
        plt.show()
    return fig


# ============================================================================
# PLOT 3: CELESTIAL SUMMARY
# ============================================================================

def plot_celestial_summary(
    constants: Optional[PhysicalConstants] = None,
    language: str = 'en',
    save: bool = True, show: bool = True,
) -> plt.Figure:
    """Four-panel celestial bodies summary."""
    if constants is None:
        constants = get_constants()
    bodies = get_solar_system_bodies(constants)
    de = language == 'de'
    planet_names = [b[0] for b in bodies]
    planet_names_de = [b[1] for b in bodies]
    masses = np.array([b[2] for b in bodies])
    radii_std = np.array([b[3] for b in bodies])
    orbits_std = np.array([b[4] for b in bodies])
    colors = [b[5] for b in bodies]
    nd = planet_names_de if de else planet_names
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 38))
    fig.subplots_adjust(hspace=0.55, top=0.96, bottom=0.04)
    st = 'Himmelskörper-Zusammenfassung' if de else 'Celestial Bodies Summary'
    fig.suptitle(st, fontsize=18, fontweight='bold', y=0.99)
    radii_10 = np.array([body_radius_scaled(R, 10.0) for R in radii_std])
    rho_lines = [1e3, 5e3, 1e4]
    rho_labels = ['1000', '5000', '10000']
    M_range = np.logspace(np.log10(masses.min()) - 0.5, np.log10(masses.max()) + 0.5, 100)
    for rv, rl in zip(rho_lines, rho_labels):
        R_line = ((3.0 * M_range) / (4.0 * np.pi * rv)) ** (1.0 / 3.0)
        ax1.loglog(M_range, R_line / constants.R_earth, '--', color=COLORS['muted'], alpha=0.5, linewidth=1)
    for i in range(len(bodies)):
        ax1.scatter(masses[i], radii_std[i] / constants.R_earth, s=100, color=colors[i], marker='o', edgecolors='black', zorder=4, label=nd[i] + ' (G$)')
        ax1.scatter(masses[i], radii_10[i] / constants.R_earth, s=80, color=colors[i], marker='^', edgecolors='black', zorder=4, alpha=0.7, label=nd[i] + ' (10G$)')
    ax1.set_xlabel('Masse (kg)' if de else 'Mass (kg)', fontsize=12)
    ax1.set_ylabel('Radius (Rextglob\\oplus$)', fontsize=12)
    ax1.set_title('1. ' + ('Masse-Radius-Diagramm' if de else 'Mass-Radius Diagram'), fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, which='both')
    ax1.tick_params(axis='both', labelsize=11)
    ax1.legend(fontsize=9, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, framealpha=0.7)
    c1 = np.array([body_compactness(masses[i], radii_std[i], constants.G, constants.c) for i in range(len(bodies))])
    c1e6 = np.array([body_compactness(masses[i], body_radius_scaled(radii_std[i], 1e6), constants.G * 1e6, constants.c) for i in range(len(bodies))])
    yp = np.arange(len(bodies))
    bh = 0.35
    ax2.barh(yp - bh/2, c1, height=bh, color=COLORS['standard'], edgecolor='black', alpha=0.8, label='G = G$')
    ax2.barh(yp + bh/2, c1e6, height=bh, color=COLORS['scaled'], edgecolor='black', alpha=0.8, label='G = 10$^6$ G$')
    ax2.set_yticks(yp)
    ax2.set_yticklabels(nd, fontsize=11)
    ax2.set_xscale('log')
    ax2.set_xlabel('Kompaktheit R$ / R' if de else 'Compactness R$ / R', fontsize=12)
    ax2.set_title('2. ' + ('Kompaktheit bei G$ vs. 10$^6$ G$' if de else 'Compactness at G$ vs. 10$^6$ G$'), fontsize=14, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3, axis='x', which='both')
    ax2.tick_params(axis='both', labelsize=11)
    ax2.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, framealpha=0.7)
    G_sc = np.logspace(0, 8, 300)
    jm = np.array([jeans_mass_at_scale(gs, 20.0, 1e-18, constants) for gs in G_sc])
    ax3.loglog(G_sc, jm / constants.M_sun, color=COLORS['primary_blue'], linewidth=2.5, label='M$ (T=20 K)')
    for i in range(len(bodies)):
        ax3.axhline(masses[i] / constants.M_sun, color=colors[i], ls=':', alpha=0.6, linewidth=1.2)
        ax3.text(G_sc[-1] * 1.1, masses[i] / constants.M_sun, nd[i], fontsize=9, va='center', color=colors[i])
    ax3.set_xlabel('G / G$', fontsize=12)
    ax3.set_ylabel('Masse (Mextglob\\odot$)' if de else 'Mass (Mextglob\\odot$)', fontsize=12)
    ax3.set_title('3. ' + ('Jeans-Masse vs. Gravitationsskala' if de else 'Jeans Mass vs. Gravity Scale'), fontsize=14, fontweight='bold', pad=15)
    ax3.grid(True, alpha=0.3, which='both')
    ax3.tick_params(axis='both', labelsize=11)
    ax3.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, framealpha=0.7)
    ruler_scales = [1, 10, 100]
    ruler_labels = ['G = G$', 'G = 10 G$', 'G = 100 G$']
    ruler_y = [3, 2, 1]
    for gs, lbl, yy in zip(ruler_scales, ruler_labels, ruler_y):
        a_au = orbits_std / (gs * constants.AU)
        ax4.hlines(yy, 0, a_au.max() * 1.05, colors=COLORS['muted'], linewidth=1, alpha=0.4)
        for i in range(len(bodies)):
            ax4.plot(a_au[i], yy, 'o', color=colors[i], markersize=8, markeredgecolor='black', zorder=4)
            off = 0.15 if i % 2 == 0 else -0.25
            ax4.annotate(nd[i], (a_au[i], yy), textcoords='offset points', xytext=(0, 10 + off * 30), fontsize=8, ha='center', color=colors[i], rotation=45)
        ax4.text(-0.5, yy, lbl, fontsize=11, va='center', ha='right', fontweight='bold')
    ax4.set_xlabel('Entfernung (AU)' if de else 'Distance (AU)', fontsize=12)
    ax4.set_xlim(-3, orbits_std.max() / constants.AU * 1.1)
    ax4.set_ylim(0.3, 3.7)
    ax4.set_yticks([])
    ax4.set_title('4. ' + ('Sonnensystem-Skalenvergleich' if de else 'Solar System Scale Comparison'), fontsize=14, fontweight='bold', pad=15)
    ax4.grid(True, alpha=0.3, axis='x')
    ax4.tick_params(axis='both', labelsize=11)
    if save:
        os.makedirs(VIS_DIR, exist_ok=True)
        suffix = '_de' if de else ''
        fp = os.path.join(VIS_DIR, 'celestial_summary' + suffix + '.png')
        fig.savefig(fp, dpi=150, bbox_inches='tight')
        print('Saved: ' + fp)
    if show:
        plt.show()
    return fig


# ============================================================================
# GENERATE ALL & VERIFY
# ============================================================================

def generate_all_celestial_plots(language: str = 'en', show: bool = False) -> List[plt.Figure]:
    """Generate all celestial-body visualizations."""
    figures = []
    print('Generating celestial body visualizations...')
    print('=' * 50)
    print('1. Planetary comparison...')
    figures.append(plot_planetary_comparison(language=language, show=show))
    print('2. Solar system positioning...')
    figures.append(plot_solar_system_positioning(language=language, show=show))
    print('3. Celestial summary...')
    figures.append(plot_celestial_summary(language=language, show=show))
    print('=' * 50)
    print(f'Generated {len(figures)} visualizations in {VIS_DIR}')
    return figures


def verify_celestial_physics() -> None:
    """Verify celestial-body calculations against known values."""
    print('=' * 70)
    print('CELESTIAL BODIES PHYSICS VERIFICATION')
    print('=' * 70)
    c = get_constants()
    print('\n1. EARTH SURFACE GRAVITY')
    print('-' * 50)
    g_e = body_surface_gravity(c.M_earth, c.R_earth, c.G)
    print(f'   g_Earth = {g_e:.4f} m/s^2')
    print(f'   Expected: ~9.81 (CHECK: {'PASS' if 9.7 < g_e < 9.9 else 'FAIL'})')
    print('\n2. JUPITER SURFACE GRAVITY')
    print('-' * 50)
    g_j = body_surface_gravity(c.M_jupiter, c.R_jupiter, c.G)
    print(f'   g_Jupiter = {g_j:.2f} m/s^2')
    print(f'   Expected: ~24.8 (CHECK: {'PASS' if 23 < g_j < 27 else 'FAIL'})')
    print('\n3. EARTH ESCAPE VELOCITY')
    print('-' * 50)
    v_e = body_escape_velocity(c.M_earth, c.R_earth, c.G)
    print(f'   v_esc = {v_e:.0f} m/s = {v_e/1e3:.2f} km/s')
    print(f'   Expected: ~11.2 km/s (CHECK: {'PASS' if 11.0e3 < v_e < 11.4e3 else 'FAIL'})')
    print('\n4. JUPITER ESCAPE VELOCITY')
    print('-' * 50)
    v_j = body_escape_velocity(c.M_jupiter, c.R_jupiter, c.G)
    print(f'   v_esc = {v_j:.0f} m/s = {v_j/1e3:.2f} km/s')
    print(f'   Expected: ~59.5 km/s (CHECK: {'PASS' if 58e3 < v_j < 62e3 else 'FAIL'})')
    print('\n5. EARTH COMPACTNESS')
    print('-' * 50)
    ce = body_compactness(c.M_earth, c.R_earth, c.G, c.c)
    print(f'   C = R_s/R = {ce:.4e}')
    print(f'   Expected: ~1.4e-9 (CHECK: {'PASS' if 1e-10 < ce < 1e-8 else 'FAIL'})')
    print('\n6. JEANS MASS (T=20 K)')
    print('-' * 50)
    mj = jeans_mass_at_scale(1.0, 20.0, 1e-18, c)
    mjs = mj / c.M_sun
    print(f'   M_J = {mj:.3e} kg = {mjs:.2f} M_sun')
    print(f'   Expected: ~few M_sun (CHECK: {'PASS' if 0.1 < mjs < 200 else 'FAIL'})')
    print('\n7. RADIUS SCALING (G_scale=10)')
    print('-' * 50)
    rs = body_radius_scaled(c.R_earth, 10.0)
    ratio = rs / c.R_earth
    expected = 10.0 ** (-0.25)
    print(f'   R(10G)/R(G) = {ratio:.4f}')
    print(f'   Expected: {expected:.4f} (CHECK: {'PASS' if abs(ratio - expected) < 0.001 else 'FAIL'})')
    print('\n' + '=' * 70)
    print('VERIFICATION COMPLETE')
    print('=' * 70)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print('=' * 60)
    print('Celestial Bodies Module - Jugend forscht 2026')
    print('=' * 60)
    verify_celestial_physics()
    print('\n')
    generate_all_celestial_plots(language='en', show=False)
    print('\nDone! Check the visualizations folder for output.')