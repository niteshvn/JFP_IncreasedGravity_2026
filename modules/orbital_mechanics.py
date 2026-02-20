"""
Orbital Mechanics Module for Jugend forscht 2026 Physics Visualization Project
Bahnmechanik-Modul fuer Jugend forscht 2026 Physik-Visualisierungsprojekt

This module visualizes how modified gravity (scaled G) affects orbital mechanics
throughout the solar system, including:
- Moon orbit around Earth and tidal effects
- Earth orbit around the Sun
- Solar system orbital summary for all planets
- Roche limits, Hill spheres, and GR precession

Key insight: When G increases, orbits shrink (for conserved angular momentum
a proportional to 1/G), tidal forces grow, and the Roche limit moves outward.
At sufficiently large G, moons are torn apart, planets spiral inward, and
the solar system becomes fundamentally unstable.

Author: Jugend forscht 2026 Project
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Optional, List
from dataclasses import dataclass

from .constants import get_constants, PhysicalConstants
from .color_scheme import COLORS, get_sequence, get_planet_colors


# Output directory for visualizations
VIS_DIR = os.path.join(os.path.dirname(__file__), '..', 'visualizations')


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class OrbitalProperties:
    """
    Container for orbital properties of a body.
    Behaelter fuer Bahneigenschaften eines Koerpers.
    """
    semi_major_axis: float    # [m]
    period: float             # [s]
    velocity: float           # [m/s]
    tidal_force: float        # [N] (for Moon on Earth)
    roche_limit: float        # [m]
    hill_sphere: float        # [m]
    binding_energy: float     # [J]
    precession_rate: float    # [rad/orbit] (GR precession)


# =============================================================================
# CALCULATION FUNCTIONS
# =============================================================================

def orbital_period(a: float, M_central: float, G: float) -> float:
    """Calculate orbital period: T = 2*pi*sqrt(a^3/(G*M)). Returns seconds."""
    return 2.0 * np.pi * np.sqrt(a**3 / (G * M_central))


def orbital_velocity(a: float, M_central: float, G: float) -> float:
    """Calculate circular orbital velocity: v = sqrt(G*M/a). Returns m/s."""
    return np.sqrt(G * M_central / a)


def tidal_force(M_source: float, R_body: float, d: float, G: float) -> float:
    """Calculate differential tidal force: F_tidal = 2*G*M*R/d^3. Returns N/kg."""
    return 2.0 * G * M_source * R_body / d**3


def roche_limit(R_primary: float, rho_primary: float, rho_secondary: float) -> float:
    """Calculate Roche limit: d = 2.44*R_p*(rho_p/rho_s)^(1/3). Returns m."""
    return 2.44 * R_primary * (rho_primary / rho_secondary) ** (1.0 / 3.0)


def hill_sphere(a: float, m_body: float, M_central: float) -> float:
    """Calculate Hill sphere radius: r_H = a*(m/(3*M))^(1/3). Returns m."""
    return a * (m_body / (3.0 * M_central)) ** (1.0 / 3.0)


def gr_precession(a: float, M_central: float, e: float, G: float, c: float) -> float:
    """Calculate GR precession: dphi = 6*pi*G*M/(a*c^2*(1-e^2)). Returns rad/orbit."""
    return 6.0 * np.pi * G * M_central / (a * c**2 * (1.0 - e**2))


def orbital_energy(m: float, M: float, a: float, G: float) -> float:
    """Calculate orbital binding energy: E = -G*M*m/(2*a). Returns J."""
    return -G * M * m / (2.0 * a)


def scaled_orbit_radius(a_standard: float, G_scale: float) -> float:
    """Scaled orbit radius for conserved angular momentum: a_new = a_old / G_scale."""
    return a_standard / G_scale


# =============================================================================
# COMPOSITE CALCULATIONS
# =============================================================================

def calculate_moon_orbit(G_scale: float,
                         constants: Optional[PhysicalConstants] = None) -> OrbitalProperties:
    """Calculate Moon orbital properties for a given G scaling."""
    if constants is None:
        constants = get_constants()
    G_eff = constants.G * G_scale
    a_moon = scaled_orbit_radius(constants.d_earth_moon, G_scale)
    T_moon = orbital_period(a_moon, constants.M_earth, G_eff)
    v_moon = orbital_velocity(a_moon, constants.M_earth, G_eff)
    F_tidal = tidal_force(constants.M_moon, constants.R_earth, a_moon, G_eff)
    rho_earth = constants.M_earth / ((4.0 / 3.0) * np.pi * constants.R_earth**3)
    rho_moon = constants.M_moon / ((4.0 / 3.0) * np.pi * constants.R_moon**3)
    d_roche = roche_limit(constants.R_earth, rho_earth, rho_moon)
    a_earth = scaled_orbit_radius(constants.a_earth, G_scale)
    r_hill = hill_sphere(a_earth, constants.M_earth, constants.M_sun)
    E_bind = orbital_energy(constants.M_moon, constants.M_earth, a_moon, G_eff)
    e_moon = 0.0549
    prec = gr_precession(a_moon, constants.M_earth, e_moon, G_eff, constants.c)
    return OrbitalProperties(
        semi_major_axis=a_moon, period=T_moon, velocity=v_moon,
        tidal_force=F_tidal, roche_limit=d_roche, hill_sphere=r_hill,
        binding_energy=E_bind, precession_rate=prec)


def calculate_earth_orbit(G_scale: float,
                          constants: Optional[PhysicalConstants] = None) -> OrbitalProperties:
    """Calculate Earth orbital properties for a given G scaling."""
    if constants is None:
        constants = get_constants()
    G_eff = constants.G * G_scale
    a_earth = scaled_orbit_radius(constants.a_earth, G_scale)
    T_earth = orbital_period(a_earth, constants.M_sun, G_eff)
    v_earth = orbital_velocity(a_earth, constants.M_sun, G_eff)
    F_tidal = tidal_force(constants.M_sun, constants.R_earth, a_earth, G_eff)
    rho_sun = constants.M_sun / ((4.0 / 3.0) * np.pi * constants.R_sun**3)
    rho_earth = constants.M_earth / ((4.0 / 3.0) * np.pi * constants.R_earth**3)
    d_roche = roche_limit(constants.R_sun, rho_sun, rho_earth)
    r_hill = hill_sphere(a_earth, constants.M_earth, constants.M_sun)
    E_bind = orbital_energy(constants.M_earth, constants.M_sun, a_earth, G_eff)
    e_earth = 0.0167
    prec = gr_precession(a_earth, constants.M_sun, e_earth, G_eff, constants.c)
    return OrbitalProperties(
        semi_major_axis=a_earth, period=T_earth, velocity=v_earth,
        tidal_force=F_tidal, roche_limit=d_roche, hill_sphere=r_hill,
        binding_energy=E_bind, precession_rate=prec)


# =============================================================================
# PLOT FUNCTIONS
# =============================================================================

def plot_moon_tidal_effects(
    constants: Optional[PhysicalConstants] = None,
    language: str = 'en',
    save: bool = True,
    show: bool = True
) -> plt.Figure:
    """Plot Moon orbital and tidal effects under modified gravity."""
    if constants is None:
        constants = get_constants()
    G_scales = np.logspace(0, 3, 500)
    moon_radii = np.array([scaled_orbit_radius(constants.d_earth_moon, g) for g in G_scales])
    G_std = constants.G
    tidal_forces = np.array([
        tidal_force(constants.M_moon, constants.R_earth,
                    scaled_orbit_radius(constants.d_earth_moon, g), G_std * g)
        for g in G_scales])
    rho_earth = constants.M_earth / ((4.0 / 3.0) * np.pi * constants.R_earth**3)
    rho_moon = constants.M_moon / ((4.0 / 3.0) * np.pi * constants.R_moon**3)
    d_roche_val = roche_limit(constants.R_earth, rho_earth, rho_moon)
    roche_limits = np.full_like(G_scales, d_roche_val)
    periods = np.array([
        orbital_period(scaled_orbit_radius(constants.d_earth_moon, g),
                       constants.M_earth, G_std * g)
        for g in G_scales])
    R_earth_val = constants.R_earth
    moon_radii_RE = moon_radii / R_earth_val
    roche_limits_RE = roche_limits / R_earth_val
    periods_days = periods / 86400.0
    std_moon_RE = constants.d_earth_moon / R_earth_val
    std_roche_RE = d_roche_val / R_earth_val
    std_period_days = orbital_period(constants.d_earth_moon, constants.M_earth, G_std) / 86400.0
    std_tidal = tidal_force(constants.M_moon, constants.R_earth, constants.d_earth_moon, G_std)
    cross_mask = moon_radii <= d_roche_val
    G_cross = G_scales[np.argmax(cross_mask)] if np.any(cross_mask) else None
    suffix = '_de' if language == 'de' else ''
    if language == 'de':
        suptitle = 'Mond und Gezeiteneffekte bei veraenderter Gravitation'
        titles = ['1. Mond-Bahnradius vs. G-Skalierung', '2. Gezeitenkraft vs. G-Skalierung',
                  '3. Roche-Grenze vs. Mond-Abstand', '4. Mond-Umlaufzeit vs. G-Skalierung']
        xlabels = ['G-Skalierungsfaktor'] * 4
        ylabels = ['Bahnradius (Erdradien)', 'Gezeitenkraft (N/kg)',
                   'Abstand (Erdradien)', 'Umlaufzeit (Tage)']
        moon_label, roche_label, std_label = 'Mond-Abstand', 'Roche-Grenze', 'Standardwert'
        danger_label = 'Gefahrenzone (Mond innerhalb Roche-Grenze)'
        tidal_label, period_label = 'Gezeitenkraft', 'Umlaufzeit'
        cross_label = 'Mond zerstoert bei G = {:.0f}x'.format(G_cross) if G_cross else None
    else:
        suptitle = 'Moon and Tidal Effects Under Modified Gravity'
        titles = ['1. Moon Orbital Radius vs. G Scaling', '2. Tidal Force vs. G Scaling',
                  '3. Roche Limit vs. Moon Distance', '4. Moon Orbital Period vs. G Scaling']
        xlabels = ['G Scaling Factor'] * 4
        ylabels = ['Orbital Radius (Earth radii)', 'Tidal Force (N/kg)',
                   'Distance (Earth radii)', 'Orbital Period (days)']
        moon_label, roche_label, std_label = 'Moon distance', 'Roche limit', 'Standard value'
        danger_label = 'Danger zone (Moon inside Roche limit)'
        tidal_label, period_label = 'Tidal force', 'Orbital period'
        cross_label = 'Moon destroyed at G = {:.0f}x'.format(G_cross) if G_cross else None
    fig, axes = plt.subplots(4, 1, figsize=(12, 32))
    fig.subplots_adjust(hspace=0.5, top=0.95, bottom=0.04)
    fig.suptitle(suptitle, fontsize=16, fontweight='bold', y=0.98)

    # Plot 1: Moon orbital radius
    ax = axes[0]
    ax.semilogx(G_scales, moon_radii_RE, '-', color=COLORS['moon'], linewidth=2.5, label=moon_label)
    ax.axhline(y=std_moon_RE, color=COLORS['standard'], linestyle='--', linewidth=1.5,
               alpha=0.7, label='{} ({:.1f} R_E)'.format(std_label, std_moon_RE))
    ax.axhline(y=std_roche_RE, color=COLORS['relativistic'], linestyle=':', linewidth=1.5,
               alpha=0.7, label='{} ({:.1f} R_E)'.format(roche_label, std_roche_RE))
    ax.set_xlabel(xlabels[0], fontsize=12)
    ax.set_ylabel(ylabels[0], fontsize=12)
    ax.set_title(titles[0], fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=11)
    ax.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, framealpha=0.7)

    # Plot 2: Tidal force
    ax = axes[1]
    ax.loglog(G_scales, tidal_forces, '-', color=COLORS['primary_blue'], linewidth=2.5, label=tidal_label)
    ax.axhline(y=std_tidal, color=COLORS['standard'], linestyle='--', linewidth=1.5,
               alpha=0.7, label='{} ({:.2e} N/kg)'.format(std_label, std_tidal))
    ax.set_xlabel(xlabels[1], fontsize=12)
    ax.set_ylabel(ylabels[1], fontsize=12)
    ax.set_title(titles[1], fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=11)
    ax.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, framealpha=0.7)

    # Plot 3: Roche limit vs Moon distance
    ax = axes[2]
    ax.semilogx(G_scales, moon_radii_RE, '-', color=COLORS['moon'], linewidth=2.5, label=moon_label)
    ax.semilogx(G_scales, roche_limits_RE, '-', color=COLORS['relativistic'], linewidth=2.5, label=roche_label)
    ax.fill_between(G_scales, 0, roche_limits_RE,
                    where=(moon_radii_RE <= roche_limits_RE),
                    color=COLORS['relativistic'], alpha=0.15, label=danger_label)
    if G_cross is not None:
        ax.axvline(x=G_cross, color=COLORS['highlight'], linestyle='--',
                   linewidth=1.5, alpha=0.8, label=cross_label)
    ax.set_xlabel(xlabels[2], fontsize=12)
    ax.set_ylabel(ylabels[2], fontsize=12)
    ax.set_title(titles[2], fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=11)
    ax.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, framealpha=0.7)

    # Plot 4: Moon orbital period
    ax = axes[3]
    ax.semilogx(G_scales, periods_days, '-', color=COLORS['primary_amber'], linewidth=2.5, label=period_label)
    ax.axhline(y=std_period_days, color=COLORS['standard'], linestyle='--', linewidth=1.5,
               alpha=0.7, label='{} ({:.1f} d)'.format(std_label, std_period_days))
    ax.set_xlabel(xlabels[3], fontsize=12)
    ax.set_ylabel(ylabels[3], fontsize=12)
    ax.set_title(titles[3], fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=11)
    ax.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, framealpha=0.7)

    if save:
        os.makedirs(VIS_DIR, exist_ok=True)
        fn = 'moon_tidal_effects{}.png'.format(suffix)
        fp = os.path.join(VIS_DIR, fn)
        fig.savefig(fp, dpi=150, bbox_inches='tight')
        print('Saved: {}'.format(fp))
    if show:
        plt.show()
    return fig


def plot_earth_orbit(
    constants: Optional[PhysicalConstants] = None,
    language: str = 'en',
    save: bool = True,
    show: bool = True
) -> plt.Figure:
    """Plot Earth orbital properties under modified gravity."""
    if constants is None:
        constants = get_constants()
    G_scales = np.logspace(0, 3, 500)
    G_std = constants.G
    AU = constants.AU
    semi_major_axes = np.array([scaled_orbit_radius(constants.a_earth, g) for g in G_scales])
    semi_major_AU = semi_major_axes / AU
    periods = np.array([orbital_period(scaled_orbit_radius(constants.a_earth, g), constants.M_sun, G_std * g) for g in G_scales])
    periods_years = periods / (365.25 * 86400.0)
    velocities = np.array([orbital_velocity(scaled_orbit_radius(constants.a_earth, g), constants.M_sun, G_std * g) for g in G_scales])
    velocities_kms = velocities / 1000.0
    e_earth = 0.0167
    precessions_rad = np.array([gr_precession(scaled_orbit_radius(constants.a_earth, g), constants.M_sun, e_earth, G_std * g, constants.c) for g in G_scales])
    precessions_arcsec_orbit = precessions_rad * (180.0 / np.pi) * 3600.0
    orbits_per_century = 100.0 / periods_years
    precessions_arcsec_century = precessions_arcsec_orbit * orbits_per_century
    std_AU = 1.0
    std_period_yr = orbital_period(constants.a_earth, constants.M_sun, G_std) / (365.25 * 86400.0)
    std_vel_kms = orbital_velocity(constants.a_earth, constants.M_sun, G_std) / 1000.0
    std_prec_rad = gr_precession(constants.a_earth, constants.M_sun, e_earth, G_std, constants.c)
    std_prec_arcsec = std_prec_rad * (180.0 / np.pi) * 3600.0 * (100.0 / std_period_yr)
    mercury_prec = 43.0
    suffix = '_de' if language == 'de' else ''
    if language == 'de':
        suptitle = 'Erdumlaufbahn bei veraenderter Gravitation'
        titles = ['1. Grosse Halbachse vs. G-Skalierung', '2. Umlaufzeit vs. G-Skalierung',
                  '3. Bahngeschwindigkeit vs. G-Skalierung', '4. GR-Praezession vs. G-Skalierung']
        xlabels = ['G-Skalierungsfaktor'] * 4
        ylabels = ['Grosse Halbachse (AU)', 'Umlaufzeit (Jahre)',
                   'Bahngeschwindigkeit (km/s)', 'GR-Praezession (Bogensekunden/Jahrhundert)']
        sma_label, period_label = 'Grosse Halbachse', 'Umlaufzeit'
        vel_label, prec_label = 'Bahngeschwindigkeit', 'Erd-Praezession'
        mercury_label = 'Merkur-Referenz ({:.0f}"/Jhd.)'.format(mercury_prec)
        std_label = 'Standardwert'
    else:
        suptitle = 'Earth Orbit Under Modified Gravity'
        titles = ['1. Semi-Major Axis vs. G Scaling', '2. Orbital Period vs. G Scaling',
                  '3. Orbital Velocity vs. G Scaling', '4. GR Precession vs. G Scaling']
        xlabels = ['G Scaling Factor'] * 4
        ylabels = ['Semi-Major Axis (AU)', 'Orbital Period (years)',
                   'Orbital Velocity (km/s)', 'GR Precession (arcsec/century)']
        sma_label, period_label = 'Semi-major axis', 'Orbital period'
        vel_label, prec_label = 'Orbital velocity', 'Earth precession'
        mercury_label = 'Mercury reference ({:.0f}"/century)'.format(mercury_prec)
        std_label = 'Standard value'
    fig, axes = plt.subplots(4, 1, figsize=(12, 32))
    fig.subplots_adjust(hspace=0.5, top=0.95, bottom=0.04)
    fig.suptitle(suptitle, fontsize=16, fontweight='bold', y=0.98)

    # Plot 1: Semi-major axis
    ax = axes[0]
    ax.loglog(G_scales, semi_major_AU, '-', color=COLORS['earth'], linewidth=2.5, label=sma_label)
    ax.axhline(y=std_AU, color=COLORS['standard'], linestyle='--', linewidth=1.5,
               alpha=0.7, label='{} ({:.1f} AU)'.format(std_label, std_AU))
    ax.set_xlabel(xlabels[0], fontsize=12)
    ax.set_ylabel(ylabels[0], fontsize=12)
    ax.set_title(titles[0], fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=11)
    ax.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, framealpha=0.7)

    # Plot 2: Orbital period
    ax = axes[1]
    ax.loglog(G_scales, periods_years, '-', color=COLORS['primary_blue'], linewidth=2.5, label=period_label)
    ax.axhline(y=std_period_yr, color=COLORS['standard'], linestyle='--', linewidth=1.5,
               alpha=0.7, label='{} ({:.2f} yr)'.format(std_label, std_period_yr))
    ax.set_xlabel(xlabels[1], fontsize=12)
    ax.set_ylabel(ylabels[1], fontsize=12)
    ax.set_title(titles[1], fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=11)
    ax.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, framealpha=0.7)

    # Plot 3: Orbital velocity
    ax = axes[2]
    ax.loglog(G_scales, velocities_kms, '-', color=COLORS['primary_amber'], linewidth=2.5, label=vel_label)
    ax.axhline(y=std_vel_kms, color=COLORS['standard'], linestyle='--', linewidth=1.5,
               alpha=0.7, label='{} ({:.1f} km/s)'.format(std_label, std_vel_kms))
    ax.set_xlabel(xlabels[2], fontsize=12)
    ax.set_ylabel(ylabels[2], fontsize=12)
    ax.set_title(titles[2], fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=11)
    ax.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, framealpha=0.7)

    # Plot 4: GR precession
    ax = axes[3]
    ax.loglog(G_scales, precessions_arcsec_century, '-', color=COLORS['relativistic'], linewidth=2.5, label=prec_label)
    ax.axhline(y=mercury_prec, color=COLORS['highlight'], linestyle='--', linewidth=1.5,
               alpha=0.7, label=mercury_label)
    ax.axhline(y=std_prec_arcsec, color=COLORS['standard'], linestyle=':', linewidth=1.5,
               alpha=0.7, label='{} ({:.3f}"/century)'.format(std_label, std_prec_arcsec))
    ax.set_xlabel(xlabels[3], fontsize=12)
    ax.set_ylabel(ylabels[3], fontsize=12)
    ax.set_title(titles[3], fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=11)
    ax.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, framealpha=0.7)

    if save:
        os.makedirs(VIS_DIR, exist_ok=True)
        fn = 'earth_orbit{}.png'.format(suffix)
        fp = os.path.join(VIS_DIR, fn)
        fig.savefig(fp, dpi=150, bbox_inches='tight')
        print('Saved: {}'.format(fp))
    if show:
        plt.show()
    return fig


def plot_orbital_summary(
    constants: Optional[PhysicalConstants] = None,
    language: str = 'en',
    save: bool = True,
    show: bool = True
) -> plt.Figure:
    """Plot solar system orbital summary under modified gravity."""
    if constants is None:
        constants = get_constants()
    G_std = constants.G
    AU = constants.AU
    planet_colors = get_planet_colors()
    planet_names_en = ['Mercury', 'Venus', 'Earth', 'Mars',
                       'Jupiter', 'Saturn', 'Uranus', 'Neptune']
    planet_names_de = ['Merkur', 'Venus', 'Erde', 'Mars',
                       'Jupiter', 'Saturn', 'Uranus', 'Neptun']
    planet_names = planet_names_de if language == 'de' else planet_names_en
    planet_masses = [constants.M_mercury, constants.M_venus, constants.M_earth,
                     constants.M_mars, constants.M_jupiter, constants.M_saturn,
                     constants.M_uranus, constants.M_neptune]
    planet_radii = [constants.R_mercury, constants.R_venus, constants.R_earth,
                    constants.R_mars, constants.R_jupiter, constants.R_saturn,
                    constants.R_uranus, constants.R_neptune]
    planet_sma = [constants.a_mercury, constants.a_venus, constants.a_earth,
                  constants.a_mars, constants.a_jupiter, constants.a_saturn,
                  constants.a_uranus, constants.a_neptune]
    G_scales = np.logspace(0, 3, 500)
    all_radii_AU = {}
    for i, name in enumerate(planet_names):
        radii = np.array([scaled_orbit_radius(planet_sma[i], g) for g in G_scales])
        all_radii_AU[name] = radii / AU
    all_hill_AU = {}
    for i, name in enumerate(planet_names):
        hills = np.array([hill_sphere(scaled_orbit_radius(planet_sma[i], g), planet_masses[i], constants.M_sun) for g in G_scales])
        all_hill_AU[name] = hills / AU
    stability_limits = [planet_sma[i] / constants.R_sun for i in range(len(planet_names))]
    G_vals_bar = [1, 10, 100]
    tidal_sun_on_earth = []
    tidal_moon_on_earth = []
    for g in G_vals_bar:
        a_e = scaled_orbit_radius(constants.a_earth, g)
        a_m = scaled_orbit_radius(constants.d_earth_moon, g)
        tidal_sun_on_earth.append(tidal_force(constants.M_sun, constants.R_earth, a_e, G_std * g))
        tidal_moon_on_earth.append(tidal_force(constants.M_moon, constants.R_earth, a_m, G_std * g))
    suffix = '_de' if language == 'de' else ''
    if language == 'de':
        suptitle = 'Sonnensystem-Bahnuebersicht'
        titles = ['1. Planetenbahnen vs. G-Skalierung', '2. Orbitale Stabilitaetskarte',
                  '3. Hill-Sphaeren vs. G-Skalierung', '4. Gezeitenkraft-Vergleich auf der Erde']
        xlabels = ['G-Skalierungsfaktor', 'G-Skalierung bei Instabilitaet', 'G-Skalierungsfaktor', '']
        ylabels = ['Bahnradius (AU)', '', 'Hill-Sphaeren-Radius (AU)', 'Gezeitenkraft (N/kg)']
        sun_radius_label = 'Sonnenradius'
        sun_tidal_label, moon_tidal_label = 'Sonne auf Erde', 'Mond auf Erde'
    else:
        suptitle = 'Solar System Orbital Summary'
        titles = ['1. Planet Orbits vs. G Scaling', '2. Orbital Stability Map',
                  '3. Hill Spheres vs. G Scaling', '4. Tidal Force Comparison on Earth']
        xlabels = ['G Scaling Factor', 'G Scaling at Instability', 'G Scaling Factor', '']
        ylabels = ['Orbital Radius (AU)', '', 'Hill Sphere Radius (AU)', 'Tidal Force (N/kg)']
        sun_radius_label = 'Sun radius'
        sun_tidal_label, moon_tidal_label = 'Sun on Earth', 'Moon on Earth'
    fig, axes = plt.subplots(4, 1, figsize=(12, 32))
    fig.subplots_adjust(hspace=0.5, top=0.95, bottom=0.04)
    fig.suptitle(suptitle, fontsize=16, fontweight='bold', y=0.98)

    # Plot 1: All planet orbital radii
    ax = axes[0]
    for i, name in enumerate(planet_names):
        ax.loglog(G_scales, all_radii_AU[name], '-', color=planet_colors[i], linewidth=2.0, label=name)
    ax.axhline(y=constants.R_sun / AU, color=COLORS['sun'], linestyle='--',
               linewidth=1.5, alpha=0.7, label=sun_radius_label)
    ax.set_xlabel(xlabels[0], fontsize=12)
    ax.set_ylabel(ylabels[0], fontsize=12)
    ax.set_title(titles[0], fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=11)
    ax.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, framealpha=0.7)

    # Plot 2: Orbital stability map
    ax = axes[1]
    y_positions = np.arange(len(planet_names))
    for i, name in enumerate(planet_names):
        bar_width = stability_limits[i] - 1.0
        ax.barh(y_positions[i], bar_width, left=1.0, height=0.6,
                color=planet_colors[i], edgecolor='black', linewidth=0.8, alpha=0.85)
        ax.text(min(stability_limits[i], 1e6) * 1.05, y_positions[i],
                'G = {:.0f}x'.format(stability_limits[i]),
                va='center', fontsize=10, fontweight='bold', color=COLORS['text_dark'])
    ax.set_xscale('log')
    ax.set_yticks(y_positions)
    ax.set_yticklabels(planet_names, fontsize=11)
    ax.set_xlabel(xlabels[1], fontsize=12)
    ax.set_ylabel(ylabels[1], fontsize=12)
    ax.set_title(titles[1], fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, axis='x')
    ax.tick_params(axis='both', labelsize=11)
    ax.set_xlim(1, None)

    # Plot 3: Hill sphere radii
    ax = axes[2]
    for i, name in enumerate(planet_names):
        ax.loglog(G_scales, all_hill_AU[name], '-', color=planet_colors[i], linewidth=2.0, label=name)
    ax.set_xlabel(xlabels[2], fontsize=12)
    ax.set_ylabel(ylabels[2], fontsize=12)
    ax.set_title(titles[2], fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=11)
    ax.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, framealpha=0.7)

    # Plot 4: Tidal force comparison
    ax = axes[3]
    x_bar = np.arange(len(G_vals_bar))
    width = 0.35
    ax.bar(x_bar - width / 2, tidal_sun_on_earth, width,
           color=COLORS['sun'], edgecolor='black', linewidth=0.8, label=sun_tidal_label)
    ax.bar(x_bar + width / 2, tidal_moon_on_earth, width,
           color=COLORS['moon'], edgecolor='black', linewidth=0.8, label=moon_tidal_label)
    ax.set_yscale('log')
    ax.set_xticks(x_bar)
    ax.set_xticklabels(['G = {}x'.format(g) for g in G_vals_bar], fontsize=11)
    ax.set_xlabel(xlabels[3], fontsize=12)
    ax.set_ylabel(ylabels[3], fontsize=12)
    ax.set_title(titles[3], fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='both', labelsize=11)
    ax.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, framealpha=0.7)

    if save:
        os.makedirs(VIS_DIR, exist_ok=True)
        fn = 'orbital_summary{}.png'.format(suffix)
        fp = os.path.join(VIS_DIR, fn)
        fig.savefig(fp, dpi=150, bbox_inches='tight')
        print('Saved: {}'.format(fp))
    if show:
        plt.show()
    return fig


# =============================================================================
# GENERATION AND VERIFICATION
# =============================================================================

def generate_all_orbital_plots(language: str = 'en', show: bool = False) -> List[plt.Figure]:
    """Generate all orbital mechanics visualizations."""
    figures = []
    print('Generating orbital mechanics visualizations...')
    print('=' * 50)
    print('1. Moon and tidal effects...')
    figures.append(plot_moon_tidal_effects(language=language, show=show))
    print('2. Earth orbit under modified gravity...')
    figures.append(plot_earth_orbit(language=language, show=show))
    print('3. Solar system orbital summary...')
    figures.append(plot_orbital_summary(language=language, show=show))
    print('=' * 50)
    print('Generated {} visualizations in {}'.format(len(figures), VIS_DIR))
    return figures


def verify_orbital_physics():
    """
    Verify orbital mechanics calculations against known values.
    Verifiziert Bahnmechanik-Berechnungen gegen bekannte Werte.

    Checks:
    - Earth orbital period ~ 365.25 days
    - Moon orbital period ~ 27.3 days
    - Mercury GR precession ~ 43 arcseconds/century
    """
    print('=' * 70)
    print('ORBITAL MECHANICS VERIFICATION')
    print('=' * 70)
    c = get_constants()
    G = c.G

    # 1. Earth orbital period
    print('\n1. EARTH ORBITAL PERIOD')
    print('-' * 50)
    T_earth = orbital_period(c.a_earth, c.M_sun, G)
    T_earth_days = T_earth / 86400.0
    expected_days = 365.25
    error_pct = abs(T_earth_days - expected_days) / expected_days * 100
    print('   T = {:.2f} days (expected {:.2f})'.format(T_earth_days, expected_days))
    print('   Error: {:.2f}% -- CHECK: {}'.format(error_pct, 'PASS' if error_pct < 1.0 else 'FAIL'))

    # 2. Moon orbital period
    print('\n2. MOON ORBITAL PERIOD')
    print('-' * 50)
    T_moon = orbital_period(c.d_earth_moon, c.M_earth, G)
    T_moon_days = T_moon / 86400.0
    expected_moon = 27.3
    error_pct = abs(T_moon_days - expected_moon) / expected_moon * 100
    print('   T = {:.2f} days (expected ~{:.1f})'.format(T_moon_days, expected_moon))
    print('   Error: {:.2f}% -- CHECK: {}'.format(error_pct, 'PASS' if error_pct < 5.0 else 'FAIL'))

    # 3. Mercury GR precession
    print('\n3. MERCURY GR PRECESSION')
    print('-' * 50)
    e_mercury = 0.2056
    prec_rad = gr_precession(c.a_mercury, c.M_sun, e_mercury, G, c.c)
    prec_arcsec = prec_rad * (180.0 / np.pi) * 3600.0
    T_mercury = orbital_period(c.a_mercury, c.M_sun, G)
    T_mercury_years = T_mercury / (365.25 * 86400.0)
    orbits_per_century = 100.0 / T_mercury_years
    prec_arcsec_century = prec_arcsec * orbits_per_century
    expected_prec = 43.0
    error_pct = abs(prec_arcsec_century - expected_prec) / expected_prec * 100
    print('   Precession: {:.2f} arcsec/century (expected ~{:.0f})'.format(prec_arcsec_century, expected_prec))
    print('   Error: {:.2f}% -- CHECK: {}'.format(error_pct, 'PASS' if error_pct < 5.0 else 'FAIL'))

    # 4. Earth orbital velocity
    print('\n4. EARTH ORBITAL VELOCITY')
    print('-' * 50)
    v_earth = orbital_velocity(c.a_earth, c.M_sun, G)
    v_earth_kms = v_earth / 1000.0
    expected_v = 29.8
    error_pct = abs(v_earth_kms - expected_v) / expected_v * 100
    print('   v = {:.2f} km/s (expected ~{:.1f})'.format(v_earth_kms, expected_v))
    print('   Error: {:.2f}% -- CHECK: {}'.format(error_pct, 'PASS' if error_pct < 2.0 else 'FAIL'))

    # 5. Scaling behavior
    print('\n5. SCALING BEHAVIOR (G x10)')
    print('-' * 50)
    a_scaled = scaled_orbit_radius(c.a_earth, 10.0)
    ratio = a_scaled / c.a_earth
    print('   a_scaled / a_standard = {:.4f} (expected 0.1)'.format(ratio))
    print('   CHECK: {}'.format('PASS' if abs(ratio - 0.1) < 1e-10 else 'FAIL'))

    print('\n' + '=' * 70)
    print('VERIFICATION COMPLETE')
    print('=' * 70)


if __name__ == "__main__":
    print('=' * 60)
    print('Orbital Mechanics Module - Jugend forscht 2026')
    print('=' * 60)
    verify_orbital_physics()
    c = get_constants()
    print('\n\nSample orbital calculations:')
    print('-' * 60)
    for G_scale in [1.0, 10.0, 100.0]:
        moon = calculate_moon_orbit(G_scale, c)
        earth = calculate_earth_orbit(G_scale, c)
        print('\nG_scale = {:.0f}x:'.format(G_scale))
        print('  Moon:  a = {:.1f} R_E, T = {:.2f} days'.format(
            moon.semi_major_axis / c.R_earth, moon.period / 86400))
        print('  Earth: a = {:.4f} AU, T = {:.4f} years, v = {:.1f} km/s'.format(
            earth.semi_major_axis / c.AU, earth.period / (365.25 * 86400), earth.velocity / 1000))
    print('\n' + '=' * 60)
    print('Generating visualizations...')
    print('=' * 60)
    generate_all_orbital_plots(language='en', show=False)
    print('\nDone! Check the visualizations folder for output.')
