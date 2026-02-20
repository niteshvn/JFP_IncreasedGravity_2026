"""
Cosmological Effects Module for Jugend forscht 2026 Physics Visualization Project
Kosmologische-Effekte-Modul fuer Jugend forscht 2026 Physik-Visualisierungsprojekt

This module explores how modifying fundamental constants (G, hbar) affects
cosmic-scale physics: Planck scales, Jeans mass/length for structure formation,
Friedmann expansion rate, and critical density of the universe.

Author: Jugend forscht 2026 Project
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Optional, List
from dataclasses import dataclass

from .constants import get_constants, PhysicalConstants
from .color_scheme import COLORS, get_sequence

VIS_DIR = os.path.join(os.path.dirname(__file__), '..', 'visualizations')


@dataclass
class CosmicProperties:
    """Container for cosmological properties under modified constants."""
    planck_mass: float              # [kg]
    planck_length: float            # [m]
    planck_temperature: float       # [K]
    planck_time: float              # [s]
    jeans_mass: float               # [kg] at reference T and rho
    jeans_length: float             # [m] at reference T and rho
    critical_density: float         # [kg/m^3] for flat universe
    schwarzschild_universe: float   # Schwarzschild radius of observable universe mass [m]


def planck_mass(constants: PhysicalConstants) -> float:
    """Calculate the Planck mass. m_P = sqrt(hbar * c / G)."""
    return np.sqrt(constants.hbar * constants.c / constants.G)


def planck_length(constants: PhysicalConstants) -> float:
    """Calculate the Planck length. l_P = sqrt(hbar * G / c^3)."""
    return np.sqrt(constants.hbar * constants.G / constants.c**3)


def planck_temperature(constants: PhysicalConstants) -> float:
    """Calculate the Planck temperature. T_P = sqrt(hbar * c^5 / (G * k_B^2))."""
    return np.sqrt(constants.hbar * constants.c**5 / (constants.G * constants.k_B**2))


def planck_time(constants: PhysicalConstants) -> float:
    """Calculate the Planck time. t_P = sqrt(hbar * G / c^5)."""
    return np.sqrt(constants.hbar * constants.G / constants.c**5)


def jeans_mass(T, rho, G, mu_mp, k_B):
    """Calculate the Jeans mass. M_J = (5kT/(G*mu*mp))^(3/2) * (3/(4*pi*rho))^(1/2)."""
    return (5.0 * k_B * T / (G * mu_mp))**1.5 * np.sqrt(3.0 / (4.0 * np.pi * rho))


def jeans_length(T, rho, G, mu_mp, k_B, gamma=5.0/3.0):
    """Calculate the Jeans length. lambda_J = sqrt(pi*cs^2/(G*rho))."""
    c_s2 = gamma * k_B * T / mu_mp
    return np.sqrt(np.pi * c_s2 / (G * rho))


def friedmann_rate(G, rho):
    """Calculate the Friedmann expansion rate. H = sqrt(8*pi*G*rho/3)."""
    return np.sqrt(8.0 * np.pi * G * rho / 3.0)


def critical_density(H, G):
    """Calculate the critical density. rho_crit = 3*H^2/(8*pi*G)."""
    return 3.0 * H**2 / (8.0 * np.pi * G)


def calculate_cosmic_properties(G_scale=1.0, hbar_scale=1.0, constants=None):
    """Calculate cosmological properties for given scaling factors."""
    if constants is None:
        constants = get_constants()
    c = get_constants(G_scale=G_scale, hbar_scale=hbar_scale)
    m_P = planck_mass(c)
    l_P = planck_length(c)
    T_P = planck_temperature(c)
    t_P = planck_time(c)
    T_ref = 1e4
    rho_ref = 1e-20
    mu = 2.0
    mu_mp = mu * c.m_p
    M_J = jeans_mass(T_ref, rho_ref, c.G, mu_mp, c.k_B)
    L_J = jeans_length(T_ref, rho_ref, c.G, mu_mp, c.k_B)
    H_0 = 70.0 * 1e3 / (3.086e22)
    rho_crit = critical_density(H_0, c.G)
    M_universe = 1e53
    R_s_universe = 2.0 * c.G * M_universe / c.c**2
    return CosmicProperties(
        planck_mass=m_P, planck_length=l_P,
        planck_temperature=T_P, planck_time=t_P,
        jeans_mass=M_J, jeans_length=L_J,
        critical_density=rho_crit,
        schwarzschild_universe=R_s_universe)


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_cosmic_scales(constants=None, language='en', save=True, show=True):
    """Plot cosmic scales under modified constants (4 vertical subplots)."""
    if constants is None:
        constants = get_constants()
    COLOR_SEQUENCE = get_sequence()
    fig, axes = plt.subplots(4, 1, figsize=(12, 32))
    fig.subplots_adjust(hspace=0.5, top=0.95, bottom=0.04)
    suffix = '_de' if language == 'de' else ''

    # -- Plot 1: Planck scales vs hbar_scale --
    ax1 = axes[0]
    hbar_scales = np.logspace(-2, 1, 200)
    m_P_vals, l_P_vals, T_P_vals, t_P_vals = [], [], [], []
    for hs in hbar_scales:
        c_scaled = get_constants(hbar_scale=hs)
        m_P_vals.append(planck_mass(c_scaled))
        l_P_vals.append(planck_length(c_scaled))
        T_P_vals.append(planck_temperature(c_scaled))
        t_P_vals.append(planck_time(c_scaled))
    m_P_vals = np.array(m_P_vals)
    l_P_vals = np.array(l_P_vals)
    T_P_vals = np.array(T_P_vals)
    t_P_vals = np.array(t_P_vals)
    std = get_constants()
    m_P_std = planck_mass(std)
    l_P_std = planck_length(std)
    T_P_std = planck_temperature(std)
    t_P_std = planck_time(std)
    if language == 'de':
        lbl_mP = f'Planck-Masse (Std: {m_P_std:.2e} kg)'
        lbl_lP = f'Planck-Laenge (Std: {l_P_std:.2e} m)'
        lbl_TP = f'Planck-Temperatur (Std: {T_P_std:.2e} K)'
        lbl_tP = f'Planck-Zeit (Std: {t_P_std:.2e} s)'
    else:
        lbl_mP = f'Planck Mass (Std: {m_P_std:.2e} kg)'
        lbl_lP = f'Planck Length (Std: {l_P_std:.2e} m)'
        lbl_TP = f'Planck Temperature (Std: {T_P_std:.2e} K)'
        lbl_tP = f'Planck Time (Std: {t_P_std:.2e} s)'
    ax1.loglog(hbar_scales, m_P_vals / m_P_std, color=COLOR_SEQUENCE[0], linewidth=2.5, label=lbl_mP)
    ax1.loglog(hbar_scales, l_P_vals / l_P_std, color=COLOR_SEQUENCE[1], linewidth=2.5, label=lbl_lP)
    ax1.loglog(hbar_scales, T_P_vals / T_P_std, color=COLOR_SEQUENCE[2], linewidth=2.5, label=lbl_TP)
    ax1.loglog(hbar_scales, t_P_vals / t_P_std, color=COLOR_SEQUENCE[3], linewidth=2.5, label=lbl_tP)
    ax1.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    ax1.scatter([1.0], [1.0], color='red', s=80, zorder=5)
    if language == 'de':
        ax1.set_xlabel('hbar-Skalierungsfaktor', fontsize=12)
        ax1.set_ylabel('Normierter Wert (bezogen auf Standardwert)', fontsize=12)
        ax1.set_title('Planck-Skalen vs. hbar-Skalierung', fontsize=14, fontweight='bold', pad=15)
    else:
        ax1.set_xlabel('hbar Scaling Factor', fontsize=12)
        ax1.set_ylabel('Normalized Value (relative to standard)', fontsize=12)
        ax1.set_title('Planck Scales vs. hbar Scaling', fontsize=14, fontweight='bold', pad=15)
    ax1.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=11)

    # -- Plot 2: Jeans mass and length vs G_scale --
    ax2 = axes[1]
    G_scales = np.logspace(-2, 2, 200)
    M_J_vals, L_J_vals = [], []
    T_ref, rho_ref, mu = 1e4, 1e-20, 2.0
    for gs in G_scales:
        c_scaled = get_constants(G_scale=gs)
        mu_mp = mu * c_scaled.m_p
        M_J_vals.append(jeans_mass(T_ref, rho_ref, c_scaled.G, mu_mp, c_scaled.k_B))
        L_J_vals.append(jeans_length(T_ref, rho_ref, c_scaled.G, mu_mp, c_scaled.k_B))
    M_J_vals = np.array(M_J_vals)
    L_J_vals = np.array(L_J_vals)
    std_c = get_constants()
    mu_mp_std = mu * std_c.m_p
    M_J_std = jeans_mass(T_ref, rho_ref, std_c.G, mu_mp_std, std_c.k_B)
    L_J_std = jeans_length(T_ref, rho_ref, std_c.G, mu_mp_std, std_c.k_B)
    if language == 'de':
        lbl_MJ = f'Jeans-Masse (Std: {M_J_std:.2e} kg)'
        lbl_LJ = f'Jeans-Laenge (Std: {L_J_std:.2e} m)'
    else:
        lbl_MJ = f'Jeans Mass (Std: {M_J_std:.2e} kg)'
        lbl_LJ = f'Jeans Length (Std: {L_J_std:.2e} m)'
    ax2.loglog(G_scales, M_J_vals / M_J_std, color=COLOR_SEQUENCE[0], linewidth=2.5, label=lbl_MJ)
    ax2.loglog(G_scales, L_J_vals / L_J_std, color=COLOR_SEQUENCE[1], linewidth=2.5, label=lbl_LJ)
    ax2.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    ax2.scatter([1.0], [1.0], color='red', s=80, zorder=5)
    if language == 'de':
        ax2.set_xlabel('G-Skalierungsfaktor', fontsize=12)
        ax2.set_ylabel('Normierter Wert', fontsize=12)
        ax2.set_title('Jeans-Masse und -Laenge vs. G-Skalierung (T=10^4 K, rho=10^-20 kg/m^3)', fontsize=14, fontweight='bold', pad=15)
    else:
        ax2.set_xlabel('G Scaling Factor', fontsize=12)
        ax2.set_ylabel('Normalized Value', fontsize=12)
        ax2.set_title('Jeans Mass and Length vs. G Scaling (T=10^4 K, rho=10^-20 kg/m^3)', fontsize=14, fontweight='bold', pad=15)
    ax2.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=11)

    # -- Plot 3: Friedmann expansion rate vs G_scale --
    ax3 = axes[2]
    rho_cosmo = 9.47e-27  # current cosmic mean density kg/m^3
    H_vals = []
    for gs in G_scales:
        c_scaled = get_constants(G_scale=gs)
        H_vals.append(friedmann_rate(c_scaled.G, rho_cosmo))
    H_vals = np.array(H_vals)
    H_std = friedmann_rate(std_c.G, rho_cosmo)
    if language == 'de':
        lbl_H = f'Friedmann-Rate H (Std: {H_std:.2e} s^-1)'
    else:
        lbl_H = f'Friedmann Rate H (Std: {H_std:.2e} s^-1)'
    ax3.loglog(G_scales, H_vals / H_std, color=COLOR_SEQUENCE[4], linewidth=2.5, label=lbl_H)
    ax3.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    ax3.scatter([1.0], [1.0], color='red', s=80, zorder=5)
    if language == 'de':
        ax3.set_xlabel('G-Skalierungsfaktor', fontsize=12)
        ax3.set_ylabel('Normierte Friedmann-Rate', fontsize=12)
        ax3.set_title('Friedmann-Expansionsrate vs. G-Skalierung', fontsize=14, fontweight='bold', pad=15)
    else:
        ax3.set_xlabel('G Scaling Factor', fontsize=12)
        ax3.set_ylabel('Normalized Friedmann Rate', fontsize=12)
        ax3.set_title('Friedmann Expansion Rate vs. G Scaling', fontsize=14, fontweight='bold', pad=15)
    ax3.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(labelsize=11)

    # -- Plot 4: Critical density vs G_scale --
    ax4 = axes[3]
    H_0 = 70.0 * 1e3 / 3.086e22
    rho_crit_vals = []
    for gs in G_scales:
        c_scaled = get_constants(G_scale=gs)
        rho_crit_vals.append(critical_density(H_0, c_scaled.G))
    rho_crit_vals = np.array(rho_crit_vals)
    rho_crit_std = critical_density(H_0, std_c.G)
    if language == 'de':
        lbl_rho = f'Kritische Dichte (Std: {rho_crit_std:.2e} kg/m^3)'
    else:
        lbl_rho = f'Critical Density (Std: {rho_crit_std:.2e} kg/m^3)'
    ax4.loglog(G_scales, rho_crit_vals / rho_crit_std, color=COLOR_SEQUENCE[5], linewidth=2.5, label=lbl_rho)
    ax4.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    ax4.scatter([1.0], [1.0], color='red', s=80, zorder=5)
    if language == 'de':
        ax4.set_xlabel('G-Skalierungsfaktor', fontsize=12)
        ax4.set_ylabel('Normierte kritische Dichte', fontsize=12)
        ax4.set_title('Kritische Dichte vs. G-Skalierung', fontsize=14, fontweight='bold', pad=15)
    else:
        ax4.set_xlabel('G Scaling Factor', fontsize=12)
        ax4.set_ylabel('Normalized Critical Density', fontsize=12)
        ax4.set_title('Critical Density vs. G Scaling', fontsize=14, fontweight='bold', pad=15)
    ax4.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(labelsize=11)

    if language == 'de':
        fig.suptitle('Kosmologische Skalen unter modifizierten Konstanten', fontsize=16, fontweight='bold', y=0.98)
    else:
        fig.suptitle('Cosmological Scales Under Modified Constants', fontsize=16, fontweight='bold', y=0.98)

    if save:
        os.makedirs(VIS_DIR, exist_ok=True)
        filepath = os.path.join(VIS_DIR, f'cosmic_scales{suffix}.png')
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        print(f'Saved: {filepath}')
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_cosmic_summary(constants=None, language='en', save=True, show=True):
    """Plot cosmic summary with structure formation and Planck trajectory (4 vertical subplots)."""
    if constants is None:
        constants = get_constants()
    COLOR_SEQUENCE = get_sequence()
    fig, axes = plt.subplots(4, 1, figsize=(12, 32))
    fig.subplots_adjust(hspace=0.5, top=0.95, bottom=0.04)
    suffix = '_de' if language == 'de' else ''

    # -- Plot 1: Structure formation - Jeans mass vs temperature --
    ax1 = axes[0]
    T_range = np.logspace(1, 6, 200)
    rho_ref = 1e-20
    mu = 2.0
    std = get_constants()
    mu_mp_std = mu * std.m_p
    G_factors = [0.1, 0.5, 1.0, 2.0, 10.0]
    for i, gf in enumerate(G_factors):
        c_sc = get_constants(G_scale=gf)
        mu_mp_sc = mu * c_sc.m_p
        MJ = jeans_mass(T_range, rho_ref, c_sc.G, mu_mp_sc, c_sc.k_B)
        lbl = f'G x {gf}'
        ax1.loglog(T_range, MJ, color=COLOR_SEQUENCE[i % len(COLOR_SEQUENCE)], linewidth=2.5, label=lbl)
    M_sun = 1.989e30
    ax1.axhline(y=M_sun, color='gray', linestyle='--', alpha=0.5, label='1 Solar Mass')
    if language == 'de':
        ax1.set_xlabel('Temperatur [K]', fontsize=12)
        ax1.set_ylabel('Jeans-Masse [kg]', fontsize=12)
        ax1.set_title('Strukturbildung: Jeans-Masse vs. Temperatur', fontsize=14, fontweight='bold', pad=15)
    else:
        ax1.set_xlabel('Temperature [K]', fontsize=12)
        ax1.set_ylabel('Jeans Mass [kg]', fontsize=12)
        ax1.set_title('Structure Formation: Jeans Mass vs. Temperature', fontsize=14, fontweight='bold', pad=15)
    ax1.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=11)

    # -- Plot 2: Planck trajectory in G-hbar space --
    ax2 = axes[1]
    hbar_grid = np.logspace(-2, 2, 50)
    G_grid = np.logspace(-2, 2, 50)
    HH, GG = np.meshgrid(hbar_grid, G_grid)
    # Planck mass ratio
    m_P_ratio = np.sqrt(HH / GG)  # proportional to sqrt(hbar/G)
    contour = ax2.contourf(np.log10(HH), np.log10(GG), np.log10(m_P_ratio), levels=20, cmap='viridis')
    cbar = fig.colorbar(contour, ax=ax2)
    if language == 'de':
        cbar.set_label('log10(Planck-Masse Verhaeltnis)', fontsize=11)
        ax2.set_xlabel('log10(hbar-Skalierung)', fontsize=12)
        ax2.set_ylabel('log10(G-Skalierung)', fontsize=12)
        ax2.set_title('Planck-Masse im G-hbar Parameterraum', fontsize=14, fontweight='bold', pad=15)
    else:
        cbar.set_label('log10(Planck Mass Ratio)', fontsize=11)
        ax2.set_xlabel('log10(hbar Scaling)', fontsize=12)
        ax2.set_ylabel('log10(G Scaling)', fontsize=12)
        ax2.set_title('Planck Mass in G-hbar Parameter Space', fontsize=14, fontweight='bold', pad=15)
    ax2.plot([0], [0], 'r*', markersize=15, label='Standard Physics')
    ax2.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    ax2.tick_params(labelsize=11)

    # -- Plot 3: Virial temperature vs G_scale --
    ax3 = axes[2]
    G_scales_v = np.logspace(-2, 2, 200)
    R_cloud = 0.1 * 3.086e16  # 0.1 parsec in meters
    M_cloud = 1.989e30  # solar mass
    T_virial_vals = []
    for gs in G_scales_v:
        c_sc = get_constants(G_scale=gs)
        T_vir = c_sc.G * M_cloud * mu * c_sc.m_p / (3.0 * c_sc.k_B * R_cloud)
        T_virial_vals.append(T_vir)
    T_virial_vals = np.array(T_virial_vals)
    T_vir_std = std.G * M_cloud * mu * std.m_p / (3.0 * std.k_B * R_cloud)
    if language == 'de':
        lbl_Tv = f'Virial-Temperatur (Std: {T_vir_std:.1f} K)'
    else:
        lbl_Tv = f'Virial Temperature (Std: {T_vir_std:.1f} K)'
    ax3.loglog(G_scales_v, T_virial_vals / T_vir_std, color=COLOR_SEQUENCE[2], linewidth=2.5, label=lbl_Tv)
    ax3.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    ax3.scatter([1.0], [1.0], color='red', s=80, zorder=5)
    if language == 'de':
        ax3.set_xlabel('G-Skalierungsfaktor', fontsize=12)
        ax3.set_ylabel('Normierte Virial-Temperatur', fontsize=12)
        ax3.set_title('Virial-Temperatur einer Sonnenmasse-Wolke vs. G-Skalierung', fontsize=14, fontweight='bold', pad=15)
    else:
        ax3.set_xlabel('G Scaling Factor', fontsize=12)
        ax3.set_ylabel('Normalized Virial Temperature', fontsize=12)
        ax3.set_title('Virial Temperature of Solar-Mass Cloud vs. G Scaling', fontsize=14, fontweight='bold', pad=15)
    ax3.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(labelsize=11)

    # -- Plot 4: Summary dashboard bar chart --
    ax4 = axes[3]
    G_test_factors = [0.1, 0.5, 1.0, 2.0, 10.0]
    bar_data = {}
    for gf in G_test_factors:
        props = calculate_cosmic_properties(G_scale=gf)
        props_std = calculate_cosmic_properties(G_scale=1.0)
        bar_data[gf] = {
            'planck_mass': props.planck_mass / props_std.planck_mass,
            'jeans_mass': props.jeans_mass / props_std.jeans_mass,
            'critical_density': props.critical_density / props_std.critical_density,
            'schwarzschild': props.schwarzschild_universe / props_std.schwarzschild_universe
        }

    x_pos = np.arange(len(G_test_factors))
    width = 0.2
    if language == 'de':
        bar_labels = ['Planck-Masse', 'Jeans-Masse', 'Krit. Dichte', 'Schwarzschild-R.']
    else:
        bar_labels = ['Planck Mass', 'Jeans Mass', 'Critical Density', 'Schwarzschild R.']
    keys = ['planck_mass', 'jeans_mass', 'critical_density', 'schwarzschild']
    for j, (key, lbl) in enumerate(zip(keys, bar_labels)):
        vals = [np.log10(bar_data[gf][key]) if bar_data[gf][key] > 0 else 0 for gf in G_test_factors]
        ax4.bar(x_pos + j * width, vals, width, label=lbl, color=COLOR_SEQUENCE[j % len(COLOR_SEQUENCE)])

    ax4.set_xticks(x_pos + 1.5 * width)
    ax4.set_xticklabels([f'G x {gf}' for gf in G_test_factors], fontsize=11)
    ax4.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    if language == 'de':
        ax4.set_xlabel('G-Skalierung', fontsize=12)
        ax4.set_ylabel('log10(Verhaeltnis zu Standard)', fontsize=12)
        ax4.set_title('Kosmologische Zusammenfassung: Auswirkung der G-Skalierung', fontsize=14, fontweight='bold', pad=15)
    else:
        ax4.set_xlabel('G Scaling', fontsize=12)
        ax4.set_ylabel('log10(Ratio to Standard)', fontsize=12)
        ax4.set_title('Cosmological Summary: Impact of G Scaling', fontsize=14, fontweight='bold', pad=15)
    ax4.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)
    ax4.grid(True, alpha=0.3, axis="y")
    ax4.tick_params(labelsize=11)

    if language == 'de':
        fig.suptitle('Kosmologische Zusammenfassung', fontsize=16, fontweight='bold', y=0.98)
    else:
        fig.suptitle('Cosmological Summary', fontsize=16, fontweight='bold', y=0.98)

    if save:
        os.makedirs(VIS_DIR, exist_ok=True)
        filepath = os.path.join(VIS_DIR, f'cosmic_summary{suffix}.png')
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        print(f'Saved: {filepath}')
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def generate_all_cosmic_plots(constants=None, save=True, show=False):
    """Generate all cosmological visualizations in both languages."""
    if constants is None:
        constants = get_constants()
    figures = []
    for lang in ['en', 'de']:
        figures.append(plot_cosmic_scales(constants=constants, language=lang, save=save, show=show))
        figures.append(plot_cosmic_summary(constants=constants, language=lang, save=save, show=show))
    return figures


def verify_cosmic_physics():
    """Verify cosmological calculations against known values."""
    print('=' * 70)
    print('COSMOLOGICAL EFFECTS MODULE - VERIFICATION')
    print('=' * 70)

    std = get_constants()
    props = calculate_cosmic_properties(G_scale=1.0, hbar_scale=1.0)

    print(f'\nPlanck Mass:        {props.planck_mass:.4e} kg   (expected ~2.18e-08 kg)')
    print(f'Planck Length:      {props.planck_length:.4e} m    (expected ~1.62e-35 m)')
    print(f'Planck Temperature: {props.planck_temperature:.4e} K    (expected ~1.42e+32 K)')
    print(f'Planck Time:        {props.planck_time:.4e} s    (expected ~5.39e-44 s)')
    print(f'Jeans Mass (ISM):   {props.jeans_mass:.4e} kg')
    print(f'Jeans Length (ISM): {props.jeans_length:.4e} m')
    print(f'Critical Density:   {props.critical_density:.4e} kg/m^3')
    print(f'Schwarzschild R (universe): {props.schwarzschild_universe:.4e} m')

    # Verification checks
    checks_passed = 0
    checks_total = 4

    # Check Planck mass
    if abs(props.planck_mass - 2.176e-8) / 2.176e-8 < 0.01:
        checks_passed += 1
        print('  [PASS] Planck mass within 1% of expected')
    else:
        print('  [FAIL] Planck mass deviation > 1%')

    # Check Planck length
    if abs(props.planck_length - 1.616e-35) / 1.616e-35 < 0.01:
        checks_passed += 1
        print('  [PASS] Planck length within 1% of expected')
    else:
        print('  [FAIL] Planck length deviation > 1%')

    # Check Planck temperature
    if abs(props.planck_temperature - 1.416e32) / 1.416e32 < 0.01:
        checks_passed += 1
        print('  [PASS] Planck temperature within 1% of expected')
    else:
        print('  [FAIL] Planck temperature deviation > 1%')

    # Check Planck time
    if abs(props.planck_time - 5.391e-44) / 5.391e-44 < 0.01:
        checks_passed += 1
        print('  [PASS] Planck time within 1% of expected')
    else:
        print('  [FAIL] Planck time deviation > 1%')

    print(f'\nVerification: {checks_passed}/{checks_total} checks passed')

    # Test scaling behavior
    print('\n--- Scaling Test ---')
    for gs in [0.1, 1.0, 10.0]:
        for hs in [0.1, 1.0, 10.0]:
            p = calculate_cosmic_properties(G_scale=gs, hbar_scale=hs)
            print(f'G_scale={gs:5.1f}, hbar_scale={hs:5.1f}: '
                  f'm_P={p.planck_mass:.3e} kg, '
                  f'l_P={p.planck_length:.3e} m, '
                  f'M_J={p.jeans_mass:.3e} kg')

    print('\n' + '=' * 70)
    return checks_passed == checks_total


if __name__ == '__main__':
    verify_cosmic_physics()
    generate_all_cosmic_plots(save=True, show=False)
    print('All cosmological visualizations generated.')
