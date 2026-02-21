"""
Fine Structure Constant Module for Jugend forscht 2026 Physics Visualization Project
Feinstrukturkonstanten-Modul fuer Jugend forscht 2026 Physik-Visualisierungsprojekt

This module explores what happens when the fine-structure constant alpha is allowed
to vary naturally with hbar scaling, rather than being held fixed as in constants.py.

Key insight: In the main project, alpha is held constant when hbar is scaled. But
physically, alpha = e^2 / (4*pi*epsilon_0*hbar*c), so reducing hbar INCREASES alpha.
This module explores the consequences:
  - Stronger electromagnetic coupling (atoms bind tighter)
  - Fine-structure splitting grows as alpha^4
  - QED perturbation theory breaks down for alpha >= 1
  - Chemistry becomes impossible for alpha > ~0.5
  - Radiative lifetimes shrink as 1/alpha^5

Formula: alpha_true = e^2 / (4*pi*epsilon_0*hbar_scaled*c) = alpha_0 / hbar_scale

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

@dataclass
class FineStructureProperties:
    """
    Container for fine-structure constant properties.
    Behaelter fuer Feinstrukturkonstanten-Eigenschaften.
    """
    alpha_standard: float       # Standard fine-structure constant (~1/137)
    alpha_modified: float       # Modified alpha (if hbar changes)
    hbar_scale: float           # The hbar scaling factor
    bohr_radius: float          # Bohr radius [m]
    binding_energy: float       # Hydrogen ground state energy [J]
    rydberg_energy: float       # Rydberg energy [J]
    fine_splitting: float       # Fine-structure splitting energy [J]
    radiative_lifetime: float   # Radiative lifetime scaling factor
    chemical_stable: bool       # Whether chemistry is still possible

# ============================================================================
# CALCULATION FUNCTIONS
# ============================================================================

def sommerfeld_alpha(hbar_scale: float, constants: PhysicalConstants) -> float:
    """
    Calculate the true Sommerfeld fine-structure constant for a given hbar scaling.
    Berechnet die wahre Sommerfeld-Feinstrukturkonstante fuer eine gegebene hbar-Skalierung.

    Formula: alpha_true = e^2 / (4 * pi * epsilon_0 * hbar_scaled * c) = alpha_0 / hbar_scale

    Args:
        hbar_scale: Scaling factor for hbar (1.0 = standard universe)
        constants: Physical constants object

    Returns:
        Modified fine-structure constant (dimensionless)
    """
    alpha_0 = constants.alpha
    return alpha_0 / hbar_scale


def binding_energy_hydrogen(alpha: float, constants: PhysicalConstants) -> float:
    """
    Calculate hydrogen ground state binding energy for a given alpha.
    Berechnet die Bindungsenergie des Wasserstoff-Grundzustands fuer ein gegebenes Alpha.

    Formula: E_1 = -(1/2) * m_e * c^2 * alpha^2
    """
    return -0.5 * constants.m_e * constants.c**2 * alpha**2


def fine_structure_splitting(alpha: float, n: int, constants: PhysicalConstants) -> float:
    """
    Calculate the fine-structure energy splitting for hydrogen.
    Berechnet die Feinstrukturaufspaltung fuer Wasserstoff.

    Formula: Delta_E = (1/2) * m_e * c^2 * alpha^4 / n^3
    """
    return 0.5 * constants.m_e * constants.c**2 * alpha**4 / n**3


def rydberg_energy(alpha: float, constants: PhysicalConstants) -> float:
    """
    Calculate the Rydberg energy for a given alpha.
    Berechnet die Rydberg-Energie fuer ein gegebenes Alpha.

    Formula: E_R = (1/2) * m_e * c^2 * alpha^2
    """
    return 0.5 * constants.m_e * constants.c**2 * alpha**2


def radiative_lifetime_factor(alpha: float) -> float:
    """
    Calculate the radiative lifetime scaling factor relative to standard alpha.
    Berechnet den Skalierungsfaktor der Strahlungslebensdauer relativ zum Standard-Alpha.

    Radiative lifetimes scale as 1/alpha^5. Normalized so standard alpha gives 1.0.
    """
    alpha_0 = 0.0072973525693  # Standard alpha
    return (alpha_0 / alpha)**5

def calculate_fine_structure(
    hbar_scale: float,
    constants: PhysicalConstants
) -> FineStructureProperties:
    """
    Calculate all fine-structure properties for a given hbar scaling.
    Berechnet alle Feinstruktureigenschaften fuer eine gegebene hbar-Skalierung.
    """
    alpha_std = constants.alpha
    alpha_mod = sommerfeld_alpha(hbar_scale, constants)

    # Bohr radius with natural alpha: a_0 proportional to hbar_scale^2
    a_0_std = constants.hbar / (constants.m_e * constants.c * alpha_std)
    bohr_r = a_0_std * hbar_scale**2

    E_bind = binding_energy_hydrogen(alpha_mod, constants)
    E_rydberg = rydberg_energy(alpha_mod, constants)
    E_splitting = fine_structure_splitting(alpha_mod, 2, constants)
    tau_factor = radiative_lifetime_factor(alpha_mod)

    # Chemistry breaks down when alpha > ~0.5
    chem_stable = alpha_mod < 0.5

    return FineStructureProperties(
        alpha_standard=alpha_std,
        alpha_modified=alpha_mod,
        hbar_scale=hbar_scale,
        bohr_radius=bohr_r,
        binding_energy=E_bind,
        rydberg_energy=E_rydberg,
        fine_splitting=E_splitting,
        radiative_lifetime=tau_factor,
        chemical_stable=chem_stable
    )

# ============================================================================
# PLOT FUNCTIONS
# ============================================================================

def plot_alpha_scaling(
    constants: Optional[PhysicalConstants] = None,
    language: str = 'en',
    save: bool = True,
    show: bool = True,
) -> plt.Figure:
    """
    Visualize how the fine-structure constant and related quantities scale with hbar.
    Visualisiert wie die Feinstrukturkonstante und verwandte Groessen mit hbar skalieren.

    Four vertical subplots:
      1. Alpha vs hbar_scale (log-log) with QED breakdown regions
      2. Hydrogen binding energy vs hbar_scale
      3. Fine-structure splitting (n=2) vs hbar_scale
      4. Bohr radius: standard vs modified

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

    # Extended to 10^-20 to show hbar×10^-18 threshold (compensates G×10^36)
    hbar_scales = np.logspace(-20, 1, 500)
    alpha_0 = constants.alpha
    eV = constants.e

    # Pre-compute arrays
    alpha_vals = alpha_0 / hbar_scales
    E_bind_vals = np.array([abs(binding_energy_hydrogen(alpha_0 / h, constants))
                            for h in hbar_scales]) / eV
    E_split_vals = np.array([fine_structure_splitting(alpha_0 / h, 2, constants)
                             for h in hbar_scales]) / eV
    # Bohr radius: standard (constant) vs modified (proportional to hbar_scale^2)
    a_0_std = constants.hbar / (constants.m_e * constants.c * alpha_0)
    bohr_standard = np.full_like(hbar_scales, a_0_std)
    bohr_modified = a_0_std * hbar_scales**2

    # ----- figure -----
    fig, axes = plt.subplots(4, 1, figsize=(12, 32))
    fig.subplots_adjust(hspace=0.5, top=0.95, bottom=0.04)

    if language == 'de':
        fig.suptitle('Feinstrukturkonstante: Skalierung mit hbar',
                     fontsize=16, fontweight='bold', y=0.98)
    else:
        fig.suptitle('Fine-Structure Constant: Scaling with hbar',
                     fontsize=16, fontweight='bold', y=0.98)

    ax1, ax2, ax3, ax4 = axes

    # --- Subplot 1: Alpha vs hbar_scale ---
    ax1.loglog(hbar_scales, alpha_vals, '-', color=COLORS['quantum'], linewidth=2.5,
               label=r'$\alpha(\hbar)$')
    ax1.axhline(y=1.0, color=COLORS['relativistic'], linestyle='--', linewidth=2,
                label=r'$\alpha=1$ (QED breakdown)')
    ax1.axhline(y=0.5, color=COLORS['primary_amber'], linestyle='--', linewidth=1.5,
                label=r'$\alpha=0.5$ (chemistry limit)')
    ax1.axvline(x=1.0, color=COLORS['standard'], linestyle=':', linewidth=1.5)
    ax1.axvline(x=1e-18, color='red', linestyle='--', linewidth=2, alpha=0.8)
    ax1.text(1e-18 * 1.5, alpha_vals.max() * 0.1, r'$\hbar \times 10^{-18}$', color='red', fontsize=10, rotation=90, va='bottom')
    ax1.fill_between(hbar_scales, alpha_vals, 1.0, where=alpha_vals > 1.0,
                     color=COLORS['relativistic'], alpha=0.15)
    ax1.set_xlabel(r'$\hbar / \hbar_0$', fontsize=12)
    if language == 'de':
        ax1.set_ylabel('Feinstrukturkonstante', fontsize=12)
        ax1.set_title('Feinstrukturkonstante vs. hbar-Skalierung', fontsize=14, fontweight='bold', pad=15)
    else:
        ax1.set_ylabel('Fine-structure constant', fontsize=12)
        ax1.set_title('Fine-Structure Constant vs. hbar Scaling', fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

    # --- Subplot 2: Binding energy ---
    ax2.loglog(hbar_scales, E_bind_vals, '-', color=COLORS['primary_blue'], linewidth=2.5,
               label=r'$|E_1|$')
    ax2.axhline(y=13.6, color=COLORS['standard'], linestyle=':', linewidth=1.5, label='13.6 eV')
    ax2.axvline(x=1.0, color=COLORS['muted'], linestyle=':', linewidth=1.0, alpha=0.5)
    ax2.set_xlabel(r'$\hbar / \hbar_0$', fontsize=12)
    if language == 'de':
        ax2.set_ylabel('Bindungsenergie [eV]', fontsize=12)
        ax2.set_title('Wasserstoff-Bindungsenergie vs. hbar', fontsize=14, fontweight='bold', pad=15)
    else:
        ax2.set_ylabel('Binding Energy [eV]', fontsize=12)
        ax2.set_title('Hydrogen Binding Energy vs. hbar', fontsize=14, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

    # --- Subplot 3: Fine-structure splitting ---
    ax3.loglog(hbar_scales, E_split_vals, '-', color=COLORS['temp_hot'], linewidth=2.5,
               label=r'$\Delta E_{fs}$ (n=2)')
    ax3.axvline(x=1.0, color=COLORS['standard'], linestyle=':', linewidth=1.5)
    ax3.set_xlabel(r'$\hbar / \hbar_0$', fontsize=12)
    if language == 'de':
        ax3.set_ylabel('Aufspaltung [eV]', fontsize=12)
        ax3.set_title('Feinstrukturaufspaltung (n=2) vs. hbar', fontsize=14, fontweight='bold', pad=15)
    else:
        ax3.set_ylabel('Splitting [eV]', fontsize=12)
        ax3.set_title('Fine-Structure Splitting (n=2) vs. hbar', fontsize=14, fontweight='bold', pad=15)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

    # --- Subplot 4: Bohr radius ---
    ax4.loglog(hbar_scales, bohr_modified * 1e12, '-', color=COLORS['primary_blue'],
               linewidth=2.5, label=r'$a_0(\hbar)$')
    ax4.axhline(y=a_0_std * 1e12, color=COLORS['standard'], linestyle=':', linewidth=1.5,
                label=f'{a_0_std*1e12:.1f} pm (standard)')
    ax4.axvline(x=1.0, color=COLORS['muted'], linestyle=':', linewidth=1.0, alpha=0.5)
    ax4.set_xlabel(r'$\hbar / \hbar_0$', fontsize=12)
    if language == 'de':
        ax4.set_ylabel('Bohr-Radius [pm]', fontsize=12)
        ax4.set_title('Bohr-Radius vs. hbar-Skalierung', fontsize=14, fontweight='bold', pad=15)
    else:
        ax4.set_ylabel('Bohr Radius [pm]', fontsize=12)
        ax4.set_title('Bohr Radius vs. hbar Scaling', fontsize=14, fontweight='bold', pad=15)
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

    if save:
        os.makedirs(VIS_DIR, exist_ok=True)
        suffix = '_de' if language == 'de' else ''
        filepath = os.path.join(VIS_DIR, f'alpha_scaling{suffix}.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f'  Saved: {filepath}')
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_alpha_consequences(
    constants: Optional[PhysicalConstants] = None,
    language: str = 'en',
    save: bool = True,
    show: bool = True,
) -> plt.Figure:
    """
    Visualize consequences of varying alpha: Rydberg energy, stability, lifetime, QED coupling.
    Visualisiert Konsequenzen der Alpha-Variation.
    """
    if constants is None:
        constants = get_constants()

    fig, axes = plt.subplots(4, 1, figsize=(12, 32))
    fig.subplots_adjust(hspace=0.5, top=0.95, bottom=0.04)

    if language == 'de':
        fig.suptitle('Konsequenzen einer veraenderten Feinstrukturkonstante',
                     fontsize=16, fontweight='bold', y=0.98)
    else:
        fig.suptitle('Consequences of Varying the Fine-Structure Constant',
                     fontsize=16, fontweight='bold', y=0.98)

    # Extended to 10^-20 to show hbar×10^-18 threshold (compensates G×10^36)
    hbar_scales = np.logspace(-20, 1, 500)
    alpha_0 = constants.alpha
    eV = constants.e
    alpha_vals = alpha_0 / hbar_scales

    # --- Subplot 1: Rydberg energy ---
    ax1 = axes[0]
    E_ryd = 0.5 * constants.m_e * constants.c**2 * alpha_vals**2
    ax1.loglog(hbar_scales, E_ryd / eV, '-', color=COLORS['quantum'], linewidth=2.5,
               label=r'$E_R(\hbar)$')
    ax1.axhline(y=13.6, color=COLORS['standard'], linestyle=':', linewidth=1.5, label='13.6 eV')
    ax1.axvline(x=1.0, color=COLORS['muted'], linestyle=':', linewidth=1.0, alpha=0.5)
    ax1.set_xlabel(r'$\hbar / \hbar_0$', fontsize=12)
    if language == 'de':
        ax1.set_ylabel('Rydberg-Energie [eV]', fontsize=12)
        ax1.set_title('Rydberg-Energie vs. hbar', fontsize=14, fontweight='bold', pad=15)
    else:
        ax1.set_ylabel('Rydberg Energy [eV]', fontsize=12)
        ax1.set_title('Rydberg Energy vs. hbar Scaling', fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

    # --- Subplot 2: Chemical stability map ---
    ax2 = axes[1]
    chem_ok = alpha_vals < 0.5
    qed_ok = alpha_vals < 1.0
    ax2.fill_between(hbar_scales, 0, 1, where=chem_ok, color=COLORS['non_relativistic'], alpha=0.4,
                     label='Chemistry possible' if language == 'en' else 'Chemie moeglich')
    ax2.fill_between(hbar_scales, 0, 1, where=(~chem_ok) & qed_ok, color=COLORS['primary_amber'], alpha=0.4,
                     label='No chemistry' if language == 'en' else 'Keine Chemie')
    ax2.fill_between(hbar_scales, 0, 1, where=~qed_ok, color=COLORS['relativistic'], alpha=0.4,
                     label='QED breakdown' if language == 'en' else 'QED-Zusammenbruch')
    ax2.set_xscale('log')
    ax2.set_ylim(0, 1)
    ax2.set_yticks([])
    hbar_chem = alpha_0 / 0.5
    hbar_qed = alpha_0 / 1.0
    ax2.axvline(x=hbar_chem, color='k', linestyle='--', linewidth=2, label=r'$\alpha=0.5$')
    ax2.axvline(x=hbar_qed, color='k', linestyle='-.', linewidth=2, label=r'$\alpha=1.0$')
    ax2.set_xlabel(r'$\hbar / \hbar_0$', fontsize=12)
    if language == 'de':
        ax2.set_title('Chemische Stabilitaetskarte', fontsize=14, fontweight='bold', pad=15)
    else:
        ax2.set_title('Chemical Stability Map', fontsize=14, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

    # --- Subplot 3: Radiative lifetime ---
    ax3 = axes[2]
    tau_factor = hbar_scales**5
    ax3.loglog(hbar_scales, tau_factor, '-', color=COLORS['primary_blue'], linewidth=2.5,
               label=r'$\tau/\tau_0 \propto \hbar^5$')
    ax3.axhline(y=1.0, color=COLORS['standard'], linestyle=':', linewidth=1.5)
    ax3.axvline(x=1.0, color=COLORS['muted'], linestyle=':', linewidth=1.0, alpha=0.5)
    ax3.set_xlabel(r'$\hbar / \hbar_0$', fontsize=12)
    if language == 'de':
        ax3.set_ylabel('Lebensdauer-Faktor', fontsize=12)
        ax3.set_title('Strahlungslebensdauer vs. hbar', fontsize=14, fontweight='bold', pad=15)
    else:
        ax3.set_ylabel('Lifetime Factor', fontsize=12)
        ax3.set_title('Radiative Lifetime vs. hbar Scaling', fontsize=14, fontweight='bold', pad=15)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

    # --- Subplot 4: QED coupling ---
    ax4 = axes[3]
    qed_coupling = alpha_vals / (4 * np.pi)
    ax4.loglog(hbar_scales, qed_coupling, '-', color=COLORS['temp_hot'], linewidth=2.5,
               label=r'$\alpha/(4\pi)$')
    ax4.axhline(y=1.0, color=COLORS['relativistic'], linestyle='--', linewidth=2,
                label='Perturbation breakdown' if language == 'en' else 'Stoerungstheorie bricht zusammen')
    ax4.axvline(x=1.0, color=COLORS['standard'], linestyle=':', linewidth=1.5)
    ax4.set_xlabel(r'$\hbar / \hbar_0$', fontsize=12)
    if language == 'de':
        ax4.set_ylabel('QED-Kopplung', fontsize=12)
        ax4.set_title('QED-Kopplungsstaerke vs. hbar', fontsize=14, fontweight='bold', pad=15)
    else:
        ax4.set_ylabel('QED Coupling', fontsize=12)
        ax4.set_title('QED Coupling Strength vs. hbar Scaling', fontsize=14, fontweight='bold', pad=15)
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

    if save:
        os.makedirs(VIS_DIR, exist_ok=True)
        suffix = '_de' if language == 'de' else ''
        filepath = os.path.join(VIS_DIR, f'alpha_consequences{suffix}.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f'  Saved: {filepath}')
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_alpha_summary(
    constants: Optional[PhysicalConstants] = None,
    language: str = 'en',
    save: bool = True,
    show: bool = True,
) -> plt.Figure:
    """
    Summary plots: bar chart, Lyman-alpha shift, EM/gravity ratio, regime diagram.
    Zusammenfassungsdiagramme.
    """
    if constants is None:
        constants = get_constants()

    colors_seq = get_sequence()
    fig, axes = plt.subplots(4, 1, figsize=(12, 32))
    fig.subplots_adjust(hspace=0.5, top=0.95, bottom=0.04)

    if language == 'de':
        fig.suptitle('Feinstrukturkonstante: Zusammenfassung',
                     fontsize=16, fontweight='bold', y=0.98)
    else:
        fig.suptitle('Fine-Structure Constant: Summary Overview',
                     fontsize=16, fontweight='bold', y=0.98)

    alpha_0 = constants.alpha
    eV = constants.e
    # Extended to 10^-20 to show hbar×10^-18 threshold (compensates G×10^36)
    hbar_scales = np.logspace(-20, 1, 500)
    alpha_vals = alpha_0 / hbar_scales

    # --- Subplot 1: Bar chart ---
    ax1 = axes[0]
    sample_s = [0.1, 0.5, 1.0, 2.0, 5.0]
    alpha_norm = [1.0 / s for s in sample_s]
    bar_labels = [f'{s}' for s in sample_s]
    bar_colors = [colors_seq[i % len(colors_seq)] for i in range(len(sample_s))]
    bars = ax1.bar(bar_labels, alpha_norm, color=bar_colors, edgecolor='black', linewidth=1.5)
    ax1.set_yscale('log')
    for bar, val in zip(bars, alpha_norm):
        ax1.text(bar.get_x() + bar.get_width() / 2, val * 1.3,
                 f'{val:.1f}x', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax1.axhline(y=1.0, color=COLORS['standard'], linestyle=':', linewidth=1.5)
    ax1.set_xlabel(r'$\hbar / \hbar_0$', fontsize=12)
    if language == 'de':
        ax1.set_ylabel(r'$\alpha / \alpha_0$', fontsize=12)
        ax1.set_title('Alpha bei verschiedenen hbar-Werten', fontsize=14, fontweight='bold', pad=15)
    else:
        ax1.set_ylabel(r'$\alpha / \alpha_0$', fontsize=12)
        ax1.set_title('Alpha at Different hbar Values', fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, axis='y')

    # --- Subplot 2: Lyman-alpha wavelength ---
    ax2 = axes[1]
    E_ly = 0.75 * 0.5 * constants.m_e * constants.c**2 * alpha_vals**2
    h_planck = 2 * np.pi * constants.hbar
    wavelength = h_planck * constants.c / E_ly * 1e9
    ax2.loglog(hbar_scales, wavelength, '-', color=COLORS['primary_blue'], linewidth=2.5,
               label=r'Lyman-$\alpha$')
    ax2.axhline(y=121.6, color=COLORS['standard'], linestyle=':', linewidth=1.5, label='121.6 nm')
    ax2.axvline(x=1.0, color=COLORS['muted'], linestyle=':', linewidth=1.0, alpha=0.5)
    ax2.set_xlabel(r'$\hbar / \hbar_0$', fontsize=12)
    if language == 'de':
        ax2.set_ylabel('Wellenlaenge [nm]', fontsize=12)
        ax2.set_title('Lyman-Alpha Wellenlaengenverschiebung', fontsize=14, fontweight='bold', pad=15)
    else:
        ax2.set_ylabel('Wavelength [nm]', fontsize=12)
        ax2.set_title('Lyman-Alpha Wavelength Shift', fontsize=14, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

    # --- Subplot 3: EM/gravity ratio ---
    ax3 = axes[2]
    alpha_G_vals = constants.G * constants.m_p**2 / (constants.hbar * hbar_scales * constants.c)
    ratio = alpha_vals / alpha_G_vals
    ax3.loglog(hbar_scales, ratio, '-', color=COLORS['quantum'], linewidth=2.5,
               label=r'$\alpha_{EM}/\alpha_G$')
    ax3.axvline(x=1.0, color=COLORS['standard'], linestyle=':', linewidth=1.5)
    ax3.set_xlabel(r'$\hbar / \hbar_0$', fontsize=12)
    if language == 'de':
        ax3.set_ylabel('Verhaeltnis', fontsize=12)
        ax3.set_title('EM/Gravitations-Kopplungsverhaeltnis', fontsize=14, fontweight='bold', pad=15)
    else:
        ax3.set_ylabel('Ratio', fontsize=12)
        ax3.set_title('EM/Gravitational Coupling Ratio', fontsize=14, fontweight='bold', pad=15)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)

    # --- Subplot 4: Regime classification ---
    ax4 = axes[3]
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 5)
    ax4.axis('off')
    regimes = [
        (1.5, 2.5, 'Standard\nChemistry' if language == 'en' else 'Standard\nChemie', COLORS['non_relativistic']),
        (5.0, 2.5, 'No Chemistry' if language == 'en' else 'Keine Chemie', COLORS['primary_amber']),
        (8.5, 2.5, 'QED Breakdown' if language == 'en' else 'QED-Zusammenbruch', COLORS['relativistic']),
    ]
    for x, y, label, color in regimes:
        rect = plt.Rectangle((x - 1.2, y - 0.8), 2.4, 1.6, facecolor=color,
                              edgecolor='black', linewidth=2, alpha=0.6)
        ax4.add_patch(rect)
        ax4.text(x, y, label, ha='center', va='center', fontsize=11, fontweight='bold', color='white')
    ax4.annotate('', xy=(3.5, 2.5), xytext=(3.0, 2.5),
                 arrowprops=dict(arrowstyle='->', color='white', lw=2))
    ax4.annotate('', xy=(7.0, 2.5), xytext=(6.5, 2.5),
                 arrowprops=dict(arrowstyle='->', color='white', lw=2))
    ax4.text(5.0, 4.2, r'Decreasing $\hbar$ $\rightarrow$' if language == 'en' else r'Abnehmendes $\hbar$ $\rightarrow$',
             ha='center', fontsize=13, fontweight='bold', color='white')
    if language == 'de':
        ax4.set_title('Regimeklassifikation', fontsize=14, fontweight='bold', pad=15)
    else:
        ax4.set_title('Regime Classification', fontsize=14, fontweight='bold', pad=15)

    if save:
        os.makedirs(VIS_DIR, exist_ok=True)
        suffix = '_de' if language == 'de' else ''
        filepath = os.path.join(VIS_DIR, f'alpha_summary{suffix}.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f'  Saved: {filepath}')
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# ============================================================================
# GENERATE ALL
# ============================================================================

def generate_all_fine_structure_plots(
    constants: Optional[PhysicalConstants] = None,
    language: str = 'en',
    save: bool = True,
    show: bool = False,
) -> List[plt.Figure]:
    """Generate all fine-structure constant plots."""
    if constants is None:
        constants = get_constants()
    figs = []
    print('Generating alpha scaling plots...')
    figs.append(plot_alpha_scaling(constants=constants, language=language, save=save, show=show))
    print('Generating alpha consequences plots...')
    figs.append(plot_alpha_consequences(constants=constants, language=language, save=save, show=show))
    print('Generating alpha summary plots...')
    figs.append(plot_alpha_summary(constants=constants, language=language, save=save, show=show))
    print(f'Done. Generated {len(figs)} fine-structure plots.')
    return figs


# ============================================================================
# VERIFICATION
# ============================================================================

def verify_fine_structure_physics() -> bool:
    """Verify fine-structure constant calculations."""
    print('=== Fine Structure Verification ===')
    constants = get_constants()
    all_pass = True

    # Test 1: alpha at hbar_scale=1
    a1 = sommerfeld_alpha(1.0, constants)
    ok1 = abs(a1 - constants.alpha) / constants.alpha < 1e-10
    print(f'  alpha(1.0) = 1/{1/a1:.2f} {"PASS" if ok1 else "FAIL"}')
    if not ok1: all_pass = False

    # Test 2: alpha at hbar_scale=0.5 = 2*alpha_0
    a2 = sommerfeld_alpha(0.5, constants)
    ok2 = abs(a2 - 2 * constants.alpha) / (2 * constants.alpha) < 1e-10
    print(f'  alpha(0.5) = 1/{1/a2:.2f} {"PASS" if ok2 else "FAIL"}')
    if not ok2: all_pass = False

    # Test 3: Binding energy scales as alpha^2
    E1 = abs(binding_energy_hydrogen(constants.alpha, constants))
    E2 = abs(binding_energy_hydrogen(2 * constants.alpha, constants))
    ok3 = abs(E2 / E1 - 4.0) < 1e-6
    print(f'  E(2a)/E(a) = {E2/E1:.4f} {"PASS" if ok3 else "FAIL"}')
    if not ok3: all_pass = False

    # Test 4: Splitting scales as alpha^4
    S1 = fine_structure_splitting(constants.alpha, 2, constants)
    S2 = fine_structure_splitting(2 * constants.alpha, 2, constants)
    ok4 = abs(S2 / S1 - 16.0) < 1e-6
    print(f'  Split(2a)/Split(a) = {S2/S1:.4f} {"PASS" if ok4 else "FAIL"}')
    if not ok4: all_pass = False

    # Test 5: Lifetime factor at standard = 1.0
    ok5 = abs(radiative_lifetime_factor(constants.alpha) - 1.0) < 1e-6
    print(f'  tau(alpha_0) = {radiative_lifetime_factor(constants.alpha):.6f} {"PASS" if ok5 else "FAIL"}')
    if not ok5: all_pass = False

    # Test 6: Binding energy ~13.6 eV
    E_eV = abs(binding_energy_hydrogen(constants.alpha, constants)) / constants.e
    ok6 = abs(E_eV - 13.6) / 13.6 < 0.01
    print(f'  E_bind = {E_eV:.2f} eV {"PASS" if ok6 else "FAIL"}')
    if not ok6: all_pass = False

    print(f'=== {"ALL PASSED" if all_pass else "SOME FAILED"} ===')
    return all_pass


if __name__ == '__main__':
    print('Fine Structure Constant Module - Jugend forscht 2026')
    print('=' * 50)
    verify_fine_structure_physics()
    print()
    print('Generating plots...')
    generate_all_fine_structure_plots(language='en', show=False)
    generate_all_fine_structure_plots(language='de', show=False)
    print('Done! Check the visualizations folder for output.')
