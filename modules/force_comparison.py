"""
Force Comparison Module for Jugend forscht 2026 Physics Visualization Project
Kraftvergleichsmodul fuer Jugend forscht 2026 Physik-Visualisierungsprojekt

This module visualizes the dramatic difference between gravitational and
electromagnetic forces at different scales, demonstrating why gravity is
negligible at atomic scales but dominant at stellar scales.

Author: Jugend forscht 2026 Project
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterMathtext
import os
from typing import Tuple, Optional, List
from dataclasses import dataclass

from .constants import get_constants, PhysicalConstants
from .color_scheme import COLORS, get_stellar_colors, get_sequence


# Output directory for visualizations
VIS_DIR = os.path.join(os.path.dirname(__file__), '..', 'visualizations')


@dataclass
class ForceCalculation:
    """
    Container for force calculation results.
    Behaelter fuer Kraftberechnungsergebnisse.
    """
    gravitational: float      # Gravitational force [N]
    coulomb: float           # Coulomb force [N]
    ratio: float             # F_grav / F_coulomb
    scale: str               # Description of the scale
    distance: float          # Distance used [m]


def calculate_forces_at_scale(
    constants: PhysicalConstants,
    scale: str,
    custom_params: Optional[dict] = None
) -> ForceCalculation:
    """
    Calculate gravitational and Coulomb forces at different physical scales.
    Berechnet Gravitations- und Coulomb-Kraefte bei verschiedenen physikalischen Skalen.

    Args:
        constants: Physical constants to use
        scale: One of 'atomic', 'molecular', 'human', 'planetary', 'stellar', 'custom'
        custom_params: Dict with 'm1', 'm2', 'q1', 'q2', 'r' for custom calculations

    Returns:
        ForceCalculation with results
    """
    # Define typical parameters for each scale
    scales = {
        'atomic': {
            'm1': constants.m_p,           # Proton mass
            'm2': constants.m_p,           # Proton mass
            'q1': constants.e,             # Elementary charge
            'q2': constants.e,             # Elementary charge
            'r': 1e-15,                    # 1 femtometer (nuclear scale)
            'description': 'Atomic: 2 protons at r=1 fm'
        },
        'molecular': {
            'm1': constants.m_p,
            'm2': constants.m_p,
            'q1': constants.e,
            'q2': constants.e,
            'r': 1e-10,                    # 1 Angstrom (atomic bond scale)
            'description': 'Molecular: 2 protons at r=1 Angstrom'
        },
        'human': {
            'm1': 70,                      # 70 kg human
            'm2': 70,                      # 70 kg human
            'q1': 1e-6,                    # 1 microcoulomb (static charge)
            'q2': 1e-6,
            'r': 1,                        # 1 meter
            'description': 'Human: 2 people (70kg each) at r=1m'
        },
        'planetary': {
            'm1': constants.M_earth,       # Earth mass
            'm2': 1000,                    # 1000 kg object on surface
            'q1': 0,                       # Electrically neutral
            'q2': 0,
            'r': constants.R_earth,        # Distance from Earth center to surface
            'description': 'Planetary: 1000kg object on Earth surface (r=R_earth from center)'
        },
        'stellar': {
            'm1': constants.M_sun,         # Sun mass
            'm2': constants.M_earth,       # Earth mass
            'q1': 0,                       # Electrically neutral
            'q2': 0,
            'r': 1.496e11,                 # 1 AU (Earth-Sun distance)
            'description': 'Stellar: Sun-Earth system at r=1 AU'
        },
        'white_dwarf': {
            'm1': 0.6 * constants.M_sun,   # Typical white dwarf mass
            'm2': constants.m_e,           # Single electron
            'q1': 0,
            'q2': constants.e,
            'r': 0.01 * constants.R_sun,   # White dwarf radius
            'description': 'White dwarf: electron at surface'
        }
    }

    if scale == 'custom' and custom_params:
        params = custom_params
        params['description'] = params.get('description', 'Custom parameters')
    elif scale in scales:
        params = scales[scale]
    else:
        raise ValueError(f"Unknown scale: {scale}. Use one of {list(scales.keys())} or 'custom'")

    # Calculate forces
    F_grav = constants.gravitational_force(params['m1'], params['m2'], params['r'])

    # Handle zero charge case
    if params['q1'] == 0 or params['q2'] == 0:
        F_coul = 0
        ratio = float('inf') if F_grav > 0 else 0
    else:
        F_coul = abs(constants.coulomb_force(params['q1'], params['q2'], params['r']))
        ratio = F_grav / F_coul if F_coul > 0 else float('inf')

    return ForceCalculation(
        gravitational=F_grav,
        coulomb=F_coul,
        ratio=ratio,
        scale=params['description'],
        distance=params['r']
    )


def plot_force_comparison_bar(
    constants: Optional[PhysicalConstants] = None,
    language: str = 'en',
    save: bool = True,
    show: bool = True
) -> plt.Figure:
    """
    Create a log-scale bar chart comparing gravitational and Coulomb forces.
    Erstellt ein logarithmisches Balkendiagramm zum Vergleich von Gravitations- und Coulomb-Kraeften.

    This visualization demonstrates the ~10^36 ratio between forces at atomic scales.

    Args:
        constants: Physical constants (uses standard if None)
        language: 'en' for English, 'de' for German labels
        save: Whether to save the figure
        show: Whether to display the figure

    Returns:
        matplotlib Figure object
    """
    if constants is None:
        constants = get_constants()

    # Calculate forces at atomic scale (proton-proton at 1 fm)
    r = 1e-15  # 1 femtometer
    F_grav = constants.gravitational_force(constants.m_p, constants.m_p, r)
    F_coul = abs(constants.coulomb_force(constants.e, constants.e, r))

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Bar positions and values
    forces = [F_grav, F_coul]
    labels_en = ['Gravitational Force', 'Coulomb Force']
    labels_de = ['Gravitationskraft', 'Coulomb-Kraft']
    labels = labels_de if language == 'de' else labels_en
    colors = [COLORS['gravity'], COLORS['electromagnetic']]

    # Create bars with individual labels for legend
    for label, force, color in zip(labels, forces, colors):
        ax.bar(label, force, color=color, edgecolor='black', linewidth=1.5, label=label)

    # Set log scale
    ax.set_yscale('log')

    # Labels and title
    if language == 'de':
        ax.set_ylabel('Kraft (N)', fontsize=12)
        ax.set_title('Gravitation vs. Elektromagnetische Kraft (Zwei Protonen, Abstand 1 fm)', fontsize=14, fontweight='bold', pad=15)
    else:
        ax.set_ylabel('Force (N)', fontsize=12)
        ax.set_title('Gravitational vs. Electromagnetic Force (Two protons, distance 1 fm)', fontsize=14, fontweight='bold', pad=15)

    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='both', labelsize=11)
    ax.legend(fontsize=11, loc='upper right', bbox_to_anchor=(1.0, -0.15), ncol=2, framealpha=0.7)

    if save:
        os.makedirs(VIS_DIR, exist_ok=True)
        filename = 'force_comparison_bar.png'
        filepath = os.path.join(VIS_DIR, filename)
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")

    if show:
        plt.show()

    return fig


def plot_force_vs_distance(
    constants: Optional[PhysicalConstants] = None,
    language: str = 'en',
    save: bool = True,
    show: bool = True
) -> plt.Figure:
    """
    Plot how gravitational and Coulomb forces vary with distance.
    Zeigt, wie Gravitations- und Coulomb-Kraefte mit dem Abstand variieren.

    Both follow inverse-square law, but with vastly different magnitudes.

    Args:
        constants: Physical constants (uses standard if None)
        language: 'en' for English, 'de' for German labels
        save: Whether to save the figure
        show: Whether to display the figure

    Returns:
        matplotlib Figure object
    """
    if constants is None:
        constants = get_constants()

    # Distance range: 1 fm to 1 nm (atomic to molecular scales)
    r = np.logspace(-15, -9, 100)  # meters

    # Calculate forces (proton-proton)
    F_grav = constants.G * constants.m_p**2 / r**2
    F_coul = constants.k_e * constants.e**2 / r**2

    # Create figure with two subplots stacked vertically
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), gridspec_kw={'hspace': 0.4})

    # Top plot: Both forces on log-log scale
    ax1.loglog(r * 1e15, F_grav, '-', color=COLORS['gravity'], linewidth=2.5, label='Gravitational' if language == 'en' else 'Gravitation')
    ax1.loglog(r * 1e15, F_coul, '-', color=COLORS['electromagnetic'], linewidth=2.5, label='Coulomb')

    if language == 'de':
        ax1.set_xlabel('Abstand (fm)', fontsize=12)
        ax1.set_ylabel('Kraft (N)', fontsize=12)
        ax1.set_title('1. Kräfte vs. Abstand (Proton-Proton)', fontsize=14, fontweight='bold', pad=15)
    else:
        ax1.set_xlabel('Distance (fm)', fontsize=12)
        ax1.set_ylabel('Force (N)', fontsize=12)
        ax1.set_title('1. Forces vs. Distance (Proton-Proton)', fontsize=14, fontweight='bold', pad=15)

    ax1.legend(fontsize=11, loc='upper right', bbox_to_anchor=(1.0, -0.15), ncol=2, framealpha=0.7)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='both', labelsize=11)
    ax1.set_xlim(1, 1e6)

    # Bottom plot: Force ratio vs distance (should be constant)
    ratio = F_coul / F_grav
    ax2.semilogx(r * 1e15, ratio, '-', color=COLORS['primary_blue'], linewidth=2.5, label='F_Coulomb / F_Gravitational' if language == 'en' else 'F_Coulomb / F_Gravitation')

    if language == 'de':
        ax2.set_xlabel('Abstand (fm)', fontsize=12)
        ax2.set_ylabel('F_Coulomb / F_Gravitation', fontsize=12)
        ax2.set_title('2. Kraftverhältnis vs. Abstand', fontsize=14, fontweight='bold', pad=15)
    else:
        ax2.set_xlabel('Distance (fm)', fontsize=12)
        ax2.set_ylabel('F_Coulomb / F_Gravitational', fontsize=12)
        ax2.set_title('2. Force Ratio vs. Distance', fontsize=14, fontweight='bold', pad=15)

    ax2.legend(fontsize=11, loc='upper right', bbox_to_anchor=(1.0, -0.15), framealpha=0.7)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', labelsize=11)
    ax2.set_xlim(1, 1e6)

    # Fix #1: Add key insight connection note
    avg_ratio = np.mean(ratio)
    if language == 'de':
        insight_text = (
            f'SCHLÜSSELERKENNTNIS:\n'
            f'In unserem Universum: F_C/F_G ≈ {avg_ratio:.2e}\n\n'
            f'Unser Szenario (G×10³⁶) macht dieses\n'
            f'Verhältnis ≈ 1—Gravitation und Elektro-\n'
            f'magnetismus werden gleich stark!'
        )
    else:
        insight_text = (
            f'KEY INSIGHT:\n'
            f'In our universe: F_C/F_G ≈ {avg_ratio:.2e}\n\n'
            f'Our scenario (G×10³⁶) makes this\n'
            f'ratio ≈ 1—gravity and electromagnetism\n'
            f'become equally strong at all distances!'
        )
    ax2.text(0.98, 0.98, insight_text, fontsize=9, va='top', ha='right',
             transform=ax2.transAxes,
             bbox=dict(boxstyle='round,pad=0.4', facecolor=COLORS['box_info'],
                      edgecolor=COLORS['primary_blue'], linewidth=2, alpha=0.95),
             color=COLORS['text_dark'], fontweight='bold')

    if save:
        os.makedirs(VIS_DIR, exist_ok=True)
        filename = 'force_vs_distance.png'
        filepath = os.path.join(VIS_DIR, filename)
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")

    if show:
        plt.show()

    return fig


def plot_force_across_scales(
    constants: Optional[PhysicalConstants] = None,
    language: str = 'en',
    save: bool = True,
    show: bool = True
) -> plt.Figure:
    """
    Visualize how gravity becomes dominant at large scales.
    Visualisiert, wie Gravitation bei grossen Skalen dominant wird.

    At atomic scales, EM dominates. At stellar scales, charges cancel
    and gravity dominates because mass always adds.

    Args:
        constants: Physical constants (uses standard if None)
        language: 'en' for English, 'de' for German labels
        save: Whether to save the figure
        show: Whether to display the figure

    Returns:
        matplotlib Figure object
    """
    if constants is None:
        constants = get_constants()

    # Define scales with their properties
    # At larger scales, assume charge neutrality (q_net ~ 0)
    scales_data = [
        {'name_en': 'Nuclear\n(1 fm)', 'name_de': 'Nuklear\n(1 fm)',
         'r': 1e-15, 'm1': constants.m_p, 'm2': constants.m_p,
         'q1': constants.e, 'q2': constants.e, 'neutral': False,
         'desc_en': '2 protons', 'desc_de': '2 Protonen'},

        {'name_en': 'Atomic\n(0.1 nm)', 'name_de': 'Atomar\n(0.1 nm)',
         'r': 1e-10, 'm1': constants.m_p, 'm2': constants.m_e,
         'q1': constants.e, 'q2': constants.e, 'neutral': False,
         'desc_en': 'proton-electron', 'desc_de': 'Proton-Elektron'},

        {'name_en': 'Molecular\n(1 nm)', 'name_de': 'Molekular\n(1 nm)',
         'r': 1e-9, 'm1': 100*constants.m_p, 'm2': 100*constants.m_p,
         'q1': constants.e, 'q2': constants.e, 'neutral': False,
         'desc_en': 'small molecules', 'desc_de': 'kleine Molekuele'},

        {'name_en': 'Human\n(1 m)', 'name_de': 'Mensch\n(1 m)',
         'r': 1, 'm1': 70, 'm2': 70,
         'q1': 1e-6, 'q2': 1e-6, 'neutral': False,
         'desc_en': '2 people (70kg)', 'desc_de': '2 Menschen (70kg)'},

        {'name_en': 'Planet\n(R_Earth)', 'name_de': 'Planet\n(R_Erde)',
         'r': constants.R_earth, 'm1': constants.M_earth, 'm2': 1000,
         'q1': 0, 'q2': 0, 'neutral': True,
         'desc_en': '1000kg on Earth', 'desc_de': '1000kg auf Erde'},

        {'name_en': 'Star\n(1 AU)', 'name_de': 'Stern\n(1 AE)',
         'r': 1.496e11, 'm1': constants.M_sun, 'm2': constants.M_earth,
         'q1': 0, 'q2': 0, 'neutral': True,
         'desc_en': 'Sun-Earth', 'desc_de': 'Sonne-Erde'},
    ]

    # Calculate forces
    F_grav_list = []
    F_coul_list = []
    names = []

    for scale in scales_data:
        F_g = constants.gravitational_force(scale['m1'], scale['m2'], scale['r'])
        F_grav_list.append(F_g)

        if scale['neutral'] or scale['q1'] == 0:
            F_coul_list.append(1e-100)  # Effectively zero, but plottable
        else:
            F_c = abs(constants.coulomb_force(scale['q1'], scale['q2'], scale['r']))
            F_coul_list.append(F_c)

        names.append(scale['name_de'] if language == 'de' else scale['name_en'])

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(names))
    width = 0.35

    # Create bars
    bars1 = ax.bar(x - width/2, F_grav_list, width, label='Gravitational' if language == 'en' else 'Gravitation',
                   color=COLORS['gravity'], edgecolor='black')
    bars2 = ax.bar(x + width/2, F_coul_list, width, label='Coulomb' if language == 'en' else 'Coulomb',
                   color=COLORS['electromagnetic'], edgecolor='black')

    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)

    if language == 'de':
        ax.set_ylabel('Kraft (N) - logarithmische Skala', fontsize=12)
        ax.set_title('Kräftevergleich über verschiedene Skalen', fontsize=14, fontweight='bold', pad=15)
    else:
        ax.set_ylabel('Force (N) - logarithmic scale', fontsize=12)
        ax.set_title('Force Comparison Across Different Scales', fontsize=14, fontweight='bold', pad=15)

    ax.legend(fontsize=11, loc='upper right', bbox_to_anchor=(1.0, -0.15), ncol=2, framealpha=0.7)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='both', labelsize=11)

    if save:
        os.makedirs(VIS_DIR, exist_ok=True)
        filename = 'force_across_scales.png'
        filepath = os.path.join(VIS_DIR, filename)
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")

    if show:
        plt.show()

    return fig


def plot_scaled_universe_comparison(
    hbar_scale: float = 0.1,
    G_scale: float = 1.0,
    language: str = 'en',
    save: bool = True,
    show: bool = True
) -> plt.Figure:
    """
    Compare forces in standard vs scaled universe AT THEIR RESPECTIVE ATOMIC SCALES.
    Vergleicht Kraefte im Standard- vs. skalierten Universum BEI IHREN JEWEILIGEN ATOMAREN SKALEN.

    IMPORTANT: When hbar changes, the Bohr radius changes. Atoms in the alternative
    universe are smaller. The meaningful comparison is at the Bohr radius of each
    universe, not at a fixed distance.

    When hbar decreases by factor X:
    - Bohr radius decreases by factor X (atoms shrink)
    - At the NEW atomic scale, forces at Bohr radius change
    - alpha_G increases by factor X (gravity relatively stronger vs quantum effects)

    NOTE (Fix #2): This page shows what happens if ONLY ℏ is changed (without G).
    This is a CONTRAST scenario to demonstrate why both constants must be scaled
    together in the main essay scenario (G×10³⁶, ℏ×10¹⁸).

    Args:
        hbar_scale: Scaling factor for hbar
        G_scale: Scaling factor for G
        language: 'en' for English, 'de' for German
        save: Whether to save the figure
        show: Whether to display the figure

    Returns:
        matplotlib Figure object
    """
    std = get_constants()
    alt = get_constants(hbar_scale=hbar_scale, G_scale=G_scale)

    # KEY INSIGHT: Compare at each universe's Bohr radius (atomic scale)
    r_std = std.a_0  # Standard Bohr radius ~5.29e-11 m
    r_alt = alt.a_0  # Alternative Bohr radius (smaller if hbar is smaller)

    # Standard universe forces at standard Bohr radius
    F_grav_std = std.gravitational_force(std.m_p, std.m_p, r_std)
    F_coul_std = abs(std.coulomb_force(std.e, std.e, r_std))
    ratio_std = F_coul_std / F_grav_std

    # Alternative universe forces at alternative Bohr radius
    # Note: m_p and e don't change, but r_alt is smaller
    F_grav_alt = alt.gravitational_force(alt.m_p, alt.m_p, r_alt)
    F_coul_alt = abs(alt.coulomb_force(alt.e, alt.e, r_alt))
    ratio_alt = F_coul_alt / F_grav_alt

    # Create figure with two subplots stacked vertically (extra height for disclaimer)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14), gridspec_kw={'hspace': 0.5})

    # Fix #2: Add IMPORTANT DISCLAIMER at the top of the figure
    if language == 'de':
        disclaimer_text = (
            '⚠️ WICHTIGER KONTEXT ⚠️\n'
            f'Diese Seite zeigt ein hypothetisches Universum, in dem NUR ℏ verändert wird (auf {hbar_scale}×),\n'
            'während G unverändert bleibt. Dies ist NICHT unser Haupt-Essay-Szenario (G×10³⁶ UND ℏ×10¹⁸).\n\n'
            'ZWECK: Diese Seite zeigt, warum die ℏ-Skalierung notwendig ist—ohne sie würde\n'
            'die Gravitation vollständig dominieren und Atome würden kollabieren.'
        )
    else:
        disclaimer_text = (
            '⚠️ IMPORTANT CONTEXT ⚠️\n'
            f'This page shows a hypothetical universe where ONLY ℏ is changed (to {hbar_scale}×),\n'
            'while G remains unchanged. This is NOT our main essay scenario (G×10³⁶ AND ℏ×10¹⁸).\n\n'
            'PURPOSE: This page demonstrates why the ℏ scaling is necessary—without it,\n'
            'gravity would completely dominate and atoms would collapse.'
        )

    fig.text(0.5, 0.97, disclaimer_text, fontsize=10, va='top', ha='center',
             bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['box_warning'],
                      edgecolor=COLORS['highlight'], linewidth=2, alpha=0.95),
             color=COLORS['primary_amber'], fontweight='bold',
             transform=fig.transFigure)

    # Top: Standard Universe
    forces_std = [F_grav_std, F_coul_std]
    labels = ['Gravity', 'Coulomb'] if language == 'en' else ['Gravitation', 'Coulomb']
    colors = [COLORS['gravity'], COLORS['electromagnetic']]

    for lbl, force, color in zip(labels, forces_std, colors):
        ax1.bar(lbl, force, color=color, edgecolor='black', linewidth=1.5, label=lbl)
    ax1.set_yscale('log')

    if language == 'de':
        ax1.set_title('1. Standarduniversum (bei Bohr-Radius)', fontsize=14, fontweight='bold', pad=15)
        ax1.set_ylabel('Kraft (N)', fontsize=12)
    else:
        ax1.set_title('1. Standard Universe (at Bohr radius)', fontsize=14, fontweight='bold', pad=15)
        ax1.set_ylabel('Force (N)', fontsize=12)

    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='both', labelsize=11)
    ax1.legend(fontsize=11, loc='upper right', bbox_to_anchor=(1.0, -0.15), ncol=2, framealpha=0.7)

    # Bottom: Alternative Universe
    forces_alt = [F_grav_alt, F_coul_alt]

    for lbl, force, color in zip(labels, forces_alt, colors):
        ax2.bar(lbl, force, color=color, edgecolor='black', linewidth=1.5, label=lbl)
    ax2.set_yscale('log')

    scale_str = f'ℏ x{hbar_scale}, G x{G_scale}'
    if language == 'de':
        ax2.set_title(f'2. Alternatives Universum ({scale_str})', fontsize=14, fontweight='bold', pad=15)
        ax2.set_ylabel('Kraft (N)', fontsize=12)
    else:
        ax2.set_title(f'2. Alternative Universe ({scale_str})', fontsize=14, fontweight='bold', pad=15)
        ax2.set_ylabel('Force (N)', fontsize=12)

    ax2.grid(True, alpha=0.3, axis='y')
    ax2.tick_params(axis='both', labelsize=11)
    ax2.legend(fontsize=11, loc='upper right', bbox_to_anchor=(1.0, -0.15), ncol=2, framealpha=0.7)

    # Make y-axis limits the same for comparison
    all_forces = forces_std + forces_alt
    y_min = min(all_forces) / 10
    y_max = max(all_forces) * 100
    ax1.set_ylim(y_min, y_max)
    ax2.set_ylim(y_min, y_max)

    # Fix #2: Add COMPARISON with main essay scenario at the bottom
    if language == 'de':
        comparison_text = (
            'VERGLEICH MIT UNSEREM SZENARIO:\n'
            f'• Nur ℏ={hbar_scale}: Atome {int(1/hbar_scale**2)}× kleiner, Gravitation dominiert → KOLLAPS\n'
            '• Unser Szenario (G×10³⁶, ℏ×10¹⁸): Gleichgewicht erhalten → NEUES GLEICHGEWICHT MÖGLICH\n\n'
            'Schlüsselerkenntnis: P_grav/P_Pauli ∝ G/ℏ² muss konstant bleiben für Stabilität!'
        )
    else:
        comparison_text = (
            'COMPARISON WITH OUR SCENARIO:\n'
            f'• ℏ={hbar_scale} only: Atoms {int(1/hbar_scale**2)}× smaller, gravity dominates → COLLAPSE\n'
            '• Our scenario (G×10³⁶, ℏ×10¹⁸): Balance maintained → NEW EQUILIBRIUM POSSIBLE\n\n'
            'Key insight: P_grav/P_Pauli ∝ G/ℏ² must remain constant for stability!'
        )

    fig.text(0.5, 0.02, comparison_text, fontsize=10, va='bottom', ha='center',
             bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['box_success'],
                      edgecolor=COLORS['standard'], linewidth=2, alpha=0.95),
             color=COLORS['equilibrium'], fontweight='bold',
             transform=fig.transFigure)

    if save:
        os.makedirs(VIS_DIR, exist_ok=True)
        filename = f'force_scaled_hbar{hbar_scale}_G{G_scale}.png'
        filepath = os.path.join(VIS_DIR, filename)
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")

    if show:
        plt.show()

    return fig


def generate_all_force_plots(language: str = 'en', show: bool = False) -> List[plt.Figure]:
    """
    Generate all force comparison visualizations.
    Erzeugt alle Kraftvergleichs-Visualisierungen.

    Args:
        language: 'en' for English, 'de' for German
        show: Whether to display plots

    Returns:
        List of matplotlib Figure objects
    """
    figures = []

    print("Generating force comparison visualizations...")
    print("=" * 50)

    # 1. Bar chart comparison
    print("1. Force comparison bar chart...")
    figures.append(plot_force_comparison_bar(language=language, show=show))

    # 2. Force vs distance
    print("2. Force vs distance plot...")
    figures.append(plot_force_vs_distance(language=language, show=show))

    # 3. Forces across scales
    print("3. Forces across different scales...")
    figures.append(plot_force_across_scales(language=language, show=show))

    # 4. Scaled universe comparison
    print("4. Scaled universe comparison...")
    figures.append(plot_scaled_universe_comparison(hbar_scale=0.1, language=language, show=show))

    print("=" * 50)
    print(f"Generated {len(figures)} visualizations in {VIS_DIR}")

    return figures


def verify_physics_calculations():
    """
    Verify that all physics calculations are correct.
    Print detailed verification of formulas and values.
    """
    print("=" * 70)
    print("PHYSICS VERIFICATION")
    print("=" * 70)

    c = get_constants()

    # 1. Verify force ratio at atomic scale
    print("\n1. FORCE RATIO (Proton-Proton at 1 fm)")
    print("-" * 50)
    r = 1e-15  # 1 fm

    F_grav = c.G * c.m_p**2 / r**2
    F_coul = c.k_e * c.e**2 / r**2
    ratio = F_coul / F_grav

    print(f"   Distance r = {r:.0e} m (1 femtometer)")
    print(f"   F_grav = G * m_p^2 / r^2")
    print(f"         = {c.G:.4e} * ({c.m_p:.4e})^2 / ({r:.0e})^2")
    print(f"         = {F_grav:.4e} N")
    print(f"   F_coul = k_e * e^2 / r^2")
    print(f"         = {c.k_e:.4e} * ({c.e:.4e})^2 / ({r:.0e})^2")
    print(f"         = {F_coul:.4e} N")
    print(f"   Ratio (Coulomb/Gravity) = {ratio:.4e}")
    print(f"   Expected: ~10^36 (CHECK: {'PASS' if 1e35 < ratio < 1e37 else 'FAIL'})")

    # 2. Verify alpha_G
    print("\n2. GRAVITATIONAL COUPLING CONSTANT (alpha_G)")
    print("-" * 50)
    alpha_G = (c.G * c.m_p**2) / (c.hbar * c.c)
    print(f"   alpha_G = (G * m_p^2) / (hbar * c)")
    print(f"          = ({c.G:.4e} * {c.m_p:.4e}^2) / ({c.hbar:.4e} * {c.c:.4e})")
    print(f"          = {alpha_G:.4e}")
    print(f"   Expected: ~5.9e-39 (CHECK: {'PASS' if 5e-39 < alpha_G < 7e-39 else 'FAIL'})")

    # 3. Verify Bohr radius
    print("\n3. BOHR RADIUS")
    print("-" * 50)
    import math
    a_0_calc = (4 * math.pi * c.epsilon_0 * c.hbar**2) / (c.m_e * c.e**2)
    print(f"   a_0 = (4*pi*epsilon_0*hbar^2) / (m_e * e^2)")
    print(f"      = {a_0_calc:.4e} m")
    print(f"   Expected: 5.29e-11 m (CHECK: {'PASS' if abs(a_0_calc - 5.29e-11)/5.29e-11 < 0.01 else 'FAIL'})")

    # 4. Verify scaling behavior
    print("\n4. SCALING VERIFICATION (hbar x0.1)")
    print("-" * 50)
    alt = get_constants(hbar_scale=0.1)
    print(f"   Standard hbar: {c.hbar:.4e}")
    print(f"   Scaled hbar:   {alt.hbar:.4e} (should be 0.1x)")
    print(f"   Standard a_0:  {c.a_0:.4e} m")
    print(f"   Scaled a_0:    {alt.a_0:.4e} m (should be 0.01x = hbar_scale^2)")
    print(f"   Actually ratio: {alt.a_0/c.a_0:.4f}")
    print(f"   Standard alpha_G: {c.alpha_G:.4e}")
    print(f"   Scaled alpha_G:   {alt.alpha_G:.4e} (should be 10x)")
    print(f"   Actually ratio: {alt.alpha_G/c.alpha_G:.2f}")

    # 5. Verify Earth surface gravity
    print("\n5. EARTH SURFACE GRAVITY")
    print("-" * 50)
    g = c.G * c.M_earth / c.R_earth**2
    print(f"   g = G * M_earth / R_earth^2")
    print(f"     = {c.G:.4e} * {c.M_earth:.4e} / ({c.R_earth:.4e})^2")
    print(f"     = {g:.4f} m/s^2")
    print(f"   Expected: ~9.8 m/s^2 (CHECK: {'PASS' if 9.7 < g < 9.9 else 'FAIL'})")

    # 6. Verify Sun-Earth force
    print("\n6. SUN-EARTH GRAVITATIONAL FORCE")
    print("-" * 50)
    r_AU = 1.496e11
    F_sun_earth = c.G * c.M_sun * c.M_earth / r_AU**2
    print(f"   F = G * M_sun * M_earth / r^2")
    print(f"     = {c.G:.4e} * {c.M_sun:.4e} * {c.M_earth:.4e} / ({r_AU:.4e})^2")
    print(f"     = {F_sun_earth:.4e} N")
    print(f"   Expected: ~3.5e22 N (CHECK: {'PASS' if 3e22 < F_sun_earth < 4e22 else 'FAIL'})")

    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    # Demo: Generate all plots and verify physics
    print("=" * 60)
    print("Force Comparison Module - Jugend forscht 2026")
    print("=" * 60)

    # Verify physics first
    verify_physics_calculations()

    # Print force calculations at different scales
    constants = get_constants()

    print("\n\nForce calculations at different scales:")
    print("-" * 60)

    for scale in ['atomic', 'molecular', 'human', 'planetary', 'stellar']:
        result = calculate_forces_at_scale(constants, scale)
        print(f"\n{result.scale}:")
        print(f"  Distance: {result.distance:.2e} m")
        print(f"  F_gravity: {result.gravitational:.2e} N")
        print(f"  F_coulomb: {result.coulomb:.2e} N")
        if result.ratio != float('inf'):
            print(f"  Ratio (G/C): {result.ratio:.2e}")
        else:
            print(f"  Ratio: Gravity only (no charge)")

    print("\n" + "=" * 60)
    print("Generating visualizations...")
    print("=" * 60)

    # Generate all plots (don't show, just save)
    generate_all_force_plots(language='en', show=False)

    print("\nDone! Check the 'visualizations' folder for output.")
