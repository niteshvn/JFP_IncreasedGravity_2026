"""
Heisenberg Uncertainty Principle Module for Jugend forscht 2026 Physics Visualization Project
Heisenberg-Unschaerferelation-Modul fuer Jugend forscht 2026 Physik-Visualisierungsprojekt

This module visualizes the connection between:
- Heisenberg uncertainty principle: Δx·Δp ≥ ℏ/2
- Confinement and momentum uncertainty
- Degeneracy pressure in compact objects (white dwarfs, neutron stars)

Key insight from essay Section 3.3:
"Wenn Δx (Position) kleiner wird, muss der Impuls (Δp) zunehmen.
 Die Elektronen beginnen sich mit unglaublicher Geschwindigkeit zu bewegen.
 Diese 'Unruhe' erzeugt den nach außen gerichteten Druck."

Author: Jugend forscht 2026 Project
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.collections import PatchCollection
import os
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass

from .constants import get_constants, PhysicalConstants
from .color_scheme import COLORS, get_sequence


# Output directory for visualizations
VIS_DIR = os.path.join(os.path.dirname(__file__), '..', 'visualizations')


@dataclass
class UncertaintyProperties:
    """
    Container for uncertainty-related properties.
    Behaelter fuer Unschaerfe-Eigenschaften.
    """
    delta_x: float              # Position uncertainty [m]
    delta_p_min: float          # Minimum momentum uncertainty [kg·m/s]
    delta_v_min: float          # Minimum velocity uncertainty [m/s]
    kinetic_energy_min: float   # Minimum kinetic energy [J]
    degeneracy_pressure: float  # Resulting degeneracy pressure [Pa]


def minimum_momentum_uncertainty(delta_x: float, constants: PhysicalConstants) -> float:
    """
    Calculate minimum momentum uncertainty from Heisenberg uncertainty principle.
    Berechnet minimale Impulsunschaerfe aus der Heisenberg-Unschaerferelation.

    Formula: Δp ≥ ℏ/(2·Δx)

    Args:
        delta_x: Position uncertainty [m]
        constants: Physical constants

    Returns:
        Minimum momentum uncertainty [kg·m/s]
    """
    return constants.hbar / (2 * delta_x)


def confinement_velocity(delta_x: float, mass: float, constants: PhysicalConstants) -> float:
    """
    Calculate minimum velocity uncertainty for a confined particle.
    Berechnet minimale Geschwindigkeitsunschaerfe fuer ein eingesperrtes Teilchen.

    Formula: Δv = Δp/m = ℏ/(2·m·Δx)

    Args:
        delta_x: Position uncertainty / confinement size [m]
        mass: Particle mass [kg]
        constants: Physical constants

    Returns:
        Minimum velocity uncertainty [m/s]
    """
    delta_p = minimum_momentum_uncertainty(delta_x, constants)
    return delta_p / mass


def confinement_kinetic_energy(delta_x: float, mass: float, constants: PhysicalConstants) -> float:
    """
    Calculate minimum kinetic energy from confinement (zero-point energy).
    Berechnet minimale kinetische Energie aus Einschluss (Nullpunktsenergie).

    Formula: E_min = (Δp)²/(2m) = ℏ²/(8·m·Δx²)

    Args:
        delta_x: Confinement size [m]
        mass: Particle mass [kg]
        constants: Physical constants

    Returns:
        Minimum kinetic energy [J]
    """
    delta_p = minimum_momentum_uncertainty(delta_x, constants)
    return delta_p**2 / (2 * mass)


def degeneracy_pressure_from_uncertainty(
    density: float,
    particle_mass: float,
    constants: PhysicalConstants,
    relativistic: bool = False
) -> float:
    """
    Calculate degeneracy pressure using uncertainty principle reasoning.
    Berechnet Entartungsdruck mit Unschaerferelation-Argumentation.

    Non-relativistic: P = (ℏ²/m) × n^(5/3)
    where n = number density ∝ ρ/m_p

    The key insight: Δx ~ n^(-1/3) (average spacing)
    Δp ~ ℏ/Δx ~ ℏ·n^(1/3)
    P ~ n × E_kin ~ n × (Δp)²/m ~ ℏ² × n^(5/3) / m

    Args:
        density: Mass density [kg/m³]
        particle_mass: Mass of degenerate particle (electron or neutron) [kg]
        constants: Physical constants
        relativistic: If True, use relativistic formula

    Returns:
        Degeneracy pressure [Pa]
    """
    # Number density of nucleons (baryons)
    n_nucleons = density / constants.m_p

    # For electrons, roughly one electron per two nucleons
    # For neutrons, roughly all particles are neutrons
    if particle_mass < constants.m_n:
        # Electron case
        n_particles = n_nucleons / 2  # Assume electron fraction Y_e ~ 0.5
    else:
        # Neutron case
        n_particles = n_nucleons

    if relativistic:
        # Ultra-relativistic: P ∝ ℏc × n^(4/3)
        K = (constants.hbar * constants.c / 4) * (3 / np.pi)**(1/3)
        return K * n_particles**(4/3)
    else:
        # Non-relativistic: P = (ℏ²/m) × (3π²)^(2/3) / 5 × n^(5/3)
        K = (constants.hbar**2 / (5 * particle_mass)) * (3 * np.pi**2)**(2/3)
        return K * n_particles**(5/3)


def calculate_uncertainty_properties(
    delta_x: float,
    particle_mass: float,
    density: float,
    constants: PhysicalConstants
) -> UncertaintyProperties:
    """
    Calculate all uncertainty-related properties for given confinement.
    Berechnet alle Unschaerfe-Eigenschaften fuer gegebenen Einschluss.

    Args:
        delta_x: Confinement length scale [m]
        particle_mass: Particle mass [kg]
        density: Material density [kg/m³]
        constants: Physical constants

    Returns:
        UncertaintyProperties object
    """
    delta_p = minimum_momentum_uncertainty(delta_x, constants)
    delta_v = confinement_velocity(delta_x, particle_mass, constants)
    E_kin = confinement_kinetic_energy(delta_x, particle_mass, constants)
    P_deg = degeneracy_pressure_from_uncertainty(density, particle_mass, constants)

    return UncertaintyProperties(
        delta_x=delta_x,
        delta_p_min=delta_p,
        delta_v_min=delta_v,
        kinetic_energy_min=E_kin,
        degeneracy_pressure=P_deg
    )


def plot_uncertainty_principle_basic(
    constants: Optional[PhysicalConstants] = None,
    language: str = 'en',
    save: bool = True,
    show: bool = True
) -> plt.Figure:
    """
    Plot the basic Heisenberg uncertainty relation Δx·Δp ≥ ℏ/2.
    Zeigt die grundlegende Heisenberg-Unschaerferelation.

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

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Range of position uncertainties
    delta_x = np.logspace(-15, -9, 100)  # From nuclear to atomic scales

    # Calculate minimum momentum uncertainty
    delta_p_min = constants.hbar / (2 * delta_x)

    # Plot 1: Δp vs Δx (log-log)
    ax1.loglog(delta_x * 1e12, delta_p_min, '-', color=COLORS['primary_blue'],
               linewidth=2.5, label='Δp_min = ℏ/(2Δx)')

    # Mark important scales
    # Atomic scale: Δx ~ Bohr radius
    a_0 = constants.a_0
    delta_p_atom = constants.hbar / (2 * a_0)
    ax1.plot(a_0 * 1e12, delta_p_atom, 'o', color=COLORS['standard'], markersize=12,
             label=f'Atom (a_0 = {a_0*1e12:.1f} pm)' if language == 'en'
                   else f'Atom (a_0 = {a_0*1e12:.1f} pm)')

    # White dwarf scale: Δx ~ 1000 fm (electron spacing)
    delta_x_wd = 1e-12  # ~1 pm
    delta_p_wd = constants.hbar / (2 * delta_x_wd)
    ax1.plot(delta_x_wd * 1e12, delta_p_wd, 's', color=COLORS['scaled'], markersize=10,
             label='White dwarf e⁻' if language == 'en' else 'Weißer Zwerg e⁻')

    # Neutron star scale: Δx ~ 1 fm
    delta_x_ns = 1e-15  # 1 fm
    delta_p_ns = constants.hbar / (2 * delta_x_ns)
    ax1.plot(delta_x_ns * 1e12, delta_p_ns, '^', color=COLORS['quantum'], markersize=10,
             label='Neutron star n' if language == 'en' else 'Neutronenstern n')

    # Shade the forbidden region (Δx·Δp < ℏ/2)
    ax1.fill_between(delta_x * 1e12, delta_p_min / 100, delta_p_min,
                     alpha=0.2, color=COLORS['quantum'],
                     label='Forbidden region' if language == 'en' else 'Verbotener Bereich')

    if language == 'de':
        ax1.set_xlabel('Positionsunsicherheit Δx (pm)', fontsize=12)
        ax1.set_ylabel('Minimalimpuls Δp_min (kg·m/s)', fontsize=12)
        ax1.set_title('Heisenberg-Unschärferelation: Δx·Δp ≥ ℏ/2', fontsize=14, fontweight='bold')
    else:
        ax1.set_xlabel('Position uncertainty Δx (pm)', fontsize=12)
        ax1.set_ylabel('Minimum momentum Δp_min (kg·m/s)', fontsize=12)
        ax1.set_title('Heisenberg Uncertainty Relation: Δx·Δp ≥ ℏ/2', fontsize=14, fontweight='bold')

    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(True, alpha=0.3, which='both')

    # Plot 2: Phase space visualization
    # Show that allowed states form a hyperbolic boundary
    delta_x_vis = np.linspace(0.1, 2, 100)
    delta_p_vis = 1 / delta_x_vis  # Normalized units

    ax2.fill_between(delta_x_vis, delta_p_vis, 5, alpha=0.3, color=COLORS['primary_blue'],
                     label='Allowed states' if language == 'en' else 'Erlaubte Zustände')
    ax2.fill_between(delta_x_vis, 0, delta_p_vis, alpha=0.2, color=COLORS['scaled'],
                     label='Forbidden' if language == 'en' else 'Verboten')
    ax2.plot(delta_x_vis, delta_p_vis, '-', color=COLORS['primary_blue'], linewidth=3,
             label='Δx·Δp = ℏ/2')

    # Mark the minimum uncertainty state
    ax2.plot(1, 1, 'o', color=COLORS['standard'], markersize=15, zorder=5,
             label='Minimum uncertainty' if language == 'en' else 'Minimale Unschärfe')

    # Add annotation
    if language == 'de':
        ax2.annotate('Minimale\nUnschärfe', xy=(1, 1), xytext=(1.5, 2),
                    fontsize=10, ha='center',
                    arrowprops=dict(arrowstyle='->', color=COLORS['text_dark']))
    else:
        ax2.annotate('Minimum\nuncertainty', xy=(1, 1), xytext=(1.5, 2),
                    fontsize=10, ha='center',
                    arrowprops=dict(arrowstyle='->', color=COLORS['text_dark']))

    ax2.set_xlim(0, 2.5)
    ax2.set_ylim(0, 5)

    if language == 'de':
        ax2.set_xlabel('Δx (normierte Einheiten)', fontsize=12)
        ax2.set_ylabel('Δp (normierte Einheiten)', fontsize=12)
        ax2.set_title('Phasenraum-Darstellung', fontsize=14, fontweight='bold')
    else:
        ax2.set_xlabel('Δx (normalized units)', fontsize=12)
        ax2.set_ylabel('Δp (normalized units)', fontsize=12)
        ax2.set_title('Phase Space Representation', fontsize=14, fontweight='bold')

    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save:
        os.makedirs(VIS_DIR, exist_ok=True)
        suffix = '_de' if language == 'de' else ''
        filepath = os.path.join(VIS_DIR, f'heisenberg_uncertainty_basic{suffix}.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")

    if show:
        plt.show()

    return fig


def plot_confinement_velocity(
    constants: Optional[PhysicalConstants] = None,
    language: str = 'en',
    save: bool = True,
    show: bool = True
) -> plt.Figure:
    """
    Plot how confinement leads to high velocities (the 'restlessness' from the essay).
    Zeigt wie Einschluss zu hohen Geschwindigkeiten fuehrt (die 'Unruhe' aus dem Essay).

    Essay quote: "Die Elektronen beginnen sich mit unglaublicher Geschwindigkeit zu bewegen"

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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Range of confinement sizes
    delta_x = np.logspace(-15, -9, 100)

    # Calculate electron velocity for each confinement
    v_electron = confinement_velocity(delta_x, constants.m_e, constants)

    # Calculate neutron velocity for comparison
    v_neutron = confinement_velocity(delta_x, constants.m_n, constants)

    # Plot 1: Velocity vs confinement
    ax1.loglog(delta_x * 1e15, v_electron / constants.c, '-', color=COLORS['primary_blue'],
               linewidth=2.5, label='Electron' if language == 'en' else 'Elektron')
    ax1.loglog(delta_x * 1e15, v_neutron / constants.c, '--', color=COLORS['scaled'],
               linewidth=2.5, label='Neutron')

    # Mark relativistic threshold (v = 0.1c)
    ax1.axhline(y=0.1, color=COLORS['quantum'], linestyle=':', linewidth=2, alpha=0.7,
                label='v = 0.1c (relativistic)' if language == 'en' else 'v = 0.1c (relativistisch)')

    # Mark speed of light limit
    ax1.axhline(y=1.0, color=COLORS['scaled'], linestyle='-', linewidth=2, alpha=0.5,
                label='c (speed limit)' if language == 'en' else 'c (Lichtgeschw.)')

    # Mark atomic scale
    a_0 = constants.a_0
    v_atom = confinement_velocity(a_0, constants.m_e, constants)
    ax1.plot(a_0 * 1e15, v_atom / constants.c, 'o', color=COLORS['standard'], markersize=12,
             label=f'Atom (v/c ≈ α ≈ 1/137)')

    # Mark white dwarf scale
    delta_x_wd = 1e-12
    v_wd = confinement_velocity(delta_x_wd, constants.m_e, constants)
    ax1.plot(delta_x_wd * 1e15, v_wd / constants.c, 's', color=COLORS['scaled'], markersize=10,
             label=f'White dwarf e⁻' if language == 'en' else f'Weißer Zwerg e⁻')

    # Mark neutron star scale
    delta_x_ns = 1e-15
    v_ns_n = confinement_velocity(delta_x_ns, constants.m_n, constants)
    ax1.plot(delta_x_ns * 1e15, v_ns_n / constants.c, '^', color=COLORS['quantum'], markersize=10,
             label='Neutron star n' if language == 'en' else 'Neutronenstern n')

    if language == 'de':
        ax1.set_xlabel('Einschlussgroesse Δx (fm)', fontsize=12)
        ax1.set_ylabel('Geschwindigkeit v/c', fontsize=12)
        ax1.set_title('„Unruhe" durch Einschluss: v ∝ ℏ/(m·Δx)', fontsize=14, fontweight='bold')
    else:
        ax1.set_xlabel('Confinement size Δx (fm)', fontsize=12)
        ax1.set_ylabel('Velocity v/c', fontsize=12)
        ax1.set_title('"Restlessness" from Confinement: v ∝ ℏ/(m·Δx)', fontsize=14, fontweight='bold')

    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.set_ylim(1e-4, 10)

    # Plot 2: Kinetic energy vs confinement
    E_electron = confinement_kinetic_energy(delta_x, constants.m_e, constants)
    E_neutron = confinement_kinetic_energy(delta_x, constants.m_n, constants)

    # Convert to eV
    E_electron_eV = E_electron / constants.e
    E_neutron_eV = E_neutron / constants.e

    ax2.loglog(delta_x * 1e15, E_electron_eV, '-', color=COLORS['primary_blue'],
               linewidth=2.5, label='Electron' if language == 'en' else 'Elektron')
    ax2.loglog(delta_x * 1e15, E_neutron_eV, '--', color=COLORS['scaled'],
               linewidth=2.5, label='Neutron')

    # Mark rest mass energies
    E_e_rest = constants.m_e * constants.c**2 / constants.e  # ~0.511 MeV
    E_n_rest = constants.m_n * constants.c**2 / constants.e  # ~939 MeV
    ax2.axhline(y=E_e_rest, color=COLORS['primary_blue'], linestyle=':', linewidth=1.5, alpha=0.5,
                label=f'm_e c² = {E_e_rest/1e6:.2f} MeV')
    ax2.axhline(y=E_n_rest, color=COLORS['scaled'], linestyle=':', linewidth=1.5, alpha=0.5,
                label=f'm_n c² = {E_n_rest/1e6:.0f} MeV')

    # Mark atomic ionization energy
    E_ion = 13.6  # eV
    ax2.axhline(y=E_ion, color=COLORS['standard'], linestyle='--', linewidth=1.5, alpha=0.7,
                label='H ionization (13.6 eV)')

    if language == 'de':
        ax2.set_xlabel('Einschlussgroesse Δx (fm)', fontsize=12)
        ax2.set_ylabel('Kinetische Energie (eV)', fontsize=12)
        ax2.set_title('Nullpunktsenergie: E ∝ ℏ²/(m·Δx²)', fontsize=14, fontweight='bold')
    else:
        ax2.set_xlabel('Confinement size Δx (fm)', fontsize=12)
        ax2.set_ylabel('Kinetic energy (eV)', fontsize=12)
        ax2.set_title('Zero-point Energy: E ∝ ℏ²/(m·Δx²)', fontsize=14, fontweight='bold')

    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()

    if save:
        os.makedirs(VIS_DIR, exist_ok=True)
        suffix = '_de' if language == 'de' else ''
        filepath = os.path.join(VIS_DIR, f'confinement_velocity{suffix}.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")

    if show:
        plt.show()

    return fig


def plot_uncertainty_to_pressure(
    constants: Optional[PhysicalConstants] = None,
    language: str = 'en',
    save: bool = True,
    show: bool = True
) -> plt.Figure:
    """
    Show the connection: Uncertainty → Momentum → Kinetic Energy → Pressure.
    Zeigt die Verbindung: Unschaerfe → Impuls → kinetische Energie → Druck.

    Essay quote: "Diese 'Unruhe' erzeugt den nach außen gerichteten Druck"

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

    # Density range for white dwarfs and neutron stars
    rho = np.logspace(6, 18, 100)  # kg/m³

    # Calculate electron degeneracy pressure (white dwarfs)
    P_e_nr = np.array([degeneracy_pressure_from_uncertainty(r, constants.m_e, constants, False)
                       for r in rho])
    P_e_r = np.array([degeneracy_pressure_from_uncertainty(r, constants.m_e, constants, True)
                      for r in rho])

    # Calculate neutron degeneracy pressure (neutron stars)
    P_n_nr = np.array([degeneracy_pressure_from_uncertainty(r, constants.m_n, constants, False)
                       for r in rho])
    P_n_r = np.array([degeneracy_pressure_from_uncertainty(r, constants.m_n, constants, True)
                      for r in rho])

    # Plot 1: Electron degeneracy pressure
    ax1 = axes[0, 0]
    ax1.loglog(rho, P_e_nr, '-', color=COLORS['primary_blue'], linewidth=2.5,
               label='Non-relativistic (P ∝ ρ^{5/3})' if language == 'en'
                     else 'Nicht-relativistisch (P ∝ ρ^{5/3})')
    ax1.loglog(rho, P_e_r, '--', color=COLORS['scaled'], linewidth=2.5,
               label='Ultra-relativistic (P ∝ ρ^{4/3})' if language == 'en'
                     else 'Ultra-relativistisch (P ∝ ρ^{4/3})')

    # Mark white dwarf density range
    ax1.axvspan(1e9, 1e11, alpha=0.2, color=COLORS['standard'],
                label='White dwarf range' if language == 'en' else 'Weißer-Zwerg-Bereich')

    if language == 'de':
        ax1.set_xlabel('Dichte ρ (kg/m³)', fontsize=11)
        ax1.set_ylabel('Elektronendruck P (Pa)', fontsize=11)
        ax1.set_title('Elektronen-Entartungsdruck', fontsize=13, fontweight='bold')
    else:
        ax1.set_xlabel('Density ρ (kg/m³)', fontsize=11)
        ax1.set_ylabel('Electron pressure P (Pa)', fontsize=11)
        ax1.set_title('Electron Degeneracy Pressure', fontsize=13, fontweight='bold')

    ax1.legend(fontsize=9, loc='upper left')
    ax1.grid(True, alpha=0.3, which='both')

    # Plot 2: Neutron degeneracy pressure
    ax2 = axes[0, 1]
    ax2.loglog(rho, P_n_nr, '-', color=COLORS['quantum'], linewidth=2.5,
               label='Non-relativistic (P ∝ ρ^{5/3})' if language == 'en'
                     else 'Nicht-relativistisch (P ∝ ρ^{5/3})')
    ax2.loglog(rho, P_n_r, '--', color=COLORS['scaled'], linewidth=2.5,
               label='Ultra-relativistic (P ∝ ρ^{4/3})' if language == 'en'
                     else 'Ultra-relativistisch (P ∝ ρ^{4/3})')

    # Mark neutron star density range
    ax2.axvspan(1e14, 1e18, alpha=0.2, color=COLORS['quantum'],
                label='Neutron star range' if language == 'en' else 'Neutronenstern-Bereich')

    if language == 'de':
        ax2.set_xlabel('Dichte ρ (kg/m³)', fontsize=11)
        ax2.set_ylabel('Neutronendruck P (Pa)', fontsize=11)
        ax2.set_title('Neutronen-Entartungsdruck', fontsize=13, fontweight='bold')
    else:
        ax2.set_xlabel('Density ρ (kg/m³)', fontsize=11)
        ax2.set_ylabel('Neutron pressure P (Pa)', fontsize=11)
        ax2.set_title('Neutron Degeneracy Pressure', fontsize=13, fontweight='bold')

    ax2.legend(fontsize=9, loc='upper left')
    ax2.grid(True, alpha=0.3, which='both')

    # Plot 3: Conceptual diagram of uncertainty → pressure chain
    ax3 = axes[1, 0]
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    ax3.axis('off')

    # Draw boxes for each step
    box_props = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=COLORS['primary_blue'],
                     linewidth=2)

    steps_en = [
        'Confinement\nΔx decreases',
        'Uncertainty\nΔp ≥ ℏ/(2Δx)',
        'Momentum\np ~ Δp',
        'Kinetic Energy\nE = p²/(2m)',
        'PRESSURE\nP = nE'
    ]
    steps_de = [
        'Einschluss\nΔx nimmt ab',
        'Unschärfe\nΔp ≥ ℏ/(2Δx)',
        'Impuls\np ~ Δp',
        'Kinetische Energie\nE = p²/(2m)',
        'DRUCK\nP = nE'
    ]
    steps = steps_de if language == 'de' else steps_en

    x_positions = [1, 3, 5, 7, 9]
    y_pos = 5

    colors = [COLORS['primary_blue'], COLORS['quantum'], COLORS['scaled'],
              COLORS['primary_amber'], COLORS['text_dark']]

    for i, (x, step, color) in enumerate(zip(x_positions, steps, colors)):
        box = dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor=color, linewidth=2)
        ax3.text(x, y_pos, step, ha='center', va='center', fontsize=10,
                bbox=box, fontweight='bold' if i == 4 else 'normal')

        # Add arrows between boxes
        if i < len(steps) - 1:
            ax3.annotate('', xy=(x + 0.7, y_pos), xytext=(x + 0.3, y_pos),
                        arrowprops=dict(arrowstyle='->', color=COLORS['text_dark'], lw=2))

    # Add essay quote
    if language == 'de':
        quote = '„Diese Unruhe erzeugt den nach außen gerichteten Druck"'
    else:
        quote = '"This restlessness creates the outward-directed pressure"'
    ax3.text(5, 2, quote, ha='center', va='center', fontsize=11, style='italic',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    if language == 'de':
        ax3.set_title('Kausalkette: Unschärfe → Druck', fontsize=13, fontweight='bold', y=0.95)
    else:
        ax3.set_title('Causal Chain: Uncertainty → Pressure', fontsize=13, fontweight='bold', y=0.95)

    # Plot 4: Effect of ℏ on pressure
    ax4 = axes[1, 1]

    # Different ℏ scaling factors
    hbar_scales = [1.0, 0.5, 0.1]

    for scale in hbar_scales:
        # Create scaled constants
        scaled_hbar = constants.hbar * scale

        # Non-relativistic pressure scales as ℏ²
        P_scaled = P_e_nr * scale**2

        ax4.loglog(rho, P_scaled, '-', linewidth=2,
                   label=f'ℏ × {scale}' + (f' (P × {scale**2:.2f})' if scale != 1 else ' (Standard)'))

    # Mark standard white dwarf conditions
    rho_wd = 1e10  # kg/m³
    P_wd_std = degeneracy_pressure_from_uncertainty(rho_wd, constants.m_e, constants, False)
    ax4.plot(rho_wd, P_wd_std, 'o', color=COLORS['standard'], markersize=12)

    if language == 'de':
        ax4.set_xlabel('Dichte ρ (kg/m³)', fontsize=11)
        ax4.set_ylabel('Entartungsdruck P (Pa)', fontsize=11)
        ax4.set_title('Effekt von ℏ: P ∝ ℏ²', fontsize=13, fontweight='bold')
    else:
        ax4.set_xlabel('Density ρ (kg/m³)', fontsize=11)
        ax4.set_ylabel('Degeneracy pressure P (Pa)', fontsize=11)
        ax4.set_title('Effect of ℏ: P ∝ ℏ²', fontsize=13, fontweight='bold')

    ax4.legend(fontsize=9, loc='upper left')
    ax4.grid(True, alpha=0.3, which='both')

    plt.tight_layout()

    if save:
        os.makedirs(VIS_DIR, exist_ok=True)
        suffix = '_de' if language == 'de' else ''
        filepath = os.path.join(VIS_DIR, f'uncertainty_to_pressure{suffix}.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")

    if show:
        plt.show()

    return fig


def plot_uncertainty_hbar_scaling(
    constants: Optional[PhysicalConstants] = None,
    language: str = 'en',
    save: bool = True,
    show: bool = True
) -> plt.Figure:
    """
    Show how changing ℏ affects the uncertainty principle and its consequences.
    Zeigt wie aenderndes ℏ die Unschaerferelation und ihre Folgen beeinflusst.

    Key insight: When ℏ increases, minimum uncertainty product increases,
    leading to stronger degeneracy pressure support.

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

    # Range of ℏ scaling factors
    hbar_scales = np.logspace(-1, 1, 50)

    # Fixed confinement (e.g., white dwarf electron spacing)
    delta_x = 1e-12  # 1 pm

    # Calculate quantities for each ℏ scaling
    delta_p_min = constants.hbar * hbar_scales / (2 * delta_x)
    v_min = delta_p_min / constants.m_e
    E_kin = delta_p_min**2 / (2 * constants.m_e)

    # Degeneracy pressure (non-relativistic) scales as ℏ²
    P_deg_ratio = hbar_scales**2

    # Plot 1: Minimum momentum uncertainty vs ℏ
    ax1 = axes[0, 0]
    ax1.loglog(hbar_scales, delta_p_min, '-', color=COLORS['primary_blue'], linewidth=2.5)
    ax1.axvline(x=1, color=COLORS['standard'], linestyle='--', alpha=0.7)
    ax1.plot(1, constants.hbar / (2 * delta_x), 'o', color=COLORS['standard'], markersize=12,
             label='Standard ℏ')

    if language == 'de':
        ax1.set_xlabel('ℏ-Skalierung', fontsize=11)
        ax1.set_ylabel('Δp_min (kg·m/s)', fontsize=11)
        ax1.set_title('Minimale Impulsunschärfe: Δp ∝ ℏ', fontsize=13, fontweight='bold')
    else:
        ax1.set_xlabel('ℏ scaling', fontsize=11)
        ax1.set_ylabel('Δp_min (kg·m/s)', fontsize=11)
        ax1.set_title('Minimum Momentum Uncertainty: Δp ∝ ℏ', fontsize=13, fontweight='bold')

    ax1.legend(fontsize=9, loc='upper left')
    ax1.grid(True, alpha=0.3, which='both')

    # Plot 2: Confinement velocity vs ℏ
    ax2 = axes[0, 1]
    ax2.loglog(hbar_scales, v_min / constants.c, '-', color=COLORS['quantum'], linewidth=2.5)
    ax2.axhline(y=1, color=COLORS['scaled'], linestyle=':', linewidth=2, alpha=0.7,
                label='Speed of light' if language == 'en' else 'Lichtgeschwindigkeit')
    ax2.axvline(x=1, color=COLORS['standard'], linestyle='--', alpha=0.7)
    ax2.plot(1, v_min[np.argmin(np.abs(hbar_scales - 1))] / constants.c, 'o',
             color=COLORS['standard'], markersize=12, label='Standard ℏ')

    if language == 'de':
        ax2.set_xlabel('ℏ-Skalierung', fontsize=11)
        ax2.set_ylabel('v/c', fontsize=11)
        ax2.set_title('Einschluss-Geschwindigkeit: v ∝ ℏ', fontsize=13, fontweight='bold')
    else:
        ax2.set_xlabel('ℏ scaling', fontsize=11)
        ax2.set_ylabel('v/c', fontsize=11)
        ax2.set_title('Confinement Velocity: v ∝ ℏ', fontsize=13, fontweight='bold')

    ax2.legend(fontsize=9, loc='upper left')
    ax2.grid(True, alpha=0.3, which='both')

    # Plot 3: Zero-point energy vs ℏ
    ax3 = axes[1, 0]
    ax3.loglog(hbar_scales, E_kin / constants.e, '-', color=COLORS['scaled'], linewidth=2.5)
    ax3.axvline(x=1, color=COLORS['standard'], linestyle='--', alpha=0.7)
    E_kin_std_idx = np.argmin(np.abs(hbar_scales - 1))
    ax3.plot(1, E_kin[E_kin_std_idx] / constants.e, 'o', color=COLORS['standard'], markersize=12,
             label='Standard ℏ')

    if language == 'de':
        ax3.set_xlabel('ℏ-Skalierung', fontsize=11)
        ax3.set_ylabel('Nullpunktsenergie (eV)', fontsize=11)
        ax3.set_title('Kinetische Energie: E ∝ ℏ²', fontsize=13, fontweight='bold')
    else:
        ax3.set_xlabel('ℏ scaling', fontsize=11)
        ax3.set_ylabel('Zero-point energy (eV)', fontsize=11)
        ax3.set_title('Kinetic Energy: E ∝ ℏ²', fontsize=13, fontweight='bold')

    ax3.legend(fontsize=9, loc='upper left')
    ax3.grid(True, alpha=0.3, which='both')

    # Plot 4: Degeneracy pressure vs ℏ
    ax4 = axes[1, 1]
    ax4.loglog(hbar_scales, P_deg_ratio, '-', color=COLORS['text_dark'], linewidth=2.5,
               label='P/P_standard')
    ax4.axvline(x=1, color=COLORS['standard'], linestyle='--', alpha=0.7)
    ax4.axhline(y=1, color=COLORS['standard'], linestyle=':', alpha=0.5)
    ax4.plot(1, 1, 'o', color=COLORS['standard'], markersize=12, label='Standard ℏ')

    # Mark the essay scaling (G × 10^36 with ℏ scaling to preserve atoms)
    # When G increases, we need ℏ to increase to maintain Bohr radius: ℏ_new = sqrt(G_new/G_old) × ℏ
    # For G × 10^36: ℏ would need to scale by 10^18 to keep atoms same size
    # But the essay discusses different scenarios

    if language == 'de':
        ax4.set_xlabel('ℏ-Skalierung', fontsize=11)
        ax4.set_ylabel('P / P_standard', fontsize=11)
        ax4.set_title('Entartungsdruck: P ∝ ℏ²', fontsize=13, fontweight='bold')
    else:
        ax4.set_xlabel('ℏ scaling', fontsize=11)
        ax4.set_ylabel('P / P_standard', fontsize=11)
        ax4.set_title('Degeneracy Pressure: P ∝ ℏ²', fontsize=13, fontweight='bold')

    ax4.legend(fontsize=9, loc='upper left')
    ax4.grid(True, alpha=0.3, which='both')

    # Add overall title
    if language == 'de':
        fig.suptitle('Effekt von ℏ auf die Heisenberg-Unschärferelation', fontsize=15, fontweight='bold', y=1.02)
    else:
        fig.suptitle('Effect of ℏ on Heisenberg Uncertainty Relation', fontsize=15, fontweight='bold', y=1.02)

    plt.tight_layout()

    if save:
        os.makedirs(VIS_DIR, exist_ok=True)
        suffix = '_de' if language == 'de' else ''
        filepath = os.path.join(VIS_DIR, f'uncertainty_hbar_scaling{suffix}.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")

    if show:
        plt.show()

    return fig


def plot_heisenberg_summary(
    constants: Optional[PhysicalConstants] = None,
    language: str = 'en',
    save: bool = True,
    show: bool = True
) -> plt.Figure:
    """
    Create a comprehensive summary of Heisenberg uncertainty and degeneracy pressure.
    Erstellt umfassende Zusammenfassung von Heisenberg-Unschaerfe und Entartungsdruck.

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

    fig = plt.figure(figsize=(16, 14))

    # Create grid layout
    gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)

    # Plot 1: Basic uncertainty relation
    ax1 = fig.add_subplot(gs[0, 0])

    delta_x = np.logspace(-15, -9, 100)
    delta_p_min = constants.hbar / (2 * delta_x)

    ax1.loglog(delta_x * 1e15, delta_p_min, '-', color=COLORS['primary_blue'], linewidth=2.5)

    # Mark key scales
    scales = {
        'Atom': (constants.a_0, COLORS['standard'], 'o'),
        'WD e⁻': (1e-12, COLORS['scaled'], 's'),
        'NS n': (1e-15, COLORS['quantum'], '^')
    }
    for name, (dx, color, marker) in scales.items():
        dp = constants.hbar / (2 * dx)
        ax1.plot(dx * 1e15, dp, marker, color=color, markersize=10, label=name)

    if language == 'de':
        ax1.set_xlabel('Δx (fm)', fontsize=10)
        ax1.set_ylabel('Δp_min (kg·m/s)', fontsize=10)
        ax1.set_title('1. Unschärferelation: Δx·Δp ≥ ℏ/2', fontsize=12, fontweight='bold')
    else:
        ax1.set_xlabel('Δx (fm)', fontsize=10)
        ax1.set_ylabel('Δp_min (kg·m/s)', fontsize=10)
        ax1.set_title('1. Uncertainty Relation: Δx·Δp ≥ ℏ/2', fontsize=12, fontweight='bold')

    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.3, which='both')

    # Plot 2: Confinement velocity
    ax2 = fig.add_subplot(gs[0, 1])

    v_e = confinement_velocity(delta_x, constants.m_e, constants)
    v_n = confinement_velocity(delta_x, constants.m_n, constants)

    ax2.loglog(delta_x * 1e15, v_e / constants.c, '-', color=COLORS['primary_blue'],
               linewidth=2, label='Electron' if language == 'en' else 'Elektron')
    ax2.loglog(delta_x * 1e15, v_n / constants.c, '--', color=COLORS['scaled'],
               linewidth=2, label='Neutron')
    ax2.axhline(y=1, color=COLORS['quantum'], linestyle=':', linewidth=1.5, alpha=0.7, label='c')

    if language == 'de':
        ax2.set_xlabel('Δx (fm)', fontsize=10)
        ax2.set_ylabel('v/c', fontsize=10)
        ax2.set_title('2. Einschluss-Geschwindigkeit', fontsize=12, fontweight='bold')
    else:
        ax2.set_xlabel('Δx (fm)', fontsize=10)
        ax2.set_ylabel('v/c', fontsize=10)
        ax2.set_title('2. Confinement Velocity', fontsize=12, fontweight='bold')

    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.set_ylim(1e-4, 10)

    # Plot 3: Degeneracy pressure vs density
    ax3 = fig.add_subplot(gs[1, 0])

    rho = np.logspace(6, 18, 100)
    P_e = np.array([degeneracy_pressure_from_uncertainty(r, constants.m_e, constants, False) for r in rho])
    P_n = np.array([degeneracy_pressure_from_uncertainty(r, constants.m_n, constants, False) for r in rho])

    ax3.loglog(rho, P_e, '-', color=COLORS['primary_blue'], linewidth=2,
               label='Electron' if language == 'en' else 'Elektron')
    ax3.loglog(rho, P_n, '--', color=COLORS['quantum'], linewidth=2, label='Neutron')

    ax3.axvspan(1e9, 1e11, alpha=0.15, color=COLORS['standard'], label='White dwarf')
    ax3.axvspan(1e14, 1e18, alpha=0.15, color=COLORS['quantum'], label='Neutron star')

    if language == 'de':
        ax3.set_xlabel('Dichte (kg/m³)', fontsize=10)
        ax3.set_ylabel('Entartungsdruck (Pa)', fontsize=10)
        ax3.set_title('3. Entartungsdruck P ∝ ρ^{5/3}', fontsize=12, fontweight='bold')
    else:
        ax3.set_xlabel('Density (kg/m³)', fontsize=10)
        ax3.set_ylabel('Degeneracy pressure (Pa)', fontsize=10)
        ax3.set_title('3. Degeneracy Pressure P ∝ ρ^{5/3}', fontsize=12, fontweight='bold')

    ax3.legend(fontsize=8, loc='upper left')
    ax3.grid(True, alpha=0.3, which='both')

    # Plot 4: Effect of ℏ
    ax4 = fig.add_subplot(gs[1, 1])

    hbar_scales = np.logspace(-1, 1, 50)
    P_ratio = hbar_scales**2

    ax4.loglog(hbar_scales, P_ratio, '-', color=COLORS['text_dark'], linewidth=2.5)
    ax4.axvline(x=1, color=COLORS['standard'], linestyle='--', alpha=0.7)
    ax4.axhline(y=1, color=COLORS['standard'], linestyle=':', alpha=0.5)
    ax4.plot(1, 1, 'o', color=COLORS['standard'], markersize=12, label='Standard ℏ')

    if language == 'de':
        ax4.set_xlabel('ℏ-Skalierung', fontsize=10)
        ax4.set_ylabel('P / P_standard', fontsize=10)
        ax4.set_title('4. Druckskalierung: P ∝ ℏ²', fontsize=12, fontweight='bold')
    else:
        ax4.set_xlabel('ℏ scaling', fontsize=10)
        ax4.set_ylabel('P / P_standard', fontsize=10)
        ax4.set_title('4. Pressure Scaling: P ∝ ℏ²', fontsize=12, fontweight='bold')

    ax4.legend(fontsize=8, loc='upper left')
    ax4.grid(True, alpha=0.3, which='both')

    # Plot 5: Conceptual summary (text box)
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')

    if language == 'de':
        summary_text = """
        HEISENBERG-UNSCHÄRFERELATION UND ENTARTUNGSDRUCK
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        KAUSALKETTE:
        1. Gravitation komprimiert Materie → Δx wird kleiner (Elektronen/Neutronen werden eingesperrt)
        2. Heisenberg-Unschärfe: Δx·Δp ≥ ℏ/2 → Δp muss zunehmen
        3. Höherer Impuls → höhere Geschwindigkeit ("Unruhe")
        4. Höhere kinetische Energie → DRUCK nach außen

        SCHLÜSSELFORMELN:
        • Unschärferelation:     Δx · Δp ≥ ℏ/2
        • Nullpunktsenergie:     E ~ ℏ²/(m·Δx²)
        • Entartungsdruck (NR):  P ∝ ℏ² · ρ^(5/3) / m
        • Entartungsdruck (UR):  P ∝ ℏc · ρ^(4/3)

        KONSEQUENZ: Diese "Unruhe" erzeugt den nach außen gerichteten Druck,
        der weiße Zwerge und Neutronensterne gegen den Gravitationskollaps stützt!
        """
    else:
        summary_text = """
        HEISENBERG UNCERTAINTY RELATION AND DEGENERACY PRESSURE
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        CAUSAL CHAIN:
        1. Gravity compresses matter → Δx decreases (electrons/neutrons get confined)
        2. Heisenberg uncertainty: Δx·Δp ≥ ℏ/2 → Δp must increase
        3. Higher momentum → higher velocity ("restlessness")
        4. Higher kinetic energy → outward PRESSURE

        KEY FORMULAS:
        • Uncertainty relation:       Δx · Δp ≥ ℏ/2
        • Zero-point energy:          E ~ ℏ²/(m·Δx²)
        • Degeneracy pressure (NR):   P ∝ ℏ² · ρ^(5/3) / m
        • Degeneracy pressure (UR):   P ∝ ℏc · ρ^(4/3)

        CONSEQUENCE: This "restlessness" creates the outward-directed pressure
        that supports white dwarfs and neutron stars against gravitational collapse!
        """

    ax5.text(0.5, 0.5, summary_text, transform=ax5.transAxes, fontsize=11,
             verticalalignment='center', horizontalalignment='center',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9,
                      edgecolor=COLORS['primary_blue'], linewidth=2))

    # Overall title
    if language == 'de':
        fig.suptitle('Heisenberg-Unschärfeprinzip: Von Quantenmechanik zu Sternenstabilität',
                    fontsize=16, fontweight='bold', y=0.98)
    else:
        fig.suptitle('Heisenberg Uncertainty Principle: From Quantum Mechanics to Stellar Stability',
                    fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout()

    if save:
        os.makedirs(VIS_DIR, exist_ok=True)
        suffix = '_de' if language == 'de' else ''
        filepath = os.path.join(VIS_DIR, f'heisenberg_summary{suffix}.png')
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved: {filepath}")

    if show:
        plt.show()

    return fig


def generate_all_heisenberg_plots(language: str = 'en', show: bool = False) -> List[plt.Figure]:
    """
    Generate all Heisenberg uncertainty visualizations.
    Erzeugt alle Heisenberg-Unschaerfe-Visualisierungen.

    Args:
        language: 'en' for English, 'de' for German
        show: Whether to display plots

    Returns:
        List of matplotlib Figure objects
    """
    figures = []

    print("Generating Heisenberg uncertainty visualizations...")
    print("=" * 50)

    # 1. Basic uncertainty relation
    print("1. Basic uncertainty relation...")
    figures.append(plot_uncertainty_principle_basic(language=language, show=show))

    # 2. Confinement velocity
    print("2. Confinement velocity ('restlessness')...")
    figures.append(plot_confinement_velocity(language=language, show=show))

    # 3. Uncertainty to pressure
    print("3. Uncertainty to pressure connection...")
    figures.append(plot_uncertainty_to_pressure(language=language, show=show))

    # 4. ℏ scaling effects
    print("4. hbar scaling effects...")
    figures.append(plot_uncertainty_hbar_scaling(language=language, show=show))

    # 5. Summary
    print("5. Comprehensive summary...")
    figures.append(plot_heisenberg_summary(language=language, show=show))

    print("=" * 50)
    print(f"Generated {len(figures)} visualizations in {VIS_DIR}")

    return figures


def verify_heisenberg_physics():
    """
    Verify Heisenberg uncertainty calculations.
    Verifiziert Heisenberg-Unschaerfe-Berechnungen.
    """
    print("=" * 70)
    print("HEISENBERG UNCERTAINTY PHYSICS VERIFICATION")
    print("=" * 70)

    c = get_constants()

    # 1. Basic uncertainty relation
    print("\n1. BASIC UNCERTAINTY RELATION")
    print("-" * 50)
    delta_x = c.a_0  # Bohr radius
    delta_p = minimum_momentum_uncertainty(delta_x, c)
    product = delta_x * delta_p
    print(f"   Δx = a_0 = {delta_x:.4e} m")
    print(f"   Δp_min = {delta_p:.4e} kg·m/s")
    print(f"   Δx·Δp = {product:.4e} J·s")
    print(f"   hbar/2 = {c.hbar/2:.4e} J*s")
    print(f"   Ratio Dx*Dp / (hbar/2) = {product / (c.hbar/2):.2f}")
    print(f"   CHECK: {'PASS' if abs(product / (c.hbar/2) - 1) < 0.01 else 'FAIL'}")

    # 2. Electron velocity in atom
    print("\n2. ELECTRON 'RESTLESSNESS' IN HYDROGEN ATOM")
    print("-" * 50)
    v_atom = confinement_velocity(c.a_0, c.m_e, c)
    print(f"   Confinement velocity v = {v_atom:.4e} m/s")
    print(f"   v/c = {v_atom/c.c:.6f}")
    print(f"   Expected: v/c ≈ α ≈ 1/137 ≈ 0.0073")
    print(f"   CHECK: {'PASS' if 0.003 < v_atom/c.c < 0.01 else 'FAIL'}")

    # 3. Zero-point energy
    print("\n3. ZERO-POINT ENERGY")
    print("-" * 50)
    E_zp = confinement_kinetic_energy(c.a_0, c.m_e, c)
    E_zp_eV = E_zp / c.e
    print(f"   Zero-point energy E = {E_zp:.4e} J = {E_zp_eV:.2f} eV")
    print(f"   Compare to H ionization energy: 13.6 eV")
    print(f"   Ratio E_zp / E_ion = {E_zp_eV / 13.6:.2f}")

    # 4. White dwarf degeneracy pressure
    print("\n4. WHITE DWARF ELECTRON DEGENERACY PRESSURE")
    print("-" * 50)
    rho_wd = 1e9  # kg/m³ (typical white dwarf)
    P_wd = degeneracy_pressure_from_uncertainty(rho_wd, c.m_e, c, False)
    print(f"   Density: {rho_wd:.0e} kg/m³")
    print(f"   Electron degeneracy pressure: {P_wd:.4e} Pa")
    print(f"   Expected: ~10²² Pa for typical WD")
    print(f"   CHECK: {'PASS' if 1e20 < P_wd < 1e25 else 'FAIL'}")

    # 5. Neutron star degeneracy pressure
    print("\n5. NEUTRON STAR NEUTRON DEGENERACY PRESSURE")
    print("-" * 50)
    rho_ns = 1e17  # kg/m³ (typical neutron star)
    P_ns = degeneracy_pressure_from_uncertainty(rho_ns, c.m_n, c, False)
    print(f"   Density: {rho_ns:.0e} kg/m³")
    print(f"   Neutron degeneracy pressure: {P_ns:.4e} Pa")
    print(f"   Expected: ~10³² Pa for typical NS")
    print(f"   CHECK: {'PASS' if 1e30 < P_ns < 1e35 else 'FAIL'}")

    # 6. Pressure scaling with ℏ
    print("\n6. PRESSURE SCALING WITH hbar")
    print("-" * 50)
    print(f"   Non-relativistic: P proportional to hbar^2")
    print(f"   If hbar doubles: P increases by factor {2**2}")
    print(f"   If hbar halves: P decreases by factor {(0.5)**2}")

    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    print("=" * 60)
    print("Heisenberg Uncertainty Module - Jugend forscht 2026")
    print("=" * 60)

    # Verify physics
    verify_heisenberg_physics()

    # Generate visualizations
    print("\n")
    generate_all_heisenberg_plots(language='en', show=False)

    print("\nDone! Check the 'visualizations' folder for output.")
