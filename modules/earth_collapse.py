"""
Earth Collapse Evolution Module for Jugend forscht 2026 Physics Visualization Project
Erdkollaps-Entwicklungsmodul fuer Jugend forscht 2026 Physik-Visualisierungsprojekt

SYNTHESIS MODULE: This module brings together all the scaling relationships from the
project to show how Earth's parameters change as hbar decreases (or equivalently as
gravity becomes relatively stronger). It tracks Earth from normalcy through compression
to potential collapse.

Key scaling relationships (when hbar_scale = s):
    - Bohr radius:          a_0  proportional to  s^2    (atoms shrink)
    - Atomic density:       rho  proportional to  1/s^6  (much denser matter)
    - Earth radius:         R    proportional to  s^2    (matter compresses)
    - Surface gravity:      g    proportional to  1/s^4
    - Escape velocity:      v_esc proportional to 1/s
    - Atmospheric height:   H    proportional to  s^4
    - Chandrasekhar mass:   M_Ch proportional to  s^(3/2)
    - Compactness:          C    proportional to  1/s^2
    - Alpha_G:              alpha_G proportional to 1/s

Visualizations produced:
    9. Earth Change Diagram   - parameter evolution, phase transitions, thresholds
    10. Earth Combined        - schematic views, pressures, bar comparison, fate diagram
    11. Earth Collapse Summary - Chandrasekhar crossing, compactness, alpha_G, summary table

Author: Jugend forscht 2026 Project
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
import os
from typing import Optional, List
from dataclasses import dataclass

from .constants import get_constants, PhysicalConstants
from .color_scheme import COLORS, get_sequence, TEMPERATURE_COLORS


# Output directory for visualizations
VIS_DIR = os.path.join(os.path.dirname(__file__), '..', 'visualizations')


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class EarthEvolutionState:
    """
    Container for Earth's physical state at a given hbar scaling factor.
    Behaelter fuer den physikalischen Zustand der Erde bei gegebenem hbar-Skalierungsfaktor.
    """
    hbar_scale: float                # Scaling factor s applied to hbar
    radius: float                    # Earth radius [m]
    density: float                   # Mean density [kg/m^3]
    surface_gravity: float           # Surface gravitational acceleration [m/s^2]
    escape_velocity: float           # Surface escape velocity [m/s]
    atmosphere_height: float         # Atmospheric scale height [m]
    compactness: float               # Schwarzschild ratio R_s / R
    alpha_G: float                   # Gravitational coupling constant
    bohr_radius: float               # Bohr radius [m]
    chandrasekhar_mass: float        # Chandrasekhar mass limit [kg]
    is_planet: bool                  # Still planet-like?
    is_white_dwarf_like: bool        # White-dwarf densities?
    is_neutron_star_like: bool       # Neutron-star densities?
    classification: str              # 'planet', 'compressed', 'white_dwarf', 'neutron_star', 'black_hole'


# =============================================================================
# CALCULATION FUNCTIONS
# =============================================================================

def chandrasekhar_mass_at_scale(hbar_scale: float, constants: PhysicalConstants) -> float:
    """
    Calculate the Chandrasekhar mass limit at a given hbar scaling.
    Berechnet die Chandrasekhar-Massengrenze bei gegebener hbar-Skalierung.

    Formula: M_Ch = 0.518 * (hbar*c/G)^(3/2) / m_p^2 * (2/mu_e)^2
    With hbar_scaled = hbar_standard * s, so M_Ch scales as s^(3/2).

    Args:
        hbar_scale: Scaling factor s applied to hbar
        constants: Physical constants (standard, unscaled)

    Returns:
        Chandrasekhar mass [kg]
    """
    mu_e = 2.0  # Mean molecular weight per electron (C/O composition)
    omega_3 = 2.018  # Lane-Emden numerical constant for n=3 polytrope

    hbar_scaled = constants.hbar * hbar_scale

    M_ch = (omega_3 * np.sqrt(3 * np.pi) / 2) *            (hbar_scaled * constants.c / constants.G) ** (3 / 2) *            (1 / (mu_e * constants.m_p)) ** 2

    return M_ch


def earth_state_at_scale(
    hbar_scale: float,
    constants: Optional[PhysicalConstants] = None
) -> EarthEvolutionState:
    """
    Calculate ALL Earth parameters at a given hbar scaling factor.
    Berechnet ALLE Erdparameter bei einem gegebenen hbar-Skalierungsfaktor.

    As hbar decreases (s < 1):
        - Atoms shrink: a_0 = a_0_std * s^2
        - Matter densifies: rho = rho_std / s^6
        - Earth compresses: R = R_earth * s^2
        - Gravity strengthens: g = G*M/R^2 = g_std / s^4
        - Escape velocity rises: v_esc = sqrt(2GM/R) = v_esc_std / s
        - Atmosphere thins: H = kT/(mu*g) = H_std * s^4
        - Chandrasekhar limit drops: M_Ch = M_Ch_std * s^(3/2)
        - Compactness grows: C = 2GM/(Rc^2) = C_std / s^2
        - Gravitational coupling grows: alpha_G = alpha_G_std / s

    Args:
        hbar_scale: Scaling factor s applied to hbar (0 < s <= 1)
        constants: Physical constants (uses standard if None)

    Returns:
        EarthEvolutionState with all computed parameters
    """
    if constants is None:
        constants = get_constants()

    s = hbar_scale
    M = constants.M_earth
    G = constants.G
    c_light = constants.c

    # --- Scaled parameters ---
    R = constants.R_earth * s ** 2
    density = M / ((4 / 3) * np.pi * R ** 3)
    g = G * M / R ** 2
    v_esc = np.sqrt(2 * G * M / R)

    # Atmospheric scale height: H = k_B * T / (mu_air * g)
    H = constants.k_B * constants.T_surface_earth / (constants.mu_air * g)

    # Compactness: C = R_schwarzschild / R = 2GM / (R * c^2)
    C = 2 * G * M / (R * c_light ** 2)

    # Bohr radius
    a_0 = constants.a_0 * s ** 2  # a_0 standard is already for s=1

    # Gravitational coupling constant
    # alpha_G_std = G * m_p^2 / (hbar_std * c), so at scale s: alpha_G = alpha_G_std / s
    alpha_G_std = (G * constants.m_p ** 2) / (constants.hbar * c_light)
    alpha_G = alpha_G_std / s

    # Chandrasekhar mass at this scale
    M_Ch = chandrasekhar_mass_at_scale(s, constants)

    # --- Classification ---
    # Density thresholds (approximate, in kg/m^3)
    rho_compressed = 1e4      # Beyond normal materials
    rho_wd = 1e9              # White dwarf regime
    rho_ns = 4e14             # Neutron star regime

    is_planet = density < rho_compressed
    is_white_dwarf_like = rho_wd <= density < rho_ns
    is_neutron_star_like = density >= rho_ns

    if C >= 1.0:
        classification = 'black_hole'
    elif density >= rho_ns:
        classification = 'neutron_star'
    elif density >= rho_wd:
        classification = 'white_dwarf'
    elif density >= rho_compressed:
        classification = 'compressed'
    else:
        classification = 'planet'

    return EarthEvolutionState(
        hbar_scale=s,
        radius=R,
        density=density,
        surface_gravity=g,
        escape_velocity=v_esc,
        atmosphere_height=H,
        compactness=C,
        alpha_G=alpha_G,
        bohr_radius=a_0,
        chandrasekhar_mass=M_Ch,
        is_planet=is_planet,
        is_white_dwarf_like=is_white_dwarf_like,
        is_neutron_star_like=is_neutron_star_like,
        classification=classification
    )


def earth_evolution_track(
    hbar_scales: Optional[np.ndarray] = None,
    constants: Optional[PhysicalConstants] = None
) -> List[EarthEvolutionState]:
    """
    Generate a sequence of Earth evolution states across hbar scales.
    Erzeugt eine Sequenz von Erdentwicklungszustaenden ueber hbar-Skalierungen.

    Args:
        hbar_scales: Array of scaling factors (default: logspace from 0.01 to 1.0)
        constants: Physical constants (uses standard if None)

    Returns:
        List of EarthEvolutionState objects
    """
    if constants is None:
        constants = get_constants()

    if hbar_scales is None:
        hbar_scales = np.logspace(-2, 0, 500)

    return [earth_state_at_scale(s, constants) for s in hbar_scales]


# =============================================================================
# HELPER FUNCTIONS FOR THRESHOLD DETECTION
# =============================================================================

def _find_threshold_scale(states: List[EarthEvolutionState], attr: str,
                          threshold: float, direction: str = 'above') -> Optional[float]:
    """
    Find the hbar_scale at which a parameter crosses a threshold.
    Searches from s=1 downward (large s to small s).

    Args:
        states: List of states (any order, will be sorted internally)
        attr: Attribute name to check
        threshold: Value to cross
        direction: 'above' if looking for value > threshold, 'below' for < threshold

    Returns:
        hbar_scale at crossing, or None if not found
    """
    # Sort by decreasing hbar_scale (from normal to collapsed)
    sorted_states = sorted(states, key=lambda st: st.hbar_scale, reverse=True)

    for i in range(1, len(sorted_states)):
        val_prev = getattr(sorted_states[i - 1], attr)
        val_curr = getattr(sorted_states[i], attr)

        if direction == 'above':
            if val_prev < threshold <= val_curr:
                s_prev = sorted_states[i - 1].hbar_scale
                s_curr = sorted_states[i].hbar_scale
                if val_curr != val_prev:
                    frac = (threshold - val_prev) / (val_curr - val_prev)
                    return s_prev + frac * (s_curr - s_prev)
                return s_curr
        else:  # below
            if val_prev > threshold >= val_curr:
                s_prev = sorted_states[i - 1].hbar_scale
                s_curr = sorted_states[i].hbar_scale
                if val_curr != val_prev:
                    frac = (threshold - val_prev) / (val_curr - val_prev)
                    return s_prev + frac * (s_curr - s_prev)
                return s_curr

    return None


def _classification_color(classification: str) -> str:
    """Return a color for a given classification string."""
    mapping = {
        'planet': COLORS['earth'],
        'compressed': COLORS['primary_amber'],
        'white_dwarf': COLORS['white_dwarf'],
        'neutron_star': COLORS['neutron_star'],
        'black_hole': COLORS['black_hole'],
    }
    return mapping.get(classification, COLORS['muted'])


# =============================================================================
# VISUALIZATION 9: EARTH CHANGE DIAGRAM
# =============================================================================

def plot_earth_change_diagram(
    constants=None, language='en', save=True, show=True):
    """
    Visualization #9 - Diagram of Change. 4 vertical subplots.
    Visualisierung #9 - Diagramm der Veraenderung.

    Subplot 1: Normalized parameter evolution (R, rho, g, v_esc, H) vs s
    Subplot 2: Phase classification bands
    Subplot 3: Cascade timeline with key events
    Subplot 4: Number line with current state markers

    Args:
        constants: Physical constants (uses standard if None)
        language: 'en' or 'de'
        save: Whether to save the figure
        show: Whether to display the figure

    Returns:
        matplotlib Figure object
    """
    if constants is None:
        constants = get_constants()

    s_values = np.logspace(-2, 0, 500)
    states = earth_evolution_track(s_values, constants)
    ref = earth_state_at_scale(1.0, constants)

    s_arr = np.array([st.hbar_scale for st in states])
    radius_norm = np.array([st.radius / ref.radius for st in states])
    density_norm = np.array([st.density / ref.density for st in states])
    gravity_norm = np.array([st.surface_gravity / ref.surface_gravity for st in states])
    vesc_norm = np.array([st.escape_velocity / ref.escape_velocity for st in states])
    height_norm = np.array([st.atmosphere_height / ref.atmosphere_height for st in states])
    compact_norm = np.array([st.compactness / ref.compactness for st in states])
    classifications = [st.classification for st in states]
    colors = get_sequence()

    fig = plt.figure(figsize=(12, 32))
    gs = fig.add_gridspec(4, 1, hspace=0.5)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    ax4 = fig.add_subplot(gs[3])

    # --- Subplot 1: Normalized parameter evolution ---
    ax1.loglog(s_arr, radius_norm, color=colors[0], linewidth=2, label='R/R_0 (radius)')
    ax1.loglog(s_arr, density_norm, color=colors[1], linewidth=2, label='rho/rho_0 (density)')
    ax1.loglog(s_arr, gravity_norm, color=colors[2], linewidth=2, label='g/g_0 (gravity)')
    ax1.loglog(s_arr, vesc_norm, color=colors[3], linewidth=2, label='v_esc/v_esc_0')
    ax1.loglog(s_arr, height_norm, color=colors[4], linewidth=2, label='H/H_0 (atm. height)')
    ax1.loglog(s_arr, compact_norm, color=colors[5], linewidth=2, label='C/C_0 (compactness)')

    if language == 'de':
        ax1.set_xlabel('hbar-Skalierungsfaktor s', fontsize=12)
        ax1.set_ylabel('Normierter Wert (bezogen auf s=1)', fontsize=12)
        ax1.set_title('1. Normierte Parameterentwicklung', fontsize=14, fontweight='bold')
    else:
        ax1.set_xlabel('hbar scaling factor s', fontsize=12)
        ax1.set_ylabel('Normalized value (relative to s=1)', fontsize=12)
        ax1.set_title('1. Normalized Parameter Evolution', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=10)
    ax1.grid(True, alpha=0.3, which='both')
    ax1.tick_params(axis='both', labelsize=11)

    # --- Subplot 2: Phase classification bands ---
    phase_colors = {
        'planet': COLORS['earth'],
        'compressed': COLORS['primary_amber'],
        'white_dwarf': COLORS['white_dwarf'],
        'neutron_star': COLORS['neutron_star'],
        'black_hole': COLORS['black_hole'],
    }
    phase_labels = {
        'planet': ('Planet', 'Planet'),
        'compressed': ('Compressed', 'Komprimiert'),
        'white_dwarf': ('White Dwarf', 'Weisser Zwerg'),
        'neutron_star': ('Neutron Star', 'Neutronenstern'),
        'black_hole': ('Black Hole', 'Schwarzes Loch'),
    }

    # Draw colored bands for each phase
    current_class = classifications[0]
    band_start = s_arr[0]
    drawn_labels = set()
    lang_idx = 1 if language == 'de' else 0

    for i in range(1, len(classifications)):
        if classifications[i] != current_class or i == len(classifications) - 1:
            band_end = s_arr[i]
            pc = phase_colors.get(current_class, COLORS['muted'])
            lbl = phase_labels.get(current_class, (current_class, current_class))[lang_idx]
            if current_class not in drawn_labels:
                ax2.axvspan(band_start, band_end, alpha=0.4, color=pc, label=lbl)
                drawn_labels.add(current_class)
            else:
                ax2.axvspan(band_start, band_end, alpha=0.4, color=pc)
            band_start = band_end
            current_class = classifications[i]

    ax2.set_xscale('log')
    if language == 'de':
        ax2.set_xlabel('hbar-Skalierungsfaktor s', fontsize=12)
        ax2.set_title('2. Phasenklassifikation', fontsize=14, fontweight='bold')
    else:
        ax2.set_xlabel('hbar scaling factor s', fontsize=12)
        ax2.set_title('2. Phase Classification', fontsize=14, fontweight='bold')
    ax2.set_yticks([])
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=10)
    ax2.tick_params(axis='both', labelsize=11)

    # --- Subplot 3: Cascade timeline with key events ---
    # Find key thresholds
    s_compressed = _find_threshold_scale(states, 'density', 1e4, direction='above')
    s_wd = _find_threshold_scale(states, 'density', 1e9, direction='above')
    s_ns = _find_threshold_scale(states, 'density', 4e14, direction='above')
    s_bh = _find_threshold_scale(states, 'compactness', 1.0, direction='above')
    s_ch_cross = _find_threshold_scale(states, 'chandrasekhar_mass',
                                        constants.M_earth, direction='below')

    events = []
    event_labels_en = []
    event_labels_de = []
    event_colors_list = []

    if s_compressed is not None:
        events.append(s_compressed)
        event_labels_en.append('Compressed matter')
        event_labels_de.append('Komprimierte Materie')
        event_colors_list.append(COLORS['primary_amber'])
    if s_ch_cross is not None:
        events.append(s_ch_cross)
        event_labels_en.append('M_Ch < M_Earth')
        event_labels_de.append('M_Ch < M_Erde')
        event_colors_list.append(COLORS['highlight'])
    if s_wd is not None:
        events.append(s_wd)
        event_labels_en.append('White dwarf density')
        event_labels_de.append('Weisser-Zwerg-Dichte')
        event_colors_list.append(COLORS['white_dwarf'])
    if s_ns is not None:
        events.append(s_ns)
        event_labels_en.append('Neutron star density')
        event_labels_de.append('Neutronenstern-Dichte')
        event_colors_list.append(COLORS['neutron_star'])
    if s_bh is not None:
        events.append(s_bh)
        event_labels_en.append('Black hole (C >= 1)')
        event_labels_de.append('Schwarzes Loch (C >= 1)')
        event_colors_list.append(COLORS['black_hole'])

    event_labels = event_labels_de if language == 'de' else event_labels_en

    for idx, (s_ev, lbl, ec) in enumerate(zip(events, event_labels, event_colors_list)):
        ax3.axvline(x=s_ev, color=ec, linewidth=2, linestyle='--')
        y_pos = 0.85 - 0.15 * idx
        ax3.annotate(lbl, xy=(s_ev, y_pos), fontsize=10,
                     color=ec, fontweight='bold',
                     ha='center', va='center',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor=ec, alpha=0.8))

    ax3.set_xscale('log')
    ax3.set_xlim(s_arr.min(), s_arr.max())
    ax3.set_ylim(0, 1)
    if language == 'de':
        ax3.set_xlabel('hbar-Skalierungsfaktor s', fontsize=12)
        ax3.set_title('3. Kaskadenzeitleiste: Schluesselereignisse', fontsize=14, fontweight='bold')
    else:
        ax3.set_xlabel('hbar scaling factor s', fontsize=12)
        ax3.set_title('3. Cascade Timeline: Key Events', fontsize=14, fontweight='bold')
    ax3.set_yticks([])
    ax3.tick_params(axis='both', labelsize=11)

    # --- Subplot 4: Number line with current state markers ---
    # Show key scales on a log number line
    key_scales = [1.0, 0.5, 0.1, 0.05, 0.01]
    for ks in key_scales:
        st = earth_state_at_scale(ks, constants)
        clr = _classification_color(st.classification)
        ax4.axvline(x=ks, color=clr, linewidth=3, alpha=0.7)

        if language == 'de':
            info = f's={ks}' + chr(10) + f'R={st.radius:.2e} m' + chr(10) + f'rho={st.density:.2e} kg/m3'
        else:
            info = f's={ks}' + chr(10) + f'R={st.radius:.2e} m' + chr(10) + f'rho={st.density:.2e} kg/m3'

        y_pos = 0.7 if key_scales.index(ks) % 2 == 0 else 0.3
        ax4.annotate(info, xy=(ks, y_pos), fontsize=9,
                     ha='center', va='center',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor=clr, alpha=0.8))

    ax4.set_xscale('log')
    ax4.set_xlim(0.005, 2)
    ax4.set_ylim(0, 1)
    if language == 'de':
        ax4.set_xlabel('hbar-Skalierungsfaktor s', fontsize=12)
        ax4.set_title('4. Skalierungs-Zahlenlinie', fontsize=14, fontweight='bold')
    else:
        ax4.set_xlabel('hbar scaling factor s', fontsize=12)
        ax4.set_title('4. Scaling Number Line', fontsize=14, fontweight='bold')
    ax4.set_yticks([])
    ax4.tick_params(axis='both', labelsize=11)

    if language == 'de':
        fig.suptitle('Erdentwicklung: Diagramm der Veraenderung', fontsize=16,
                     fontweight='bold', y=0.98)
    else:
        fig.suptitle('Earth Evolution: Diagram of Change', fontsize=16,
                     fontweight='bold', y=0.98)

    if save:
        os.makedirs(VIS_DIR, exist_ok=True)
        suffix = '_de' if language == 'de' else ''
        fpath = os.path.join(VIS_DIR, f'earth_change_diagram{suffix}.png')
        fig.savefig(fpath, dpi=150, bbox_inches='tight')
        print(f"Saved: {fpath}")
    if show:
        plt.show()
    return fig


# =============================================================================
# VISUALIZATION 10: EARTH COMBINED
# =============================================================================

def plot_earth_combined(
    constants=None, language='en', save=True, show=True):
    """
    Visualization #10 - Combined Earth Evolution. 4 vertical subplots.
    Visualisierung #10 - Kombinierte Erdentwicklung.

    Subplot 1: Earth schematic circles at different scales
    Subplot 2: Pressure comparison (degeneracy vs gravity)
    Subplot 3: Grouped bar chart of parameters at key scales
    Subplot 4: Fate trajectory in density-compactness space

    Args:
        constants: Physical constants (uses standard if None)
        language: 'en' or 'de'
        save: Whether to save the figure
        show: Whether to display the figure

    Returns:
        matplotlib Figure object
    """
    if constants is None:
        constants = get_constants()

    s_values = np.logspace(-2, 0, 500)
    states = earth_evolution_track(s_values, constants)
    ref = earth_state_at_scale(1.0, constants)
    colors = get_sequence()

    fig = plt.figure(figsize=(12, 32))
    gs = fig.add_gridspec(4, 1, hspace=0.5)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    ax4 = fig.add_subplot(gs[3])

    # --- Subplot 1: Earth schematic circles at different scales ---
    key_scales = [1.0, 0.5, 0.2, 0.1, 0.05]
    x_positions = np.linspace(0.1, 0.9, len(key_scales))

    # Normalize radii for display: largest circle = 0.15 in axis coords
    radii = [earth_state_at_scale(s, constants).radius for s in key_scales]
    max_r = max(radii)

    for idx, (s, xp) in enumerate(zip(key_scales, x_positions)):
        st = earth_state_at_scale(s, constants)
        display_r = 0.15 * (st.radius / max_r)
        clr = _classification_color(st.classification)
        circle = Circle((xp, 0.5), display_r, color=clr, alpha=0.6,
                        transform=ax1.transAxes)
        ax1.add_patch(circle)

        if language == 'de':
            label = f's={s}' + chr(10) + f'R={st.radius:.2e} m' + chr(10) + f'{st.classification}'
        else:
            label = f's={s}' + chr(10) + f'R={st.radius:.2e} m' + chr(10) + f'{st.classification}'

        ax1.text(xp, 0.12, label, transform=ax1.transAxes,
                ha='center', va='center', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                         edgecolor=clr, alpha=0.8))

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_aspect('equal')
    ax1.set_xticks([])
    ax1.set_yticks([])
    if language == 'de':
        ax1.set_title('1. Erdgroesse bei verschiedenen hbar-Skalierungen',
                      fontsize=14, fontweight='bold')
    else:
        ax1.set_title('1. Earth Size at Different hbar Scalings',
                      fontsize=14, fontweight='bold')

    # --- Subplot 2: Pressure comparison (degeneracy vs gravity) ---
    s_arr = np.array([st.hbar_scale for st in states])

    # Electron degeneracy pressure (non-relativistic): P_deg = K_nr * rho^(5/3)
    # K_nr = (hbar^2 / (5 * m_e)) * (3 / (8*pi*m_p))^(2/3)
    # At scale s: hbar -> hbar*s, so K_nr scales as s^2
    K_nr_std = (constants.hbar**2 / (5 * constants.m_e)) *                (3 / (8 * np.pi * constants.m_p))**(2.0/3.0)

    P_deg = np.array([K_nr_std * (s**2) * (st.density**(5.0/3.0))
                      for s, st in zip(s_arr, states)])

    # Gravitational pressure: P_grav ~ G * M^2 / (4*pi*R^4)
    M = constants.M_earth
    P_grav = np.array([constants.G * M**2 / (4 * np.pi * st.radius**4)
                       for st in states])

    ax2.loglog(s_arr, P_deg, color=colors[0], linewidth=2.5,
              label='Electron degeneracy pressure' if language != 'de'
              else 'Elektronenentartungsdruck')
    ax2.loglog(s_arr, P_grav, color=colors[3], linewidth=2.5,
              label='Gravitational pressure' if language != 'de'
              else 'Gravitationsdruck')

    # Shade regions
    ax2.fill_between(s_arr, P_deg, P_grav,
                     where=P_deg >= P_grav, alpha=0.15, color=colors[0])
    ax2.fill_between(s_arr, P_deg, P_grav,
                     where=P_deg < P_grav, alpha=0.15, color=colors[3])

    if language == 'de':
        ax2.set_xlabel('hbar-Skalierungsfaktor s', fontsize=12)
        ax2.set_ylabel('Druck [Pa]', fontsize=12)
        ax2.set_title('2. Druckvergleich: Entartung vs. Gravitation',
                      fontsize=14, fontweight='bold')
    else:
        ax2.set_xlabel('hbar scaling factor s', fontsize=12)
        ax2.set_ylabel('Pressure [Pa]', fontsize=12)
        ax2.set_title('2. Pressure Comparison: Degeneracy vs. Gravity',
                      fontsize=14, fontweight='bold')
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=10)
    ax2.grid(True, alpha=0.3, which='both')
    ax2.tick_params(axis='both', labelsize=11)

    # --- Subplot 3: Grouped bar chart of parameters at key scales ---
    bar_scales = [1.0, 0.3, 0.1, 0.03, 0.01]
    bar_states = [earth_state_at_scale(s, constants) for s in bar_scales]

    params = ['radius', 'density', 'surface_gravity', 'escape_velocity', 'compactness']
    if language == 'de':
        param_labels = ['Radius', 'Dichte', 'Oberfl.-Schwerkraft', 'Fluchtgeschw.', 'Kompaktheit']
    else:
        param_labels = ['Radius', 'Density', 'Surface gravity', 'Escape velocity', 'Compactness']

    # Normalize each parameter to its s=1 value
    bar_data = np.zeros((len(params), len(bar_scales)))
    for j, s in enumerate(bar_scales):
        st = bar_states[j]
        bar_data[0, j] = st.radius / ref.radius
        bar_data[1, j] = st.density / ref.density
        bar_data[2, j] = st.surface_gravity / ref.surface_gravity
        bar_data[3, j] = st.escape_velocity / ref.escape_velocity
        bar_data[4, j] = st.compactness / ref.compactness

    x = np.arange(len(bar_scales))
    width = 0.15
    for i in range(len(params)):
        offset = (i - len(params)/2 + 0.5) * width
        bars = ax3.bar(x + offset, bar_data[i], width, color=colors[i % len(colors)],
                      label=param_labels[i], alpha=0.8)

    ax3.set_xticks(x)
    ax3.set_xticklabels([f's={s}' for s in bar_scales])
    ax3.set_yscale('log')
    if language == 'de':
        ax3.set_ylabel('Normierter Wert (bezogen auf s=1)', fontsize=12)
        ax3.set_title('3. Gruppierte Parametervergleiche', fontsize=14, fontweight='bold')
    else:
        ax3.set_ylabel('Normalized value (relative to s=1)', fontsize=12)
        ax3.set_title('3. Grouped Parameter Comparison', fontsize=14, fontweight='bold')
    ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.tick_params(axis='both', labelsize=11)

    # --- Subplot 4: Fate trajectory in density-compactness space ---
    density_arr = np.array([st.density for st in states])
    compact_arr = np.array([st.compactness for st in states])

    # Color the trajectory by classification
    for i in range(len(states) - 1):
        clr = _classification_color(states[i].classification)
        ax4.loglog([density_arr[i], density_arr[i+1]],
                  [compact_arr[i], compact_arr[i+1]],
                  color=clr, linewidth=2.5)

    # Mark key points
    for s_mark in [1.0, 0.1, 0.01]:
        st = earth_state_at_scale(s_mark, constants)
        clr = _classification_color(st.classification)
        ax4.plot(st.density, st.compactness, 'o', color=clr, markersize=10, zorder=5)
        ax4.annotate(f's={s_mark}', xy=(st.density, st.compactness),
                    xytext=(10, 10), textcoords='offset points', fontsize=9,
                    fontweight='bold', color=clr)

    # Black hole threshold line
    ax4.axhline(y=1.0, color=COLORS['black_hole'], linestyle='--', linewidth=1.5,
               label='Black hole threshold (C=1)' if language != 'de'
               else 'Schwarzes-Loch-Schwelle (C=1)')

    if language == 'de':
        ax4.set_xlabel('Dichte [kg/m3]', fontsize=12)
        ax4.set_ylabel('Kompaktheit C = Rs/R', fontsize=12)
        ax4.set_title('4. Schicksal der Erde: Dichte-Kompaktheits-Trajektorie',
                      fontsize=14, fontweight='bold')
    else:
        ax4.set_xlabel('Density [kg/m3]', fontsize=12)
        ax4.set_ylabel('Compactness C = Rs/R', fontsize=12)
        ax4.set_title('4. Fate of Earth: Density-Compactness Trajectory',
                      fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, which='both')
    ax4.tick_params(axis='both', labelsize=11)

    if language == 'de':
        fig.suptitle('Kombinierte Erdentwicklung', fontsize=16, fontweight='bold', y=0.98)
    else:
        fig.suptitle('Combined Earth Evolution', fontsize=16, fontweight='bold', y=0.98)
    if save:
        os.makedirs(VIS_DIR, exist_ok=True)
        suffix = '_de' if language == 'de' else ''
        fpath = os.path.join(VIS_DIR, f'earth_combined{suffix}.png')
        fig.savefig(fpath, dpi=150, bbox_inches='tight')
        print(f"Saved: {fpath}")
    if show:
        plt.show()
    return fig


# =============================================================================
# VISUALIZATION 11: EARTH COLLAPSE SUMMARY
# =============================================================================

def plot_earth_collapse_summary(
    constants=None, language='en', save=True, show=True):
    """
    Visualization #11 - Collapse Summary. 4 vertical subplots.
    Visualisierung #11 - Kollapszusammenfassung.

    Subplot 1: Chandrasekhar mass crossing diagram
    Subplot 2: Compactness evolution with thresholds
    Subplot 3: Gravitational coupling alpha_G vs electromagnetic alpha
    Subplot 4: Summary table of key results

    Args:
        constants: Physical constants (uses standard if None)
        language: 'en' or 'de'
        save: Whether to save the figure
        show: Whether to display the figure

    Returns:
        matplotlib Figure object
    """
    if constants is None:
        constants = get_constants()

    s_values = np.logspace(-2, 0, 500)
    states = earth_evolution_track(s_values, constants)
    ref = earth_state_at_scale(1.0, constants)

    s_arr = np.array([st.hbar_scale for st in states])
    mch_arr = np.array([st.chandrasekhar_mass for st in states])
    compact_arr = np.array([st.compactness for st in states])
    alpha_g_arr = np.array([st.alpha_G for st in states])
    colors = get_sequence()

    M_earth = constants.M_earth
    M_sun = constants.M_sun

    fig = plt.figure(figsize=(12, 32))
    gs = fig.add_gridspec(4, 1, hspace=0.5)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])
    ax4 = fig.add_subplot(gs[3])

    # --- Subplot 1: Chandrasekhar mass crossing ---
    mch_solar = mch_arr / M_sun
    m_earth_solar = M_earth / M_sun

    ax1.loglog(s_arr, mch_solar, color=colors[0], linewidth=2.5,
              label='Chandrasekhar mass M_Ch(s)' if language != 'de'
              else 'Chandrasekhar-Masse M_Ch(s)')
    ax1.axhline(y=m_earth_solar, color=colors[3], linestyle='--', linewidth=1.5,
               label='Earth mass' if language != 'de' else 'Erdmasse')

    # Find crossing
    s_cross = _find_threshold_scale(states, 'chandrasekhar_mass', M_earth,
                                     direction='below')
    if s_cross is not None:
        ax1.axvline(x=s_cross, color=colors[4], linestyle=':', linewidth=1.5, alpha=0.7)
        mch_at_cross = chandrasekhar_mass_at_scale(s_cross, constants) / M_sun
        ax1.plot(s_cross, mch_at_cross, 'o', color=colors[4], markersize=10, zorder=5)
        cross_label = f's = {s_cross:.4f}' if language != 'de' else f's = {s_cross:.4f}'
        ax1.annotate(cross_label, xy=(s_cross, mch_at_cross),
                    xytext=(15, 15), textcoords='offset points', fontsize=10,
                    fontweight='bold', color=colors[4],
                    arrowprops=dict(arrowstyle='->', color=colors[4]))

    ax1.fill_between(s_arr, mch_solar, m_earth_solar,
                     where=mch_solar >= m_earth_solar,
                     alpha=0.15, color=colors[0],
                     label='M_Ch > M_Earth (stable)' if language != 'de'
                     else 'M_Ch > M_Erde (stabil)')
    ax1.fill_between(s_arr, mch_solar, m_earth_solar,
                     where=mch_solar < m_earth_solar,
                     alpha=0.15, color=colors[4],
                     label='M_Ch < M_Earth (collapse)' if language != 'de'
                     else 'M_Ch < M_Erde (Kollaps)')

    if language == 'de':
        ax1.set_xlabel('hbar-Skalierungsfaktor s', fontsize=12)
        ax1.set_ylabel('Masse [Sonnenmassen]', fontsize=12)
        ax1.set_title('1. Chandrasekhar-Massengrenze vs. Erdmasse',
                      fontsize=14, fontweight='bold')
    else:
        ax1.set_xlabel('hbar scaling factor s', fontsize=12)
        ax1.set_ylabel('Mass [solar masses]', fontsize=12)
        ax1.set_title('1. Chandrasekhar Mass Limit vs. Earth Mass',
                      fontsize=14, fontweight='bold')
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=10)
    ax1.grid(True, alpha=0.3, which='both')
    ax1.tick_params(axis='both', labelsize=11)

    # Fix #4: Add Chandrasekhar Limit calculation annotation for scaled universe
    # Updated based on reviewer feedback - show actual mass value (2,800 kg)!
    if language == 'de':
        ch_calc_text = (
            'SKALIERUNG DER CHANDRASEKHAR-GRENZE:\n'
            'Formel: M_Ch ∝ (ℏc/G)^(3/2)\n\n'
            'In unserem Szenario: G → G×10³⁶, ℏ → ℏ×10¹⁸\n'
            'M_Ch(skaliert) = 1.4 M☉ × (10¹⁸/10³⁶)^(3/2)\n'
            '               = 1.4 M☉ × (10⁻¹⁸)^(3/2)\n'
            '               = 1.4 M☉ × 10⁻²⁷\n'
            '               = 2.800 kg (Masse eines Autos!)\n\n'
            'BEDEUTUNG: Erde (6×10²⁴ kg) ist 10²¹-mal\n'
            'ÜBER der neuen Chandrasekhar-Grenze!\n\n'
            'Die Erde wird zum entarteten Objekt – NICHT\n'
            'weil sie die alte Grenze erreicht, sondern\n'
            'weil die Grenze selbst so stark gesunken ist!'
        )
    else:
        ch_calc_text = (
            'CHANDRASEKHAR LIMIT SCALING:\n'
            'Formula: M_Ch ∝ (ℏc/G)^(3/2)\n\n'
            'In our scenario: G → G×10³⁶, ℏ → ℏ×10¹⁸\n'
            'M_Ch(scaled) = 1.4 M☉ × (10¹⁸/10³⁶)^(3/2)\n'
            '             = 1.4 M☉ × (10⁻¹⁸)^(3/2)\n'
            '             = 1.4 M☉ × 10⁻²⁷\n'
            '             = 2,800 kg (mass of a small car!)\n\n'
            'IMPLICATION: Earth (6×10²⁴ kg) is 10²¹ times\n'
            'ABOVE the new Chandrasekhar limit!\n\n'
            'Earth becomes degenerate – NOT because it\n'
            'reaches the original limit, but because\n'
            'the limit itself has dropped dramatically!'
        )

    # Position the text box in the upper right area of the plot
    ax1.text(0.98, 0.98, ch_calc_text, fontsize=8, va='top', ha='right',
             transform=ax1.transAxes,
             bbox=dict(boxstyle='round,pad=0.4', facecolor=COLORS['box_error'],
                      edgecolor=COLORS['collapse'], linewidth=2, alpha=0.95),
             color=COLORS['collapse'], fontweight='normal', family='monospace')

    # --- Subplot 2: Compactness evolution with thresholds ---
    ax2.loglog(s_arr, compact_arr, color=colors[1], linewidth=2.5,
              label='Compactness C(s)' if language != 'de' else 'Kompaktheit C(s)')

    # Threshold lines
    ax2.axhline(y=1.0, color=COLORS['black_hole'], linestyle='--', linewidth=1.5,
               label='Black hole (C=1)' if language != 'de' else 'Schwarzes Loch (C=1)')
    ax2.axhline(y=0.5, color=COLORS['neutron_star'], linestyle=':', linewidth=1.5,
               label='Neutron star range' if language != 'de' else 'Neutronenstern-Bereich')
    ax2.axhline(y=1e-4, color=COLORS['white_dwarf'], linestyle=':', linewidth=1.0, alpha=0.7,
               label='White dwarf range' if language != 'de' else 'Weisser-Zwerg-Bereich')

    # Mark where compactness reaches key values
    s_c05 = _find_threshold_scale(states, 'compactness', 0.5, direction='above')
    s_c1 = _find_threshold_scale(states, 'compactness', 1.0, direction='above')

    if s_c05 is not None:
        ax2.plot(s_c05, 0.5, 'o', color=COLORS['neutron_star'], markersize=10, zorder=5)
    if s_c1 is not None:
        ax2.plot(s_c1, 1.0, 's', color=COLORS['black_hole'], markersize=12, zorder=5)

    if language == 'de':
        ax2.set_xlabel('hbar-Skalierungsfaktor s', fontsize=12)
        ax2.set_ylabel('Kompaktheit C = 2GM/(Rc^2)', fontsize=12)
        ax2.set_title('2. Kompaktheitsentwicklung', fontsize=14, fontweight='bold')
    else:
        ax2.set_xlabel('hbar scaling factor s', fontsize=12)
        ax2.set_ylabel('Compactness C = 2GM/(Rc^2)', fontsize=12)
        ax2.set_title('2. Compactness Evolution', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=10)
    ax2.grid(True, alpha=0.3, which='both')
    ax2.tick_params(axis='both', labelsize=11)

    # --- Subplot 3: Gravitational coupling alpha_G vs electromagnetic alpha ---
    alpha_em = constants.alpha  # ~1/137, constant

    ax3.loglog(s_arr, alpha_g_arr, color=colors[2], linewidth=2.5,
              label='alpha_G(s)' if language != 'de' else 'alpha_G(s)')
    ax3.axhline(y=alpha_em, color=colors[4], linestyle='--', linewidth=1.5,
               label='alpha_EM = 1/137' if language != 'de' else 'alpha_EM = 1/137')

    # Find where alpha_G crosses alpha_EM
    s_alpha_cross = _find_threshold_scale(states, 'alpha_G', alpha_em, direction='above')
    if s_alpha_cross is not None:
        ax3.axvline(x=s_alpha_cross, color=colors[5], linestyle=':', linewidth=1.5, alpha=0.7)
        ax3.plot(s_alpha_cross, alpha_em, 'D', color=colors[5], markersize=10, zorder=5)
        cross_label = f's = {s_alpha_cross:.2e}'
        ax3.annotate(cross_label, xy=(s_alpha_cross, alpha_em),
                    xytext=(15, 15), textcoords='offset points', fontsize=10,
                    fontweight='bold', color=colors[5],
                    arrowprops=dict(arrowstyle='->', color=colors[5]))

    # Shade regions
    ax3.fill_between(s_arr, alpha_g_arr, alpha_em,
                     where=alpha_g_arr < alpha_em,
                     alpha=0.1, color=colors[4],
                     label='EM dominates' if language != 'de' else 'EM dominiert')
    ax3.fill_between(s_arr, alpha_g_arr, alpha_em,
                     where=alpha_g_arr >= alpha_em,
                     alpha=0.1, color=colors[2],
                     label='Gravity dominates' if language != 'de' else 'Gravitation dominiert')

    if language == 'de':
        ax3.set_xlabel('hbar-Skalierungsfaktor s', fontsize=12)
        ax3.set_ylabel('Kopplungskonstante', fontsize=12)
        ax3.set_title('3. Gravitationskopplung vs. elektromagnetische Kopplung',
                      fontsize=14, fontweight='bold')
    else:
        ax3.set_xlabel('hbar scaling factor s', fontsize=12)
        ax3.set_ylabel('Coupling constant', fontsize=12)
        ax3.set_title('3. Gravitational Coupling vs. Electromagnetic Coupling',
                      fontsize=14, fontweight='bold')
    ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=10)
    ax3.grid(True, alpha=0.3, which='both')
    ax3.tick_params(axis='both', labelsize=11)

    # --- Subplot 4: Summary table of key results ---
    ax4.axis('off')

    # Compute key thresholds
    s_cross_ch = _find_threshold_scale(states, 'chandrasekhar_mass',
                                        M_earth, direction='below')
    s_wd = _find_threshold_scale(states, 'density', 1e9, direction='above')
    s_ns = _find_threshold_scale(states, 'density', 4e14, direction='above')
    s_bh = _find_threshold_scale(states, 'compactness', 1.0, direction='above')

    if language == 'de':
        headers = ['Ereignis', 'hbar-Skala s', 'Erdradius', 'Dichte [kg/m3]', 'Klassifikation']
        rows_data = [
            ['Normalzustand', '1.0',
             f'{ref.radius:.2e} m', f'{ref.density:.2e}', ref.classification],
        ]
    else:
        headers = ['Event', 'hbar scale s', 'Earth radius', 'Density [kg/m3]', 'Classification']
        rows_data = [
            ['Normal state', '1.0',
             f'{ref.radius:.2e} m', f'{ref.density:.2e}', ref.classification],
        ]

    # Add threshold rows
    # Each entry: (scale, label_en, label_de, expected_classification)
    # We use expected_classification to avoid floating-point precision issues
    threshold_info = [
        (s_cross_ch, 'M_Ch < M_Earth', 'M_Ch < M_Erde', 'compressed'),
        (s_wd, 'White dwarf density', 'Weisser-Zwerg-Dichte', 'white_dwarf'),
        (s_ns, 'Neutron star density', 'Neutronenstern-Dichte', 'neutron_star'),
        (s_bh, 'Black hole (C>=1)', 'Schwarzes Loch (C>=1)', 'black_hole'),
    ]
    for s_val, label_en, label_de, expected_class in threshold_info:
        if s_val is not None:
            st = earth_state_at_scale(s_val, constants)
            label = label_de if language == 'de' else label_en
            rows_data.append([label, f'{s_val:.4f}',
                            f'{st.radius:.2e} m', f'{st.density:.2e}',
                            expected_class])

    table = ax4.table(cellText=rows_data, colLabels=headers,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.8)

    # Style header
    for j in range(len(headers)):
        cell = table[0, j]
        cell.set_facecolor(COLORS['primary_blue'])
        cell.set_text_props(color='white', fontweight='bold')

    # Color rows by classification
    phase_bg = {
        'planet': '#E8F5E9',
        'compressed': '#FFF3E0',
        'white_dwarf': '#F3E5F5',
        'neutron_star': '#E3F2FD',
        'black_hole': '#FFEBEE',
    }
    for i, row in enumerate(rows_data):
        cls = row[-1]
        bg = phase_bg.get(cls, '#FFFFFF')
        for j in range(len(headers)):
            table[i + 1, j].set_facecolor(bg)

    if language == 'de':
        ax4.set_title('4. Zusammenfassungstabelle: Schluesselergebnisse',
                      fontsize=14, fontweight='bold', pad=20)
    else:
        ax4.set_title('4. Summary Table: Key Results',
                      fontsize=14, fontweight='bold', pad=20)

    if language == 'de':
        fig.suptitle('Erdkollaps-Zusammenfassung', fontsize=16, fontweight='bold', y=0.98)
    else:
        fig.suptitle('Earth Collapse Summary', fontsize=16, fontweight='bold', y=0.98)

    if save:
        os.makedirs(VIS_DIR, exist_ok=True)
        suffix = '_de' if language == 'de' else ''
        fpath = os.path.join(VIS_DIR, f'earth_collapse_summary{suffix}.png')
        fig.savefig(fpath, dpi=150, bbox_inches='tight')
        print(f"Saved: {fpath}")
    if show:
        plt.show()
    return fig


# =============================================================================
# GENERATION AND VERIFICATION
# =============================================================================

def generate_all_earth_collapse_plots(
    constants=None, language='en', save=True, show=False):
    """
    Generate all Earth collapse visualizations (9, 10, 11).
    Erzeugt alle Erdkollaps-Visualisierungen.

    Args:
        constants: Physical constants (uses standard if None)
        language: 'en' or 'de'
        save: Whether to save figures
        show: Whether to display figures

    Returns:
        List of matplotlib Figure objects
    """
    if constants is None:
        constants = get_constants()

    figs = []
    print(f"Generating Earth collapse plots (language={language})...")

    print("  [9] Earth Change Diagram...")
    figs.append(plot_earth_change_diagram(constants, language, save, show))

    print("  [10] Earth Combined...")
    figs.append(plot_earth_combined(constants, language, save, show))

    print("  [11] Earth Collapse Summary...")
    figs.append(plot_earth_collapse_summary(constants, language, save, show))

    print(f"Done. Generated {len(figs)} figures.")
    return figs


def verify_earth_collapse_physics(constants=None):
    """
    Verify key physical relationships in the Earth collapse module.
    Ueberprueft die wichtigsten physikalischen Beziehungen im Erdkollapsmodul.

    Runs a series of consistency checks on the scaling relations and
    computed values. Prints PASS/FAIL for each check.

    Args:
        constants: Physical constants (uses standard if None)

    Returns:
        True if all checks pass, False otherwise
    """
    if constants is None:
        constants = get_constants()

    all_pass = True

    def check(name, condition):
        nonlocal all_pass
        status = "PASS" if condition else "FAIL"
        if not condition:
            all_pass = False
        print(f"  [{status}] {name}")

    print("=" * 60)
    print("Earth Collapse Module Verification")
    print("=" * 60)

    # Check 1: At s=1, state should match standard Earth values
    ref = earth_state_at_scale(1.0, constants)
    check("s=1 radius matches R_earth",
          abs(ref.radius - constants.R_earth) / constants.R_earth < 1e-10)
    check("s=1 classification is planet",
          ref.classification == 'planet')

    # Check 2: Radius scaling R ~ s^2
    s_test = 0.5
    st_half = earth_state_at_scale(s_test, constants)
    expected_ratio = s_test ** 2
    actual_ratio = st_half.radius / ref.radius
    check(f"Radius scales as s^2 (s={s_test})",
          abs(actual_ratio - expected_ratio) / expected_ratio < 1e-10)

    # Check 3: Density scaling rho ~ 1/s^6
    expected_density_ratio = 1.0 / s_test ** 6
    actual_density_ratio = st_half.density / ref.density
    check(f"Density scales as 1/s^6 (s={s_test})",
          abs(actual_density_ratio - expected_density_ratio) / expected_density_ratio < 1e-6)

    # Check 4: Surface gravity scaling g ~ 1/s^4
    expected_g_ratio = 1.0 / s_test ** 4
    actual_g_ratio = st_half.surface_gravity / ref.surface_gravity
    check(f"Surface gravity scales as 1/s^4 (s={s_test})",
          abs(actual_g_ratio - expected_g_ratio) / expected_g_ratio < 1e-10)

    # Check 5: Escape velocity scaling v_esc ~ 1/s
    expected_vesc_ratio = 1.0 / s_test
    actual_vesc_ratio = st_half.escape_velocity / ref.escape_velocity
    check(f"Escape velocity scales as 1/s (s={s_test})",
          abs(actual_vesc_ratio - expected_vesc_ratio) / expected_vesc_ratio < 1e-10)

    # Check 6: Compactness scaling C ~ 1/s^2
    expected_c_ratio = 1.0 / s_test ** 2
    actual_c_ratio = st_half.compactness / ref.compactness
    check(f"Compactness scales as 1/s^2 (s={s_test})",
          abs(actual_c_ratio - expected_c_ratio) / expected_c_ratio < 1e-10)

    # Check 7: Chandrasekhar mass scaling M_Ch ~ s^(3/2)
    mch_1 = chandrasekhar_mass_at_scale(1.0, constants)
    mch_half = chandrasekhar_mass_at_scale(s_test, constants)
    expected_mch_ratio = s_test ** 1.5
    actual_mch_ratio = mch_half / mch_1
    check(f"Chandrasekhar mass scales as s^(3/2) (s={s_test})",
          abs(actual_mch_ratio - expected_mch_ratio) / expected_mch_ratio < 1e-10)

    # Check 8: alpha_G scaling ~ 1/s
    expected_ag_ratio = 1.0 / s_test
    actual_ag_ratio = st_half.alpha_G / ref.alpha_G
    check(f"alpha_G scales as 1/s (s={s_test})",
          abs(actual_ag_ratio - expected_ag_ratio) / expected_ag_ratio < 1e-10)

    # Check 9: At very small s, should eventually become black hole
    st_tiny = earth_state_at_scale(0.01, constants)
    check("s=0.01 has elevated compactness (C > 1e-6)",
          st_tiny.compactness > 1e-6)

    # Check 10: Chandrasekhar mass at s=1 is reasonable (~1.4 solar masses)
    mch_solar = mch_1 / constants.M_sun
    check(f"M_Ch(s=1) ~ 1.4 solar masses (got {mch_solar:.3f})",
          1.0 < mch_solar < 2.0)

    # Check 11: Evolution track produces correct number of states
    track = earth_evolution_track(np.array([0.01, 0.1, 1.0]), constants)
    check("Evolution track returns correct count",
          len(track) == 3)

    # Check 12: Atmosphere height scaling H ~ s^4
    expected_h_ratio = s_test ** 4
    actual_h_ratio = st_half.atmosphere_height / ref.atmosphere_height
    check(f"Atmosphere height scales as s^4 (s={s_test})",
          abs(actual_h_ratio - expected_h_ratio) / expected_h_ratio < 1e-6)

    print("=" * 60)
    print(f"Result: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
    print("=" * 60)
    return all_pass


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')

    print('Earth Collapse Evolution Module')
    print('=' * 40)

    # Run verification
    passed = verify_earth_collapse_physics()

    if passed:
        # Generate all plots
        generate_all_earth_collapse_plots(language='en', save=True, show=False)
        generate_all_earth_collapse_plots(language='de', save=True, show=False)
        print('All plots generated successfully.')
    else:
        print('Verification failed. Fix issues before generating plots.')
