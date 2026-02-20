"""
Hydrostatic Equilibrium Module for Jugend forscht 2026 Physics Visualization Project
Hydrostatisches-Gleichgewicht-Modul fuer Jugend forscht 2026 Physik-Visualisierungsprojekt

This module models hydrostatic equilibrium: the balance between gravity pulling
inward and pressure pushing outward inside a self-gravitating body. It solves
the Lane-Emden equation numerically for different polytropic indices and
gravitational constant values, then maps the dimensionless solutions to
physical units.

Key visualizations:
1. Pressure, density, enclosed-mass, and gravity profiles for polytropic stars
2. How central pressure, equilibrium radius, and density change with G
3. Phase diagrams, stability maps, Lane-Emden solutions, and TOV corrections

The central question: How does scaling G affect the internal structure
and stability of self-gravitating bodies?

Author: Jugend forscht 2026 Project
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Optional, List, Tuple
from dataclasses import dataclass

from .constants import get_constants, PhysicalConstants
from .color_scheme import COLORS, get_sequence


# Output directory for visualizations
VIS_DIR = os.path.join(os.path.dirname(__file__), '..', 'visualizations')


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class HydrostaticProfile:
    """
    Container for a hydrostatic equilibrium profile.
    Behaelter fuer ein hydrostatisches Gleichgewichtsprofil.
    """
    radii: 'np.ndarray'            # Radial positions [m]
    pressures: 'np.ndarray'        # Pressure profile [Pa]
    densities: 'np.ndarray'        # Density profile [kg/m^3]
    masses_enclosed: 'np.ndarray'  # Enclosed mass profile [kg]
    total_radius: float            # Surface radius (where P=0) [m]
    central_pressure: float        # Central pressure [Pa]
    central_density: float         # Central density [kg/m^3]
    polytropic_index: float        # n value used
    is_stable: bool                # Whether equilibrium exists


# ---------------------------------------------------------------------------
# Lane-Emden solver
# ---------------------------------------------------------------------------

def lane_emden_solve(n: float, num_points: int = 1000) -> 'Tuple[np.ndarray, np.ndarray, float]':
    """
    Solve the Lane-Emden equation numerically using 4th-order Runge-Kutta.
    Loest die Lane-Emden-Gleichung numerisch mit Runge-Kutta 4. Ordnung.

    The Lane-Emden equation is:
        (1/xi^2) d/dxi (xi^2 d theta / d xi) = - theta^n

    which is equivalent to the system:
        d theta / d xi  = phi
        d phi   / d xi  = -theta^n  -  2 phi / xi

    Boundary conditions: theta(0) = 1, theta'(0) = 0.
    Integration starts at xi = 0.001 to avoid the coordinate singularity at
    xi = 0 and proceeds until theta <= 0 (the stellar surface) or a maximum
    xi is reached.

    For n >= 5 the solution never reaches theta = 0 (infinite radius); in
    that case the integration is terminated at xi_max = 50.

    Args:
        n: Polytropic index (0 <= n < 5 for finite radius solutions).
        num_points: Number of integration steps.

    Returns:
        (xi_array, theta_array, xi_surface) where xi_surface is the first
        zero of theta (or the maximum xi reached for n >= 5).
    """
    xi_start = 1e-3
    xi_max = 50.0
    dxi = (xi_max - xi_start) / num_points

    # Initial conditions at xi_start using series expansion
    # theta ~ 1 - xi^2/6 + n xi^4/120 ...
    theta0 = 1.0 - xi_start**2 / 6.0
    phi0 = -xi_start / 3.0  # d theta/d xi ~ -xi/3

    xi_list = [xi_start]
    theta_list = [theta0]

    theta = theta0
    phi = phi0
    xi = xi_start
    xi_surface = xi_max  # default for n >= 5

    def derivs(xi_val, theta_val, phi_val):
        """Return (d_theta/d_xi, d_phi/d_xi)."""
        if theta_val < 0:
            theta_val = 0.0
        d_theta = phi_val
        d_phi = -theta_val**n - 2.0 * phi_val / xi_val
        return d_theta, d_phi

    for _ in range(num_points):
        # RK4
        k1_t, k1_p = derivs(xi, theta, phi)
        k2_t, k2_p = derivs(xi + 0.5 * dxi, theta + 0.5 * dxi * k1_t,
                            phi + 0.5 * dxi * k1_p)
        k3_t, k3_p = derivs(xi + 0.5 * dxi, theta + 0.5 * dxi * k2_t,
                            phi + 0.5 * dxi * k2_p)
        k4_t, k4_p = derivs(xi + dxi, theta + dxi * k3_t,
                            phi + dxi * k3_p)

        theta_new = theta + (dxi / 6.0) * (k1_t + 2 * k2_t + 2 * k3_t + k4_t)
        phi_new = phi + (dxi / 6.0) * (k1_p + 2 * k2_p + 2 * k3_p + k4_p)
        xi_new = xi + dxi

        if theta_new <= 0:
            # Linear interpolation to find exact surface
            frac = theta / (theta - theta_new) if (theta - theta_new) != 0 else 1.0
            xi_surface = xi + frac * dxi
            xi_list.append(xi_surface)
            theta_list.append(0.0)
            break

        xi_list.append(xi_new)
        theta_list.append(theta_new)
        theta = theta_new
        phi = phi_new
        xi = xi_new

    return np.array(xi_list), np.array(theta_list), xi_surface


# ---------------------------------------------------------------------------
# Physical profile from Lane-Emden solution
# ---------------------------------------------------------------------------

def polytropic_pressure_profile(
    mass: float,
    radius: float,
    n: float,
    G: float,
    num_points: int = 200
) -> 'HydrostaticProfile':
    """
    Map a Lane-Emden solution to physical units for a body of given mass and radius.
    Bildet eine Lane-Emden-Loesung auf physikalische Einheiten ab.

    The mapping is:
        r_n      = radius / xi_surface        (length scale)
        rho_c    = M / (4 pi r_n^3 |xi_1^2 theta'(xi_1)|)
        P_c      = (4 pi G rho_c^2 r_n^2) / (n + 1)
        r(xi)    = r_n * xi
        rho(xi)  = rho_c * theta^n
        P(xi)    = P_c * theta^(n+1)
        m(xi)    = integrated mass

    Args:
        mass: Total mass of the body [kg].
        radius: Total (surface) radius [m].
        n: Polytropic index.
        G: Gravitational constant [m^3 kg^-1 s^-2].
        num_points: Resolution of the Lane-Emden integration.

    Returns:
        HydrostaticProfile with arrays in physical units.
    """
    xi_arr, theta_arr, xi1 = lane_emden_solve(n, num_points=num_points)

    # Length scale
    r_n = radius / xi1

    # Central density from total mass normalisation.
    # M = 4 pi rho_c r_n^3 |xi_1^2 theta'(xi_1)|
    if len(xi_arr) >= 2:
        d_theta_surface = (theta_arr[-1] - theta_arr[-2]) / (xi_arr[-1] - xi_arr[-2])
    else:
        d_theta_surface = -1.0 / 3.0
    mass_integral = abs(xi1**2 * d_theta_surface)
    rho_c = mass / (4.0 * np.pi * r_n**3 * mass_integral) if mass_integral > 0 else 1.0

    # Central pressure
    P_c = (4.0 * np.pi * G * rho_c**2 * r_n**2) / (n + 1.0)

    # Physical arrays
    radii = r_n * xi_arr
    theta_safe = np.maximum(theta_arr, 0.0)
    densities = rho_c * theta_safe**n
    pressures = P_c * theta_safe**(n + 1.0)

    # Enclosed mass via numerical integration  m(r) = 4 pi int_0^r rho r'^2 dr'
    masses_enclosed = np.zeros_like(radii)
    for i in range(1, len(radii)):
        dr = radii[i] - radii[i - 1]
        masses_enclosed[i] = masses_enclosed[i - 1] + 4.0 * np.pi * densities[i] * radii[i]**2 * dr

    # Stability: finite radius polytropes with n < 5 are stable
    is_stable = (n < 5.0) and (xi1 < 49.0)

    return HydrostaticProfile(
        radii=radii,
        pressures=pressures,
        densities=densities,
        masses_enclosed=masses_enclosed,
        total_radius=radius,
        central_pressure=P_c,
        central_density=rho_c,
        polytropic_index=n,
        is_stable=is_stable,
    )


# ---------------------------------------------------------------------------
# TOV correction
# ---------------------------------------------------------------------------

def tov_correction_factor(
    r: float,
    m: float,
    P: float,
    rho: float,
    c_light: float,
    G: float
) -> float:
    """
    Compute the Tolman-Oppenheimer-Volkoff correction factor.
    Berechnet den TOV-Korrekturfaktor.

    The TOV equation modifies Newtonian hydrostatic equilibrium by
    three multiplicative relativistic correction terms:

        factor = (1 + P/(rho c^2)) * (1 + 4 pi r^3 P / (m c^2))
                 / (1 - 2 G m / (r c^2))

    Args:
        r: Radial coordinate [m].
        m: Enclosed mass at r [kg].
        P: Pressure at r [Pa].
        rho: Density at r [kg/m^3].
        c_light: Speed of light [m/s].
        G: Gravitational constant [m^3 kg^-1 s^-2].

    Returns:
        Dimensionless correction factor (>= 1 for bound objects).
    """
    c2 = c_light**2
    if rho <= 0 or m <= 0 or r <= 0:
        return 1.0

    term1 = 1.0 + P / (rho * c2)
    term2 = 1.0 + 4.0 * np.pi * r**3 * P / (m * c2)
    denom = 1.0 - 2.0 * G * m / (r * c2)

    if denom <= 0:
        # Inside or at Schwarzschild radius -- return a large number
        return 1e10

    return term1 * term2 / denom


# ---------------------------------------------------------------------------
# Simple central pressure estimate
# ---------------------------------------------------------------------------

def hydrostatic_central_pressure(mass: float, radius: float, G: float) -> float:
    """
    Rough central-pressure estimate for a uniform-density sphere.
    Grobe Abschaetzung des Zentraldrucks fuer eine Kugel gleichmaessiger Dichte.

    Formula: P_c = (3 / 8 pi) G M^2 / R^4

    Args:
        mass: Total mass [kg].
        radius: Radius [m].
        G: Gravitational constant [m^3 kg^-1 s^-2].

    Returns:
        Central pressure [Pa].
    """
    return (3.0 / (8.0 * np.pi)) * G * mass**2 / radius**4


# ---------------------------------------------------------------------------
# Profile plots
# ---------------------------------------------------------------------------

def plot_hydrostatic_profiles(
    constants=None,
    language='en',
    save=True,
    show=True,
):
    """
    Plot P(r), rho(r), m(r) profiles for polytropic stars and g(r) for scaled G.
    Zeichnet P(r)-, rho(r)-, m(r)-Profile fuer polytropische Sterne und g(r)
    fuer skaliertes G.

    Four subplots:
        1. Normalized pressure   P(r)/P_c   vs r/R  for n=0, 3/2, 3
        2. Normalized density    rho(r)/rho_c vs r/R
        3. Normalized enclosed mass  m(r)/M  vs r/R
        4. Gravitational acceleration g(r) for different G scalings (n=1.5)

    Args:
        constants: PhysicalConstants instance (default: get_constants()).
        language: 'en' or 'de' for bilingual labels.
        save: Whether to save the figure to VIS_DIR.
        show: Whether to display the figure interactively.

    Returns:
        matplotlib Figure.
    """
    if constants is None:
        constants=get_constants()
    colors=get_sequence()

    M=constants.M_earth
    R=constants.R_earth
    G=constants.G

    # Define polytropes
    polytropes=[
        (0,   'n=0 (uniform)',     '-'),
        (1.5, 'n=3/2 (convective)','--'),
        (3,   'n=3 (Eddington)',   '-.'),
    ]

    # Compute profiles
    profiles=[]
    for n_val,label,ls in polytropes:
        prof=polytropic_pressure_profile(M,R,n_val,G,num_points=500)
        profiles.append(prof)

    fig,axes=plt.subplots(4,1,figsize=(12,32))
    fig.subplots_adjust(hspace=0.5,top=0.95,bottom=0.04)

    # ---- Subplot 1: Normalized pressure ----
    ax1=axes[0]
    for idx,((n_val,label,ls),prof) in enumerate(zip(polytropes,profiles)):
        r_norm=prof.radii/prof.total_radius
        P_norm=prof.pressures/prof.central_pressure if prof.central_pressure>0 else prof.pressures
        ax1.plot(r_norm,P_norm,ls,color=colors[idx%len(colors)],linewidth=2.5,label=label)
    if language=='de':
        ax1.set_xlabel('r / R',fontsize=12)
        ax1.set_ylabel('P(r) / P_c',fontsize=12)
        ax1.set_title('Druckprofil',fontsize=14,fontweight='bold',pad=15)
    else:
        ax1.set_xlabel('r / R',fontsize=12)
        ax1.set_ylabel('P(r) / P_c',fontsize=12)
        ax1.set_title('Pressure Profile',fontsize=14,fontweight='bold',pad=15)
    ax1.grid(True,alpha=0.3)
    ax1.legend(fontsize=11,loc='upper center',bbox_to_anchor=(0.5,-0.15),ncol=3)

    # ---- Subplot 2: Normalized density ----
    ax2=axes[1]
    for idx,((n_val,label,ls),prof) in enumerate(zip(polytropes,profiles)):
        r_norm=prof.radii/prof.total_radius
        rho_norm=prof.densities/prof.central_density if prof.central_density>0 else prof.densities
        ax2.plot(r_norm,rho_norm,ls,color=colors[idx%len(colors)],linewidth=2.5,label=label)
    if language=='de':
        ax2.set_xlabel('r / R',fontsize=12)
        ax2.set_ylabel('Dichte',fontsize=12)
        ax2.set_title('Dichteprofil',fontsize=14,fontweight='bold',pad=15)
    else:
        ax2.set_xlabel('r / R',fontsize=12)
        ax2.set_ylabel('Density',fontsize=12)
        ax2.set_title('Density Profile',fontsize=14,fontweight='bold',pad=15)
    ax2.grid(True,alpha=0.3)
    ax2.legend(fontsize=11,loc='upper center',bbox_to_anchor=(0.5,-0.15),ncol=3)

    # ---- Subplot 3: Enclosed mass ----
    ax3=axes[2]
    for idx,((n_val,label,ls),prof) in enumerate(zip(polytropes,profiles)):
        r_norm=prof.radii/prof.total_radius
        m_total=prof.masses_enclosed[-1] if prof.masses_enclosed[-1]>0 else 1.0
        m_norm=prof.masses_enclosed/m_total
        ax3.plot(r_norm,m_norm,ls,color=colors[idx%len(colors)],linewidth=2.5,label=label)
    if language=='de':
        ax3.set_xlabel('r / R',fontsize=12)
        ax3.set_ylabel('m(r) / M',fontsize=12)
        ax3.set_title('Eingeschlossene Masse',fontsize=14,fontweight='bold',pad=15)
    else:
        ax3.set_xlabel('r / R',fontsize=12)
        ax3.set_ylabel('m(r) / M',fontsize=12)
        ax3.set_title('Enclosed Mass',fontsize=14,fontweight='bold',pad=15)
    ax3.grid(True,alpha=0.3)
    ax3.legend(fontsize=11,loc='upper center',bbox_to_anchor=(0.5,-0.15),ncol=3)

    # ---- Subplot 4: Gravitational acceleration ----
    ax4=axes[3]
    g_scales_list=[1,10,100,1000]
    n_plot=1.5
    for idx,gs in enumerate(g_scales_list):
        G_eff=G*gs
        prof=polytropic_pressure_profile(M,R,n_plot,G_eff,num_points=500)
        r_norm=prof.radii/prof.total_radius
        with np.errstate(divide='ignore',invalid='ignore'):
            g_acc=np.where(prof.radii>0,G_eff*prof.masses_enclosed/prof.radii**2,0.0)
        ax4.plot(r_norm,g_acc,'-',color=colors[idx%len(colors)],linewidth=2.5,label=f'G x {gs}')
    if language=='de':
        ax4.set_xlabel('r / R',fontsize=12)
        ax4.set_ylabel('g(r) [m/s^2]',fontsize=12)
        ax4.set_title('Gravitationsbeschleunigung bei skaliertem G (n=1.5)',fontsize=14,fontweight='bold',pad=15)
    else:
        ax4.set_xlabel('r / R',fontsize=12)
        ax4.set_ylabel('g(r) [m/s^2]',fontsize=12)
        ax4.set_title('Gravitational Acceleration for Scaled G (n=1.5)',fontsize=14,fontweight='bold',pad=15)
    ax4.grid(True,alpha=0.3)
    ax4.legend(fontsize=11,loc='upper center',bbox_to_anchor=(0.5,-0.15),ncol=3)

    if save:
        os.makedirs(VIS_DIR,exist_ok=True)
        suffix='_de' if language=='de' else ''
        filepath=os.path.join(VIS_DIR,f'hydrostatic_profiles{suffix}.png')
        fig.savefig(filepath,dpi=150,bbox_inches='tight')
        print(f'  Saved: {filepath}')
    if not show:
        plt.close(fig)
    return fig



def plot_hydrostatic_comparison(
    constants=None,
    language='en',
    save=True,
    show=True,
):
    """
    Compare hydrostatic properties as a function of the gravitational scaling factor.
    Vergleicht hydrostatische Eigenschaften als Funktion des Gravitationsskalierungsfaktors.
    """
    if constants is None:
        constants=get_constants()
    colors=get_sequence()
    M=constants.M_earth
    R_base=constants.R_earth
    G0=constants.G
    n_poly=1.5

    g_scales=np.logspace(0,6,60)
    P_central=np.zeros_like(g_scales)
    R_eq=np.zeros_like(g_scales)
    rho_central=np.zeros_like(g_scales)

    for i,gs in enumerate(g_scales):
        G_eff=G0*gs
        R_scaled=R_base*gs**(-0.25)
        P_central[i]=hydrostatic_central_pressure(M,R_scaled,G_eff)
        rho_central[i]=M/((4.0/3.0)*np.pi*R_scaled**3)
        R_eq[i]=R_scaled

    fig,axes=plt.subplots(4,1,figsize=(12,32))
    fig.subplots_adjust(hspace=0.5,top=0.95,bottom=0.04)

    ax1=axes[0]
    ax1.loglog(g_scales,P_central,'-',color=COLORS['scaled'],linewidth=2.5,label='P_c(G)')
    if language=='de':
        ax1.set_xlabel('G-Skalierungsfaktor',fontsize=12)
        ax1.set_ylabel('Zentraldruck P_c [Pa]',fontsize=12)
        ax1.set_title('Zentraldruck vs. Gravitationsskalierung',fontsize=14,fontweight='bold',pad=15)
    else:
        ax1.set_xlabel('G Scaling Factor',fontsize=12)
        ax1.set_ylabel('Central Pressure P_c [Pa]',fontsize=12)
        ax1.set_title('Central Pressure vs. Gravitational Scaling',fontsize=14,fontweight='bold',pad=15)
    ax1.grid(True,alpha=0.3)
    ax1.legend(fontsize=11,loc='upper center',bbox_to_anchor=(0.5,-0.15),ncol=3)

    ax2=axes[1]
    ax2.loglog(g_scales,R_eq/1e3,'-',color=COLORS['standard'],linewidth=2.5,label='R_eq(G)')
    if language=='de':
        ax2.set_xlabel('G-Skalierungsfaktor',fontsize=12)
        ax2.set_ylabel('Gleichgewichtsradius [km]',fontsize=12)
        ax2.set_title('Gleichgewichtsradius vs. Gravitationsskalierung',fontsize=14,fontweight='bold',pad=15)
    else:
        ax2.set_xlabel('G Scaling Factor',fontsize=12)
        ax2.set_ylabel('Equilibrium Radius [km]',fontsize=12)
        ax2.set_title('Equilibrium Radius vs. Gravitational Scaling',fontsize=14,fontweight='bold',pad=15)
    ax2.grid(True,alpha=0.3)
    ax2.legend(fontsize=11,loc='upper center',bbox_to_anchor=(0.5,-0.15),ncol=3)

    ax3=axes[2]
    ax3.loglog(g_scales,rho_central,'-',color=COLORS['quantum'],linewidth=2.5,label='rho_c(G)')
    if language=='de':
        ax3.set_xlabel('G-Skalierungsfaktor',fontsize=12)
        ax3.set_ylabel('Zentraldichte [kg/m^3]',fontsize=12)
        ax3.set_title('Zentraldichte vs. Gravitationsskalierung',fontsize=14,fontweight='bold',pad=15)
    else:
        ax3.set_xlabel('G Scaling Factor',fontsize=12)
        ax3.set_ylabel('Central Density [kg/m^3]',fontsize=12)
        ax3.set_title('Central Density vs. Gravitational Scaling',fontsize=14,fontweight='bold',pad=15)
    ax3.grid(True,alpha=0.3)
    ax3.legend(fontsize=11,loc='upper center',bbox_to_anchor=(0.5,-0.15),ncol=3)

    ax4=axes[3]
    overlay_scales=[1,1e2,1e4,1e6]
    for idx,gs in enumerate(overlay_scales):
        G_eff=G0*gs
        R_scaled=R_base*gs**(-0.25)
        prof=polytropic_pressure_profile(M,R_scaled,n_poly,G_eff,num_points=500)
        r_norm=prof.radii/prof.total_radius
        P_norm=prof.pressures/prof.central_pressure if prof.central_pressure>0 else prof.pressures
        lbl=f'G x {gs:.0e}' if gs>=10 else f'G x {int(gs)}'
        ax4.plot(r_norm,P_norm,'-',color=colors[idx%len(colors)],linewidth=2.5,label=lbl)
    if language=='de':
        ax4.set_xlabel('r / R',fontsize=12)
        ax4.set_ylabel('P(r) / P_c',fontsize=12)
        ax4.set_title('Druckprofile bei verschiedenen G-Skalierungen',fontsize=14,fontweight='bold',pad=15)
    else:
        ax4.set_xlabel('r / R',fontsize=12)
        ax4.set_ylabel('P(r) / P_c',fontsize=12)
        ax4.set_title('Pressure Profiles at Various G Scalings',fontsize=14,fontweight='bold',pad=15)
    ax4.grid(True,alpha=0.3)
    ax4.legend(fontsize=11,loc='upper center',bbox_to_anchor=(0.5,-0.15),ncol=3)

    if save:
        os.makedirs(VIS_DIR,exist_ok=True)
        suffix='_de' if language=='de' else ''
        filepath=os.path.join(VIS_DIR,f'hydrostatic_comparison{suffix}.png')
        fig.savefig(filepath,dpi=150,bbox_inches='tight')
        print(f'  Saved: {filepath}')
    if not show:
        plt.close(fig)
    return fig


def plot_hydrostatic_summary(
    constants=None,
    language='en',
    save=True,
    show=True,
):
    """
    Summary plots: phase diagram, stability map, Lane-Emden curves, TOV comparison.
    Zusammenfassungsdiagramme: Phasendiagramm, Stabilitaetskarte, Lane-Emden-Kurven, TOV-Vergleich.
    """
    if constants is None:
        constants=get_constants()
    colors=get_sequence()
    fig,axes=plt.subplots(4,1,figsize=(12,32))
    fig.subplots_adjust(hspace=0.5,top=0.95,bottom=0.04)

    # ---- Subplot 1: Phase diagram ----
    ax1=axes[0]
    log_rho=np.linspace(2,12,300)
    rho_arr=10.0**log_rho
    mu_e=2.0
    K_nr=(constants.hbar**2/(5*constants.m_e))*(3*np.pi**2)**(2/3)/(mu_e*constants.m_p)**(5/3)
    T_boundary_nr=K_nr*rho_arr**(2.0/3.0)*(2.0*constants.m_p)/constants.k_B
    K_r=(constants.hbar*constants.c/4)*(3*np.pi**2)**(1/3)/(mu_e*constants.m_p)**(4/3)
    rho_cross=(K_r/K_nr)**3 if K_nr>0 else 1e10
    log_rho_cross=np.log10(rho_cross) if rho_cross>0 else 10
    log_T_boundary=np.log10(np.maximum(T_boundary_nr,1.0))
    log_T_min=3.0;log_T_max=11.0
    ax1.fill_between(log_rho,log_T_boundary,log_T_max,alpha=0.25,color=COLORS['temp_hot'],
                     label='Thermal pressure dominates' if language=='en' else 'Thermischer Druck dominiert')
    mask_nr=log_rho<log_rho_cross
    ax1.fill_between(log_rho[mask_nr],log_T_min,log_T_boundary[mask_nr],alpha=0.25,color=COLORS['non_relativistic'],
                     label='Degeneracy pressure dominates' if language=='en' else 'Entartungsdruck dominiert')
    mask_r=log_rho>=log_rho_cross
    if np.any(mask_r):
        ax1.fill_between(log_rho[mask_r],log_T_min,log_T_boundary[mask_r],alpha=0.25,color=COLORS['relativistic'],
                         label='Relativistic degeneracy' if language=='en' else 'Relativistische Entartung')
    ax1.plot(log_rho,log_T_boundary,'-',color=COLORS['text_dark'],linewidth=2)
    if log_rho_cross<12:
        ax1.axvline(log_rho_cross,color=COLORS['muted'],linestyle='--',linewidth=1.5,alpha=0.7)
    ax1.set_xlim(2,12);ax1.set_ylim(log_T_min,log_T_max)
    if language=='de':
        ax1.set_xlabel('log10(rho / kg/m^3)',fontsize=12)
        ax1.set_ylabel('log10(T / K)',fontsize=12)
        ax1.set_title('Phasendiagramm: Druckregime',fontsize=14,fontweight='bold',pad=15)
    else:
        ax1.set_xlabel('log10(rho / kg/m^3)',fontsize=12)
        ax1.set_ylabel('log10(T / K)',fontsize=12)
        ax1.set_title('Phase Diagram: Pressure Regimes',fontsize=14,fontweight='bold',pad=15)
    ax1.grid(True,alpha=0.3)
    ax1.legend(fontsize=11,loc='upper center',bbox_to_anchor=(0.5,-0.15),ncol=3)

    # ---- Subplot 2: Stability map ----
    ax2=axes[1]
    log_G_scale=np.linspace(0,8,80)
    log_M_solar=np.linspace(-2,2,80)
    GG,MM=np.meshgrid(log_G_scale,log_M_solar)
    M_ch_std=constants.chandrasekhar_limit
    stability=np.zeros_like(GG)
    for i in range(GG.shape[0]):
        for j in range(GG.shape[1]):
            gs=10.0**GG[i,j]
            m_sol=10.0**MM[i,j]
            m_ch_scaled=M_ch_std/gs**1.5
            stability[i,j]=1.0 if m_sol<m_ch_scaled else 0.0
    ax2.contourf(GG,MM,stability,levels=[-0.5,0.5,1.5],
                 colors=[COLORS['relativistic'],COLORS['non_relativistic']],alpha=0.5)
    if language=='de':
        ax2.set_xlabel('log10(G-Skalierung)',fontsize=12)
        ax2.set_ylabel('log10(M / M_sun)',fontsize=12)
        ax2.set_title('Gleichgewichts-Existenzkarte',fontsize=14,fontweight='bold',pad=15)
        ax2.text(1.0,1.5,'Instabil',fontsize=11,ha='center',color=COLORS['relativistic'],fontweight='bold')
        ax2.text(1.0,-1.5,'Stabil',fontsize=11,ha='center',color=COLORS['non_relativistic'],fontweight='bold')
    else:
        ax2.set_xlabel('log10(G Scaling)',fontsize=12)
        ax2.set_ylabel('log10(M / M_sun)',fontsize=12)
        ax2.set_title('Equilibrium Existence Map',fontsize=14,fontweight='bold',pad=15)
        ax2.text(1.0,1.5,'Unstable',fontsize=11,ha='center',color=COLORS['relativistic'],fontweight='bold')
        ax2.text(1.0,-1.5,'Stable',fontsize=11,ha='center',color=COLORS['non_relativistic'],fontweight='bold')
    g_line=np.linspace(0,8,200)
    m_ch_line=np.log10(M_ch_std)-1.5*g_line
    valid=m_ch_line>=-2
    ax2.plot(g_line[valid],m_ch_line[valid],'k-',linewidth=2.5,
             label='Chandrasekhar limit' if language=='en' else 'Chandrasekhar-Grenze')
    ax2.grid(True,alpha=0.3)
    ax2.legend(fontsize=11,loc='upper center',bbox_to_anchor=(0.5,-0.15),ncol=3)


    # ---- Subplot 3: Lane-Emden solutions ----
    ax3=axes[2]
    le_indices=[0,1,1.5,2,3,4]
    le_styles=['-','--','-.',':','-','--']
    for idx_le,n_le in enumerate(le_indices):
        xi_arr,theta_arr,xi1=lane_emden_solve(n_le,num_points=2000)
        c_col=colors[idx_le % len(colors)]
        lbl='n='+str(n_le)
        ax3.plot(xi_arr,theta_arr,le_styles[idx_le % len(le_styles)],color=c_col,linewidth=2.5,label=lbl)
    ax3.axhline(0,color=COLORS['muted'],linestyle=':',linewidth=1,alpha=0.5)
    ax3.set_xlim(0,15)
    ax3.set_ylim(-0.5,1.1)
    if language=='de':
        ax3.set_xlabel('xi (dimensionslos)',fontsize=12)
        ax3.set_ylabel('theta(xi)',fontsize=12)
        ax3.set_title('Lane-Emden-Loesungen fuer verschiedene n',fontsize=14,fontweight='bold',pad=15)
    else:
        ax3.set_xlabel('xi (dimensionless)',fontsize=12)
        ax3.set_ylabel('theta(xi)',fontsize=12)
        ax3.set_title('Lane-Emden Solutions for Different n',fontsize=14,fontweight='bold',pad=15)
    ax3.grid(True,alpha=0.3)
    ax3.legend(fontsize=11,loc='upper center',bbox_to_anchor=(0.5,-0.15),ncol=3)

    # ---- Subplot 4: Newton vs TOV comparison ----
    ax4=axes[3]
    compactness=0.1
    M_obj=1.4*constants.M_sun
    R_obj=2*constants.G*M_obj/(compactness*constants.c**2)
    prof_tov=polytropic_pressure_profile(M_obj,R_obj,1.5,constants.G,num_points=500)
    r_norm_tov=prof_tov.radii/prof_tov.total_radius
    P_newton=prof_tov.pressures/prof_tov.central_pressure if prof_tov.central_pressure>0 else prof_tov.pressures
    P_tov=np.copy(P_newton)
    for k in range(len(prof_tov.radii)):
        r_k=prof_tov.radii[k]
        m_k=prof_tov.masses_enclosed[k]
        P_k=prof_tov.pressures[k]
        rho_k=prof_tov.densities[k]
        if r_k>0 and m_k>0:
            corr=tov_correction_factor(r_k,m_k,P_k,rho_k,constants.c,constants.G)
            P_tov[k]=P_newton[k]*corr
    ax4.plot(r_norm_tov,P_newton,'-',color=COLORS['non_relativistic'],linewidth=2.5,
             label='Newtonian' if language=='en' else 'Newtonsch')
    ax4.plot(r_norm_tov,P_tov,'--',color=COLORS['relativistic'],linewidth=2.5,
             label='TOV (GR corrected)' if language=='en' else 'TOV (ART-korrigiert)')
    if language=='de':
        ax4.set_xlabel('r / R',fontsize=12)
        ax4.set_ylabel('P / P_c (normiert)',fontsize=12)
        ax4.set_title('Newtonsch vs. TOV-Druckprofil',fontsize=14,fontweight='bold',pad=15)
    else:
        ax4.set_xlabel('r / R',fontsize=12)
        ax4.set_ylabel('P / P_c (normalized)',fontsize=12)
        ax4.set_title('Newtonian vs. TOV Pressure Profile',fontsize=14,fontweight='bold',pad=15)
    ax4.grid(True,alpha=0.3)
    ax4.legend(fontsize=11,loc='upper center',bbox_to_anchor=(0.5,-0.15),ncol=3)

    suffix='_de' if language=='de' else ''
    if save:
        os.makedirs(VIS_DIR,exist_ok=True)
        filepath=os.path.join(VIS_DIR,'hydrostatic_summary'+suffix+'.png')
        fig.savefig(filepath,dpi=150,bbox_inches='tight')
        print(f'Saved: {filepath}')
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


# ===================================================================
# Generate all plots
# ===================================================================

def generate_all_hydrostatic_plots(
    constants=None,
    language='en',
    save=True,
    show=False,
):
    """
    Generate all hydrostatic equilibrium plots.
    Erzeugt alle hydrostatischen Gleichgewichtsdiagramme.
    """
    if constants is None:
        constants=get_constants()
    figs=[]
    print('Generating hydrostatic profiles...')
    figs.append(plot_hydrostatic_profiles(constants=constants,language=language,save=save,show=show))
    print('Generating hydrostatic comparison...')
    figs.append(plot_hydrostatic_comparison(constants=constants,language=language,save=save,show=show))
    print('Generating hydrostatic summary...')
    figs.append(plot_hydrostatic_summary(constants=constants,language=language,save=save,show=show))
    print(f'Done. Generated {len(figs)} hydrostatic plots.')
    return figs


# ===================================================================
# Verification
# ===================================================================

def verify_hydrostatic_physics():
    """
    Verify Lane-Emden solutions and hydrostatic physics.
    Ueberprueft Lane-Emden-Loesungen und hydrostatische Physik.
    """
    print('=== Hydrostatic Equilibrium Verification ===')
    all_pass=True

    # Test 1: Lane-Emden n=0 => xi1 = sqrt(6) ~ 2.449
    _,_,xi1_n0=lane_emden_solve(0,num_points=5000)
    expected_n0=np.sqrt(6)
    err_n0=abs(xi1_n0-expected_n0)/expected_n0
    ok1=err_n0<0.01
    print(f'  n=0: xi1={xi1_n0:.4f}, expected={expected_n0:.4f}, err={err_n0:.4f} {"PASS" if ok1 else "FAIL"}')
    if not ok1: all_pass=False

    # Test 2: Lane-Emden n=1 => xi1 = pi ~ 3.14159
    _,_,xi1_n1=lane_emden_solve(1,num_points=5000)
    expected_n1=np.pi
    err_n1=abs(xi1_n1-expected_n1)/expected_n1
    ok2=err_n1<0.01
    print(f'  n=1: xi1={xi1_n1:.4f}, expected={expected_n1:.4f}, err={err_n1:.4f} {"PASS" if ok2 else "FAIL"}')
    if not ok2: all_pass=False

    # Test 3: Lane-Emden n=5 should never reach zero (xi1 ~ inf)
    xi_arr_n5,theta_arr_n5,xi1_n5=lane_emden_solve(5,num_points=5000)
    ok3=xi1_n5>=49.0
    print(f'  n=5: xi1={xi1_n5:.1f} (should be large/inf) {"PASS" if ok3 else "FAIL"}')
    if not ok3: all_pass=False

    # Test 4: Central pressure positive
    constants=get_constants()
    Pc=hydrostatic_central_pressure(constants.M_earth,constants.R_earth,constants.G)
    ok4=Pc>0
    print(f'  Central pressure P_c={Pc:.3e} Pa {"PASS" if ok4 else "FAIL"}')
    if not ok4: all_pass=False

    # Test 5: TOV correction >= 1 for compact object
    r_test=1e4
    m_test=1.4*constants.M_sun
    P_test=1e30
    rho_test=1e17
    corr=tov_correction_factor(r_test,m_test,P_test,rho_test,constants.c,constants.G)
    ok5=corr>=1.0
    print(f'  TOV correction={corr:.4f} (>=1 expected) {"PASS" if ok5 else "FAIL"}')
    if not ok5: all_pass=False

    # Test 6: Polytropic profile returns valid data
    prof=polytropic_pressure_profile(constants.M_earth,constants.R_earth,1.5,constants.G)
    ok6=prof.central_pressure>0 and prof.central_density>0 and prof.is_stable
    print(f'  Polytropic profile: Pc={prof.central_pressure:.3e}, rho_c={prof.central_density:.3e} {"PASS" if ok6 else "FAIL"}')
    if not ok6: all_pass=False

    print(f'=== Overall: {"ALL PASSED" if all_pass else "SOME FAILED"} ===')
    return all_pass
