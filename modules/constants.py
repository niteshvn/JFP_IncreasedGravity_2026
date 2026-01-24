"""
Constants Module for Jugend forscht 2026 Physics Visualization Project
Konstantenmodul fuer Jugend forscht 2026 Physik-Visualisierungsprojekt

This module loads physical constants from config/constants.json and provides
utilities for scaling constants to simulate alternative universes.

Author: Jugend forscht 2026 Project
"""

import json
import os
from dataclasses import dataclass
from typing import Optional
import math


# Path to the configuration file
CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config', 'constants.json')


@dataclass
class PhysicalConstants:
    """
    Container for physical constants with scaling support.
    Behaelter fuer physikalische Konstanten mit Skalierungsunterstuetzung.
    """
    # Fundamental constants (SI units)
    hbar: float          # Reduced Planck constant [J·s]
    G: float             # Gravitational constant [m³·kg⁻¹·s⁻²]
    c: float             # Speed of light [m/s]
    m_p: float           # Proton mass [kg]
    m_e: float           # Electron mass [kg]
    e: float             # Elementary charge [C]
    k_e: float           # Coulomb constant [N·m²·C⁻²]
    epsilon_0: float     # Vacuum permittivity [F/m]
    k_B: float           # Boltzmann constant [J/K]

    # Derived constants
    alpha: float         # Fine-structure constant (dimensionless)
    a_0: float           # Bohr radius [m]
    alpha_G: float       # Gravitational coupling constant (dimensionless)

    # Atmospheric/thermal constants
    c_p_air: float       # Specific heat of air [J/(kg·K)]
    mu_air: float        # Mean molecular mass of air [kg]
    T_surface_earth: float  # Earth surface temperature [K]
    T_core_earth: float     # Earth core temperature [K]
    
    # Astronomical data
    M_sun: float         # Solar mass [kg]
    R_sun: float         # Solar radius [m]
    M_earth: float       # Earth mass [kg]
    R_earth: float       # Earth radius [m]
    chandrasekhar_limit: float  # In solar masses
    
    # Scaling factors applied
    hbar_scale: float = 1.0
    G_scale: float = 1.0
    
    def calculate_alpha_G(self) -> float:
        """
        Calculate the gravitational coupling constant.
        Berechnet die Gravitationskopplungskonstante.
        
        Formula: alpha_G = (G * m_p^2) / (hbar * c)
        
        This dimensionless constant represents the ratio of gravitational
        to electromagnetic force strength between two protons.
        """
        return (self.G * self.m_p**2) / (self.hbar * self.c)
    
    def calculate_bohr_radius(self) -> float:
        """
        Calculate the Bohr radius based on current constants.
        Berechnet den Bohrschen Radius basierend auf aktuellen Konstanten.
        
        Formula: a_0 = (4 * pi * epsilon_0 * hbar^2) / (m_e * e^2)
        
        When alpha is kept constant, this simplifies to: a_0 proportional to hbar/m_e
        """
        return (4 * math.pi * self.epsilon_0 * self.hbar**2) / (self.m_e * self.e**2)
    
    def schwarzschild_radius(self, mass: float) -> float:
        """
        Calculate the Schwarzschild radius for a given mass.
        Berechnet den Schwarzschild-Radius fuer eine gegebene Masse.
        
        Formula: R_s = 2 * G * M / c^2
        
        Args:
            mass: Mass of the object [kg]
        
        Returns:
            Schwarzschild radius [m]
        """
        return (2 * self.G * mass) / (self.c**2)
    
    def escape_velocity(self, mass: float, radius: float) -> float:
        """
        Calculate escape velocity from a spherical body.
        Berechnet die Fluchtgeschwindigkeit von einem sphaerischen Koerper.
        
        Formula: v_esc = sqrt(2 * G * M / R)
        
        Args:
            mass: Mass of the body [kg]
            radius: Radius of the body [m]
        
        Returns:
            Escape velocity [m/s]
        """
        return math.sqrt(2 * self.G * mass / radius)
    
    def surface_gravity(self, mass: float, radius: float) -> float:
        """
        Calculate surface gravitational acceleration.
        Berechnet die Gravitationsbeschleunigung an der Oberflaeche.
        
        Formula: g = G * M / R^2
        
        Args:
            mass: Mass of the body [kg]
            radius: Radius of the body [m]
        
        Returns:
            Surface gravity [m/s²]
        """
        return self.G * mass / radius**2
    
    def compactness(self, mass: float, radius: float) -> float:
        """
        Calculate the compactness parameter (R_s / R).
        Berechnet den Kompaktheitsparameter.
        
        This dimensionless ratio indicates how close an object is to
        becoming a black hole. For Earth: ~10^-9, for white dwarfs: ~10^-4
        
        Args:
            mass: Mass of the body [kg]
            radius: Radius of the body [m]
        
        Returns:
            Compactness ratio (dimensionless)
        """
        R_s = self.schwarzschild_radius(mass)
        return R_s / radius
    
    def coulomb_force(self, q1: float, q2: float, r: float) -> float:
        """
        Calculate Coulomb force between two charges.
        Berechnet die Coulomb-Kraft zwischen zwei Ladungen.
        
        Formula: F = k_e * q1 * q2 / r^2
        
        Args:
            q1: First charge [C]
            q2: Second charge [C]
            r: Distance between charges [m]
        
        Returns:
            Force [N] (positive = repulsion, negative = attraction)
        """
        return self.k_e * q1 * q2 / r**2
    
    def gravitational_force(self, m1: float, m2: float, r: float) -> float:
        """
        Calculate gravitational force between two masses.
        Berechnet die Gravitationskraft zwischen zwei Massen.
        
        Formula: F = G * m1 * m2 / r^2
        
        Args:
            m1: First mass [kg]
            m2: Second mass [kg]
            r: Distance between masses [m]
        
        Returns:
            Force [N]
        """
        return self.G * m1 * m2 / r**2
    
    def force_ratio_protons(self, r: float) -> float:
        """
        Calculate the ratio of gravitational to electromagnetic force between two protons.
        Berechnet das Verhaeltnis von Gravitations- zu elektromagnetischer Kraft zwischen zwei Protonen.
        
        This demonstrates why gravity is negligible at atomic scales (~10^-36 in our universe).
        
        Args:
            r: Distance between protons [m]
        
        Returns:
            F_gravity / F_coulomb (dimensionless)
        """
        F_grav = self.gravitational_force(self.m_p, self.m_p, r)
        F_coul = abs(self.coulomb_force(self.e, self.e, r))
        return F_grav / F_coul
    
    def summary(self, language: str = 'en') -> str:
        """
        Generate a summary of the current constants.
        Erzeugt eine Zusammenfassung der aktuellen Konstanten.
        
        Args:
            language: 'en' for English, 'de' for German
        
        Returns:
            Formatted string summary
        """
        if language == 'de':
            return f"""
=== Physikalische Konstanten ===
Skalierungsfaktoren: hbar x{self.hbar_scale}, G x{self.G_scale}

Fundamentale Konstanten:
  hbar     = {self.hbar:.4e} J*s
  G        = {self.G:.4e} m^3/(kg*s^2)
  c        = {self.c:.4e} m/s
  m_p      = {self.m_p:.4e} kg
  m_e      = {self.m_e:.4e} kg
  e        = {self.e:.4e} C
  k_e      = {self.k_e:.4e} N*m^2/C^2

Abgeleitete Konstanten:
  alpha       = {self.alpha:.6f} (~1/137)
  alpha_G     = {self.alpha_G:.4e}
  Bohr-Radius = {self.a_0:.4e} m

Kraftverhaeltnis (Gravitation/Coulomb) bei 1 fm:
  {self.force_ratio_protons(1e-15):.4e}
"""
        else:
            return f"""
=== Physical Constants ===
Scaling factors: hbar x{self.hbar_scale}, G x{self.G_scale}

Fundamental Constants:
  hbar     = {self.hbar:.4e} J*s
  G        = {self.G:.4e} m^3/(kg*s^2)
  c        = {self.c:.4e} m/s
  m_p      = {self.m_p:.4e} kg
  m_e      = {self.m_e:.4e} kg
  e        = {self.e:.4e} C
  k_e      = {self.k_e:.4e} N*m^2/C^2

Derived Constants:
  alpha       = {self.alpha:.6f} (~1/137)
  alpha_G     = {self.alpha_G:.4e}
  Bohr radius = {self.a_0:.4e} m

Force ratio (gravity/Coulomb) at 1 fm:
  {self.force_ratio_protons(1e-15):.4e}
"""


def load_constants(config_path: Optional[str] = None) -> dict:
    """
    Load constants from JSON configuration file.
    Laedt Konstanten aus der JSON-Konfigurationsdatei.
    
    Args:
        config_path: Path to config file (uses default if None)
    
    Returns:
        Dictionary containing all configuration data
    """
    path = config_path or CONFIG_PATH
    
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_constants(
    hbar_scale: float = 1.0,
    G_scale: float = 1.0,
    config_path: Optional[str] = None
) -> PhysicalConstants:
    """
    Get physical constants with optional scaling factors.
    Gibt physikalische Konstanten mit optionalen Skalierungsfaktoren zurueck.
    
    This function allows you to simulate alternative universes by scaling
    fundamental constants. To increase effective gravity:
    - Increase G_scale (direct approach, causes collapse)
    - Decrease hbar_scale (maintains atomic structure but increases alpha_G)
    
    Args:
        hbar_scale: Scaling factor for reduced Planck constant (default 1.0)
        G_scale: Scaling factor for gravitational constant (default 1.0)
        config_path: Optional path to configuration file
    
    Returns:
        PhysicalConstants object with scaled values
    
    Example:
        # Standard universe
        std = get_constants()
        
        # Universe with 10x stronger gravity
        alt = get_constants(G_scale=10.0)
        
        # Universe with smaller hbar (atoms shrink, gravity relatively stronger)
        alt2 = get_constants(hbar_scale=0.1)
    """
    config = load_constants(config_path)

    fund = config['fundamental_constants']
    derived = config['derived_constants']
    astro = config['astronomical_data']
    atmo = config.get('atmospheric_constants', {})
    
    # Apply scaling to fundamental constants
    hbar = fund['hbar']['value'] * hbar_scale
    G = fund['G']['value'] * G_scale
    
    # Constants that don't scale
    c = fund['c']['value']
    m_p = fund['m_p']['value']
    m_e = fund['m_e']['value']
    e = fund['e']['value']
    k_e = fund['k_e']['value']
    epsilon_0 = fund['epsilon_0']['value']
    k_B = fund['k_B']['value']

    # Atmospheric constants
    c_p_air = atmo.get('c_p_air', {}).get('value', 1005)
    mu_air = atmo.get('mu_air', {}).get('value', 4.81e-26)
    T_surface_earth = atmo.get('T_surface_earth', {}).get('value', 288)
    T_core_earth = atmo.get('T_core_earth', {}).get('value', 5778)
    
    # Calculate derived constants with scaled values
    alpha = derived['alpha']['value']  # Keep alpha constant for atomic stability
    
    # Recalculate Bohr radius with scaled hbar
    a_0 = (4 * math.pi * epsilon_0 * hbar**2) / (m_e * e**2)
    
    # Recalculate gravitational coupling constant
    alpha_G = (G * m_p**2) / (hbar * c)
    
    return PhysicalConstants(
        hbar=hbar,
        G=G,
        c=c,
        m_p=m_p,
        m_e=m_e,
        e=e,
        k_e=k_e,
        epsilon_0=epsilon_0,
        k_B=k_B,
        alpha=alpha,
        a_0=a_0,
        alpha_G=alpha_G,
        c_p_air=c_p_air,
        mu_air=mu_air,
        T_surface_earth=T_surface_earth,
        T_core_earth=T_core_earth,
        M_sun=astro['M_sun']['value'],
        R_sun=astro['R_sun']['value'],
        M_earth=astro['M_earth']['value'],
        R_earth=astro['R_earth']['value'],
        chandrasekhar_limit=astro['chandrasekhar_limit']['value'],
        hbar_scale=hbar_scale,
        G_scale=G_scale
    )


def compare_universes(
    hbar_scale: float = 1.0,
    G_scale: float = 1.0,
    language: str = 'en'
) -> str:
    """
    Compare standard universe with a scaled alternative.
    Vergleicht das Standarduniversum mit einer skalierten Alternative.
    
    Args:
        hbar_scale: Scaling factor for hbar
        G_scale: Scaling factor for G
        language: 'en' or 'de'
    
    Returns:
        Formatted comparison string
    """
    std = get_constants()
    alt = get_constants(hbar_scale=hbar_scale, G_scale=G_scale)
    
    ratio_alpha_G = alt.alpha_G / std.alpha_G
    ratio_a0 = alt.a_0 / std.a_0
    
    if language == 'de':
        return f"""
=== Universumsvergleich ===
Skalierung: hbar x{hbar_scale}, G x{G_scale}

                        Standard        Alternativ      Verhaeltnis
alpha_G:                {std.alpha_G:.4e}    {alt.alpha_G:.4e}    {ratio_alpha_G:.2e}x
Bohr-Radius:            {std.a_0:.4e} m  {alt.a_0:.4e} m  {ratio_a0:.4f}x
Erd-Schwarzschild-R:    {std.schwarzschild_radius(std.M_earth):.4e} m  {alt.schwarzschild_radius(alt.M_earth):.4e} m

Interpretation:
- Gravitation ist {ratio_alpha_G:.2e}x staerker relativ zu Quanteneffekten
- Atome sind {ratio_a0:.4f}x so gross wie im Standarduniversum
"""
    else:
        return f"""
=== Universe Comparison ===
Scaling: hbar x{hbar_scale}, G x{G_scale}

                        Standard        Alternative     Ratio
alpha_G:                {std.alpha_G:.4e}    {alt.alpha_G:.4e}    {ratio_alpha_G:.2e}x
Bohr radius:            {std.a_0:.4e} m  {alt.a_0:.4e} m  {ratio_a0:.4f}x
Earth Schwarzschild R:  {std.schwarzschild_radius(std.M_earth):.4e} m  {alt.schwarzschild_radius(alt.M_earth):.4e} m

Interpretation:
- Gravity is {ratio_alpha_G:.2e}x stronger relative to quantum effects
- Atoms are {ratio_a0:.4f}x the size of standard universe
"""


# Module-level instance for convenience
STANDARD = get_constants()


if __name__ == "__main__":
    # Demo: Print constants and compare universes
    print("=" * 60)
    print("Jugend forscht 2026 - Physics Constants Module")
    print("=" * 60)
    
    # Standard universe
    std = get_constants()
    print(std.summary())
    
    # Compare with alternative universe (hbar reduced by 10x)
    print("\n" + "=" * 60)
    print("Alternative Universe Comparison")
    print("=" * 60)
    print(compare_universes(hbar_scale=0.1))
    
    # Compare with increased G
    print(compare_universes(G_scale=1e10))
