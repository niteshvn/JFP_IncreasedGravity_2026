"""
Physics Visualization Modules for Jugend forscht 2026
Physik-Visualisierungsmodule fuer Jugend forscht 2026

Project: How does increasing gravitational pressure modify the balance between
         electron degeneracy pressure (Pauli) and Coulomb interactions?
"""

from .constants import (
    PhysicalConstants,
    get_constants,
    load_constants,
    compare_universes,
    STANDARD
)

from .force_comparison import (
    ForceCalculation,
    calculate_forces_at_scale,
    plot_force_comparison_bar,
    plot_force_vs_distance,
    plot_force_across_scales,
    plot_scaled_universe_comparison,
    generate_all_force_plots
)

from .white_dwarf import (
    WhiteDwarfProperties,
    electron_degeneracy_pressure_nr,
    electron_degeneracy_pressure_r,
    chandrasekhar_mass,
    white_dwarf_radius,
    calculate_white_dwarf,
    plot_pressure_vs_density,
    plot_mass_radius_relation,
    plot_chandrasekhar_limit,
    plot_white_dwarf_summary,
    generate_all_white_dwarf_plots
)

from .spacetime_curvature import (
    CompactObject,
    calculate_compact_object,
    get_stellar_objects,
    gravitational_potential,
    spacetime_embedding,
    plot_potential_well_2d,
    plot_potential_well_3d,
    plot_compactness_comparison,
    plot_escape_velocity,
    plot_spacetime_summary,
    generate_all_spacetime_plots,
    # Light bending / gravitational lensing
    light_deflection_angle,
    einstein_radius,
    calculate_light_path,
    plot_light_bending,
    create_light_bending_animation,
    # Penrose-Carter diagrams
    schwarzschild_to_kruskal,
    kruskal_to_penrose,
    calculate_constant_r_curve,
    calculate_infalling_worldline,
    plot_penrose_carter_diagram,
    plot_penrose_comparison,
    # Time dilation
    gravitational_time_dilation_factor,
    plot_time_dilation_comparison,
    plot_time_dilation_scaling,
    plot_time_dilation_summary,
)

from .atomic_scale import (
    AtomicProperties,
    calculate_atomic_properties,
    energy_level,
    orbital_radius,
    plot_bohr_radius_scaling,
    plot_atom_size_comparison,
    plot_energy_levels,
    plot_quantum_gravity_connection,
    plot_atomic_summary,
    generate_all_atomic_plots
)

from .thermal_physics import (
    ThermalProperties,
    atmospheric_scale_height,
    adiabatic_lapse_rate,
    fermi_temperature,
    thermal_pressure,
    degeneracy_pressure_nr,
    virial_temperature,
    calculate_thermal_properties,
    temperature_vs_altitude,
    plot_temperature_atmosphere,
    plot_temperature_degeneracy,
    plot_temperature_summary,
    generate_all_thermal_plots,
    verify_thermal_physics
)

from .neutron_star import (
    NeutronStarProperties,
    neutron_degeneracy_pressure,
    neutron_degeneracy_pressure_relativistic,
    tov_mass_limit,
    neutron_star_radius,
    gravitational_time_dilation,
    calculate_neutron_star,
    plot_tov_limit_comparison,
    plot_neutron_star_structure,
    plot_electron_capture,
    plot_neutron_star_summary,
    generate_all_neutron_star_plots,
    verify_neutron_star_physics
)

from .heisenberg_uncertainty import (
    UncertaintyProperties,
    minimum_momentum_uncertainty,
    confinement_velocity,
    confinement_kinetic_energy,
    degeneracy_pressure_from_uncertainty,
    calculate_uncertainty_properties,
    plot_uncertainty_principle_basic,
    plot_confinement_velocity,
    plot_uncertainty_to_pressure,
    plot_uncertainty_hbar_scaling,
    plot_heisenberg_summary,
    generate_all_heisenberg_plots,
    verify_heisenberg_physics
)

from .gravity_pauli_balance import (
    PlanetaryEquilibrium,
    gravitational_central_pressure,
    electron_degeneracy_pressure_simple,
    coulomb_thermal_pressure,
    calculate_planetary_equilibrium,
    plot_earth_structural_effects,
    plot_gravity_vs_pauli,
    plot_hypothesis_summary,
    generate_all_gravity_pauli_plots,
    verify_gravity_pauli_physics
)

# Interactive 3D visualizations (requires plotly)
try:
    from .interactive_3d import (
        plot_spacetime_curvature_3d_interactive,
        plot_multiple_masses_3d_interactive,
        plot_atom_scaling_3d_interactive,
        plot_force_ratio_3d_interactive,
        plot_temperature_profile_3d_interactive,
        plot_light_bending_3d_interactive,
        plot_neutron_star_3d_interactive,
        plot_heisenberg_3d_interactive,
        plot_time_dilation_3d_interactive,
        plot_gravity_pauli_3d_interactive,
        generate_all_interactive_plots,
        PLOTLY_AVAILABLE
    )
except ImportError:
    PLOTLY_AVAILABLE = False

__all__ = [
    # Constants
    'PhysicalConstants',
    'get_constants',
    'load_constants',
    'compare_universes',
    'STANDARD',
    # Force comparison
    'ForceCalculation',
    'calculate_forces_at_scale',
    'plot_force_comparison_bar',
    'plot_force_vs_distance',
    'plot_force_across_scales',
    'plot_scaled_universe_comparison',
    'generate_all_force_plots',
    # White dwarf
    'WhiteDwarfProperties',
    'electron_degeneracy_pressure_nr',
    'electron_degeneracy_pressure_r',
    'chandrasekhar_mass',
    'white_dwarf_radius',
    'calculate_white_dwarf',
    'plot_pressure_vs_density',
    'plot_mass_radius_relation',
    'plot_chandrasekhar_limit',
    'plot_white_dwarf_summary',
    'generate_all_white_dwarf_plots',
    # Spacetime curvature
    'CompactObject',
    'calculate_compact_object',
    'get_stellar_objects',
    'gravitational_potential',
    'spacetime_embedding',
    'plot_potential_well_2d',
    'plot_potential_well_3d',
    'plot_compactness_comparison',
    'plot_escape_velocity',
    'plot_spacetime_summary',
    'generate_all_spacetime_plots',
    # Light bending / gravitational lensing
    'light_deflection_angle',
    'einstein_radius',
    'calculate_light_path',
    'plot_light_bending',
    'create_light_bending_animation',
    # Penrose-Carter diagrams
    'schwarzschild_to_kruskal',
    'kruskal_to_penrose',
    'calculate_constant_r_curve',
    'calculate_infalling_worldline',
    'plot_penrose_carter_diagram',
    'plot_penrose_comparison',
    # Time dilation
    'gravitational_time_dilation_factor',
    'plot_time_dilation_comparison',
    'plot_time_dilation_scaling',
    'plot_time_dilation_summary',
    # Atomic scale
    'AtomicProperties',
    'calculate_atomic_properties',
    'energy_level',
    'orbital_radius',
    'plot_bohr_radius_scaling',
    'plot_atom_size_comparison',
    'plot_energy_levels',
    'plot_quantum_gravity_connection',
    'plot_atomic_summary',
    'generate_all_atomic_plots',
    # Thermal physics
    'ThermalProperties',
    'atmospheric_scale_height',
    'adiabatic_lapse_rate',
    'fermi_temperature',
    'thermal_pressure',
    'degeneracy_pressure_nr',
    'virial_temperature',
    'calculate_thermal_properties',
    'temperature_vs_altitude',
    'plot_temperature_atmosphere',
    'plot_temperature_degeneracy',
    'plot_temperature_summary',
    'generate_all_thermal_plots',
    'verify_thermal_physics',
    # Neutron star physics
    'NeutronStarProperties',
    'neutron_degeneracy_pressure',
    'neutron_degeneracy_pressure_relativistic',
    'tov_mass_limit',
    'neutron_star_radius',
    'gravitational_time_dilation',
    'calculate_neutron_star',
    'plot_tov_limit_comparison',
    'plot_neutron_star_structure',
    'plot_electron_capture',
    'plot_neutron_star_summary',
    'generate_all_neutron_star_plots',
    'verify_neutron_star_physics',
    # Heisenberg uncertainty
    'UncertaintyProperties',
    'minimum_momentum_uncertainty',
    'confinement_velocity',
    'confinement_kinetic_energy',
    'degeneracy_pressure_from_uncertainty',
    'calculate_uncertainty_properties',
    'plot_uncertainty_principle_basic',
    'plot_confinement_velocity',
    'plot_uncertainty_to_pressure',
    'plot_uncertainty_hbar_scaling',
    'plot_heisenberg_summary',
    'generate_all_heisenberg_plots',
    'verify_heisenberg_physics',
    # Gravity vs Pauli balance
    'PlanetaryEquilibrium',
    'gravitational_central_pressure',
    'electron_degeneracy_pressure_simple',
    'coulomb_thermal_pressure',
    'calculate_planetary_equilibrium',
    'plot_earth_structural_effects',
    'plot_gravity_vs_pauli',
    'plot_hypothesis_summary',
    'generate_all_gravity_pauli_plots',
    'verify_gravity_pauli_physics',
    # Interactive 3D (optional, requires plotly)
    'plot_spacetime_curvature_3d_interactive',
    'plot_multiple_masses_3d_interactive',
    'plot_atom_scaling_3d_interactive',
    'plot_force_ratio_3d_interactive',
    'plot_temperature_profile_3d_interactive',
    'plot_light_bending_3d_interactive',
    'plot_neutron_star_3d_interactive',
    'plot_heisenberg_3d_interactive',
    'plot_time_dilation_3d_interactive',
    'plot_gravity_pauli_3d_interactive',
    'generate_all_interactive_plots',
    'PLOTLY_AVAILABLE'
]
