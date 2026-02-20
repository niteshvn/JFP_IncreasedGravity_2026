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

from .celestial_bodies import (
    CelestialBodyProperties,
    get_solar_system_bodies,
    body_radius_scaled,
    body_surface_gravity,
    body_escape_velocity,
    body_compactness,
    collapse_threshold,
    jeans_mass_at_scale,
    calculate_body_properties,
    plot_planetary_comparison,
    plot_solar_system_positioning,
    plot_celestial_summary,
    generate_all_celestial_plots,
    verify_celestial_physics
)

from .orbital_mechanics import (
    OrbitalProperties,
    orbital_period,
    orbital_velocity,
    tidal_force,
    roche_limit,
    hill_sphere,
    gr_precession,
    orbital_energy,
    scaled_orbit_radius,
    calculate_moon_orbit,
    calculate_earth_orbit,
    plot_moon_tidal_effects,
    plot_earth_orbit,
    plot_orbital_summary,
    generate_all_orbital_plots,
    verify_orbital_physics
)

from .earth_collapse import (
    EarthEvolutionState,
    chandrasekhar_mass_at_scale,
    earth_state_at_scale,
    earth_evolution_track,
    plot_earth_change_diagram,
    plot_earth_combined,
    plot_earth_collapse_summary,
    generate_all_earth_collapse_plots,
    verify_earth_collapse_physics
)

from .solar_physics import (
    SolarProperties,
    solar_luminosity,
    solar_radius,
    solar_core_temperature,
    solar_surface_temperature,
    gamow_fusion_factor,
    eddington_luminosity,
    solar_lifetime,
    solar_central_pressure,
    calculate_solar_properties,
    plot_solar_structure,
    plot_solar_lifetime,
    plot_solar_summary,
    generate_all_solar_plots,
    verify_solar_physics
)

from .hydrostatic_equilibrium import (
    HydrostaticProfile,
    lane_emden_solve,
    polytropic_pressure_profile,
    tov_correction_factor,
    hydrostatic_central_pressure,
    plot_hydrostatic_profiles,
    plot_hydrostatic_comparison,
    plot_hydrostatic_summary,
    generate_all_hydrostatic_plots,
    verify_hydrostatic_physics
)

from .fine_structure import (
    FineStructureProperties,
    sommerfeld_alpha,
    binding_energy_hydrogen,
    fine_structure_splitting,
    rydberg_energy,
    radiative_lifetime_factor,
    calculate_fine_structure,
    plot_alpha_scaling,
    plot_alpha_consequences,
    plot_alpha_summary,
    generate_all_fine_structure_plots,
    verify_fine_structure_physics
)

from .cosmological_effects import (
    CosmicProperties,
    planck_mass,
    planck_length,
    planck_temperature,
    planck_time,
    jeans_mass,
    jeans_length,
    friedmann_rate,
    critical_density,
    calculate_cosmic_properties,
    plot_cosmic_scales,
    plot_cosmic_summary,
    generate_all_cosmic_plots,
    verify_cosmic_physics
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
        plot_solar_system_3d_interactive,
        plot_solar_luminosity_3d_interactive,
        plot_hydrostatic_3d_interactive,
        plot_fine_structure_3d_interactive,
        plot_earth_evolution_3d_interactive,
        plot_cosmic_scales_3d_interactive,
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
    # Celestial bodies
    'CelestialBodyProperties',
    'get_solar_system_bodies',
    'body_radius_scaled',
    'body_surface_gravity',
    'body_escape_velocity',
    'body_compactness',
    'collapse_threshold',
    'jeans_mass_at_scale',
    'calculate_body_properties',
    'plot_planetary_comparison',
    'plot_solar_system_positioning',
    'plot_celestial_summary',
    'generate_all_celestial_plots',
    'verify_celestial_physics',
    # Orbital mechanics
    'OrbitalProperties',
    'orbital_period',
    'orbital_velocity',
    'tidal_force',
    'roche_limit',
    'hill_sphere',
    'gr_precession',
    'orbital_energy',
    'scaled_orbit_radius',
    'calculate_moon_orbit',
    'calculate_earth_orbit',
    'plot_moon_tidal_effects',
    'plot_earth_orbit',
    'plot_orbital_summary',
    'generate_all_orbital_plots',
    'verify_orbital_physics',
    # Earth collapse
    'EarthEvolutionState',
    'chandrasekhar_mass_at_scale',
    'earth_state_at_scale',
    'earth_evolution_track',
    'plot_earth_change_diagram',
    'plot_earth_combined',
    'plot_earth_collapse_summary',
    'generate_all_earth_collapse_plots',
    'verify_earth_collapse_physics',
    # Solar physics
    'SolarProperties',
    'solar_luminosity',
    'solar_radius',
    'solar_core_temperature',
    'solar_surface_temperature',
    'gamow_fusion_factor',
    'eddington_luminosity',
    'solar_lifetime',
    'solar_central_pressure',
    'calculate_solar_properties',
    'plot_solar_structure',
    'plot_solar_lifetime',
    'plot_solar_summary',
    'generate_all_solar_plots',
    'verify_solar_physics',
    # Hydrostatic equilibrium
    'HydrostaticProfile',
    'lane_emden_solve',
    'polytropic_pressure_profile',
    'tov_correction_factor',
    'hydrostatic_central_pressure',
    'plot_hydrostatic_profiles',
    'plot_hydrostatic_comparison',
    'plot_hydrostatic_summary',
    'generate_all_hydrostatic_plots',
    'verify_hydrostatic_physics',
    # Fine structure
    'FineStructureProperties',
    'sommerfeld_alpha',
    'binding_energy_hydrogen',
    'fine_structure_splitting',
    'rydberg_energy',
    'radiative_lifetime_factor',
    'calculate_fine_structure',
    'plot_alpha_scaling',
    'plot_alpha_consequences',
    'plot_alpha_summary',
    'generate_all_fine_structure_plots',
    'verify_fine_structure_physics',
    # Cosmological effects
    'CosmicProperties',
    'planck_mass',
    'planck_length',
    'planck_temperature',
    'planck_time',
    'jeans_mass',
    'jeans_length',
    'friedmann_rate',
    'critical_density',
    'calculate_cosmic_properties',
    'plot_cosmic_scales',
    'plot_cosmic_summary',
    'generate_all_cosmic_plots',
    'verify_cosmic_physics',
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
    'plot_solar_system_3d_interactive',
    'plot_solar_luminosity_3d_interactive',
    'plot_hydrostatic_3d_interactive',
    'plot_fine_structure_3d_interactive',
    'plot_earth_evolution_3d_interactive',
    'plot_cosmic_scales_3d_interactive',
    'generate_all_interactive_plots',
    'PLOTLY_AVAILABLE'
]
