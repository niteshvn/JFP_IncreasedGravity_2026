#!/usr/bin/env python3
"""
Jugend forscht 2026 - Physics Visualization Suite
=================================================

Project: How does increasing gravitational pressure modify the balance between
         electron degeneracy pressure (Pauli) and Coulomb interactions?

Projekt: Wie veraendert zunehmender Gravitationsdruck das Gleichgewicht zwischen
         Elektronenentartungsdruck (Pauli) und Coulomb-Wechselwirkungen?

This main.py serves as the entry point for generating all physics visualizations.
Run with --help for usage information.

Author: Jugend forscht 2026 Team
"""

import argparse
import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules import (
    # Constants
    PhysicalConstants,
    get_constants,
    load_constants,
    compare_universes,
    STANDARD,
    # Force comparison
    generate_all_force_plots,
    calculate_forces_at_scale,
    # White dwarf
    generate_all_white_dwarf_plots,
    calculate_white_dwarf,
    chandrasekhar_mass,
    # Spacetime curvature
    generate_all_spacetime_plots,
    get_stellar_objects,
    calculate_compact_object,
    # Atomic scale
    generate_all_atomic_plots,
    calculate_atomic_properties,
    # Interactive 3D (optional)
    PLOTLY_AVAILABLE
)

# Conditionally import interactive 3D module
if PLOTLY_AVAILABLE:
    from modules import generate_all_interactive_plots


def print_header():
    """Print project header."""
    print("=" * 70)
    print("  JUGEND FORSCHT 2026 - Physics Visualization Suite")
    print("  Gravitational Effects on Quantum Matter")
    print("=" * 70)
    print()


def verify_physics():
    """Verify all physics calculations are correct."""
    print("\n" + "=" * 70)
    print("  PHYSICS VERIFICATION")
    print("=" * 70)

    constants = get_constants()
    all_passed = True

    # 1. Fundamental constants check
    print("\n[1] Fundamental Constants:")
    print(f"    Speed of light: c = {constants.c:.3e} m/s")
    print(f"    Planck constant: h-bar = {constants.hbar:.3e} J*s")
    print(f"    Electron mass: m_e = {constants.m_e:.3e} kg")
    print(f"    Proton mass: m_p = {constants.m_p:.3e} kg")
    print(f"    Elementary charge: e = {constants.e:.3e} C")
    print(f"    Gravitational constant: G = {constants.G:.3e} m^3/(kg*s^2)")

    # 2. Chandrasekhar mass
    print("\n[2] Chandrasekhar Mass:")
    M_ch = chandrasekhar_mass(constants)
    M_ch_solar = M_ch / constants.M_sun
    expected_ch = 1.44  # Solar masses
    passed = abs(M_ch_solar - expected_ch) / expected_ch < 0.05
    status = "PASS" if passed else "FAIL"
    print(f"    Calculated: {M_ch_solar:.3f} M_sun")
    print(f"    Expected: ~{expected_ch} M_sun")
    print(f"    Status: [{status}]")
    all_passed = all_passed and passed

    # 3. Bohr radius
    print("\n[3] Bohr Radius:")
    atomic = calculate_atomic_properties(constants)
    a0_pm = atomic.bohr_radius * 1e12
    expected_a0 = 52.92  # pm
    passed = abs(a0_pm - expected_a0) / expected_a0 < 0.01
    status = "PASS" if passed else "FAIL"
    print(f"    Calculated: {a0_pm:.2f} pm")
    print(f"    Expected: {expected_a0} pm")
    print(f"    Status: [{status}]")
    all_passed = all_passed and passed

    # 4. Hydrogen ground state energy
    print("\n[4] Hydrogen Ground State Energy:")
    E1_eV = atomic.ground_state_energy / constants.e  # Convert J to eV
    expected_E1 = -13.6  # eV
    passed = abs(E1_eV - expected_E1) / abs(expected_E1) < 0.01
    status = "PASS" if passed else "FAIL"
    print(f"    Calculated: {E1_eV:.2f} eV")
    print(f"    Expected: {expected_E1} eV")
    print(f"    Status: [{status}]")
    all_passed = all_passed and passed

    # 5. Fine structure constant
    print("\n[5] Fine Structure Constant:")
    alpha = atomic.fine_structure_constant
    expected_alpha = 1/137.036
    passed = abs(alpha - expected_alpha) / expected_alpha < 0.001
    status = "PASS" if passed else "FAIL"
    print(f"    Calculated: 1/{1/alpha:.2f}")
    print(f"    Expected: 1/137.04")
    print(f"    Status: [{status}]")
    all_passed = all_passed and passed

    # 6. Gravitational coupling constant
    print("\n[6] Gravitational Coupling Constant:")
    alpha_G = (constants.G * constants.m_p**2) / (constants.hbar * constants.c)
    expected_alpha_G = 5.9e-39
    passed = abs(alpha_G - expected_alpha_G) / expected_alpha_G < 0.1
    status = "PASS" if passed else "FAIL"
    print(f"    Calculated: alpha_G = {alpha_G:.2e}")
    print(f"    Expected: ~{expected_alpha_G:.1e}")
    print(f"    Status: [{status}]")
    all_passed = all_passed and passed

    # 7. Force ratio at atomic scale (two protons)
    print("\n[7] Electromagnetic/Gravitational Force Ratio (proton-proton):")
    forces = calculate_forces_at_scale(constants, 'atomic')
    ratio = forces.coulomb / forces.gravitational if forces.gravitational > 0 else float('inf')
    # For two protons: F_em/F_grav = k_e * e^2 / (G * m_p^2) ≈ 1.24e36
    expected_ratio = 1.24e36
    passed = abs(ratio - expected_ratio) / expected_ratio < 0.1
    status = "PASS" if passed else "FAIL"
    print(f"    Calculated: F_em/F_grav = {ratio:.2e}")
    print(f"    Expected: ~{expected_ratio:.1e}")
    print(f"    Status: [{status}]")
    all_passed = all_passed and passed

    # 8. Schwarzschild radius of Sun
    print("\n[8] Schwarzschild Radius of Sun:")
    sun = calculate_compact_object("Sun", "Sonne", constants.M_sun, constants.R_sun, constants)
    R_s_km = sun.schwarzschild_radius / 1000
    expected_Rs = 2.95  # km
    passed = abs(R_s_km - expected_Rs) / expected_Rs < 0.02
    status = "PASS" if passed else "FAIL"
    print(f"    Calculated: R_s = {R_s_km:.2f} km")
    print(f"    Expected: ~{expected_Rs} km")
    print(f"    Status: [{status}]")
    all_passed = all_passed and passed

    # Summary
    print("\n" + "-" * 70)
    if all_passed:
        print("  ALL PHYSICS VERIFICATIONS PASSED")
    else:
        print("  SOME VERIFICATIONS FAILED - Please check calculations!")
    print("-" * 70)

    return all_passed


def compare_scaled_universes():
    """Show how physics changes in scaled universes."""
    print("\n" + "=" * 70)
    print("  SCALED UNIVERSE COMPARISON")
    print("=" * 70)

    standard = get_constants()

    # Compare with h-bar scaled to 0.1
    scaled = get_constants(hbar_scale=0.1)

    print("\n  Standard Universe vs. h-bar = 0.1 * h-bar_0:")
    print("-" * 70)

    # Bohr radius comparison
    a0_std = calculate_atomic_properties(standard).bohr_radius
    a0_scaled = calculate_atomic_properties(scaled).bohr_radius
    print(f"\n  Bohr radius:")
    print(f"    Standard: {a0_std*1e12:.2f} pm")
    print(f"    Scaled:   {a0_scaled*1e12:.2f} pm")
    print(f"    Ratio:    {a0_scaled/a0_std:.4f} (expected: 0.01 = 0.1^2)")

    # Chandrasekhar mass comparison
    M_ch_std = chandrasekhar_mass(standard)
    M_ch_scaled = chandrasekhar_mass(scaled)
    print(f"\n  Chandrasekhar mass:")
    print(f"    Standard: {M_ch_std/standard.M_sun:.3f} M_sun")
    print(f"    Scaled:   {M_ch_scaled/scaled.M_sun:.3f} M_sun")
    print(f"    Ratio:    {M_ch_scaled/M_ch_std:.4f} (scales as h-bar^1.5)")

    # Gravitational coupling
    alpha_G_std = (standard.G * standard.m_p**2) / (standard.hbar * standard.c)
    alpha_G_scaled = (scaled.G * scaled.m_p**2) / (scaled.hbar * scaled.c)
    print(f"\n  Gravitational coupling alpha_G:")
    print(f"    Standard: {alpha_G_std:.2e}")
    print(f"    Scaled:   {alpha_G_scaled:.2e}")
    print(f"    Ratio:    {alpha_G_scaled/alpha_G_std:.1f} (expected: 10 = 1/0.1)")

    # Matter density at electron degeneracy
    rho_std = standard.m_e / a0_std**3
    rho_scaled = scaled.m_e / a0_scaled**3
    print(f"\n  Typical atomic density:")
    print(f"    Standard: {rho_std:.2e} kg/m^3")
    print(f"    Scaled:   {rho_scaled:.2e} kg/m^3")
    print(f"    Ratio:    {rho_scaled/rho_std:.0f} (scales as 1/h-bar^6)")

    print("\n" + "-" * 70)
    print("  Key insight: Reducing h-bar makes atoms smaller, gravity stronger,")
    print("  and enables gravitational collapse at much lower masses.")
    print("-" * 70)


def generate_all_visualizations(language: str = 'en', verbose: bool = True,
                                interactive: bool = False):
    """Generate all physics visualizations."""
    output_dir = str(PROJECT_ROOT / "visualizations")
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        print("\n" + "=" * 70)
        print("  GENERATING VISUALIZATIONS")
        print("=" * 70)

    total_plots = 0

    # Force comparison plots
    if verbose:
        print("\n[1] Force Comparison Module:")
    figures = generate_all_force_plots(language=language, show=False)
    total_plots += len(figures)
    if verbose:
        print(f"    Generated {len(figures)} plots")

    # White dwarf plots
    if verbose:
        print("\n[2] White Dwarf Module:")
    figures = generate_all_white_dwarf_plots(language=language, show=False)
    total_plots += len(figures)
    if verbose:
        print(f"    Generated {len(figures)} plots")

    # Spacetime curvature plots
    if verbose:
        print("\n[3] Spacetime Curvature Module:")
    figures = generate_all_spacetime_plots(language=language, show=False)
    total_plots += len(figures)
    if verbose:
        print(f"    Generated {len(figures)} plots")

    # Atomic scale plots
    if verbose:
        print("\n[4] Atomic Scale Module:")
    figures = generate_all_atomic_plots(language=language, show=False)
    total_plots += len(figures)
    if verbose:
        print(f"    Generated {len(figures)} plots")

    # Interactive 3D plots (optional)
    if interactive:
        if PLOTLY_AVAILABLE:
            if verbose:
                print("\n[5] Interactive 3D Module:")
            figures = generate_all_interactive_plots(language=language)
            total_plots += len(figures)
            if verbose:
                print(f"    Generated {len(figures)} interactive HTML files")
        else:
            if verbose:
                print("\n[5] Interactive 3D Module: SKIPPED (plotly not installed)")
                print("    Install with: pip install plotly")

    if verbose:
        print("\n" + "-" * 70)
        print(f"  Total: {total_plots} visualizations generated in {output_dir}")
        print("-" * 70)

    return total_plots


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Jugend forscht 2026 - Physics Visualization Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Generate all visualizations
  python main.py --verify           # Run physics verification only
  python main.py --compare          # Show scaled universe comparison
  python main.py --all              # Run everything
  python main.py --german           # Generate German language plots
  python main.py --interactive      # Include interactive 3D HTML plots
        """
    )

    parser.add_argument(
        "--verify", "-v",
        action="store_true",
        help="Run physics verification checks"
    )

    parser.add_argument(
        "--compare", "-c",
        action="store_true",
        help="Show scaled universe comparison"
    )

    parser.add_argument(
        "--generate", "-g",
        action="store_true",
        help="Generate all visualizations (default if no flags)"
    )

    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Run verification, comparison, and generation"
    )

    parser.add_argument(
        "--german", "-de",
        action="store_true",
        help="Generate visualizations with German labels"
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output"
    )

    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Generate interactive 3D plots (requires plotly)"
    )

    args = parser.parse_args()

    # Default to generate if no flags specified
    if not (args.verify or args.compare or args.generate or args.all):
        args.generate = True

    print_header()

    # Determine language
    language = 'de' if args.german else 'en'

    # Run requested operations
    if args.all or args.verify:
        verify_physics()

    if args.all or args.compare:
        compare_scaled_universes()

    if args.all or args.generate:
        generate_all_visualizations(
            language=language,
            verbose=not args.quiet,
            interactive=args.interactive
        )

    print("\n" + "=" * 70)
    print("  Complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
