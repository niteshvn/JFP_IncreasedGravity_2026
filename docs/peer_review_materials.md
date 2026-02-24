# PEER REVIEW MATERIALS PACKAGE
## Jugend forscht 2026 - Physics Visualization Project

---

## 1. PHYSICAL CONSTANTS (Exact Values Used)

### Speed of Light (c) - SPECIFICALLY REQUESTED
| Constant | Value | Unit |
|----------|-------|------|
| **c (Speed of Light)** | **299,792,458** | **m/s** |

### All Fundamental Constants
| Constant | Symbol | Value | Unit |
|----------|--------|-------|------|
| Reduced Planck constant | ℏ | 1.054571817 × 10⁻³⁴ | J·s |
| Gravitational constant | G | 6.67430 × 10⁻¹¹ | m³·kg⁻¹·s⁻² |
| Speed of light | c | 2.99792458 × 10⁸ | m/s |
| Proton mass | m_p | 1.67262192369 × 10⁻²⁷ | kg |
| Electron mass | m_e | 9.1093837015 × 10⁻³¹ | kg |
| Elementary charge | e | 1.602176634 × 10⁻¹⁹ | C |
| Coulomb constant | k_e | 8.9875517923 × 10⁹ | N·m²·C⁻² |
| Boltzmann constant | k_B | 1.380649 × 10⁻²³ | J/K |
| Fine-structure constant | α | 0.0072973526 (~1/137) | dimensionless |
| Bohr radius | a₀ | 5.29177210903 × 10⁻¹¹ | m |
| Gravitational coupling | α_G | 5.9 × 10⁻³⁹ | dimensionless |

**IMPORTANT: c is NOT modified in the essay scenario. Only G and ℏ are scaled.**

---

## 2. KEY SCREENSHOTS

### A. Essay Hypothesis Summary
**File:** `visualizations/hypothesis_summary.png`
- **Panel 1**: Object stability under standard physics (Earth, Jupiter, White Dwarf, Neutron Star)
- **Panel 2**: Pressure scaling with G (constant ℏ) - shows G×10³⁶ marker
- **Panel 3**: Stability ratio vs G for different ℏ scalings (1, 10⁹, 10¹⁸)
- **Panel 4**: Required ℏ for balance: **ℏ ∝ √G** (key relationship)

### B. Gravity vs. Pauli Pressure
**File:** `visualizations/gravity_vs_pauli.png`
- **Panel 1**: Gravitational Pressure vs Pauli Pressure across densities
- **Panel 2**: Pressure Ratio vs Mass (at white dwarf density)
- **Panel 3**: Pressure Ratio vs G Scaling
- **Panel 4**: Balance at G × 10³⁶ → shows ℏ × 10¹⁸ restores stability

### C. Force vs Distance (Proton-Proton)
**File:** `visualizations/force_vs_distance.png`
- Shows gravitational (blue) vs Coulomb (orange) force
- **Ratio: ~1.24 × 10³⁶** (Coulomb/Gravitational) - constant with distance

### D. Earth Structure Under Modified Constants
**File:** `visualizations/earth_structural_effects.png`
- **Panel 1**: Earth radius vs G scaling factor
- **Panel 2**: Pressure components (gravity, thermal, degeneracy)
- **Panel 3**: Degeneracy pressure vs ℏ at G × 10⁶
- **Panel 4**: Stability map in G-ℏ space (blue=stable, red=collapse)

### E. Atom Size Comparison
**File:** `visualizations/atom_size_comparison.png`
- Shows Bohr radius scaling: **a₀ ∝ ℏ²**
- Standard: 52.92 pm
- ℏ × 0.5: 13.23 pm (4× smaller)
- ℏ × 0.1: 0.53 pm (100× smaller)
- **Altered Universe (G×10³⁶, ℏ×10¹⁸): a₀ = 5.29×10²⁵ m = 10³⁶ × Standard**

### F. Orbital Precession Animation
**File:** `visualizations/orbital_precession_animation.gif`
- Animated GIF showing relativistic orbital precession

---

## 3. CORE PHYSICS FORMULAS (Code Implementation)

### A. Gravitational Central Pressure
```python
def gravitational_central_pressure(mass, radius, constants):
    """
    Formula: P_c ≈ (3/8π) × G × M² / R⁴
    """
    return (3 / (8 * np.pi)) * constants.G * mass**2 / radius**4
```

### B. Electron Degeneracy Pressure (Non-Relativistic)
```python
def electron_degeneracy_pressure_simple(density, constants):
    """
    Formula: P = K × (ρ/μ_e m_p)^(5/3)
    where K = (ℏ²/m_e) × (3π²)^(2/3) / 5

    KEY SCALING: P_deg ∝ ℏ²
    """
    mu_e = 2.0  # Mean molecular weight per electron
    n_e = density / (mu_e * constants.m_p)
    K = (constants.hbar**2 / (5 * constants.m_e)) * (3 * np.pi**2)**(2/3)
    return K * n_e**(5/3)
```

### C. Gravitational Coupling Constant
```python
def calculate_alpha_G(self):
    """
    Formula: α_G = (G × m_p²) / (ℏ × c)

    This dimensionless constant represents the ratio of gravitational
    to electromagnetic force strength between two protons.
    """
    return (self.G * self.m_p**2) / (self.hbar * self.c)
```

### D. Bohr Radius
```python
def calculate_bohr_radius(self):
    """
    Formula: a₀ = (4π × ε₀ × ℏ²) / (m_e × e²)

    KEY SCALING: a₀ ∝ ℏ²
    """
    return (4 * math.pi * self.epsilon_0 * self.hbar**2) / (self.m_e * self.e**2)
```

### E. Force Calculations
```python
def gravitational_force(self, m1, m2, r):
    """Formula: F = G × m1 × m2 / r²"""
    return self.G * m1 * m2 / r**2

def coulomb_force(self, q1, q2, r):
    """Formula: F = k_e × q1 × q2 / r²"""
    return self.k_e * q1 * q2 / r**2
```

### F. Schwarzschild Radius
```python
def schwarzschild_radius(self, mass):
    """Formula: R_s = 2GM / c²"""
    return (2 * self.G * mass) / (self.c**2)
```

### G. Orbital Mechanics
```python
def orbital_period(self, semi_major_axis, constants):
    """Formula: T = 2π√(a³/GM)"""
    return 2 * np.pi * np.sqrt(semi_major_axis**3 / (constants.G * self.central_mass))

def gr_precession(self, semi_major_axis, eccentricity, constants):
    """
    Formula: δφ = 6πGM / (ac²(1-e²))
    Per orbit precession in radians
    """
    return (6 * np.pi * constants.G * self.central_mass) / \
           (semi_major_axis * constants.c**2 * (1 - eccentricity**2))
```

---

## 4. CRITICAL SCALING RELATIONSHIPS

### The Core Hypothesis
| Relationship | Formula | Meaning |
|-------------|---------|---------|
| Pressure Balance | **P_grav / P_Pauli ∝ G / ℏ²** | Gravity vs quantum support |
| For Stability | G_scale = ℏ_scale² | Must maintain this ratio |
| Essay Scenario | G × 10³⁶ requires ℏ × 10¹⁸ | √(10³⁶) = 10¹⁸ |

### How Constants Scale
| Quantity | Scales as | At G×10³⁶, ℏ×10¹⁸ |
|----------|-----------|-------------------|
| Bohr radius a₀ | ∝ ℏ² | × 10³⁶ (atoms huge) |
| Degeneracy pressure | ∝ ℏ² | × 10³⁶ (compensates G) |
| Gravitational coupling α_G | ∝ G/ℏ | × 10¹⁸ (still stronger) |
| Chandrasekhar mass | ∝ (ℏc/G)^(3/2) | Modified threshold |

---

## 5. DATA TABLE: Force Comparison at Atomic Scales

| Distance | F_gravitational (N) | F_coulomb (N) | Ratio F_c/F_g |
|----------|---------------------|---------------|---------------|
| 1 fm | ~10⁻³⁴ | ~230 N | ~10³⁶ |
| 10 fm | ~10⁻³⁶ | ~2.3 N | ~10³⁶ |
| 100 fm | ~10⁻³⁸ | ~0.023 N | ~10³⁶ |
| 1 pm | ~10⁻⁴⁰ | ~2.3×10⁻⁴ N | ~10³⁶ |
| 1 Å | ~10⁻⁴⁶ | ~2.3×10⁻⁸ N | ~10³⁶ |

**The ratio is constant (~1.24 × 10³⁶) regardless of distance because both forces follow 1/r² law.**

---

## 6. PROJECT FILE STRUCTURE

```
Project/
├── config/
│   └── constants.json              # All physical constants (JSON)
├── modules/
│   ├── constants.py                # Core constants & scaling functions
│   ├── gravity_pauli_balance.py    # ESSAY CORE MODULE
│   ├── force_comparison.py         # Force visualizations
│   ├── orbital_mechanics.py        # Orbital dynamics
│   ├── atomic_scale.py             # Quantum scale effects
│   ├── spacetime_curvature.py      # GR effects
│   ├── solar_physics.py            # Star physics
│   ├── hydrostatic_equilibrium.py  # Internal structure
│   ├── white_dwarf.py              # White dwarf physics
│   ├── neutron_star.py             # Neutron star physics
│   └── ...                         # Additional modules
├── visualizations/
│   ├── hypothesis_summary.png
│   ├── gravity_vs_pauli.png
│   ├── force_vs_distance.png
│   ├── earth_structural_effects.png
│   ├── atom_size_comparison.png
│   ├── orbital_precession_animation.gif
│   └── ... (122 PNG + 32 HTML + 4 GIF files)
├── docs/
│   ├── formeln.md                  # Complete formula documentation
│   └── peer_review_materials.md    # This file
└── main.py                         # Entry point
```

---

## 7. SUMMARY FOR REVIEWER

### Essay Core Claim (Verified by Code)
1. **Standard Universe**: F_gravity/F_coulomb ≈ 10⁻³⁶ at atomic scales
2. **If G increases by 10³⁶**: Gravity would equal electromagnetic force
3. **To maintain atomic stability**: Need ℏ × 10¹⁸ because:
   - Bohr radius a₀ ∝ ℏ² (atoms scale up)
   - Degeneracy pressure P_deg ∝ ℏ² (support increases)
   - Balance: P_grav/P_Pauli ∝ G/ℏ² stays constant when ℏ² = G_scale

### Key Result from Graphs
The stability map (Panel 4 of earth_structural_effects.png) shows:
- **Blue region**: Earth remains stable
- **Red region**: Earth collapses
- The boundary line follows ℏ ∝ √G exactly as predicted

### Physical Constants Source
All values from **CODATA 2018** recommended values and **IAU 2015** astronomical constants.

---

## 8. VERIFICATION CHECKLIST

- [ ] Speed of light c = 299,792,458 m/s (standard, unmodified)
- [ ] Gravitational constant G = 6.67430 × 10⁻¹¹ m³·kg⁻¹·s⁻²
- [ ] Reduced Planck constant ℏ = 1.054571817 × 10⁻³⁴ J·s
- [ ] Force ratio F_coulomb/F_gravity ≈ 10³⁶ at atomic scales
- [ ] Bohr radius formula: a₀ = 4πε₀ℏ²/(m_e·e²)
- [ ] Degeneracy pressure formula: P ∝ ℏ² × ρ^(5/3)
- [ ] Central pressure formula: P_c ∝ G × M²/R⁴
- [ ] Balance condition: P_grav/P_Pauli ∝ G/ℏ²
- [ ] Essay scaling: G × 10³⁶ compensated by ℏ × 10¹⁸

---

*Generated for peer review - Jugend forscht 2026*
*Last updated: February 2026*
