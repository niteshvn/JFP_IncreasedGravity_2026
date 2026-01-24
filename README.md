# Jugend forscht 2026 - Physics Visualization Suite
# Physik-Visualisierungsprojekt

## Research Question / Forschungsfrage

**English:** How does increasing gravitational pressure modify the balance between electron degeneracy pressure (Pauli) and Coulomb interactions in dense stellar matter?

**Deutsch:** Wie veraendert zunehmender Gravitationsdruck das Gleichgewicht zwischen Elektronenentartungsdruck (Pauli) und Coulomb-Wechselwirkungen in dichter Sternmaterie?

---

## Project Overview / Projektuebersicht

This Python visualization suite explores the interplay between quantum mechanics and gravity, demonstrating why gravity is negligible at atomic scales but becomes dominant in extreme astrophysical environments like white dwarf stars.

Dieses Python-Visualisierungspaket untersucht das Zusammenspiel von Quantenmechanik und Gravitation und zeigt, warum die Gravitation auf atomaren Skalen vernachlaessigbar ist, aber in extremen astrophysikalischen Umgebungen wie Weissen Zwergen dominant wird.

---

## Key Physics Concepts / Wichtige Physikkonzepte

### The Gravitational Coupling Constant / Die Gravitationskopplungskonstante

The dimensionless gravitational coupling constant measures gravity's strength relative to quantum effects:

```
alpha_G = (G * m_p^2) / (hbar * c) = 5.9 * 10^-39
```

This incredibly small number explains why gravity is irrelevant at atomic scales - electromagnetic forces are ~10^36 times stronger!

### The Chandrasekhar Limit / Die Chandrasekhar-Grenze

When a star exhausts its nuclear fuel, electron degeneracy pressure (from Pauli's exclusion principle) can support it against gravitational collapse - but only up to ~1.4 solar masses:

```
M_Ch = (hbar * c / G)^(3/2) / m_p^2 = 1.44 M_sun
```

Above this limit, gravity wins and the star collapses further.

### Scaling Relations / Skalierungsbeziehungen

When we decrease hbar (Planck's constant):
- Atoms shrink: a_0 proportional to hbar^2
- Gravity coupling increases: alpha_G proportional to 1/hbar
- Matter density increases: rho proportional to 1/hbar^6
- Gravity becomes relatively more important!

---

## Installation / Installation

### Requirements / Anforderungen

```bash
pip install numpy matplotlib plotly
```

### Project Structure / Projektstruktur

```
Project/
├── config/
│   └── constants.json          # Physical constants configuration
├── modules/
│   ├── __init__.py            # Module exports
│   ├── constants.py           # Constants and scaling
│   ├── force_comparison.py    # Force comparison visualizations
│   ├── white_dwarf.py         # White dwarf physics
│   ├── spacetime_curvature.py # Spacetime curvature effects
│   └── atomic_scale.py        # Atomic scale effects
├── visualizations/            # Generated plots (18 total)
├── Analysis_docs/             # Project documentation
├── main.py                    # Entry point
└── README.md                  # This file
```

---

## Usage / Verwendung

### Generate All Visualizations / Alle Visualisierungen erzeugen

```bash
python main.py
```

### Command Line Options / Befehlszeilenoptionen

| Flag | Description (EN) | Beschreibung (DE) |
|------|------------------|-------------------|
| `--verify` or `-v` | Run physics verification | Physik-Verifizierung ausfuehren |
| `--compare` or `-c` | Show scaled universe comparison | Skaliertes Universum vergleichen |
| `--generate` or `-g` | Generate all visualizations | Alle Visualisierungen erzeugen |
| `--all` or `-a` | Run everything | Alles ausfuehren |
| `--german` or `-de` | German language plots | Deutsche Beschriftungen |
| `--quiet` or `-q` | Suppress verbose output | Ausgabe unterdruecken |

### Examples / Beispiele

```bash
# Generate all plots with English labels
python main.py

# Generate plots with German labels
python main.py --german

# Run physics verification only
python main.py --verify

# Run everything (verify + compare + generate)
python main.py --all
```

---

## Modules / Module

### Module 1: Constants (`constants.py`)

Defines all physical constants with support for scaling:

| Constant | Symbol | Value | Description |
|----------|--------|-------|-------------|
| Speed of light | c | 2.998 * 10^8 m/s | Lichtgeschwindigkeit |
| Planck constant | hbar | 1.055 * 10^-34 J*s | Planck-Konstante |
| Gravitational constant | G | 6.674 * 10^-11 m^3/(kg*s^2) | Gravitationskonstante |
| Electron mass | m_e | 9.109 * 10^-31 kg | Elektronenmasse |
| Proton mass | m_p | 1.673 * 10^-27 kg | Protonenmasse |
| Elementary charge | e | 1.602 * 10^-19 C | Elementarladung |
| Fine structure | alpha | 1/137 | Feinstrukturkonstante |

### Module 2: Force Comparison (`force_comparison.py`)

Visualizes the dramatic difference between gravitational and electromagnetic forces:

- **force_comparison_bar.png**: Bar chart showing ~10^36 force ratio
- **force_vs_distance.png**: How forces scale with distance
- **force_across_scales.png**: Forces at atomic, molecular, human, planetary scales
- **force_scaled_*.png**: How forces change in alternative universes

### Module 3: White Dwarf Physics (`white_dwarf.py`)

Explores electron degeneracy pressure and the Chandrasekhar limit:

- **pressure_vs_density.png**: Non-relativistic (rho^5/3) vs relativistic (rho^4/3) regimes
- **mass_radius_relation.png**: Why more massive white dwarfs are smaller
- **chandrasekhar_limit.png**: The 1.44 solar mass stability limit
- **white_dwarf_summary.png**: Comprehensive overview

### Module 4: Spacetime Curvature (`spacetime_curvature.py`)

Visualizes gravitational effects on spacetime:

- **potential_well_2d.png**: Gravitational potential comparison
- **potential_well_3d.png**: 3D "rubber sheet" visualization
- **compactness_comparison.png**: R_schwarzschild / R for different objects
- **escape_velocity.png**: Escape velocities (log scale)
- **spacetime_summary.png**: Comprehensive overview

### Module 5: Atomic Scale (`atomic_scale.py`)

Shows how quantum mechanics determines atomic structure:

- **bohr_radius_scaling.png**: How a_0 changes with hbar
- **atom_size_comparison.png**: Visual comparison of atom sizes
- **energy_levels.png**: Hydrogen energy levels and orbital radii
- **quantum_gravity_connection.png**: How hbar connects QM and gravity
- **atomic_summary.png**: Comprehensive overview

---

## Physics Verification / Physik-Verifizierung

Run `python main.py --verify` to check all calculations:

| Property | Calculated | Expected | Status |
|----------|------------|----------|--------|
| Chandrasekhar mass | 1.435 M_sun | ~1.44 M_sun | PASS |
| Bohr radius | 52.92 pm | 52.92 pm | PASS |
| H ground state | -13.61 eV | -13.6 eV | PASS |
| Fine structure | 1/137.04 | 1/137 | PASS |
| alpha_G | 5.91 * 10^-39 | ~5.9 * 10^-39 | PASS |
| Force ratio (p-p) | 1.24 * 10^36 | ~1.2 * 10^36 | PASS |
| Sun R_schwarzschild | 2.95 km | ~2.95 km | PASS |

---

## Key Formulas / Wichtige Formeln

### Gravitational Force / Gravitationskraft
```
F_grav = G * (m1 * m2) / r^2
```

### Coulomb Force / Coulomb-Kraft
```
F_coulomb = k_e * (q1 * q2) / r^2
```

### Bohr Radius / Bohr-Radius
```
a_0 = 4 * pi * epsilon_0 * hbar^2 / (m_e * e^2) = 52.9 pm
```

### Electron Degeneracy Pressure / Elektronenentartungsdruck
```
P_e = (hbar^2 / m_e) * (N_e / V)^(5/3)    [non-relativistic]
P_e = hbar * c * (N_e / V)^(4/3)          [relativistic]
```

### Chandrasekhar Mass / Chandrasekhar-Masse
```
M_Ch = (hbar * c / G)^(3/2) / m_p^2 = 1.44 M_sun
```

### Schwarzschild Radius / Schwarzschild-Radius
```
R_s = 2 * G * M / c^2
```

### Escape Velocity / Fluchtgeschwindigkeit
```
v_esc = sqrt(2 * G * M / R)
```

---

## Educational Context / Bildungskontext

This project is designed to help explain to competition judges why:

1. **Gravity is normally irrelevant at quantum scales** - The electromagnetic force between a proton and electron is ~10^39 times stronger than gravity!

2. **Gravity becomes dominant in extreme environments** - In white dwarfs, matter is so compressed that gravitational pressure overcomes electron degeneracy pressure above 1.4 solar masses.

3. **The Chandrasekhar limit connects quantum mechanics and gravity** - The formula M_Ch = (hbar*c/G)^(3/2) / m_p^2 beautifully combines Planck's constant (quantum), speed of light (relativity), and G (gravity).

4. **Changing fundamental constants has dramatic effects** - Reducing hbar by 10x makes atoms 100x smaller, gravity 10x relatively stronger, and matter density 1,000,000x higher!

---

## License / Lizenz

This project was created for Jugend forscht 2026.

---

## Authors / Autoren

Jugend forscht 2026 Team

Created with assistance from Claude (Anthropic)
