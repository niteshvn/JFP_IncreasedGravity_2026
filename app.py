"""
Jugend forscht 2026 - Interactive Physics Dashboard
====================================================

A comprehensive Streamlit web application for presenting physics visualizations
to competition judges. Works on any device including iPad.

Features:
- All 22 visualizations (18 PNG + 4 interactive HTML)
- Language toggle (German/English)
- Formula documentation integrated with each section
- Responsive design

Run with: streamlit run app.py

Author: Navya Nahta
Supervisor: Hr. Seuferling
"""

import streamlit as st
import os
from pathlib import Path

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Gravitation & Quantenmechanik | Jugend forscht 2026",
    page_icon="⭐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# LOAD EXTERNAL CSS
# =============================================================================

def load_css():
    """Load CSS from external file for consistent styling across all pages."""
    css_file = Path(__file__).parent / "static" / "style.css"
    if css_file.exists():
        with open(css_file, 'r', encoding='utf-8') as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

load_css()

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent
VIS_DIR = PROJECT_ROOT / "visualizations"

# =============================================================================
# LANGUAGE DICTIONARY
# =============================================================================

LANG = {
    'en': {
        'title': "Gravity vs Quantum Mechanics",
        'subtitle': "What happens when gravitational force increases?",
        'author': "Project by Navya Nahta | Supervisor: Hr. Seuferling",

        # Navigation
        'nav_intro': "🏠 Introduction",
        'nav_forces': "⚡ Force Comparison",
        'nav_whitedwarf': "⭐ White Dwarfs",
        'nav_spacetime': "🌌 Spacetime",
        'nav_atoms': "⚛️ Atomic Scale",
        'nav_thermal': "🌡️ Temperature Physics",
        'nav_interactive': "🎮 3D Interactive",

        # Introduction
        'intro_title': "Research Question",
        'intro_question': """**What happens when gravitational force on Earth increases?
What does this mean for related factors (e.g., Pauli's principle, Coulomb's law, spacetime curvature)?
Which physical constants need to scale together?**""",
        'hypothesis_title': "Hypothesis",
        'hypothesis_text': """By increasing gravitational force (by 10³⁶, with scaling of ℏ to prevent atomic collapse):
- Earth's radius becomes smaller
- Objects become shorter/compressed
- Spacetime curvature becomes more prominent""",
        'motivation_title': "Motivation",
        'motivation_text': """I've always wondered if gravitational force could be increased. Usually we only talk about
*low* or *no* gravity (in space, on the Moon). I thought the question of what happens with
*increased* gravity was pointless.

However, during summer vacation I read a NASA article about **White Dwarfs**, **Neutron Stars**,
and **Black Holes** - and my interest was rekindled!""",
        'key_insight': "Key Insight",
        'key_insight_text': """In our universe, gravity is incredibly weak compared to electromagnetic forces at atomic scales -
about **10³⁶ times weaker**! This is why atoms don't collapse under their own gravity.

But in extreme environments like white dwarf stars, matter is so compressed that gravity finally
becomes important. The **Chandrasekhar limit** (1.44 solar masses) marks where gravity wins.""",

        # Section titles
        'force_section': "Force Comparison Visualizations",
        'wd_section': "White Dwarf Physics Visualizations",
        'spacetime_section': "Spacetime Curvature Visualizations",
        'atomic_section': "Atomic Scale Visualizations",
        'thermal_section': "Temperature Physics Visualizations",
        'interactive_section': "Interactive 3D Visualizations",
    },
    'de': {
        'title': "Gravitation vs Quantenmechanik",
        'subtitle': "Was passiert bei erhöhter Gravitationskraft?",
        'author': "Projekt von Navya Nahta | Betreuer: Hr. Seuferling",

        # Navigation
        'nav_intro': "🏠 Einführung",
        'nav_forces': "⚡ Kräftevergleich",
        'nav_whitedwarf': "⭐ Weiße Zwerge",
        'nav_spacetime': "🌌 Raumzeit",
        'nav_atoms': "⚛️ Atomare Skala",
        'nav_thermal': "🌡️ Temperaturphysik",
        'nav_interactive': "🎮 3D Interaktiv",

        # Introduction
        'intro_title': "Forschungsfrage",
        'intro_question': """**Was geschieht bei erhöhter Gravitationskraft auf der Erde?
Was bedeutet das für zusammenhängende Faktoren (z.B. Paulis Prinzip, Coulombs Gesetz, Raumzeitkrümmung)?
Welche physikalischen Konstanten müssen mitskaliert werden?**""",
        'hypothesis_title': "Hypothese",
        'hypothesis_text': """Durch eine Erhöhung der Gravitationskraft (um 10³⁶, mit Skalierung von ℏ damit es nicht
zum atomischen Kollaps kommt):
- Der Radius der Erde wird geringer
- Lebewesen/Objekte werden kürzer (komprimiert)
- Die Raumzeitkrümmung wird prominenter""",
        'motivation_title': "Motivation",
        'motivation_text': """Ich habe mich schon immer gefragt, ob sich die Gravitationskraft erhöhen könnte. Immer wurde
von *wenig* oder *keiner* Gravitationskraft gesprochen (im Weltall, z.B. Mond). Ich dachte,
die Frage nach *erhöhter* Gravitation sei sinnlos.

Jedoch habe ich in den Sommerferien einen NASA-Artikel über **Weiße Zwerge**, **Neutronensterne**
und **Schwarze Löcher** gelesen - und mein Interesse wurde geweckt!""",
        'key_insight': "Wichtige Erkenntnis",
        'key_insight_text': """In unserem Universum ist die Gravitation auf atomaren Skalen unglaublich schwach im Vergleich
zu elektromagnetischen Kräften - etwa **10³⁶ mal schwächer**! Deshalb kollabieren Atome nicht
unter ihrer eigenen Schwerkraft.

Aber in extremen Umgebungen wie Weißen Zwergen ist die Materie so komprimiert, dass die
Gravitation endlich wichtig wird. Die **Chandrasekhar-Grenze** (1,44 Sonnenmassen) markiert,
wo die Gravitation gewinnt.""",

        # Section titles
        'force_section': "Kräftevergleich-Visualisierungen",
        'wd_section': "Physik der Weißen Zwerge",
        'spacetime_section': "Raumzeitkrümmungs-Visualisierungen",
        'atomic_section': "Atomare Skalen-Visualisierungen",
        'thermal_section': "Temperaturphysik-Visualisierungen",
        'interactive_section': "Interaktive 3D-Visualisierungen",
    }
}

# =============================================================================
# IMAGE CATALOG
# =============================================================================

IMAGES = {
    'forces': [
        ('force_comparison_bar.png', 'Force Comparison (Bar Chart)', 'Kräftevergleich (Balkendiagramm)',
         'Compares gravitational and Coulomb forces between two protons at 1 fm distance.',
         'Vergleicht Gravitations- und Coulomb-Kräfte zwischen zwei Protonen bei 1 fm Abstand.'),
        ('force_vs_distance.png', 'Forces vs Distance', 'Kräfte vs Abstand',
         'Shows how both forces follow the inverse-square law but differ by 10³⁶.',
         'Zeigt wie beide Kräfte dem 1/r²-Gesetz folgen, aber sich um 10³⁶ unterscheiden.'),
        ('force_across_scales.png', 'Forces Across Scales', 'Kräfte über verschiedene Skalen',
         'Demonstrates why gravity dominates at stellar scales (charges cancel).',
         'Zeigt warum Gravitation bei stellaren Skalen dominiert (Ladungen heben sich auf).'),
        ('force_scaled_hbar0.1_G1.0.png', 'Scaled Universe Comparison', 'Skaliertes Universum',
         'Compares forces in standard vs alternative universe with ℏ×0.1.',
         'Vergleicht Kräfte im Standard- vs. alternativem Universum mit ℏ×0.1.'),
    ],
    'whitedwarf': [
        ('pressure_vs_density.png', 'Pressure vs Density', 'Druck vs Dichte',
         'Electron degeneracy pressure in non-relativistic and relativistic regimes.',
         'Elektronenentartungsdruck im nicht-relativistischen und relativistischen Bereich.'),
        ('mass_radius_relation.png', 'Mass-Radius Relation', 'Masse-Radius-Beziehung',
         'Shows why more massive white dwarfs are SMALLER (R ∝ M⁻¹/³).',
         'Zeigt warum massivere Weiße Zwerge KLEINER sind (R ∝ M⁻¹/³).'),
        ('chandrasekhar_limit.png', 'Chandrasekhar Limit', 'Chandrasekhar-Grenze',
         'Visualizes why the 1.44 M☉ limit exists - degeneracy vs gravity.',
         'Visualisiert warum die 1,44 M☉-Grenze existiert - Entartung vs. Gravitation.'),
        ('white_dwarf_summary.png', 'White Dwarf Summary', 'Zusammenfassung Weiße Zwerge',
         'Comprehensive overview of white dwarf physics.',
         'Umfassende Übersicht der Physik Weißer Zwerge.'),
    ],
    'spacetime': [
        ('potential_well_2d.png', '2D Potential Wells', '2D Potentialmulden',
         'Cross-section of gravitational potential for different stellar objects.',
         'Querschnitt des Gravitationspotentials für verschiedene Sternobjekte.'),
        ('potential_well_3d.png', '3D Spacetime Curvature', '3D Raumzeitkrümmung',
         'The classic "rubber sheet" visualization of curved spacetime.',
         'Die klassische "Gummituch"-Visualisierung der gekrümmten Raumzeit.'),
        ('compactness_comparison.png', 'Compactness Comparison', 'Kompaktheitsvergleich',
         'Compares R_schwarzschild/R for Earth, Sun, white dwarf, neutron star.',
         'Vergleicht R_schwarzschild/R für Erde, Sonne, Weißer Zwerg, Neutronenstern.'),
        ('escape_velocity.png', 'Escape Velocity', 'Fluchtgeschwindigkeit',
         'Shows how escape velocity approaches c for compact objects.',
         'Zeigt wie die Fluchtgeschwindigkeit sich c nähert für kompakte Objekte.'),
        ('spacetime_summary.png', 'Spacetime Summary', 'Raumzeit-Zusammenfassung',
         'Comprehensive overview of spacetime curvature concepts.',
         'Umfassende Übersicht der Raumzeitkrümmungskonzepte.'),
    ],
    'atoms': [
        ('bohr_radius_scaling.png', 'Bohr Radius Scaling', 'Bohr-Radius-Skalierung',
         'How Bohr radius changes with ℏ: a₀ ∝ ℏ².',
         'Wie sich der Bohr-Radius mit ℏ ändert: a₀ ∝ ℏ².'),
        ('atom_size_comparison.png', 'Atom Size Comparison', 'Atomgrößenvergleich',
         'Visual comparison of atom sizes in different universes.',
         'Visueller Vergleich der Atomgrößen in verschiedenen Universen.'),
        ('energy_levels.png', 'Energy Levels', 'Energieniveaus',
         'Hydrogen energy levels and spectral transitions.',
         'Wasserstoff-Energieniveaus und Spektralübergänge.'),
        ('quantum_gravity_connection.png', 'Quantum-Gravity Connection', 'Quanten-Gravitations-Verbindung',
         'How ℏ connects quantum mechanics and gravity.',
         'Wie ℏ Quantenmechanik und Gravitation verbindet.'),
        ('atomic_summary.png', 'Atomic Summary', 'Atomare Zusammenfassung',
         'Comprehensive overview of atomic scale effects.',
         'Umfassende Übersicht der atomaren Skaleneffekte.'),
    ],
    'thermal': [
        ('temperature_atmosphere.png', 'Atmospheric Temperature Profile', 'Atmosphärisches Temperaturprofil',
         'Shows temperature vs altitude for different gravity values and scale height compression.',
         'Zeigt Temperatur vs. Höhe für verschiedene Gravitationswerte und Skalenhöhenkompression.'),
        ('temperature_degeneracy.png', 'Degeneracy vs Thermal Pressure', 'Entartungs- vs. Thermischer Druck',
         'Fermi temperature and the crossover between thermal and degeneracy pressure regimes.',
         'Fermi-Temperatur und der Übergang zwischen thermischem und Entartungsdruck-Regime.'),
        ('temperature_summary.png', 'Temperature Physics Summary', 'Temperaturphysik-Zusammenfassung',
         'Comprehensive overview of temperature effects with increasing gravity.',
         'Umfassende Übersicht der Temperatureffekte bei steigender Gravitation.'),
    ],
}

INTERACTIVE = [
    ('spacetime_3d_interactive.html', 'Spacetime Curvature 3D', 'Raumzeitkrümmung 3D'),
    ('spacetime_comparison_3d_interactive.html', 'Multiple Masses Comparison', 'Vergleich mehrerer Massen'),
    ('atom_scaling_3d_interactive.html', 'Atom Scaling Surface', 'Atomskalierung-Oberfläche'),
    ('force_ratio_3d_interactive.html', 'Force Ratio Surface', 'Kräfteverhältnis-Oberfläche'),
    ('temperature_profile_3d_interactive.html', 'Temperature vs Altitude & Gravity', 'Temperatur vs. Höhe & Gravitation'),
]

# =============================================================================
# FORMULAS CONTENT - Organized by section
# =============================================================================

FORMULAS = {
    'forces': {
        'en': [
            {  # force_comparison_bar.png
                'title': 'Force Comparison',
                'formula': r'F_C = k_e \frac{q_1 q_2}{r^2} \quad \text{vs} \quad F_G = G \frac{m_1 m_2}{r^2}',
                'description': '''Both follow 1/r² law. At atomic scales: **F_C/F_G ≈ 10³⁶** (Coulomb dominates!)''',
            },
            {  # force_vs_distance.png
                'title': 'Inverse Square Law',
                'formula': r'F \propto \frac{1}{r^2}',
                'description': '''Both forces decrease with distance squared. The **ratio stays constant** at all distances.''',
            },
            {  # force_across_scales.png
                'title': 'Why Gravity Wins at Large Scales',
                'formula': r'\sum q_i = 0 \quad \text{but} \quad \sum m_i > 0',
                'description': '''Charges cancel in bulk matter. Mass always adds up → **Gravity dominates in stars!**''',
            },
            {  # force_scaled_hbar0.1_G1.0.png
                'title': 'Gravitational Coupling',
                'formula': r'\alpha_G = \frac{G m_p^2}{\hbar c} \propto \frac{1}{\hbar}',
                'description': '''When ℏ decreases → α_G increases → **Gravity becomes relatively stronger!**''',
            },
        ],
        'de': [
            {  # force_comparison_bar.png
                'title': 'Kräftevergleich',
                'formula': r'F_C = k_e \frac{q_1 q_2}{r^2} \quad \text{vs} \quad F_G = G \frac{m_1 m_2}{r^2}',
                'description': '''Beide folgen dem 1/r²-Gesetz. Bei atomaren Skalen: **F_C/F_G ≈ 10³⁶** (Coulomb dominiert!)''',
            },
            {  # force_vs_distance.png
                'title': 'Abstandsquadratgesetz',
                'formula': r'F \propto \frac{1}{r^2}',
                'description': '''Beide Kräfte nehmen mit dem Quadrat des Abstands ab. Das **Verhältnis bleibt konstant**.''',
            },
            {  # force_across_scales.png
                'title': 'Warum Gravitation bei großen Skalen gewinnt',
                'formula': r'\sum q_i = 0 \quad \text{aber} \quad \sum m_i > 0',
                'description': '''Ladungen heben sich in Materie auf. Masse addiert sich → **Gravitation dominiert in Sternen!**''',
            },
            {  # force_scaled_hbar0.1_G1.0.png
                'title': 'Gravitationskopplung',
                'formula': r'\alpha_G = \frac{G m_p^2}{\hbar c} \propto \frac{1}{\hbar}',
                'description': '''Wenn ℏ sinkt → α_G steigt → **Gravitation wird relativ stärker!**''',
            },
        ],
    },
    'whitedwarf': {
        'en': [
            {  # pressure_vs_density.png
                'title': 'Degeneracy Pressure',
                'formula': r'P_{nr} \propto \rho^{5/3} \quad \text{vs} \quad P_r \propto \rho^{4/3}',
                'description': '''Low density: steeper slope → more stable. High density: shallower slope → less stable. Pauli principle creates pressure against gravitational collapse!''',
            },
            {  # mass_radius_relation.png
                'title': 'Mass-Radius Relation',
                'formula': r'R \propto M^{-1/3}',
                'description': '''More mass → Smaller radius! Why? More mass = more gravity → electrons squeezed tightly → higher degeneracy pressure at smaller volume.''',
            },
            {  # chandrasekhar_limit.png
                'title': 'Chandrasekhar Mass',
                'formula': r'M_{Ch} \approx 1.44 M_\odot \propto \left(\frac{\hbar c}{G}\right)^{3/2}',
                'description': '''Maximum mass for stable white dwarf. Above this → collapse to neutron star!''',
            },
            {  # white_dwarf_summary.png
                'title': 'Key Physics',
                'formula': r'\text{Stability:} \quad P_{degeneracy} \geq P_{gravity}',
                'description': '''Pauli principle: electrons forced into higher energy states → degeneracy pressure. M_Ch connects quantum mechanics (ℏ) with gravity (G)!''',
            },
        ],
        'de': [
            {  # pressure_vs_density.png
                'title': 'Entartungsdruck',
                'formula': r'P_{nr} \propto \rho^{5/3} \quad \text{vs} \quad P_r \propto \rho^{4/3}',
                'description': '''Niedrige Dichte: steilerer Anstieg → stabiler. Hohe Dichte: flacherer Anstieg → instabiler. Pauli-Prinzip erzeugt Druck gegen Gravitationskollaps!''',
            },
            {  # mass_radius_relation.png
                'title': 'Masse-Radius-Beziehung',
                'formula': r'R \propto M^{-1/3}',
                'description': '''Mehr Masse → Kleinerer Radius! Warum? Mehr Masse = mehr Gravitation → Elektronen zusammengepresst → höherer Entartungsdruck bei kleinerem Volumen.''',
            },
            {  # chandrasekhar_limit.png
                'title': 'Chandrasekhar-Masse',
                'formula': r'M_{Ch} \approx 1,44 M_\odot \propto \left(\frac{\hbar c}{G}\right)^{3/2}',
                'description': '''Maximale Masse für stabilen Weißen Zwerg. Darüber → Kollaps zum Neutronenstern!''',
            },
            {  # white_dwarf_summary.png
                'title': 'Schlüsselphysik',
                'formula': r'\text{Stabilität:} \quad P_{Entartung} \geq P_{Gravitation}',
                'description': '''Pauli-Prinzip: Elektronen in höhere Energiezustände gezwungen → Entartungsdruck. M_Ch verbindet Quantenmechanik (ℏ) mit Gravitation (G)!''',
            },
        ],
    },
    'spacetime': {
        'en': [
            {  # potential_well_2d.png
                'title': 'Gravitational Potential',
                'formula': r'\Phi = -\frac{GM}{r}',
                'description': '''Deeper potential = stronger spacetime curvature. Neutron stars ~1000× deeper than white dwarfs!''',
            },
            {  # potential_well_3d.png
                'title': 'Schwarzschild Radius',
                'formula': r'R_s = \frac{2GM}{c^2}',
                'description': '''Radius where escape velocity = c. The "rubber sheet" shows how mass curves spacetime.''',
            },
            {  # compactness_comparison.png
                'title': 'Compactness Parameter',
                'formula': r'C = \frac{R_s}{R} = \frac{2GM}{Rc^2}',
                'description': '''C → 1: Object becomes a Black Hole. More compact = stronger spacetime curvature! Earth: C ≈ 10⁻⁹ | White dwarf: C ≈ 10⁻⁴ | Neutron star: C ≈ 0.3''',
            },
            {  # escape_velocity.png
                'title': 'Escape Velocity',
                'formula': r'v_{esc} = \sqrt{\frac{2GM}{R}} = c\sqrt{C}',
                'description': '''When v_esc = c → nothing can escape → Black hole!''',
            },
            {  # spacetime_summary.png
                'title': 'Spacetime Curvature Summary',
                'formula': r'R_s = \frac{2GM}{c^2}, \quad C = \frac{R_s}{R}, \quad v_{esc} = \sqrt{\frac{2GM}{R}}',
                'description': '''**Schwarzschild radius:** R_s is where escape velocity = c. Earth: R_s ≈ 9 mm | Sun: R_s ≈ 3 km. **Compactness:** Earth C ≈ 10⁻⁹ | White dwarf C ≈ 10⁻⁴ | Neutron star C ≈ 0.2-0.4 | Black hole C = 1. **Key insight:** The more compact an object, the more it curves spacetime → stronger gravitational effects. White dwarfs are compact enough that relativistic effects become important (Chandrasekhar limit)!''',
            },
        ],
        'de': [
            {  # potential_well_2d.png
                'title': 'Gravitationspotential',
                'formula': r'\Phi = -\frac{GM}{r}',
                'description': '''Tieferes Potential = stärkere Raumzeitkrümmung. Neutronensterne ~1000× tiefer als Weiße Zwerge!''',
            },
            {  # potential_well_3d.png
                'title': 'Schwarzschild-Radius',
                'formula': r'R_s = \frac{2GM}{c^2}',
                'description': '''Radius bei dem Fluchtgeschwindigkeit = c. Das "Gummituch" zeigt wie Masse die Raumzeit krümmt.''',
            },
            {  # compactness_comparison.png
                'title': 'Kompaktheitsparameter',
                'formula': r'C = \frac{R_s}{R} = \frac{2GM}{Rc^2}',
                'description': '''C → 1: Objekt wird zum Schwarzen Loch. Je kompakter, desto stärker die Raumzeitkrümmung! Erde: C ≈ 10⁻⁹ | Weißer Zwerg: C ≈ 10⁻⁴ | Neutronenstern: C ≈ 0,3''',
            },
            {  # escape_velocity.png
                'title': 'Fluchtgeschwindigkeit',
                'formula': r'v_{esc} = \sqrt{\frac{2GM}{R}} = c\sqrt{C}',
                'description': '''Wenn v_esc = c → nichts kann entkommen → Schwarzes Loch!''',
            },
            {  # spacetime_summary.png
                'title': 'Raumzeitkrümmung Zusammenfassung',
                'formula': r'R_s = \frac{2GM}{c^2}, \quad C = \frac{R_s}{R}, \quad v_{esc} = \sqrt{\frac{2GM}{R}}',
                'description': '''**Schwarzschild-Radius:** R_s ist wo Fluchtgeschwindigkeit = c. Erde: R_s ≈ 9 mm | Sonne: R_s ≈ 3 km. **Kompaktheit:** Erde C ≈ 10⁻⁹ | Weißer Zwerg C ≈ 10⁻⁴ | Neutronenstern C ≈ 0,2-0,4 | Schwarzes Loch C = 1. **Kernaussage:** Je kompakter ein Objekt, desto stärker krümmt es die Raumzeit → stärkere Gravitationseffekte. Weiße Zwerge sind kompakt genug, dass relativistische Effekte wichtig werden (Chandrasekhar-Grenze)!''',
            },
        ],
    },
    'atoms': {
        'en': [
            {  # bohr_radius_scaling.png
                'title': 'Bohr Radius Scaling',
                'formula': r'a_0 = \frac{4\pi\epsilon_0 \hbar^2}{m_e e^2} \propto \hbar^2',
                'description': '''If ℏ decreases by 10× → atoms shrink by 100×!''',
            },
            {  # atom_size_comparison.png
                'title': 'Atom Size',
                'formula': r'a_0 \approx 52.9 \text{ pm (standard)}',
                'description': '''Smaller ℏ → smaller atoms → higher density → gravity becomes more important!''',
            },
            {  # energy_levels.png
                'title': 'Hydrogen Energy Levels',
                'formula': r'E_n = -\frac{13.6 \text{ eV}}{n^2}, \quad r_n = n^2 \times a_0',
                'description': '''**Quantization:** Only certain energies allowed (E_n ∝ 1/n², r_n ∝ n²). **Ground state (n=1):** E₁ = -13.6 eV, r₁ = 52.9 pm. Transitions between levels produce spectral lines (Lyman, Balmer series).''',
            },
            {  # quantum_gravity_connection.png
                'title': 'Quantum-Gravity Connection',
                'formula': r'\alpha_G = \frac{G m_p^2}{\hbar c} \approx 5.9 \times 10^{-39}',
                'description': '''**At ℏ × 0.1:** Atoms 100× smaller (a₀ ∝ ℏ²), gravity 10× stronger (α_G ∝ 1/ℏ), density 10⁶× higher (ρ ∝ 1/ℏ⁶), gravity importance 10⁷× greater! **Key insight:** In a universe with smaller ℏ, gravity would become important at much smaller masses.''',
            },
            {  # atomic_summary.png
                'title': 'Key Insight',
                'formula': r'\frac{a_0}{\lambda_C} = \frac{1}{\alpha} \approx 137',
                'description': '''**Bohr radius** a₀ = 4πε₀ℏ²/(m_e×e²) ≈ 52.9 pm determines atomic size (depends on ℏ: a₀ ∝ ℏ²). **Quantization:** Electrons can only have certain energies - a purely quantum effect (Pauli + Heisenberg). **Connection to white dwarfs:** In white dwarfs, electron degeneracy pressure (Pauli) fights gravity. The Chandrasekhar limit connects ℏ, G, c, and m_p!''',
            },
        ],
        'de': [
            {  # bohr_radius_scaling.png
                'title': 'Bohr-Radius-Skalierung',
                'formula': r'a_0 = \frac{4\pi\epsilon_0 \hbar^2}{m_e e^2} \propto \hbar^2',
                'description': '''Wenn ℏ um 10× sinkt → Atome schrumpfen um 100×!''',
            },
            {  # atom_size_comparison.png
                'title': 'Atomgröße',
                'formula': r'a_0 \approx 52,9 \text{ pm (Standard)}',
                'description': '''Kleineres ℏ → kleinere Atome → höhere Dichte → Gravitation wird wichtiger!''',
            },
            {  # energy_levels.png
                'title': 'Wasserstoff-Energieniveaus',
                'formula': r'E_n = -\frac{13,6 \text{ eV}}{n^2}, \quad r_n = n^2 \times a_0',
                'description': '''**Quantisierung:** Nur bestimmte Energien erlaubt (E_n ∝ 1/n², r_n ∝ n²). **Grundzustand (n=1):** E₁ = -13,6 eV, r₁ = 52,9 pm. Übergänge zwischen Niveaus erzeugen Spektrallinien (Lyman-, Balmer-Serie).''',
            },
            {  # quantum_gravity_connection.png
                'title': 'Quanten-Gravitations-Verbindung',
                'formula': r'\alpha_G = \frac{G m_p^2}{\hbar c} \approx 5,9 \times 10^{-39}',
                'description': '''**Bei ℏ × 0,1:** Atome 100× kleiner (a₀ ∝ ℏ²), Gravitation 10× stärker (α_G ∝ 1/ℏ), Dichte 10⁶× höher (ρ ∝ 1/ℏ⁶), Gravitations-Bedeutung 10⁷× größer! **Kernaussage:** In einem Universum mit kleinerem ℏ würde Gravitation bei viel kleineren Massen wichtig werden.''',
            },
            {  # atomic_summary.png
                'title': 'Schlüsselerkenntnis',
                'formula': r'\frac{a_0}{\lambda_C} = \frac{1}{\alpha} \approx 137',
                'description': '''**Bohr-Radius** a₀ = 4πε₀ℏ²/(m_e×e²) ≈ 52,9 pm bestimmt Atomgröße (abhängig von ℏ: a₀ ∝ ℏ²). **Quantisierung:** Elektronen können nur bestimmte Energien haben - ein rein quantenmechanischer Effekt (Pauli + Heisenberg). **Verbindung zu Weißen Zwergen:** In Weißen Zwergen kämpft Elektronen-Entartungsdruck (Pauli) gegen Gravitation. Die Chandrasekhar-Grenze verbindet ℏ, G, c und m_p!''',
            },
        ],
    },
    'thermal': {
        'en': [
            {  # temperature_atmosphere.png
                'title': 'Atmospheric Scale Height',
                'formula': r'H = \frac{k_B T}{\mu g}',
                'description': '''Scale height is the altitude over which pressure drops by factor e. **At 10× g:** Atmosphere 10× thinner! Standard Earth: H ≈ 8.5 km. Higher gravity compresses atmosphere dramatically.''',
            },
            {  # temperature_degeneracy.png
                'title': 'Fermi Temperature',
                'formula': r'T_F = \frac{E_F}{k_B} = \frac{\hbar^2}{2m_e k_B}(3\pi^2 n_e)^{2/3}',
                'description': '''When T << T_F: **degeneracy pressure dominates** (quantum effects rule). In white dwarfs: T ~ 10⁷ K but T_F ~ 10⁹ K → electrons are degenerate → Pauli provides stability!''',
            },
            {  # temperature_summary.png
                'title': 'Temperature Physics Summary',
                'formula': r'\frac{dT}{dz} = -\frac{g}{c_p} \quad \text{(Adiabatic Lapse Rate)}',
                'description': '''**Lapse rate:** How temperature changes with altitude. Standard: -9.8 K/km. **At 10× g:** -98 K/km (atmosphere cools 10× faster with height). **Virial theorem:** Gravitational compression heats interior: T_core ∝ g^(1/3).''',
            },
        ],
        'de': [
            {  # temperature_atmosphere.png
                'title': 'Atmosphärische Skalenhöhe',
                'formula': r'H = \frac{k_B T}{\mu g}',
                'description': '''Skalenhöhe ist die Höhe, über die der Druck um Faktor e sinkt. **Bei 10× g:** Atmosphäre 10× dünner! Standard-Erde: H ≈ 8,5 km. Höhere Gravitation komprimiert die Atmosphäre dramatisch.''',
            },
            {  # temperature_degeneracy.png
                'title': 'Fermi-Temperatur',
                'formula': r'T_F = \frac{E_F}{k_B} = \frac{\hbar^2}{2m_e k_B}(3\pi^2 n_e)^{2/3}',
                'description': '''Wenn T << T_F: **Entartungsdruck dominiert** (Quanteneffekte herrschen). In Weißen Zwergen: T ~ 10⁷ K aber T_F ~ 10⁹ K → Elektronen sind entartet → Pauli sorgt für Stabilität!''',
            },
            {  # temperature_summary.png
                'title': 'Temperaturphysik-Zusammenfassung',
                'formula': r'\frac{dT}{dz} = -\frac{g}{c_p} \quad \text{(Adiabatische Abkühlung)}',
                'description': '''**Abkühlungsrate:** Wie sich Temperatur mit Höhe ändert. Standard: -9,8 K/km. **Bei 10× g:** -98 K/km (Atmosphäre kühlt 10× schneller mit Höhe). **Virialsatz:** Gravitationskompression heizt das Innere: T_Kern ∝ g^(1/3).''',
            },
        ],
    },
}

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("## ⭐ Jugend forscht 2026")
    st.markdown("---")

    # Language selector
    lang = st.radio(
        "🌐 Language / Sprache",
        options=['de', 'en'],
        format_func=lambda x: "🇩🇪 Deutsch" if x == 'de' else "🇬🇧 English",
        horizontal=True
    )
    L = LANG[lang]

    st.markdown("---")

    # Navigation (removed formulas page - now integrated)
    st.markdown("### 📚 Navigation")
    page = st.radio(
        "Select section:",
        options=['intro', 'forces', 'whitedwarf', 'spacetime', 'atoms', 'thermal', 'interactive'],
        format_func=lambda x: L[f'nav_{x}'],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.caption(L['author'])

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def display_image_grid(category: str, lang_code: str):
    """Display images in a grid with descriptions and formulas side by side."""
    images = IMAGES.get(category, [])
    formulas = FORMULAS.get(category, {}).get(lang_code, [])

    for idx, (img_file, title_en, title_de, desc_en, desc_de) in enumerate(images):
        img_path = VIS_DIR / img_file
        title = title_de if lang_code == 'de' else title_en
        desc = desc_de if lang_code == 'de' else desc_en

        # Create two columns: image (larger) and details (smaller)
        col_img, col_details = st.columns([3, 2])

        with col_img:
            if img_path.exists():
                st.image(str(img_path), caption=title, use_container_width=True)
            else:
                st.warning(f"Image not found: {img_file}")

        with col_details:
            # Details section at top of column
            st.markdown(f"### ℹ️ {'Details' if lang_code == 'en' else 'Details'}")
            st.markdown(f"**{title}**")
            st.markdown(desc)
            # Show relevant formula if available
            if idx < len(formulas):
                f = formulas[idx]
                st.markdown("---")
                st.markdown(f"**{f['title']}**")
                st.latex(f['formula'])
                st.markdown(f['description'])

        st.markdown("---")

# =============================================================================
# HEADER
# =============================================================================

st.markdown(f"# ⭐ {L['title']}")
st.markdown(f"### {L['subtitle']}")
st.markdown("---")

# =============================================================================
# PAGE: INTRODUCTION
# =============================================================================

if page == 'intro':
    st.header(f"🔬 {L['intro_title']}")
    st.markdown(L['intro_question'])

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"💡 {L['hypothesis_title']}")
        st.info(L['hypothesis_text'])

    with col2:
        st.subheader(f"⚡ {L['key_insight']}")
        st.success(L['key_insight_text'])

    st.markdown("<br>", unsafe_allow_html=True)

    st.subheader(f"❤️ {L['motivation_title']}")
    st.markdown(L['motivation_text'])

    st.markdown("<br>", unsafe_allow_html=True)

    # Key Numbers
    st.subheader("📊 " + ("Key Numbers" if lang == 'en' else "Wichtige Zahlen"))

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(label="α_G", value="5.9 × 10⁻³⁹")
        st.caption("Gravitational coupling" if lang == 'en' else "Gravitationskopplung")

    with col2:
        st.metric(label="F_em / F_grav", value="~10³⁶")
        st.caption("Force ratio" if lang == 'en' else "Kräfteverhältnis")

    with col3:
        st.metric(label="M_Ch", value="1.44 M☉")
        st.caption("Chandrasekhar limit" if lang == 'en' else "Chandrasekhar-Grenze")

    with col4:
        st.metric(label="a₀", value="52.9 pm")
        st.caption("Bohr radius" if lang == 'en' else "Bohr-Radius")

# =============================================================================
# PAGE: FORCE COMPARISON
# =============================================================================

elif page == 'forces':
    st.header(f"⚡ {L['force_section']}")
    display_image_grid('forces', lang)

# =============================================================================
# PAGE: WHITE DWARF
# =============================================================================

elif page == 'whitedwarf':
    st.header(f"⭐ {L['wd_section']}")
    display_image_grid('whitedwarf', lang)

# =============================================================================
# PAGE: SPACETIME
# =============================================================================

elif page == 'spacetime':
    st.header(f"🌌 {L['spacetime_section']}")
    display_image_grid('spacetime', lang)

# =============================================================================
# PAGE: ATOMIC SCALE
# =============================================================================

elif page == 'atoms':
    st.header(f"⚛️ {L['atomic_section']}")
    display_image_grid('atoms', lang)

# =============================================================================
# PAGE: TEMPERATURE PHYSICS
# =============================================================================

elif page == 'thermal':
    st.header(f"🌡️ {L['thermal_section']}")
    display_image_grid('thermal', lang)

# =============================================================================
# PAGE: INTERACTIVE 3D
# =============================================================================

elif page == 'interactive':
    st.header(f"🎮 {L['interactive_section']}")

    # Create tabs for each 3D visualization
    tab_titles = [title_de if lang == 'de' else title_en for _, title_en, title_de in INTERACTIVE]
    tabs = st.tabs(tab_titles)

    for idx, tab in enumerate(tabs):
        html_file, title_en, title_de = INTERACTIVE[idx]
        html_path = VIS_DIR / html_file
        title = title_de if lang == 'de' else title_en

        with tab:
            if html_path.exists():
                # Read and embed HTML at full width and large height
                with open(html_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()

                # Display at full viewport height
                st.components.v1.html(html_content, height=1100, scrolling=True)
            else:
                st.warning(f"File not found: {html_file}")
                st.markdown(f"Run `python main.py --interactive` to generate this file.")

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown(f"""
<div style='text-align: center; padding: 1rem;'>
    <p><strong>Jugend forscht 2026</strong> | {L['author']}</p>
    <p style='font-size: 0.8rem;'>Interactive Physics Visualization Dashboard</p>
    <p style='font-size: 0.7rem;'>21 PNG visualizations + 5 interactive 3D plots</p>
</div>
""", unsafe_allow_html=True)
