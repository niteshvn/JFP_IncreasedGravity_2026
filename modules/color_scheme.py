"""
Unified Color Scheme for Jugend forscht 2026 Physics Visualization Project
Einheitliches Farbschema fuer Jugend forscht 2026 Physik-Visualisierungsprojekt

This module defines a consistent, professional color palette used across all
visualizations for a cohesive look suitable for scientific presentation.

Author: Jugend forscht 2026 Project
"""

# =============================================================================
# PROFESSIONAL SCIENTIFIC COLOR PALETTE
# =============================================================================
# Muted, colorblind-friendly colors suitable for scientific visualization

COLORS = {
    # Primary colors for main data
    'primary_blue': '#1E40AF',      # Deep blue - gravity, main data
    'primary_teal': '#0F766E',      # Teal - secondary data
    'primary_amber': '#B45309',     # Amber - highlights, warnings

    # Secondary colors for comparisons
    'gravity': '#1E40AF',           # Deep blue for gravitational
    'electromagnetic': '#B45309',   # Amber for electromagnetic/Coulomb
    'quantum': '#7C3AED',           # Purple for quantum effects
    'relativistic': '#DC2626',      # Red for relativistic
    'non_relativistic': '#2563EB',  # Blue for non-relativistic

    # Stellar objects
    'earth': '#059669',             # Green
    'sun': '#D97706',               # Orange/gold
    'white_dwarf': '#3B82F6',       # Blue
    'neutron_star': '#7C3AED',      # Purple
    'black_hole': '#1F2937',        # Dark gray/black

    # UI and annotations
    'standard': '#059669',          # Green for standard/reference
    'scaled': '#DC2626',            # Red for scaled/modified
    'highlight': '#F59E0B',         # Amber for highlights
    'muted': '#6B7280',             # Gray for secondary info

    # Temperature scale colors
    'temp_cold': '#1E40AF',         # Deep blue for cold
    'temp_cool': '#3B82F6',         # Light blue for cool
    'temp_mild': '#10B981',         # Green for mild
    'temp_warm': '#F59E0B',         # Amber for warm
    'temp_hot': '#DC2626',          # Red for hot
    'temp_extreme': '#7C2D12',      # Dark red for extreme

    # Planets and solar system
    'moon': '#9CA3AF',              # Gray
    'mercury': '#6B7280',           # Dark gray
    'venus': '#10B981',             # Green
    'mars': '#DC2626',              # Red
    'jupiter': '#D97706',           # Orange
    'saturn': '#F59E0B',            # Gold
    'uranus': '#06B6D4',            # Cyan
    'neptune': '#3B82F6',           # Blue

    # Module categories
    'cosmic': '#4F46E5',            # Indigo for cosmic effects
    'equilibrium': '#059669',       # Green for equilibrium
    'collapse': '#991B1B',          # Dark red for collapse

    # Backgrounds and boxes
    'box_info': '#DBEAFE',          # Light blue
    'box_warning': '#FEF3C7',       # Light amber
    'box_success': '#D1FAE5',       # Light green
    'box_error': '#FEE2E2',         # Light red

    # Text colors
    'text_dark': '#1F2937',         # Dark gray
    'text_light': '#F9FAFB',        # Light gray/white
    'text_muted': '#6B7280',        # Muted gray
}

# Ordered color sequences for bar charts and multiple series
COLOR_SEQUENCE = [
    '#1E40AF',  # Deep blue
    '#B45309',  # Amber
    '#059669',  # Green
    '#7C3AED',  # Purple
    '#DC2626',  # Red
    '#0F766E',  # Teal
]

# Stellar objects sequence (for compactness plots etc.)
STELLAR_COLORS = [
    '#059669',  # Earth - green
    '#D97706',  # Sun - orange
    '#3B82F6',  # White dwarf - blue
    '#7C3AED',  # Neutron star - purple
    '#1F2937',  # Black hole - dark
]

# Matplotlib style settings for consistent look
MPL_STYLE = {
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#374151',
    'axes.labelcolor': '#1F2937',
    'axes.titlecolor': '#1F2937',
    'xtick.color': '#374151',
    'ytick.color': '#374151',
    'text.color': '#1F2937',
    'grid.color': '#E5E7EB',
    'grid.alpha': 0.5,
}

def get_color(name: str) -> str:
    """Get a color by name from the palette."""
    return COLORS.get(name, '#6B7280')

def get_stellar_colors() -> list:
    """Get the stellar objects color sequence."""
    return STELLAR_COLORS.copy()

def get_sequence() -> list:
    """Get the general color sequence for multiple data series."""
    return COLOR_SEQUENCE.copy()

# Temperature scale sequence (cold to hot)
TEMPERATURE_COLORS = [
    '#1E40AF',  # Deep blue - very cold
    '#3B82F6',  # Light blue - cold
    '#10B981',  # Green - mild
    '#F59E0B',  # Amber - warm
    '#DC2626',  # Red - hot
    '#7C2D12',  # Dark red - extreme
]

def get_temperature_colors() -> list:
    """Get the temperature color sequence (cold to hot)."""
    return TEMPERATURE_COLORS.copy()

# Planet color sequence (Mercury to Neptune)
PLANET_COLORS = [
    '#6B7280',  # Mercury - dark gray
    '#10B981',  # Venus - green
    '#059669',  # Earth - deeper green
    '#DC2626',  # Mars - red
    '#D97706',  # Jupiter - orange
    '#F59E0B',  # Saturn - gold
    '#06B6D4',  # Uranus - cyan
    '#3B82F6',  # Neptune - blue
]

def get_planet_colors() -> list:
    """Get the planet color sequence (Mercury to Neptune)."""
    return PLANET_COLORS.copy()
