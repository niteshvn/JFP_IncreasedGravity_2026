# Graph Standards Document
## Jugend forscht 2026 - Physics Visualization Project

This document outlines the standards and guidelines for creating visualizations in this project.

---

## 1. File Naming Conventions

### Static Images (PNG)
```
{visualization_name}.png          # English version
{visualization_name}_de.png       # German version
```

### Animations (GIF)
```
{visualization_name}_animation.png      # English version
{visualization_name}_animation_de.png   # German version
```

### Interactive 3D (HTML)
```
{visualization_name}_3d_interactive.html      # English version
{visualization_name}_3d_interactive_de.html   # German version
```

### Examples
```
light_bending.png
light_bending_de.png
light_bending_animation.gif
light_bending_animation_de.gif
light_bending_3d_interactive.html
light_bending_3d_interactive_de.html
```

---

## 2. Output Directories

| Directory | Purpose |
|-----------|---------|
| `visualizations/` | Source generated files |
| `public/visualizations/` | Netlify deployment (copy here after generation) |

Always generate to `visualizations/` first, then copy to `public/visualizations/` for deployment.

---

## 3. Language Support

### Required
- **Every visualization MUST have both English (EN) and German (DE) versions**
- Use `language` parameter in all plot functions: `language: str = 'en'`

### Implementation Pattern
```python
def plot_example(constants, language='en', save=True, show=False):
    # Titles
    title = 'English Title' if language == 'en' else 'German Title'

    # Axis labels
    xlabel = 'Distance (m)' if language == 'en' else 'Abstand (m)'

    # Legend labels
    label = 'Data Series' if language == 'en' else 'Datenreihe'

    # Save with correct suffix
    suffix = '_de' if language == 'de' else ''
    filename = f'example{suffix}.png'
```

### Common Translations
| English | German |
|---------|--------|
| Distance | Abstand |
| Time | Zeit |
| Mass | Masse |
| Radius | Radius |
| Velocity | Geschwindigkeit |
| Pressure | Druck |
| Temperature | Temperatur |
| Density | Dichte |
| Force | Kraft |
| Energy | Energie |
| Light | Licht |
| Gravity | Gravitation |
| Spacetime | Raumzeit |
| Black Hole | Schwarzes Loch |
| Event Horizon | Ereignishorizont |
| Standard Universe | Standarduniversum |
| Alternative Universe | Alternatives Universum |

---

## 4. Legend and Colorbar Placement

### Rule: Always place at the BOTTOM of the graph

### Matplotlib (Static)
```python
# Legend - bottom center, horizontal
ax.legend(
    loc='upper center',
    bbox_to_anchor=(0.5, -0.12),  # Below the plot
    ncol=3,                        # Horizontal layout
    framealpha=0.9
)

# Colorbar - bottom, horizontal
cbar = fig.colorbar(mappable, ax=ax, orientation='horizontal',
                    fraction=0.046, pad=0.15)
```

### Plotly (Interactive 3D)
```python
# Legend - bottom center, horizontal
fig.update_layout(
    legend=dict(
        x=0.5,
        y=0.12,              # Near bottom
        xanchor='center',
        yanchor='top',
        orientation='h',     # Horizontal
        bgcolor='rgba(255,255,255,0.95)',
        bordercolor='rgba(180,180,180,0.8)',
        borderwidth=1
    )
)

# Colorbar - bottom, horizontal
colorbar=dict(
    orientation='h',
    x=0.5,
    y=0.02,
    xanchor='center',
    yanchor='top',
    len=0.5,
    thickness=18
)
```

### Adjust spacing for legend count
```python
# More items = more vertical space needed
if len(legend_items) > 4:
    plt.subplots_adjust(bottom=0.25)
else:
    plt.subplots_adjust(bottom=0.18)
```

---

## 5. Color Scheme

### Use project color scheme from `modules/color_scheme.py`

```python
from .color_scheme import COLORS, get_stellar_colors

# Available colors
COLORS = {
    'primary_blue': '#1E40AF',
    'secondary_blue': '#3B82F6',
    'accent_purple': '#7C3AED',
    'highlight': '#F59E0B',      # Amber - for emphasis
    'standard': '#16A34A',        # Green - standard universe
    'alternative': '#DC2626',     # Red - alternative universe
    'relativistic': '#DC2626',    # Red - horizons, singularities
    'quantum': '#8B5CF6',         # Purple - quantum effects
    'thermal': '#F97316',         # Orange - temperature
    'neutral': '#6B7280',         # Gray - reference lines
}
```

### Consistent Usage
| Element | Color |
|---------|-------|
| Standard universe data | `COLORS['standard']` (green) |
| Alternative universe data | `COLORS['alternative']` (red) |
| Event horizons, singularities | `COLORS['relativistic']` (red) |
| Light rays | `COLORS['primary_blue']` (blue) |
| Observers, highlights | `COLORS['highlight']` (amber) |
| Reference lines | `COLORS['neutral']` (gray) |
| Quantum effects | `COLORS['quantum']` (purple) |

---

## 6. Text and Annotations

### DO NOT include on graphs:
- Detailed explanations or descriptions
- Mathematical formulas
- Notes or footnotes
- Physics interpretations

### CAN include on graphs:
- Axis labels with units
- Concise legend labels
- Region labels (e.g., "Region I", "Event Horizon")
- Data point annotations (sparingly)

### Where detailed content goes:
All descriptions, formulas, and explanations go in the **web UI** (`public/index.html`):
- Description in `<p class="description">` element
- Formulas in `<div class="formula">` with KaTeX rendering

---

## 7. Figure Size and Layout

### Standard sizes
```python
# Single plot
fig, ax = plt.subplots(figsize=(10, 8))

# Two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Summary with multiple plots
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
```

### Margins and spacing
```python
plt.tight_layout()
plt.subplots_adjust(bottom=0.18)  # Space for legend
```

---

## 8. Animation Standards (GIF)

### Settings
```python
from matplotlib.animation import FuncAnimation, PillowWriter

# Animation parameters
frames = 120          # Total frames
interval = 50         # ms between frames (20 fps)
fps = 20              # For PillowWriter

# Create animation
anim = FuncAnimation(fig, update_func, frames=frames, interval=interval, blit=True)

# Save
writer = PillowWriter(fps=fps)
anim.save(filepath, writer=writer)
```

### Best practices
- Keep file size reasonable (< 2MB if possible)
- Use `blit=True` for performance
- Clear previous frame data to avoid artifacts
- Include a brief pause at start/end states

---

## 9. Interactive 3D Standards (Plotly)

### Layout settings
```python
fig.update_layout(
    title=dict(text=title, x=0.5),
    height=900,
    margin=dict(l=0, r=0, t=50, b=10),
    template='plotly_white',
    scene=dict(
        xaxis_title='X Label',
        yaxis_title='Y Label',
        zaxis_title='Z Label',
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
        aspectmode='data',
        domain=dict(x=[0, 1], y=[0.22, 1])  # Leave space for legend
    )
)
```

### Save configuration
```python
fig.write_html(filepath, config={
    'displaylogo': False,
    'displayModeBar': True
})
```

---

## 10. Function Signature Pattern

### Standard pattern for plot functions
```python
def plot_visualization_name(
    constants: Optional[PhysicalConstants] = None,
    language: str = 'en',
    save: bool = True,
    show: bool = False,
    # Additional parameters as needed
) -> plt.Figure:
    """
    Brief English description.
    Kurze deutsche Beschreibung.

    Args:
        constants: Physical constants (uses default if None)
        language: 'en' for English, 'de' for German
        save: Whether to save the figure
        show: Whether to display the figure

    Returns:
        Matplotlib Figure object
    """
    if constants is None:
        constants = get_constants()

    # ... implementation ...

    if save:
        suffix = '_de' if language == 'de' else ''
        filepath = os.path.join(VIS_DIR, f'visualization_name{suffix}.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Saved: {filepath}")

    if show:
        plt.show()

    return fig
```

---

## 11. Integration Checklist

When adding a new visualization:

- [ ] Create function in appropriate module (`spacetime_curvature.py`, `force_comparison.py`, etc.)
- [ ] Generate both EN and DE versions
- [ ] Export function in `modules/__init__.py`
- [ ] Add to `generate_all_*_plots()` function
- [ ] Copy files to `public/visualizations/`
- [ ] Add to `public/index.html`:
  - [ ] Add `<img>` with `data-img-base` attribute
  - [ ] Add description (EN and DE with `data-en`/`data-de`)
  - [ ] Add KaTeX formulas
- [ ] Test language switching works
- [ ] Verify legends/colorbars are at bottom

---

## 12. Quality Checklist

Before committing a new visualization:

- [ ] No text boxes or annotations cluttering the graph
- [ ] Legend at bottom, horizontal layout
- [ ] Colorbar at bottom (if applicable)
- [ ] Both EN and DE versions generated
- [ ] Correct file naming convention
- [ ] Uses project color scheme
- [ ] Reasonable file size
- [ ] Renders correctly in web UI
- [ ] Language toggle works in web UI

---

## 13. Example: Complete Implementation

See `modules/spacetime_curvature.py`:
- `plot_light_bending()` - static visualization with animation
- `plot_penrose_carter_diagram()` - conformal diagram

See `modules/interactive_3d.py`:
- `plot_light_bending_3d_interactive()` - 3D interactive

See `public/index.html`:
- Spacetime section with sub-tabs for organization
- KaTeX formula rendering
- Language switching implementation

---

*Last updated: January 2025*
