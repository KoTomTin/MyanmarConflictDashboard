# components/colors.py
"""Single source of truth for color systems used across pages."""

# ── Key events (12 categories) ────────────────────────────────
KEY_EVENT_COLORS = {
    # ── combat (tab10 palette — maximally distinguishable) ──
    "Armed Clash":                  "#1f77b4",  # tab10 blue
    "Shelling/Artillery":           "#ff7f0e",  # tab10 orange
    "IED/Mine":                     "#2ca02c",  # tab10 green
    "Air Strike":                   "#9467bd",  # tab10 purple
    "Drone Strike":                 "#d62728",  # tab10 red
    # ── civilian violence ──
    "Massacres":                    "#7f1d1d",  # very dark red
    "Massacre":                     "#7f1d1d",  # alias (singular)
    "Attack on Civilians":          "#e377c2",  # tab10 pink
    # ── political / other ──
    "Protests":                     "#d97706",  # amber
    "Arrests":                      "#059669",  # emerald
    "Looting/property destruction": "#8c564b",  # tab10 brown
    "Displacement":                 "#17becf",  # tab10 cyan
    "Others":                       "#9ca3af",  # light grey
}

# Ordered list for consistent display (aliases excluded from order)
KEY_EVENT_ORDER = [
    "Armed Clash",
    "Shelling/Artillery",
    "IED/Mine",
    "Air Strike",
    "Drone Strike",
    "Massacres",
    "Attack on Civilians",
    "Protests",
    "Arrests",
    "Looting/property destruction",
    "Displacement",
    "Others",
]

# ── Actor type colours ────────────────────────────────────────
ACTOR_TYPE_COLORS = {
    "Myanmar Military Regime": "#dc2626",
    "ERO":                     "#1d4ed8",
    "People's Defense Force":  "#059669",
    "Pyu Saw Htee":            "#d97706",
}

# ── Alert / status colors ─────────────────────────────────────
# Used in both Plotly map fills (Python) and CSS chip classes (vars in :root).
# Keep these in sync with --alert-* CSS variables in assets/style.css.
ALERT_CATEGORY_COLORS = {
    "No unusual change": "#d7dde5",
    "Activity surge":    "#2b6cb0",
    "Fatality surge":    "#dc2626",
    "Both surges":       "#7c3aed",
}

# ── Sequential scales ─────────────────────────────────────────
# Zero stop is a slightly bluer grey (#dde3ea) so empty townships read as
# "measured zero" rather than "missing" against the cream page background.
# Zero stop uses the warm page-background cream (#f0ece3) so townships with
# zero activity visually dissolve into the page — clearly distinct from the
# lightest blue (which still means "a small number was recorded").
SEQUENTIAL_BLUES_ZERO_GREY = [
    (0.00, "#f0ece3"),
    (0.01, "#deebf7"),
    (0.20, "#c6dbef"),
    (0.40, "#9ecae1"),
    (0.60, "#6baed6"),
    (0.80, "#3182bd"),
    (1.00, "#08519c"),
]

SEQUENTIAL_REDS_ZERO_GREY = [
    (0.00, "#f0ece3"),
    (0.01, "#fee0d2"),
    (0.20, "#fcbba1"),
    (0.40, "#fc9272"),
    (0.60, "#fb6a4a"),
    (0.80, "#de2d26"),
    (1.00, "#a50f15"),
]
