# components/colors.py
"""Single source of truth for color systems used across pages."""

# ── Key events (9 categories) ─────────────────────────────────
KEY_EVENT_COLORS = {
    "Ground-based attack":          "#1d4ed8",  # deep blue
    "Air attack":                   "#7c3aed",  # purple
    "Drone attack":                 "#475569",  # slate grey
    "Massacres":                    "#7f1d1d",  # very dark red
    "Violence against civilians":   "#dc2626",  # red
    "Protests":                     "#d97706",  # amber
    "Arrests":                      "#059669",  # emerald
    "Looting/property destruction": "#92400e",  # brown
    "Displacement":                 "#0891b2",  # cyan
    "Others":                       "#9ca3af",  # light grey
}

# Ordered list for consistent display
KEY_EVENT_ORDER = list(KEY_EVENT_COLORS.keys())

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
SEQUENTIAL_BLUES_ZERO_GREY = [
    (0.00, "#dde3ea"),
    (0.01, "#deebf7"),
    (0.20, "#c6dbef"),
    (0.40, "#9ecae1"),
    (0.60, "#6baed6"),
    (0.80, "#3182bd"),
    (1.00, "#08519c"),
]

# Reds with explicit zero → cool grey (actor footprint)
SEQUENTIAL_REDS_ZERO_GREY = [
    (0.00, "#dde3ea"),
    (0.01, "#fee0d2"),
    (0.20, "#fcbba1"),
    (0.40, "#fc9272"),
    (0.60, "#fb6a4a"),
    (0.80, "#de2d26"),
    (1.00, "#a50f15"),
]
