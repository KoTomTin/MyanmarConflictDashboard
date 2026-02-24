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

# ── Sequential scales ─────────────────────────────────────────
# Blues with explicit zero → grey (for choropleths of counts)
SEQUENTIAL_BLUES_ZERO_GREY = [
    (0.00, "#d9d9d9"),
    (0.01, "#deebf7"),
    (0.20, "#c6dbef"),
    (0.40, "#9ecae1"),
    (0.60, "#6baed6"),
    (0.80, "#3182bd"),
    (1.00, "#08519c"),
]

# Reds with explicit zero → grey (actor footprint)
SEQUENTIAL_REDS_ZERO_GREY = [
    (0.00, "#d9d9d9"),
    (0.01, "#fee0d2"),
    (0.20, "#fcbba1"),
    (0.40, "#fc9272"),
    (0.60, "#fb6a4a"),
    (0.80, "#de2d26"),
    (1.00, "#a50f15"),
]
