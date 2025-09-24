# components/colors.py
"""
Single source of truth for color systems used across pages.
"""

# --- Categorical palettes ---

# Key events (5)
KEY_EVENT_COLORS = {
    "Armed conflict":      "#1f77b4",  # blue
    "Arrests":             "#ff7f0e",  # orange
    "Displacement":        "#2ca02c",  # green
    "Protests":            "#9467bd",  # purple
    "Violence against civilians": "#d62728",  # red
}

# Anomaly groups (4)
ANOMALY_COLORS = {
    "normal":        "#9aa0a6",
    "event surge":   "#2f5fbd",
    "fatality surge":"#d35e60",
    "both surges":   "#8c6bb1",
}

# Clusters (k = 2..5). Reuse the first N for k.
CLUSTER_COLORS = ["#1f77b4","#ff7f0e","#2ca02c","#9467bd","#d62728"]

# --- Sequential scales ---

# Sequential blues with explicit zero grey (for choropleths / tiles of counts)
# Use as Plotly "colorscale" argument (list of [position, color])
SEQUENTIAL_BLUES_ZERO_GREY = [
    (0.00, "#d9d9d9"),  # zero = light grey (not white)
    (0.01, "#deebf7"),
    (0.20, "#c6dbef"),
    (0.40, "#9ecae1"),
    (0.60, "#6baed6"),
    (0.80, "#3182bd"),
    (1.00, "#08519c"),
]

# For forecast/predicted windows (next 3 months)
SEQUENTIAL_ORANGE = [
    (0.00, "#fff5eb"),
    (0.20, "#fee6ce"),
    (0.40, "#fdd0a2"),
    (0.60, "#fdae6b"),
    (0.80, "#fd8d3c"),
    (1.00, "#e6550d"),
]