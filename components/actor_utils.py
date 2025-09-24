# components/actor_utils.py
from __future__ import annotations
from collections import Counter
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import networkx as nx

# ---- canonical 6 tactics (fixed order on x-axis) ----
TACTICS = [
    "Armed clash",
    "Remote explosive/landmine/IED",
    "Shelling/artillery/missile attack",
    "Air strike",
    "Drone strike",
    "Grenade",
]

# robust mapping to 6 tactics (case-insensitive substring)
TACTIC_RULES = [
    ("armed clash",                         "Armed clash"),
    ("remote explosive",                    "Remote explosive/landmine/IED"),
    ("landmine",                            "Remote explosive/landmine/IED"),
    ("ied",                                 "Remote explosive/landmine/IED"),
    ("shelling",                            "Shelling/artillery/missile attack"),
    ("artillery",                           "Shelling/artillery/missile attack"),
    ("missile",                             "Shelling/artillery/missile attack"),
    ("air strike",                          "Air strike"),
    ("airstrike",                           "Air strike"),
    ("drone",                               "Drone strike"),
    ("grenade",                             "Grenade"),
]

def normalize_tactic(s: str) -> str | None:
    if not isinstance(s, str): return None
    t = s.lower().strip()
    for needle, label in TACTIC_RULES:
        if needle in t:
            return label
    return None

def collapse_pdf(name: str) -> str:
    """Collapse any variant containing “People's Defense Force” to a single label."""
    if isinstance(name, str) and "people's defense force" in name.lower():
        return "People's Defense Force"
    return name or ""

def is_unidentified(name: str) -> bool:
    return isinstance(name, str) and ("unidentified" in name.lower())

# ---- simple min–max + power scaling for node size contrast ----
def power_scale(values, min_size=16, max_size=90, gamma=1.6):
    arr = np.asarray(values, dtype=float)
    vmin, vmax = float(np.min(arr)), float(np.max(arr))
    if vmax <= vmin:
        return np.full_like(arr, (min_size + max_size) / 2.0)
    norm = (arr - vmin) / (vmax - vmin + 1e-12)
    scaled = np.power(norm, gamma)
    return min_size + (max_size - min_size) * scaled

# ---- build network figure (top-10 edges, blue-only theme) ----
def build_network_figure(df_window: pd.DataFrame) -> go.Figure:
    """
    df_window must already be time-filtered and key_event == 'Armed conflict'.
    We:
      - drop blank actors & any actor containing 'Unidentified'
      - collapse PDF variants
      - take top-10 undirected edges by weight
      - size nodes by log-like power scaling of involvement
      - keep a single blue color theme
    """
    tmp = df_window.copy()
    tmp["primary_actor"] = tmp["primary_actor"].apply(collapse_pdf)
    tmp["secondary_actor"] = tmp["secondary_actor"].apply(collapse_pdf)
    tmp = tmp[(tmp["primary_actor"] != "") & (tmp["secondary_actor"] != "")]
    tmp = tmp[~(tmp["primary_actor"].apply(is_unidentified) | tmp["secondary_actor"].apply(is_unidentified))]
    if tmp.empty:
        fig = go.Figure(); fig.update_layout(template="plotly_white", title="Actor interaction network (no data)")
        return fig

    pairs = tmp.apply(lambda r: tuple(sorted((r["primary_actor"], r["secondary_actor"]))), axis=1)
    counts = Counter(pairs)
    top_edges = counts.most_common(10)
    if not top_edges:
        fig = go.Figure(); fig.update_layout(template="plotly_white", title="Actor interaction network (no data)")
        return fig

    nodes = sorted({a for (a, b), _ in top_edges} | {b for (a, b), _ in top_edges})
    involvement = Counter(tmp["primary_actor"]) + Counter(tmp["secondary_actor"])

    G = nx.Graph()
    for n in nodes:
        G.add_node(n, involvement=int(involvement.get(n, 0)))
    for (a, b), w in top_edges:
        G.add_edge(a, b, weight=int(w))

    # Layout tuned for separation
    pos = nx.spring_layout(G, seed=7, k=0.9, weight="weight", iterations=200)

    # Edges (single trace; blue-ish thicker lines)
    edge_x, edge_y = [], []
    mid_x, mid_y, mid_text = [], [], []
    for a, b, data in G.edges(data=True):
        x0, y0 = pos[a]; x1, y1 = pos[b]
        edge_x += [x0, x1, None]; edge_y += [y0, y1, None]
        mid_x.append((x0 + x1) / 2); mid_y.append((y0 + y1) / 2)
        mid_text.append(f"{a} ↔ {b}<br>Interactions: {data['weight']:,}")

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(width=3.0, color="rgba(60, 100, 170, 0.55)"),
        hoverinfo="none", showlegend=False
    )
    mid_trace = go.Scatter(
        x=mid_x, y=mid_y, mode="markers",
        marker=dict(size=1), opacity=0,
        hoverinfo="text", hovertext=mid_text, showlegend=False
    )

    # Node sizes and labels
    sizes = power_scale([G.nodes[n]["involvement"] for n in G.nodes()], min_size=18, max_size=95, gamma=1.6)
    # Label only top N by involvement to avoid clutter
    label_top = min(10, len(G.nodes()))
    top_nodes = set([n for n, _ in sorted(G.nodes(data=True), key=lambda kv: kv[1]["involvement"], reverse=True)[:label_top]])
    node_text = [n if n in top_nodes else "" for n in G.nodes()]
    node_hover = [f"{n}<br>Involvements: {G.nodes[n]['involvement']:,}" for n in G.nodes()]

    node_trace = go.Scatter(
        x=[pos[n][0] for n in G.nodes()],
        y=[pos[n][1] for n in G.nodes()],
        mode="markers+text",
        text=node_text,
        textposition="middle center",
        textfont=dict(color="#0e2a52", size=12),
        marker=dict(
            size=sizes,
            color="#1f77b4",
            line=dict(width=3, color="rgba(15, 35, 70, 0.9)"),
            opacity=0.98
        ),
        hoverinfo="text",
        hovertext=node_hover,
        showlegend=False
    )

    fig = go.Figure([edge_trace, mid_trace, node_trace])
    fig.update_layout(
        template="plotly_white",
        title="Actor interaction network (top 10 connections)",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(l=10, r=10, t=50, b=10),
        height=720,
        hoverlabel=dict(bgcolor="white", font_color="#111", bordercolor="#999"),
    )
    return fig

# ---- build tiles (column% and row%) for tactics × top-10 primary actors ----
def build_tactic_tiles(df_window: pd.DataFrame) -> tuple[go.Figure, go.Figure]:
    work = df_window.copy()
    work["primary_actor"] = work["primary_actor"].apply(collapse_pdf)

    # exclude unidentified from tiles as well (per spec)
    work = work[~work["primary_actor"].apply(is_unidentified)]

    # top-10 primary actors (ignore blank)
    base = work[work["primary_actor"] != ""].copy()
    if base.empty:
        empty = go.Figure(); empty.update_layout(template="plotly_white", title="No data")
        return empty, empty

    top_primary = base["primary_actor"].value_counts().head(10).index.tolist()
    work = work[work["primary_actor"].isin(top_primary)].copy()

    work["tactic6"] = work["detailed_event"].apply(normalize_tactic)
    work = work[work["tactic6"].notna()]
    if work.empty:
        empty = go.Figure(); empty.update_layout(template="plotly_white", title="No tactics in this range")
        return empty, empty

    counts = (work.groupby(["primary_actor","tactic6"]).size()
                    .unstack(fill_value=0)
                    .reindex(columns=TACTICS, fill_value=0))
    counts = counts.loc[counts.sum(axis=1).sort_values(ascending=False).index]

    # Column %
    col_tot = counts.sum(axis=0).replace(0, np.nan)
    col_pct = counts / col_tot * 100.0

    # Build explicit hover text to avoid literal %{...} rendering on some setups
    actors = counts.index.tolist()
    z_col = col_pct.values
    cd_col = counts.values
    text_col = [[
        f"Actor: {actors[i]}<br>Tactic: {TACTICS[j]}<br>Count: {int(cd_col[i, j])}<br>Share within tactic: "
        f"{(0 if np.isnan(z_col[i, j]) else round(float(z_col[i, j]), 1))}%"
        for j in range(len(TACTICS))
    ] for i in range(len(actors))]

    fig_col = go.Figure(go.Heatmap(
        z=z_col, x=TACTICS, y=actors,
        colorscale="Blues", zmin=0, zmax=100,
        colorbar=dict(title="Share within tactic"),
        text=text_col,
        hovertemplate="%{text}<extra></extra>",
        hoverongaps=False,
    ))
    fig_col.update_layout(template="plotly_white",
                          title="Tactics by actor (column-normalized %)",
                          height=340, margin=dict(l=10, r=10, t=50, b=10))

    # Row %
    row_tot = counts.sum(axis=1).replace(0, np.nan)
    row_pct = counts.div(row_tot, axis=0) * 100.0

    z_row = row_pct.values
    cd_row = counts.values
    text_row = [[
        f"Actor: {actors[i]}<br>Tactic: {TACTICS[j]}<br>Count: {int(cd_row[i, j])}<br>Share within actor: "
        f"{(0 if np.isnan(z_row[i, j]) else round(float(z_row[i, j]), 1))}%"
        for j in range(len(TACTICS))
    ] for i in range(len(actors))]

    fig_row = go.Figure(go.Heatmap(
        z=z_row, x=TACTICS, y=actors,
        colorscale="Blues", zmin=0, zmax=100,
        colorbar=dict(title="Share within actor"),
        text=text_row,
        hovertemplate="%{text}<extra></extra>",
        hoverongaps=False,
    ))
    fig_row.update_layout(template="plotly_white",
                          title="Actor profiles (row-normalized %)",
                          height=340, margin=dict(l=10, r=10, t=50, b=10))

    return fig_col, fig_row