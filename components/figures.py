# components/figures.py
import plotly.graph_objects as go

def apply_white(fig: go.Figure, title: str | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        title=title,
        margin=dict(l=10, r=10, t=48, b=10),
    )
    return fig

def empty_fig(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white",
        title=title,
        xaxis={"visible": False},
        yaxis={"visible": False},
        annotations=[dict(text="No data", x=0.5, y=0.5, showarrow=False)],
        margin=dict(l=10, r=10, t=48, b=10),
    )
    return fig