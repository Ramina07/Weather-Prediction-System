"""Plotly figure builders with one consistent style."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .data import AQ_CATEGORIES, MODEL_COLORS, MODEL_SHORT, WHO_24H, aq_category

FONT = "Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif"
GRID = "#E2E8F0"


def style(fig: go.Figure, height: int = 380, legend: bool = True, title: str | None = None) -> go.Figure:
    fig.update_layout(
        height=height, template="plotly_white", font=dict(family=FONT, size=13, color="#0F172A"),
        margin=dict(l=10, r=10, t=50 if title else 20, b=10), paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", hoverlabel=dict(font_family=FONT, bgcolor="white"),
        showlegend=legend, title=dict(text=title, x=0, font=dict(size=16)) if title else None,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    return fig


def who_line(fig: go.Figure, label: bool = True) -> None:
    fig.add_hline(y=WHO_24H, line=dict(color="#DC2626", width=1.2, dash="dot"),
                  annotation_text="WHO 24-h guideline 15 µg/m³" if label else None,
                  annotation_position="top left", annotation_font=dict(size=11, color="#DC2626"))


def gauge(value: float, title: str = "", max_value: float = 60.0, height: int = 260) -> go.Figure:
    cat = aq_category(value)
    steps = [dict(range=[lo, min(hi, max_value)], color=c + "40") for lo, hi, _, c, _ in AQ_CATEGORIES
             if lo < max_value]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value, number=dict(suffix=" µg/m³", font=dict(size=34, family=FONT)),
        title=dict(text=title, font=dict(size=14)),
        gauge=dict(axis=dict(range=[0, max(max_value, value * 1.1)], tickwidth=1, tickcolor="#94A3B8"),
                   bar=dict(color=cat["color"], thickness=0.28), steps=steps, borderwidth=0,
                   threshold=dict(line=dict(color="#DC2626", width=3), thickness=0.8, value=WHO_24H))))
    fig.update_layout(height=height, margin=dict(l=25, r=25, t=40, b=5), paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(family=FONT))
    return fig


def bar_compare(labels: list[str], values: list[float], colors: list[str], height: int = 300,
                title: str | None = None, text_suffix: str = "") -> go.Figure:
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker_color=colors,
                           text=[f"{v:.2f}{text_suffix}" for v in values], textposition="outside",
                           hovertemplate="%{y}: %{x:.2f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(range=[0, max(values) * 1.5 if len(values) else 1])
    return style(fig, height, legend=False, title=title)


def location_map(lat: float, lon: float, value: float | None = None, height: int = 300) -> go.Figure:
    color = aq_category(value)["color"] if value is not None else "#0F766E"
    text = f"Forecast point<br>{lat:.4f} N, {lon:.4f} E" + (f"<br>{value:.1f} µg/m³" if value is not None else "")
    try:
        fig = go.Figure(go.Scattermap(lat=[lat], lon=[lon], mode="markers", text=[text], hoverinfo="text",
                                      marker=dict(size=26, color=color, opacity=0.85)))
        fig.update_layout(map=dict(style="open-street-map", center=dict(lat=lat, lon=lon), zoom=10.3))
    except (AttributeError, ValueError):  # older plotly
        fig = go.Figure(go.Scattermapbox(lat=[lat], lon=[lon], mode="markers", text=[text], hoverinfo="text",
                                         marker=dict(size=26, color=color, opacity=0.85)))
        fig.update_layout(mapbox=dict(style="open-street-map", center=dict(lat=lat, lon=lon), zoom=10.3))
    fig.update_layout(height=height, margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
    return fig


def timeseries(df: pd.DataFrame, cols: Iterable[str], colors: dict[str, str] | None = None,
               names: dict[str, str] | None = None, height: int = 400, title: str | None = None,
               rangeslider: bool = False, who: bool = False) -> go.Figure:
    fig = go.Figure()
    for c in cols:
        if c not in df.columns:
            continue
        is_actual = c == "actual"
        fig.add_trace(go.Scatter(
            x=df.index, y=df[c], name=(names or {}).get(c, MODEL_SHORT.get(c, c)), mode="lines",
            line=dict(width=2.6 if is_actual else 1.6, color=(colors or MODEL_COLORS).get(c),
                      dash=None if is_actual or c != "persistence" else "dot"),
            hovertemplate="%{x|%d %b %Y}: %{y:.2f} µg/m³<extra>%{fullData.name}</extra>"))
    if who:
        who_line(fig)
    fig.update_yaxes(title_text="PM2.5 (µg/m³)")
    if rangeslider:
        fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.06))
    return style(fig, height, title=title)


def scatter_actual_pred(actual: pd.Series, pred: pd.Series, color: str, name: str, height: int = 380) -> go.Figure:
    m = float(np.nanmax([actual.max(), pred.max()])) * 1.05
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, m], y=[0, m], mode="lines", line=dict(color="#94A3B8", dash="dash"),
                             name="Perfect forecast", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=actual, y=pred, mode="markers", name=name,
                             marker=dict(color=color, size=7, opacity=0.6, line=dict(width=0.5, color="white")),
                             text=[d.strftime("%d %b %Y") for d in actual.index],
                             hovertemplate="%{text}<br>actual %{x:.2f} · predicted %{y:.2f}<extra></extra>"))
    fig.update_xaxes(title_text="Actual daily-mean PM2.5 (µg/m³)", range=[0, m])
    fig.update_yaxes(title_text="Predicted (µg/m³)", range=[0, m])
    return style(fig, height)


def heatmap(z: pd.DataFrame, title: str | None = None, height: int = 380, colorscale: str = "Teal",
            fmt: str = ".2f", reverse: bool = False, zmid: float | None = None) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=z.values, x=list(z.columns), y=list(z.index), colorscale=colorscale, reversescale=reverse, zmid=zmid,
        text=np.vectorize(lambda v: "" if pd.isna(v) else format(v, fmt), otypes=[object])(z.values), texttemplate="%{text}",
        hovertemplate="%{y} · %{x}: %{z:.3f}<extra></extra>", colorbar=dict(thickness=12)))
    fig.update_yaxes(autorange="reversed")
    return style(fig, height, legend=False, title=title)


def importance_bars(df: pd.DataFrame, value_col: str, label_col: str, color_col: str | None = None,
                    color_map: dict[str, str] | None = None, error_col: str | None = None,
                    height: int = 480, title: str | None = None) -> go.Figure:
    d = df.sort_values(value_col)
    fig = px.bar(d, x=value_col, y=label_col, orientation="h", color=color_col, color_discrete_map=color_map,
                 error_x=error_col)
    fig.update_layout(yaxis_title=None, legend_title_text=None)
    return style(fig, height, legend=color_col is not None, title=title)


def violin_errors(df: pd.DataFrame, models: list[str], height: int = 380) -> go.Figure:
    fig = go.Figure()
    for m in models:
        if m not in df.columns:
            continue
        err = df[m] - df["actual"]
        fig.add_trace(go.Box(y=err, name=MODEL_SHORT.get(m, m), marker_color=MODEL_COLORS.get(m),
                             boxmean=True, boxpoints=False))
    fig.add_hline(y=0, line=dict(color="#0F172A", width=1))
    fig.update_yaxes(title_text="Forecast error = predicted − actual (µg/m³)")
    return style(fig, height, legend=False)
