"""Shared look & feel: global CSS, page headers, cards and small HTML components."""

from __future__ import annotations

import html
from typing import Iterable

import streamlit as st

from . import paths
from .data import aq_category

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');
:root { --ink:#0B1220; --muted:#5B6476; --line:#E6E8EE; --soft:#F5F7FA; --accent:#0F766E; --accent2:#14B8A6; }
html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea, select, label {
  font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif; }
h1, h2, h3, h4, .brand, .kpi .value, .fc .big, .ph h1 { font-family: 'Manrope', 'Inter', sans-serif; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1240px; }
h3 { font-size: 1.15rem !important; font-weight: 700 !important; color: var(--ink); letter-spacing: -0.01em; }
h4 { font-size: 1rem !important; font-weight: 700 !important; color: var(--ink); }
[data-testid="stHeader"] { background: transparent; }

/* sidebar */
[data-testid="stSidebar"] { background: #0B1220; border-right: 1px solid #111A2E; }
[data-testid="stSidebar"] * { color: #C9D1E0 !important; }
[data-testid="stSidebarNav"] a { border-radius: 8px; margin: 1px 0; }
[data-testid="stSidebarNav"] a[aria-current="page"] { background: rgba(20,184,166,0.14); }
[data-testid="stSidebarNav"] a[aria-current="page"] span { color: #FFFFFF !important; font-weight: 600; }
[data-testid="stSidebarNav"] a:hover { background: rgba(255,255,255,0.05); }
[data-testid="stSidebarNavSeparator"] { border-color: #1C2740; }
header [data-testid="stSidebarNavItems"] span, [data-testid="stNavSectionHeader"] {
  text-transform: uppercase; letter-spacing: .1em; font-size: .68rem !important; color: #6B7894 !important; }
.brand { font-size: 1.2rem; font-weight: 800; color: #FFFFFF !important; letter-spacing: -0.02em; }
.brand-sub { font-size: .78rem; color: #7D8AA6 !important; margin-top: 2px; }
.sys { font-size: .76rem; line-height: 2; color: #8B97B0 !important; margin-top: 1.2rem;
  border-top: 1px solid #1C2740; padding-top: .8rem; }
.dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-right: 8px; vertical-align: middle; }

/* widgets */
[data-testid="stMetric"] { background: #FFF; border: 1px solid var(--line); border-radius: 14px; padding: 14px 18px; }
div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 14px; border-color: var(--line) !important; }
.stTabs [data-baseweb="tab-list"] { gap: 2px; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] { padding: 8px 14px; font-weight: 500; color: var(--muted); }
.stTabs [aria-selected="true"] { color: var(--ink) !important; }
.stButton button[kind="primary"] { border-radius: 10px; font-weight: 600; padding: .5rem 1.2rem; }
.stButton button { border-radius: 10px; }
[data-testid="stCaptionContainer"] { color: #7A8395; }

/* page header */
.ph { margin: 0 0 1.6rem 0; }
.ph .eb { text-transform: uppercase; letter-spacing: .14em; font-size: .7rem; font-weight: 700; color: var(--accent); }
.ph h1 { font-size: 2rem; font-weight: 800; margin: .25rem 0 0 0; color: var(--ink); padding: 0; letter-spacing: -0.03em; }
.ph p { color: var(--muted); font-size: .98rem; margin: .4rem 0 0 0; max-width: 720px; line-height: 1.5; }

/* cards */
.card { background: #FFF; border: 1px solid var(--line); border-radius: 14px; padding: 18px; height: 100%; }
.card .n { font-family: 'Manrope', sans-serif; font-size: .78rem; font-weight: 800; color: var(--accent);
  letter-spacing: .08em; }
.card h4 { margin: .45rem 0 .3rem 0; font-size: .98rem; color: var(--ink); font-weight: 700; }
.card p { margin: 0; color: var(--muted); font-size: .86rem; line-height: 1.5; }
.kpi { background: #FFF; border: 1px solid var(--line); border-radius: 14px; padding: 16px 18px; height: 100%; }
.kpi .label { color: var(--muted); font-size: .74rem; font-weight: 600; text-transform: uppercase; letter-spacing: .08em; }
.kpi .value { font-size: 1.7rem; font-weight: 800; color: var(--ink); line-height: 1.15; margin-top: 6px;
  letter-spacing: -0.02em; }
.kpi .sub { color: #8A93A5; font-size: .8rem; margin-top: 3px; }
.kpi.accent { border-color: transparent; background: linear-gradient(180deg, #F0FDFA, #FFFFFF);
  box-shadow: inset 0 0 0 1px #CCF1EB; }
.kpi .tag { display:inline-block; font-size:.68rem; font-weight:700; color: var(--accent); background:#E6F7F4;
  padding: 2px 8px; border-radius: 999px; margin-left: 6px; vertical-align: middle; }

/* forecast card */
.fc { border-radius: 18px; padding: 24px 26px; background: #FFF; border: 1px solid var(--line); position: relative;
  overflow: hidden; }
.fc:before { content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 5px; background: var(--c); }
.fc .t { color: var(--muted); font-size: .8rem; font-weight: 600; text-transform: uppercase; letter-spacing: .08em; }
.fc .d { font-size: 1.02rem; font-weight: 600; color: var(--ink); margin: 4px 0 14px 0; }
.fc .big { font-size: 3.9rem; font-weight: 800; line-height: 1; letter-spacing: -0.04em; color: var(--ink); }
.fc .unit { font-size: 1rem; font-weight: 600; color: var(--muted); margin-left: 8px; }
.fc .cat { display: inline-flex; align-items: center; gap: 8px; margin-top: 14px; font-weight: 700; font-size: .92rem;
  color: var(--ink); }
.fc .cat i { width: 10px; height: 10px; border-radius: 50%; background: var(--c); display: inline-block; }
.fc .meta { margin-top: 10px; font-size: .88rem; color: var(--muted); line-height: 1.55; }

.pill { display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: .74rem; font-weight: 600;
  margin: 0 6px 6px 0; border: 1px solid var(--line); color: #3B4456; background: #FFF; }
.pill.ok { color: #0F766E; border-color: #BFE9E2; background: #F0FDFA; }
.pill.warn { color: #9A3412; border-color: #FED7AA; background: #FFF7ED; }
.pill.info { color: #1E3A8A; border-color: #C7D2FE; background: #EEF2FF; }

.scale { display: flex; border-radius: 6px; overflow: hidden; height: 6px; margin: 14px 0 4px 0; gap: 2px; }
.scale div { flex: 1; }
.scale-labels { display: flex; justify-content: space-between; font-size: .68rem; color: #8A93A5; }
.ins { border-left: 3px solid var(--accent); padding: 8px 14px; color: #273043; font-size: .92rem;
  margin: 8px 0 14px 0; background: #F7FBFA; border-radius: 0 10px 10px 0; line-height: 1.5; }
.ins.partial { border-color: #D97706; background: #FFFBF3; }
.ins.no { border-color: #DC2626; background: #FFF6F6; }
.ins b { color: var(--ink); }
.foot { color: #A0A8B8; font-size: .74rem; margin-top: 2.5rem; padding-top: 1rem; border-top: 1px solid var(--line); }
</style>
"""


def setup_page() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(eyebrow: str, title: str, text: str = "") -> None:
    st.markdown(
        f"<div class='ph'><div class='eb'>{html.escape(eyebrow)}</div><h1>{html.escape(title)}</h1>"
        + (f"<p>{text}</p>" if text else "") + "</div>", unsafe_allow_html=True)


def kpi(label: str, value: str, sub: str = "", accent: str | None = None) -> str:
    style = " accent" if accent else ""
    return (f"<div class='kpi{style}'><div class='label'>{html.escape(label)}</div>"
            f"<div class='value'>{value}</div><div class='sub'>{sub}</div></div>")


def kpi_row(items: Iterable[tuple[str, str, str]], accent: str | None = None) -> None:
    items = list(items)
    cols = st.columns(len(items))
    for c, (label, value, sub) in zip(cols, items):
        c.markdown(kpi(label, value, sub, accent), unsafe_allow_html=True)


def card(num: str, title: str, text: str) -> str:
    return f"<div class='card'><div class='n'>{num}</div><h4>{html.escape(title)}</h4><p>{text}</p></div>"


def pill(text: str, kind: str = "gray") -> str:
    return f"<span class='pill {kind}'>{html.escape(text)}</span>"


def verdict(text: str, kind: str = "yes") -> None:
    cls = {"yes": "", "partial": " partial", "no": " no"}[kind]
    st.markdown(f"<div class='ins{cls}'>{text}</div>", unsafe_allow_html=True)


def aq_scale_html() -> str:
    from .data import AQ_CATEGORIES

    bars = "".join(f"<div style='background:{c[3]}' title='{c[2]}'></div>" for c in AQ_CATEGORIES)
    short = ["Good", "Moderate", "Sensitive", "Unhealthy", "Very unhealthy", "Hazardous"]
    labels = "".join(f"<span>{n}</span>" for n in short)
    return f"<div class='scale'>{bars}</div><div class='scale-labels'>{labels}</div>"


def forecast_card(value: float, date_label: str, subtitle: str, extra: str = "") -> None:
    cat = aq_category(value)
    st.markdown(
        f"""<div class='fc' style='--c:{cat["color"]}'>
        <div class='t'>{html.escape(subtitle)}</div><div class='d'>{html.escape(date_label)}</div>
        <div><span class='big'>{value:.1f}</span><span class='unit'>µg/m³</span></div>
        <div class='cat'><i></i>{html.escape(cat["name"])}</div>
        <div class='meta'>{html.escape(cat["advice"])}{extra}</div></div>""", unsafe_allow_html=True)


def hero_svg() -> str:
    p = paths.ASSETS / "hero_dark.svg"
    return p.read_text(encoding="utf-8") if p.exists() else ""


def fmt(x, nd: int = 2, suffix: str = "") -> str:
    try:
        return f"{float(x):.{nd}f}{suffix}"
    except (TypeError, ValueError):
        return "–"


def footer() -> None:
    st.markdown("<div class='foot'>Data: OpenWeatherMap Air Pollution API, Open-Meteo. Concentrations are "
                "provider-modelled values for 51.17 N, 71.45 E.</div>", unsafe_allow_html=True)


# ------------------------------------------------------------------ version-safe wrappers
def _st_version() -> tuple[int, int]:
    try:
        parts = st.__version__.split(".")
        return int(parts[0]), int(parts[1])
    except Exception:  # noqa: BLE001
        return (1, 40)


_NEW_WIDTH = _st_version() >= (1, 46)


def plot(fig, key: str | None = None, **kw) -> None:
    """st.plotly_chart stretched to the container width on every Streamlit version."""
    if _NEW_WIDTH:
        st.plotly_chart(fig, width="stretch", key=key, config={"displaylogo": False}, **kw)
    else:
        st.plotly_chart(fig, use_container_width=True, key=key, config={"displaylogo": False}, **kw)


def table(df, **kw) -> None:
    """st.dataframe stretched to the container width on every Streamlit version."""
    kw.setdefault("hide_index", True)
    if _NEW_WIDTH:
        st.dataframe(df, width="stretch", **kw)
    else:
        st.dataframe(df, use_container_width=True, **kw)
