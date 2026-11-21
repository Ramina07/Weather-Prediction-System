from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from lib import charts, data, paths, ui

ui.page_header("Research", "Data",
               "Hourly air pollution from OpenWeatherMap and weather from Open-Meteo, 2021–2025.")

audit = data.metric_json("dataset_audit.json")
h = data.hourly_pollution()
w = data.hourly_weather()
f = data.daily_features()

# ------------------------------------------------------------------ datasets
a, b, c = st.columns(3)
with a:
    st.markdown(ui.card("POLLUTION", "OpenWeatherMap Air Pollution API",
                        f"<b>{audit.get('n_records', 0):,}</b> hourly records · 8 pollutants + AQI<br>"
                        f"{str(audit.get('first_timestamp_local', ''))[:10]} to "
                        f"{str(audit.get('last_timestamp_local', ''))[:10]} · "
                        f"{audit.get('coverage_pct', 0):.1f}% coverage"), unsafe_allow_html=True)
with b:
    st.markdown(ui.card("WEATHER", "Open-Meteo (ERA5 + forecast)",
                        f"<b>{len(w):,}</b> hourly rows<br>Temperature, humidity, pressure, precipitation, "
                        f"cloud cover, wind"), unsafe_allow_html=True)
with c:
    n_s = data.model_selection().get("n_samples", {})
    st.markdown(ui.card("MODEL TABLE", "Daily feature matrix",
                        f"<b>{len(f):,}</b> days × <b>{f.shape[1]}</b> columns<br>"
                        f"Train {n_s.get('train', '–')} · validation {n_s.get('validation', '–')} · "
                        f"test {n_s.get('test', '–')}"),
                unsafe_allow_html=True)

st.write("")
tabs = st.tabs(["Time series", "Seasonality", "Correlations", "Distributions", "Quality", "Figures", "Raw data"])

POLL = {"pm2_5": "PM2.5", "pm10": "PM10", "no2": "NO₂", "co": "CO", "o3": "O₃", "so2": "SO₂", "nh3": "NH₃",
        "no": "NO"}

with tabs[0]:
    c1, c2, c3 = st.columns([2, 1, 1])
    pols = c1.multiselect("Pollutants", list(POLL), default=["pm2_5", "pm10"], format_func=POLL.get)
    res = c2.segmented_control("Resolution", ["Hourly", "Daily", "Weekly", "Monthly"], default="Daily")
    add_w = c3.toggle("Overlay temperature", value=False)
    rule = {"Hourly": None, "Daily": "D", "Weekly": "W", "Monthly": "MS"}[res or "Daily"]
    s = h.set_index("timestamp_utc")[pols] if pols else pd.DataFrame()
    if rule:
        s = s.resample(rule).mean()
    if s.empty:
        st.caption("Select at least one pollutant.")
    else:
        fig = go.Figure()
        palette = px.colors.qualitative.Bold
        for i, p in enumerate(pols):
            fig.add_trace(go.Scattergl(x=s.index, y=s[p], name=POLL[p], mode="lines",
                                       line=dict(width=1.3 if rule in (None, "D") else 2, color=palette[i % 10])))
        if add_w and not w.empty:
            ws = w.set_index("timestamp_utc")["temp"]
            ws = ws.resample(rule).mean() if rule else ws
            fig.add_trace(go.Scattergl(x=ws.index, y=ws, name="Temperature (°C)", yaxis="y2", mode="lines",
                                       line=dict(color="#94A3B8", width=1)))
            fig.update_layout(yaxis2=dict(overlaying="y", side="right", title="°C", showgrid=False))
        fig.update_yaxes(title_text="Concentration (µg/m³)")
        fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.05))
        if "pm2_5" in pols:
            charts.who_line(fig)
        ui.plot(charts.style(fig, 470))


with tabs[1]:
    d = data.daily_pollution()
    c1, c2 = st.columns(2)
    with c1:
        mm = data.metric_table("eda_monthly_mean_by_year.csv")
        if not mm.empty:
            mm = mm.set_index("month")
            fig = go.Figure()
            for i, y in enumerate([c for c in mm.columns if mm[c].notna().sum() > 3]):
                fig.add_trace(go.Scatter(x=[pd.Timestamp(2000, int(m), 1).strftime("%b") for m in mm.index],
                                         y=mm[y], name=y, mode="lines+markers",
                                         line=dict(width=3 if y == "2025" else 1.8)))
            fig.update_yaxes(title_text="Mean PM2.5 (µg/m³)")
            ui.plot(charts.style(fig, 380, title="Monthly mean PM2.5 by year"))
    with c2:
        hh = h.dropna(subset=["pm2_5"]).copy()
        hh["month"] = hh["date_local"].dt.month
        z = hh.pivot_table(index="month", columns="hour_local", values="pm2_5", aggfunc="mean")
        z.index = [pd.Timestamp(2000, i, 1).strftime("%b") for i in z.index]
        fig = go.Figure(go.Heatmap(z=z.values, x=z.columns, y=z.index, colorscale="YlOrRd",
                                   colorbar=dict(title="µg/m³", thickness=12),
                                   hovertemplate="%{y}, %{x}:00 → %{z:.1f} µg/m³<extra></extra>"))
        fig.update_xaxes(title_text="Local hour", dtick=3)
        fig.update_yaxes(autorange="reversed")
        ui.plot(charts.style(fig, 380, legend=False, title="Average PM2.5 by month and hour"))
    c3, c4 = st.columns(2)
    with c3:
        dd = d.copy()
        dd["dow"] = dd.index.dayofweek
        g = dd.groupby("dow")["pm2_5"].mean()
        fig = go.Figure(go.Bar(x=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], y=g.values,
                               marker_color="#14B8A6", text=g.round(2).values, textposition="outside"))
        fig.update_yaxes(title_text="Mean PM2.5 (µg/m³)")
        ui.plot(charts.style(fig, 320, legend=False, title="Day of week"))
    with c4:
        if not f.empty:
            ff = f[f["target_pm25"].notna()]
            fig = px.scatter(ff, x="wxf_temp_mean", y="target_pm25", color="wxf_wind_mean",
                             color_continuous_scale="Viridis_r", opacity=0.6,
                             labels={"wxf_temp_mean": "Mean temperature of the day (°C)",
                                     "target_pm25": "Daily-mean PM2.5 (µg/m³)", "wxf_wind_mean": "Wind (m/s)"})
            ui.plot(charts.style(fig, 320, title="Temperature, wind and PM2.5"))
    ui.verdict("Winter levels are 3–4× summer levels with an evening peak; weekday effects are small.")

with tabs[2]:
    sp = data.metric_table("eda_spearman_correlation.csv")
    if not sp.empty:
        sp = sp.set_index(sp.columns[0])
        sp.index = [POLL.get(i, i.upper()) for i in sp.index]
        sp.columns = [POLL.get(i, i.upper()) for i in sp.columns]
        a, b = st.columns([1.2, 1])
        with a:
            ui.plot(charts.heatmap(sp, height=460, colorscale="RdBu", reverse=True, zmid=0,
                                   title="Spearman correlation between hourly pollutants"))
        with b:
            if not f.empty:
                cols = ["target_pm25", "pm25_mean_24h", "pm25_last", "pm25_lag_24h", "wxf_temp_mean", "wxf_wind_mean",
                        "wxf_calm_hours", "wxf_hdd", "wx_rh_mean_24h", "wxf_precip_sum"]
                cc = f[cols].corr(method="spearman")["target_pm25"].drop("target_pm25").sort_values()
                fig = go.Figure(go.Bar(x=cc.values, y=[data.feature_text(i) for i in cc.index], orientation="h",
                                       marker_color=["#2563EB" if v < 0 else "#DC2626" for v in cc.values]))
                fig.update_xaxes(title_text="Spearman ρ with next-day PM2.5", range=[-1, 1])
                ui.plot(charts.style(fig, 460, legend=False, title="Correlation with next-day PM2.5"))


with tabs[3]:
    a, b = st.columns(2)
    with a:
        dd = data.daily_pollution()
        fig = go.Figure(go.Histogram(x=dd["pm2_5"], nbinsx=60, marker_color="#0F766E", opacity=0.85))
        fig.add_vline(x=data.WHO_24H, line=dict(color="#DC2626", dash="dot"), annotation_text="WHO 15")
        fig.update_xaxes(title_text="Daily-mean PM2.5 (µg/m³)")
        fig.update_yaxes(title_text="Days")
        ui.plot(charts.style(fig, 360, legend=False, title="Daily-mean distribution"))
        share = (dd["pm2_5"] > data.WHO_24H).mean() * 100
        st.caption(f"{share:.1f}% of days exceed the WHO daily guideline.")
    with b:
        if not f.empty:
            ff = f[f["split"].isin(["train", "validation", "test"]) & f["target_pm25"].notna()]
            fig = px.box(ff, x="split", y="target_pm25", color="split", points=False,
                         category_orders={"split": ["train", "validation", "test"]},
                         color_discrete_map={"train": "#0F766E", "validation": "#7C3AED", "test": "#EA580C"},
                         labels={"target_pm25": "Daily-mean PM2.5 (µg/m³)", "split": ""})
            ui.plot(charts.style(fig, 360, legend=False, title="Train / validation / test"))
            shift = data.metric_table("eda_split_distribution_shift.csv")
            if not shift.empty:
                ui.table(shift.round(2))

with tabs[4]:
    a, b = st.columns([1.3, 1])
    with a:
        comp = data.metric_table("eda_monthly_completeness.csv")
        if not comp.empty:
            fig = go.Figure(go.Bar(x=comp["timestamp_local"], y=comp["pct_hours_observed"],
                                   marker_color=np.where(comp["pct_hours_observed"] < 95, "#F59E0B", "#14B8A6")))
            fig.update_yaxes(title_text="% of hours observed", range=[0, 105])
            ui.plot(charts.style(fig, 340, legend=False, title="Monthly completeness"))
    with b:
        ui.kpi_row([("Coverage", ui.fmt(audit.get("coverage_pct"), 1, " %"), "of expected hours"),
                    ("Valid days", str(audit.get("n_valid_days_min_20h", "–")), "≥ 20 valid hours")])
        st.write("")
        neg = audit.get("n_negative_values", {})
        st.markdown(f"- Duplicated timestamps: **{audit.get('n_duplicated_timestamps', 0)}**\n"
                    f"- Missing hours (kept as gaps): **{audit.get('missing_hours', 0)}**\n"
                    f"- Negative values set to missing: **{sum(neg.values()) if neg else 0}**\n"
                    f"- PM2.5 range: **{audit.get('pm2_5_min', 0)} – {audit.get('pm2_5_max', 0)} µg/m³**\n"
                    "- Extreme values flagged, not removed")
    qp = paths.METRICS / "data_quality_report.md"
    if qp.exists():
        with st.expander("Quality report"):
            st.markdown(qp.read_text(encoding="utf-8"))

with tabs[5]:
    figs = data.figure_files()
    if not figs:
        st.caption("No figures in `artifacts/figures/`.")
    else:
        names = {p: p.stem.split("_", 1)[1].replace("_", " ").capitalize() if "_" in p.stem else p.stem for p in figs}
        pick = st.selectbox("Figure", figs, format_func=lambda p: f"Figure {p.stem.split('_')[0]} · {names[p]}")
        st.image(str(pick))


with tabs[6]:
    which = st.radio("Table", ["Hourly air pollution (clean)", "Hourly weather", "Daily feature table"],
                     horizontal=True)
    df = {"Hourly air pollution (clean)": h, "Hourly weather": w, "Daily feature table": f}[which]
    st.caption(f"{len(df):,} rows × {df.shape[1]} columns · last 500 shown")
    ui.table(df.tail(500), height=380)
    st.download_button("Download full table (CSV)", df.to_csv(index=False).encode("utf-8"),
                       file_name=which.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".csv",
                       mime="text/csv", icon=":material/download:")

ui.footer()
