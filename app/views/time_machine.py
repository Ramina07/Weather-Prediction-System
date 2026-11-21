from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib import charts, data, ui

ui.page_header("Models", "Replay 2025",
               "Pick any day of the held-out test year and compare every model with what was observed.")

wide = data.selected_test_predictions("full_weather")
if wide.empty:
    st.error("Test predictions not found (`artifacts/predictions/predictions_test.csv`).")
    st.stop()
models = [m for m in data.MODEL_ORDER if m in wide.columns]
dates = list(wide.index)

# interesting presets
top = wide["actual"].idxmax()
err_ens = (wide.get("ensemble", wide[models[0]]) - wide["actual"]).abs()
best_day = err_ens.idxmin()
worst_day = err_ens.idxmax()
presets = {"Peak day": top, "Best forecast": best_day, "Worst forecast": worst_day,
           "Summer": min(dates, key=lambda d: abs((d - pd.Timestamp("2025-07-15")).days))}

if "tm_date" not in st.session_state:
    st.session_state["tm_date"] = top

pc = st.columns(len(presets))
for c, (label, d) in zip(pc, presets.items()):
    c.button(f"{label} · {d:%d %b}", on_click=st.session_state.update, kwargs={"tm_date": d})

day = st.select_slider("Target day", options=dates, format_func=lambda d: d.strftime("%a %d %b %Y"),
                       key="tm_date")
row = wide.loc[day]
actual = float(row["actual"])

# ------------------------------------------------------------------ result cards
st.write("")
left, right = st.columns([1, 1.6])
with left:
    cat = data.aq_category(actual)
    ui.forecast_card(actual, f"{day:%A, %d %B %Y}", "Observed daily mean",
                     f"<br>Forecasts issued {day - pd.Timedelta(days=1):%d %b}, 18:00.")
with right:
    preds = {m: float(row[m]) for m in models if pd.notna(row[m])}
    errs = {m: abs(v - actual) for m, v in preds.items()}
    winner = min((m for m in errs if m != "persistence"), key=errs.get)
    labels = [data.MODEL_SHORT[m] for m in preds]
    fig = go.Figure(go.Bar(x=list(preds.values()), y=labels, orientation="h",
                           marker_color=[data.MODEL_COLORS[m] for m in preds],
                           text=[f"{v:.2f}  ({v - actual:+.2f})" for v in preds.values()],
                           textposition="outside", hovertemplate="%{y}: %{x:.2f} µg/m³<extra></extra>"))
    fig.add_vline(x=actual, line=dict(color="#0F172A", width=2.5, dash="dash"),
                  annotation_text=f"observed {actual:.2f}", annotation_position="top")
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title_text="µg/m³", range=[0, max(list(preds.values()) + [actual]) * 1.45])
    ui.plot(charts.style(fig, 330, legend=False, title="Forecasts"))

ui.verdict(f"Closest: <b>{data.MODEL_LABELS[winner]}</b>, error {preds[winner] - actual:+.2f} µg/m³. "
           f"Persistence: {preds.get('persistence', float('nan')):.2f} µg/m³.")

# ------------------------------------------------------------------ context window + hourly profile
a, b = st.columns([1.5, 1])
with a:
    st.markdown("#### Surrounding three weeks")
    sel_models = st.pills("Models", models, default=["ensemble", "persistence"] if "ensemble" in models else models[:2],
                          selection_mode="multi", format_func=lambda m: data.MODEL_SHORT[m])
    win = wide.loc[day - pd.Timedelta(days=14): day + pd.Timedelta(days=7)]
    fig = charts.timeseries(win, ["actual"] + list(sel_models or []), height=360, who=True)
    fig.add_vrect(x0=day - pd.Timedelta(hours=12), x1=day + pd.Timedelta(hours=12), fillcolor="#FDE68A",
                  opacity=0.35, line_width=0)
    ui.plot(fig)
with b:
    st.markdown("#### Hourly profile")
    h = data.hourly_pollution()
    hd = h[h["date_local"] == day] if not h.empty else h
    if hd.empty:
        st.caption("No hourly data for this day.")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hd["hour_local"], y=hd["pm2_5"], mode="lines+markers", name="Hourly PM2.5",
                                 line=dict(color="#0F172A", width=2), fill="tozeroy",
                                 fillcolor="rgba(15,23,42,0.06)"))
        fig.add_hline(y=actual, line=dict(color="#0F172A", dash="dash"), annotation_text="daily mean")
        if "ensemble" in preds:
            fig.add_hline(y=preds["ensemble"], line=dict(color=data.MODEL_COLORS["ensemble"], width=2),
                          annotation_text="ensemble forecast", annotation_position="bottom right")
        fig.update_xaxes(title_text="Local hour", dtick=3)
        fig.update_yaxes(title_text="PM2.5 (µg/m³)", rangemode="tozero")
        ui.plot(charts.style(fig, 360, legend=False))


# ------------------------------------------------------------------ weather context
f = data.daily_features()
if not f.empty:
    fr = f[f["target_date"] == day]
    if len(fr):
        r = fr.iloc[0]
        st.markdown("#### Inputs at issue time")
        ui.kpi_row([
            ("PM2.5, last 24 h", ui.fmt(r.get("pm25_mean_24h"), 1, " µg/m³"), "mean at 17:00"),
            ("Temperature", ui.fmt(r.get("wxf_temp_mean"), 1, " °C"), f"min {ui.fmt(r.get('wxf_temp_min'), 1)} °C"),
            ("Wind", ui.fmt(r.get("wxf_wind_mean"), 1, " m/s"), f"{ui.fmt(r.get('wxf_calm_hours'), 0)} calm hours"),
            ("Precipitation", ui.fmt(r.get("wxf_precip_sum"), 1, " mm"), "target day"),
            ("Heating season", "Yes" if r.get("target_is_heating_season") == 1 else "No", "15 Oct – 15 Apr"),
        ])

with st.expander("Closest model by month"):
    w = wide.copy()
    ml = [m for m in models if m != "persistence"]
    w["winner"] = (w[ml].sub(w["actual"], axis=0)).abs().idxmin(axis=1)
    cnt = w.groupby([w.index.month, "winner"]).size().unstack(fill_value=0)
    cnt.index = [pd.Timestamp(2025, i, 1).strftime("%b") for i in cnt.index]
    fig = go.Figure()
    for m in [x for x in ml if x in cnt.columns]:
        fig.add_trace(go.Bar(x=cnt.index, y=cnt[m], name=data.MODEL_SHORT[m], marker_color=data.MODEL_COLORS[m]))
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title_text="Days closest")
    ui.plot(charts.style(fig, 360))
    tot = w["winner"].value_counts()
    st.caption("Full year: " + ", ".join(f"{data.MODEL_SHORT[k]} {v}" for k, v in tot.items()) + " days.")

ui.footer()
