from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib import charts, data, modeling, ui

ui.page_header("Models", "Scenario Lab",
               "Start from a real 2025 evening, change conditions and see how every trained model responds.")

models = modeling.load_benchmark_models()
if not models:
    st.error("Could not load the saved models from `artifacts/models/benchmark/`.")
    st.stop()
missing = [m for m in modeling.BENCHMARK_MODELS if m not in models]
if missing:
    st.caption("Unavailable: " + ", ".join(data.MODEL_SHORT[m] for m in missing) + " (XGBoost not installed).")

f = data.daily_features()
test = f[(f["split"] == "test") & f["target_pm25"].notna()].set_index("target_date").sort_index()
if test.empty:
    st.error("Feature table not found (`data/processed/daily_pm25_features.csv`).")
    st.stop()

# ------------------------------------------------------------------ base day
presets = {
    "Winter episode": test["target_pm25"].idxmax(),
    "Autumn day": min(test.index, key=lambda d: abs((d - pd.Timestamp("2025-10-20")).days)),
    "Summer day": min(test.index, key=lambda d: abs((d - pd.Timestamp("2025-07-10")).days)),
}
c1, c2 = st.columns([1, 2])
with c1:
    preset = st.radio("Base case", list(presets) + ["Custom date"], index=0)
with c2:
    if preset == "Custom date":
        base_day = st.select_slider("Target date", options=list(test.index),
                                    value=presets["Winter episode"], format_func=lambda d: d.strftime("%d %b %Y"))
    else:
        base_day = presets[preset]
    base = test.loc[base_day]
    st.markdown(f"**{base_day:%d %B %Y}** · observed **{base['target_pm25']:.1f} µg/m³**")


def _val(col: str, default: float) -> float:
    v = base.get(col, np.nan)
    return float(v) if pd.notna(v) else default


# ------------------------------------------------------------------ controls
st.markdown("#### Conditions")
k1, k2, k3 = st.columns(3)
with k1:
    pm = st.slider("PM2.5, last 24 h (µg/m³)", 0.5, 60.0, round(_val("pm25_mean_24h", 5.0), 1), 0.5,
                   help="Scales the whole recent pollution history consistently.")
    heating = st.toggle("Heating season", value=bool(_val("target_is_heating_season", 0)))
with k2:
    temp = st.slider("Temperature (°C)", -35.0, 30.0, round(_val("wxf_temp_mean", 0.0), 1), 0.5)
    precip = st.slider("Precipitation (mm)", 0.0, 20.0, round(_val("wxf_precip_sum", 0.0), 1), 0.5)
with k3:
    wind = st.slider("Wind speed (m/s)", 0.0, 12.0, round(_val("wxf_wind_mean", 3.0), 1), 0.1)
    calm = st.slider("Calm hours (< 2 m/s)", 0, 24, int(round(_val("wxf_calm_hours", 4))), 1)

scen = modeling.apply_scenario(base, pm_level=pm, temp=temp, wind=wind, calm_hours=float(calm),
                               precip=precip, heating=heating)
base_df = pd.DataFrame([base])
P = modeling.predict_all(models, pd.concat([base_df, scen], ignore_index=True))
order = [m for m in data.MODEL_ORDER if m in P.columns]

# ------------------------------------------------------------------ results
st.write("")
left, right = st.columns([1, 1.5])
with left:
    main = "ensemble" if "ensemble" in P.columns else order[0]
    val = float(P.loc[1, main])
    delta = val - float(P.loc[0, main])
    ui.forecast_card(val, data.MODEL_LABELS[main], "Scenario forecast", f"<br>{delta:+.2f} µg/m³ vs base case")
    votes = modeling.ensemble_members(models, scen)
    if votes:
        st.markdown("<div style='font-size:.85rem;color:#5B6476;margin-top:10px'>Ensemble = mean of " + ", ".join(
            f"<b style='color:{data.MODEL_COLORS[k]}'>{data.MODEL_SHORT[k]} {v:.2f}</b>"
            for k, v in votes.items()) + "</div>", unsafe_allow_html=True)
with right:
    fig = go.Figure()
    labels = [data.MODEL_SHORT[m] for m in order]
    fig.add_trace(go.Bar(y=labels, x=P.loc[0, order], orientation="h", name="Base case",
                         marker_color="#CBD5E1"))
    fig.add_trace(go.Bar(y=labels, x=P.loc[1, order], orientation="h", name="Scenario",
                         marker_color=[data.MODEL_COLORS[m] for m in order],
                         text=[f"{v:.2f}" for v in P.loc[1, order]], textposition="outside"))
    fig.add_vline(x=data.WHO_24H, line=dict(color="#DC2626", dash="dot"), annotation_text="WHO 15")
    fig.update_layout(barmode="group")
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title_text="Forecast daily-mean PM2.5 (µg/m³)")
    ui.plot(charts.style(fig, 360))

# ------------------------------------------------------------------ sensitivity curves
st.divider()
st.markdown("#### Response curves")
var = st.segmented_control("Vary", list(modeling.SWEEPS), default="temp",
                           format_func=lambda v: modeling.SWEEPS[v][0].split(" (")[0])
var = var or "temp"
fixed = dict(pm_level=pm, temp=temp, wind=wind, calm_hours=float(calm), precip=precip, heating=heating)
fixed.pop(var, None)
sw = modeling.sweep(models, base, var, **fixed)
fig = go.Figure()
for m in order:
    fig.add_trace(go.Scatter(x=sw[var], y=sw[m], mode="lines", name=data.MODEL_SHORT[m],
                             line=dict(color=data.MODEL_COLORS[m], width=3 if m == "ensemble" else 2,
                                       shape="linear")))
cur = {"pm_level": pm, "temp": temp, "wind": wind, "calm_hours": calm}[var]
fig.add_vline(x=cur, line=dict(color="#0F172A", dash="dash"), annotation_text="current setting")
charts.who_line(fig, label=False)
fig.update_xaxes(title_text=modeling.SWEEPS[var][0])
fig.update_yaxes(title_text="Forecast PM2.5 (µg/m³)", rangemode="tozero")
ui.plot(charts.style(fig, 420))
shapes = [("Ridge", "Linear", "A straight line: weighted sum of inputs."),
          ("RF · XGBoost", "Stepwise", "Trees split inputs at learned thresholds; flat outside the training range."),
          ("SVR", "Smooth", "Curved response from the RBF kernel."),
          ("Ensemble", "Averaged", "Sits between its members and is less jumpy.")]
for c, (m, t, x) in zip(st.columns(4), shapes):
    c.markdown(ui.card(m.upper(), t, x), unsafe_allow_html=True)
st.caption("Models trained on 2021–2023. Related features move together. Responses are associations, "
           "not causal effects.")

ui.footer()
