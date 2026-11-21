from __future__ import annotations

import streamlit as st

from lib import data, store, ui

h = data.headline()
best = h.get("best_model", "ensemble")
n_rec = h.get("n_records") or 0

st.html(f"""
<div style="position:relative;border-radius:20px;overflow:hidden;min-height:330px;background:#0B1220;
            margin-bottom:1.6rem;font-family:Inter,system-ui,sans-serif;">
  <div style="position:absolute;inset:0;">{ui.hero_svg()}</div>
  <div style="position:relative;padding:46px 46px 40px 46px;max-width:560px;">
    <div style="color:#5EEAD4;font-size:.72rem;font-weight:700;letter-spacing:.16em;text-transform:uppercase;">
      Astana · Next-day PM2.5</div>
    <h1 style="font-family:Manrope,Inter,sans-serif;font-size:2.7rem;line-height:1.06;font-weight:800;color:#FFFFFF;
               margin:14px 0 14px 0;letter-spacing:-0.035em;">Tomorrow's air,<br>forecast tonight.</h1>
    <p style="font-size:1rem;color:#A9B4C8;line-height:1.55;margin:0 0 26px 0;max-width:470px;">
      Five machine-learning models trained on five years of pollution and weather data
      predict the daily mean PM2.5 every evening at 18:00.</p>
    <div style="display:flex;gap:34px;">
      <div><div style="font-family:Manrope,sans-serif;font-size:1.5rem;font-weight:800;color:#fff;">
        {ui.fmt(h.get('test_mae'), 2)}</div><div style="font-size:.72rem;color:#7D8AA6;text-transform:uppercase;
        letter-spacing:.08em;">MAE, µg/m³</div></div>
      <div><div style="font-family:Manrope,sans-serif;font-size:1.5rem;font-weight:800;color:#fff;">
        +{ui.fmt(h.get('skill_pct'), 1)}%</div><div style="font-size:.72rem;color:#7D8AA6;text-transform:uppercase;
        letter-spacing:.08em;">vs baseline</div></div>
      <div><div style="font-family:Manrope,sans-serif;font-size:1.5rem;font-weight:800;color:#fff;">
        {n_rec / 1000:.0f}k</div><div style="font-size:.72rem;color:#7D8AA6;text-transform:uppercase;
        letter-spacing:.08em;">hourly records</div></div>
    </div>
  </div>
</div>
""")

left, right = st.columns([1.05, 1], gap="large")
with left:
    fc = store.forecasts_frame()
    live = fc[fc["kind"] != "hindcast"] if not fc.empty else fc
    if not live.empty:
        r = live.sort_values(["target_date", "created_at_utc"], ascending=False).iloc[0]
        ui.forecast_card(float(r["predicted"]), r["target_date"], "Latest forecast",
                         f"<br>{data.MODEL_SHORT.get(r['model'], r['model'])} · issued "
                         f"{str(r['issue_time_local'])[:16].replace('T', ' ')}")
    else:
        lf = data.latest_forecast_file()
        if lf:
            ui.forecast_card(float(lf["predicted_pm25"]), lf["prediction_date"], "Latest forecast")
        else:
            st.info("No forecast yet.")
    st.markdown(ui.aq_scale_html(), unsafe_allow_html=True)
    st.write("")
    st.page_link("views/live_forecast.py", label="Run a new forecast", icon=":material/arrow_forward:")
with right:
    k1, k2 = st.columns(2)
    k1.markdown(ui.kpi("Selected model", data.MODEL_SHORT.get(best, best), "RF + XGBoost + SVR", True),
                unsafe_allow_html=True)
    k2.markdown(ui.kpi("Test R²", ui.fmt(h.get("test_r2"), 2), "346 unseen days of 2025"), unsafe_allow_html=True)
    st.write("")
    k3, k4 = st.columns(2)
    k3.markdown(ui.kpi("Data coverage", ui.fmt(h.get("coverage_pct"), 1, "%"), "2021–2025, hourly"),
                unsafe_allow_html=True)
    k4.markdown(ui.kpi("Input features", "61", "pollution, calendar, weather"), unsafe_allow_html=True)
    st.write("")
    with st.container(border=True):
        a, b = st.columns(2)
        with a:
            st.page_link("views/model_arena.py", label="Leaderboard", icon=":material/leaderboard:")
            st.page_link("views/time_machine.py", label="Replay 2025", icon=":material/history:")
        with b:
            st.page_link("views/what_if.py", label="Scenario Lab", icon=":material/tune:")
            st.page_link("views/track_record.py", label="Track Record", icon=":material/verified:")

st.write("")
st.markdown("### How it works")
steps = [
    ("01", "Collect", "Hourly pollutants from OpenWeatherMap and weather from Open-Meteo."),
    ("02", "Clean", "Gaps kept as gaps, nothing interpolated, 20+ valid hours per day."),
    ("03", "Engineer", "61 leakage-free features available at 18:00."),
    ("04", "Learn", "Four tuned regressors and their ensemble, chronological validation."),
    ("05", "Forecast", "Daily-mean PM2.5 with a 90% band, verified against reality."),
]
for c, (n, t, x) in zip(st.columns(5), steps):
    c.markdown(ui.card(n, t, x), unsafe_allow_html=True)

ui.footer()
