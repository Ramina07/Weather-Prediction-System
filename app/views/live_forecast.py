from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from lib import charts, data, modeling, store, ui

ui.page_header("Forecast", "Live forecast",
               "Latest OpenWeatherMap observations and the Open-Meteo weather forecast, run through the "
               "production model.")

status = modeling.backend_status()
now_local = pd.Timestamp.now(tz="Asia/Almaty")
after_issue = now_local.hour >= 18

with st.container(border=True):
    c1, c2, c3 = st.columns([1.4, 1, 1])
    with c1:
        target = now_local + pd.Timedelta(days=1) if after_issue else now_local
        st.markdown(f"**Target day: {target:%A, %d %B}**")
        st.caption(f"Astana time {now_local:%H:%M}. Forecasts are issued at 18:00"
                   + ("." if after_issue else "; before that, the previous evening's issue is used."))
    with c2:
        with_provider = st.toggle("Compare with provider forecast", value=True)
    with c3:
        with st.popover("Options"):
            model_choice = st.radio("Model artifact", ["auto", "benchmark", "operational"],
                                    help="auto = operational model if it was trained "
                                         "(`train-operational`), otherwise the benchmark model.")
            force = st.checkbox("Force a forecast even if live data are stale (flagged as unreliable)")
    run = st.button("Run forecast", type="primary",
                    disabled=not status.get("backend"))

if not status.get("api_key"):
    st.caption("API key not configured: add `OPENWEATHER_API_KEY=...` to `.env` to run live forecasts. "
               "Showing the latest saved forecast.")

if run:
    with st.status("Running forecast", expanded=True) as s:
        st.write("Downloading the last 120 hours of observations")
        st.write("Fetching the weather forecast")
        st.write("Building features and predicting")
        res, err = modeling.run_live_forecast("auto", model_choice, with_provider, force)
        if err:
            s.update(label="Forecast failed", state="error")
            st.error(err)
        else:
            s.update(label="Forecast ready", state="complete", expanded=False)
            st.session_state["live_result"] = res
            data.latest_forecast_file.clear()

result = st.session_state.get("live_result") or data.latest_forecast_file()
if not result:
    st.info("No forecast available yet.")
    st.stop()

is_new = "live_result" in st.session_state
pred = float(result["predicted_pm25"])
pi = result.get("prediction_interval") or {}
prov = (result.get("provider_forecast") or {}).get("mean_pm25_target_date")
recent = result.get("recent_pm25_mean_24h")

st.write("")
left, right = st.columns([1.1, 1])
with left:
    extra = ""
    if pi:
        extra += f"<br>90% range {pi['lower']:.1f}–{pi['upper']:.1f} µg/m³"
    pct = pred / data.WHO_24H * 100
    extra += f" · {pct:.0f}% of WHO daily limit"
    ui.forecast_card(pred, f"{pd.Timestamp(result['prediction_date']):%A, %d %B %Y}",
                     "Daily mean PM2.5" + (" · new" if is_new else ""), extra)
    st.markdown(ui.aq_scale_html(), unsafe_allow_html=True)
    pills = [ui.pill("Data complete" if result.get("reliable") else "Flagged",
                     "ok" if result.get("reliable") else "warn"),
             ui.pill(data.MODEL_SHORT.get(result.get('model'), str(result.get('model'))), "info"),
             ui.pill(data.CONFIG_LABELS.get(result.get("feature_config"), str(result.get("feature_config"))), "gray"),
             ui.pill(f"72 h coverage {result.get('pm25_coverage_last_72h_pct')}%", "gray")]
    st.markdown("".join(pills), unsafe_allow_html=True)
    for w in result.get("warnings") or []:
        st.warning(w)
with right:
    ui.plot(charts.gauge(pred, ""))
    labels, values, colors = ["AirCast"], [pred], [data.MODEL_COLORS["ensemble"]]
    if prov is not None:
        labels.append("OpenWeatherMap")
        values.append(float(prov))
        colors.append("#F59E0B")
    if recent is not None:
        labels.append("Last 24 h")
        values.append(float(recent))
        colors.append(data.MODEL_COLORS["persistence"])
    ui.plot(charts.bar_compare(labels, values, colors, height=200, text_suffix=" µg/m³"))

# ------------------------------------------------------------------ how the forecast was made
st.write("")
st.markdown("### How this forecast was made")
ex = result.get("explain") or {}
mname = result.get("model") or "ensemble"
mc = data.model_comparison()
mrow = mc[mc["model"] == mname].iloc[0] if not mc.empty and (mc["model"] == mname).any() else None
fresh = str(result.get("data_freshness_latest_observation_local", ""))[11:16]
steps = [
    ("01", "Observations", f"Last 120 h from OpenWeatherMap, newest record {fresh or '–'}, "
                           f"{result.get('pm25_coverage_last_72h_pct', '–')}% complete."),
    ("02", "Weather", "Open-Meteo forecast for the target day." if "Open-Meteo" in str(result.get("weather_data"))
     else str(result.get("weather_data") or "–")),
    ("03", "Features", f"{ex.get('n_features', len(data.model_card().get('features', [])) or 61)} inputs built "
                       "with the same code as in training."),
    ("04", "Model", f"{data.MODEL_LABELS.get(mname, mname)}, trained on "
                    f"{'2021–2024' if result.get('model_fitted_on') == 'train+validation' else '2021–2023'}."
                    + (f" Test MAE {mrow['test_mae']:.2f} µg/m³." if mrow is not None else "")),
]
for c, (n, t, x) in zip(st.columns(4), steps):
    c.markdown(ui.card(n, t, x), unsafe_allow_html=True)

if ex:
    st.write("")
    inp = ex.get("inputs", {})
    ui.kpi_row([
        ("Last hour PM2.5", ui.fmt(inp.get("pm25_last"), 1, " µg/m³"), "17:00 record"),
        ("Last 24 h mean", ui.fmt(inp.get("pm25_mean_24h"), 1, " µg/m³"), f"max {ui.fmt(inp.get('pm25_max_24h'), 1)}"),
        ("Temperature", ui.fmt(inp.get("wxf_temp_mean"), 1, " °C"), "target-day forecast"),
        ("Wind", ui.fmt(inp.get("wxf_wind_mean"), 1, " m/s"), f"{ui.fmt(inp.get('wxf_calm_hours'), 0)} calm hours"),
        ("Heating season", "Yes" if inp.get("target_is_heating_season") == 1 else "No", "15 Oct – 15 Apr"),
    ])
    st.write("")
    v1, v2 = st.columns(2, gap="large")
    members = ex.get("members") or {}
    with v1:
        if members:
            labels = [data.MODEL_SHORT.get(k, k) for k in members] + ["Ensemble (mean)"]
            vals = list(members.values()) + [pred]
            cols_ = [data.MODEL_COLORS.get(k, "#94A3B8") for k in members] + [data.MODEL_COLORS["ensemble"]]
            ui.plot(charts.bar_compare(labels, vals, cols_, height=260, title="Ensemble members",
                                       text_suffix=" µg/m³"))
            st.caption("The production forecast is the average of these three models.")
    with v2:
        others = ex.get("others") or {}
        cmp_l, cmp_v, cmp_c = ["AirCast ensemble"], [pred], [data.MODEL_COLORS["ensemble"]]
        if "ridge" in others:
            cmp_l.append("Ridge (linear)")
            cmp_v.append(others["ridge"])
            cmp_c.append(data.MODEL_COLORS["ridge"])
        if prov is not None:
            cmp_l.append("OpenWeatherMap")
            cmp_v.append(float(prov))
            cmp_c.append("#F59E0B")
        if "persistence" in others:
            cmp_l.append("Persistence (last 24 h)")
            cmp_v.append(others["persistence"])
            cmp_c.append(data.MODEL_COLORS["persistence"])
        if len(cmp_l) > 1:
            ui.plot(charts.bar_compare(cmp_l, cmp_v, cmp_c, height=260, title="Alternative approaches",
                                       text_suffix=" µg/m³"))
            st.caption("Same inputs, simpler methods and the provider's own forecast.")
else:
    st.caption("Run a forecast to see the inputs and each model's vote.")


st.write("")
a, b = st.columns([1, 1])
with a:
    st.markdown("#### Location")
    co = result.get("coordinates") or {"lat": 51.1694, "lon": 71.4491}
    ui.plot(charts.location_map(co["lat"], co["lon"], pred))
with b:
    st.markdown("#### Details")
    rows = {
        "Location": result.get("location"),
        "Issued (local)": str(result.get("forecast_issued_local", ""))[:16].replace("T", " "),
        "Generated (local)": str(result.get("forecast_generated_at_local", ""))[:16].replace("T", " "),
        "Target": result.get("target"),
        "Protocol": result.get("issue_protocol"),
        "Newest observation": str(result.get("data_freshness_latest_observation_local", ""))[:16].replace("T", " "),
        "Weather inputs": result.get("weather_data"),
        "Model file": result.get("model_artifact"),
        "Model fitted on": result.get("model_fitted_on"),
        "Data source": result.get("data_source"),
    }
    ui.table(pd.DataFrame({"Field": list(rows.keys()), "Value": [str(v) for v in rows.values()]}), height=388)
    st.download_button("Download JSON", json.dumps(result, indent=2, ensure_ascii=False, default=str),
                       file_name=f"forecast_{result['prediction_date']}.json", mime="application/json",
                       icon=":material/download:")

st.divider()
st.markdown("#### Recent forecasts")
fc = store.forecasts_frame()
if fc.empty:
    st.caption("No forecasts stored yet.")
else:
    show = fc.head(10)[["target_date", "kind", "model", "predicted", "lower", "upper", "provider_pred", "actual",
                        "issue_time_local"]].copy()
    show["model"] = show["model"].map(lambda m: data.MODEL_SHORT.get(m, m))
    ui.table(show, column_config={
        "target_date": "Target day", "kind": "Type", "model": "Model",
        "predicted": st.column_config.NumberColumn("Forecast", format="%.2f"),
        "lower": st.column_config.NumberColumn("Low", format="%.1f"),
        "upper": st.column_config.NumberColumn("High", format="%.1f"),
        "provider_pred": st.column_config.NumberColumn("OWM forecast", format="%.2f"),
        "actual": st.column_config.NumberColumn("Actual", format="%.2f"),
        "issue_time_local": "Issued"})
    st.page_link("views/track_record.py", label="Full track record", icon=":material/arrow_forward:")

ui.footer()
