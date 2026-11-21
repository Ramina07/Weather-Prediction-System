from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib import charts, data, modeling, paths, store, ui

ui.page_header("Forecast", "Track record",
               "Every forecast is stored and scored against the observed daily mean once the day is over.")

status = modeling.backend_status()

# automatic verification once per session when finished days are waiting
if status.get("api_key") and not st.session_state.get("auto_verified"):
    st.session_state["auto_verified"] = True
    today = pd.Timestamp.now(tz="Asia/Almaty").strftime("%Y-%m-%d")
    if store.pending_dates(today):
        with st.spinner("Verifying finished days"):
            n_auto, _ = modeling.verify_pending()
        if n_auto:
            st.toast(f"{n_auto} day(s) verified")

c1, c2, c3 = st.columns([1, 1.3, 1.2])
with c1:
    if st.button("Verify finished days", type="primary",
                 disabled=not status.get("api_key"),
                 help="Downloads the observed daily mean for every finished, unverified day."):
        with st.spinner("Downloading observations"):
            n, msgs = modeling.verify_pending()
        if n:
            st.success(f"{n} day(s) verified.")
        for m in msgs:
            st.caption(m)
with c2:
    days = st.select_slider("Hindcast window (days)", options=[1, 2, 3, 4, 5], value=3,
                            help="Re-creates the 18:00 forecasts of the last N days using only data available "
                                 "at each issue time.")
with c3:
    st.write("")
    if st.button("Run hindcast", disabled=not status.get("api_key")):
        with st.spinner("Re-creating past forecasts"):
            res, errs = modeling.run_hindcast(days)
        if res:
            st.success(f"{len(res)} hindcast(s) saved.")
        for e in errs:
            st.caption(e)

if not status.get("api_key"):
    st.caption("Verification and hindcasts require the OpenWeatherMap key in `.env`.")

df = store.forecasts_frame()
if df.empty:
    st.info("No forecasts stored yet.")
    st.stop()

kinds = st.pills("Show", ["live", "hindcast", "imported"], default=["live", "hindcast", "imported"],
                 selection_mode="multi", format_func=lambda k: {"live": "Live", "hindcast": "Hindcast",
                                                              "imported": "CLI"}[k])
sel = df[df["kind"].isin(kinds or [])]
# one forecast per target day: the most recent run
one = sel.sort_values("created_at_utc").groupby("target_date").tail(1).sort_values("target_date")
s = store.summary(one)

st.write("")
ui.kpi_row([
    ("Target days", str(len(one)), f"{len(sel)} forecasts stored"),
    ("Verified", str(s.get("n_verified", 0)), f"{s.get('n_pending', 0)} pending"),
    ("Live MAE", ui.fmt(s.get("mae"), 2, " µg/m³"), f"bias {ui.fmt(s.get('bias'), 2)}"),
    ("Provider MAE", ui.fmt(s.get("provider_mae"), 2, " µg/m³"), "OpenWeatherMap forecast"),
    ("Within 90% band", ui.fmt(100 * s["interval_hit_rate"], 0, "%") if "interval_hit_rate" in s else "–",
     "of verified days"),
])

st.write("")
fig = go.Figure()
x = one["target_date"]
if one["upper"].notna().any():
    fig.add_trace(go.Scatter(x=list(x) + list(x[::-1]), y=list(one["upper"]) + list(one["lower"].fillna(0)[::-1]),
                             fill="toself", fillcolor="rgba(15,118,110,0.12)", line=dict(width=0),
                             name="90 % band", hoverinfo="skip"))
fig.add_trace(go.Scatter(x=x, y=one["predicted"], name="AirCast", mode="lines+markers",
                         line=dict(color=data.MODEL_COLORS["ensemble"], width=2.5), marker=dict(size=9)))
if one["provider_pred"].notna().any():
    fig.add_trace(go.Scatter(x=x, y=one["provider_pred"], name="OpenWeatherMap", mode="markers",
                             marker=dict(color="#F59E0B", size=9, symbol="diamond")))
if one["actual"].notna().any():
    fig.add_trace(go.Scatter(x=x, y=one["actual"], name="Observed", mode="lines+markers",
                             line=dict(color="#0F172A", width=2, dash="dot"), marker=dict(size=10, symbol="x")))
charts.who_line(fig)
fig.update_yaxes(title_text="Daily-mean PM2.5 (µg/m³)", rangemode="tozero")
ui.plot(charts.style(fig, 420))

st.markdown("#### Forecast log")
show = sel[["id", "target_date", "kind", "model", "feature_config", "predicted", "lower", "upper", "provider_pred",
            "actual", "abs_error", "reliable", "issue_time_local", "created_at_utc"]].copy()
show["model"] = show["model"].map(lambda m: data.MODEL_SHORT.get(m, m))
ui.table(show, column_config={
    "id": "ID", "target_date": "Target day", "kind": "Type", "model": "Model", "feature_config": "Features",
    "predicted": st.column_config.NumberColumn("Forecast", format="%.2f"),
    "lower": st.column_config.NumberColumn("Low", format="%.1f"),
    "upper": st.column_config.NumberColumn("High", format="%.1f"),
    "provider_pred": st.column_config.NumberColumn("OWM", format="%.2f"),
    "actual": st.column_config.NumberColumn("Actual", format="%.2f"),
    "abs_error": st.column_config.NumberColumn("|Error|", format="%.2f"),
    "reliable": st.column_config.CheckboxColumn("Reliable"),
    "issue_time_local": "Issued", "created_at_utc": "Saved (UTC)"})

d1, d2 = st.columns([1, 2])
with d1:
    st.download_button("Export CSV", sel.drop(columns=["payload_json"]).to_csv(index=False),
                       file_name="forecast_track_record.csv", mime="text/csv", icon=":material/download:")
with d2:
    with st.popover("Delete entry"):
        fid = st.selectbox("Forecast ID", sel["id"].tolist())
        if st.button("Delete permanently", type="secondary"):
            store.delete_forecast(int(fid))
            st.rerun()

with st.expander("Storage"):
    st.markdown(f"""
SQLite database `{paths.STORE_DB.relative_to(paths.ROOT).as_posix()}`

| Table | Content |
|---|---|
| `forecasts` | issue time, target day, model, prediction, 90% band, provider forecast, quality flags, full JSON |
| `observations` | observed daily mean per finished day (20+ valid hours) |
| `events` | audit log |
""")
    st.markdown("**Activity**")
    ui.table(store.events_frame(30))

ui.footer()
